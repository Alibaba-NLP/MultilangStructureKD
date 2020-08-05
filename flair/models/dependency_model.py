import warnings
import logging
from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import Result, store_embeddings
from .biaffine_attention import BiaffineAttention

from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import pdb
import copy
import time

import sys
# sys.path.insert(0,'/home/wangxy/workspace/flair/parser')
# sys.path.append('./flair/parser/modules')

from flair.parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TrilinearScorer
from flair.parser.modules.dropout import IndependentDropout, SharedDropout
from flair.parser.utils.alg import eisner, crf
from flair.parser.utils.metric import Metric
from flair.parser.utils.fn import ispunct, istree, numericalize_arcs
# from flair.parser.utils.fn import ispunct
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)

import torch.nn.functional as F
from .mst_decoder import MST_inference
def process_potential(log_potential):
	# (batch, sent_len+1, sent_len+1) or (batch, sent_len+1, sent_len+1, labels)
	
	# (batch, sent_len)
	root_score = log_potential[:,1:,0]
	# convert (dependency, head) to (head, dependency)
	# (batch, sent_len, sent_len)
	log_potential = log_potential.transpose(1,2)[:,1:,1:]
	batch, sent_len = log_potential.shape[:2]
	# Remove the <ROOT> and put the root probability in the diagonal part 
	log_potential[:,torch.arange(sent_len),torch.arange(sent_len)] = root_score
	return log_potential


def get_struct_predictions(dist):
	# (batch, sent_len, sent_len) | head, dep
	argmax_val = dist.argmax
	batch, sent_len, _ = argmax_val.shape
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]).type_as(argmax_val)
	res_val[:,1:,1:] = argmax_val
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val.argmax(-1)

def convert_score_back(marginals):
	# (batch, sent_len, sent_len) | head, dep
	batch = marginals.shape[0]
	sent_len = marginals.shape[1]
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]+list(marginals.shape[3:])).type_as(marginals)
	res_val[:,1:,1:] = marginals
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val



def is_punctuation(word, pos, punct_set=None):
	if punct_set is None:
		return is_uni_punctuation(word)
	else:
		return pos in punct_set

import uuid
uid = uuid.uuid4().hex[:6]
  


log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
	return var.view(-1).detach().tolist()[0]


def argmax(vec):
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)


def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
	_, idx = torch.max(vecs, 1)
	return idx


def log_sum_exp_batch(vecs):
	maxi = torch.max(vecs, 1)[0]
	maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
	recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
	return maxi + recti_

def log_sum_exp_vb(vec, m_size):
	"""
	calculate log of exp sum

	args:
		vec (batch_size, vanishing_dim, hidden_dim) : input tensor
		m_size : hidden_dim
	return:
		batch_size, hidden_dim
	"""
	_, idx = torch.max(vec, 1)  # B * 1 * M
	max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

	return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
																												m_size)  # B * M

def pad_tensors(tensor_list):
	ml = max([x.shape[0] for x in tensor_list])
	shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
	template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
	lens_ = [x.shape[0] for x in tensor_list]
	for i, tensor in enumerate(tensor_list):
		template[i, : lens_[i]] = tensor

	return template, lens_


# Part of Codes are from https://github.com/yzhangcs/biaffine-parser
class SemanticDependencyParser(flair.nn.Model):
	def __init__(
		self,
		hidden_size: int,
		embeddings: TokenEmbeddings,
		tag_dictionary: Dictionary,
		tag_type: str,
		use_crf: bool = False,
		use_rnn: bool = False,
		train_initial_hidden_state: bool = False,
		punct: bool = False, # ignore all punct in default
		tree: bool = False, # keep the dpendency with tree structure
		n_mlp_arc = 500,
		n_mlp_rel = 100,
		mlp_dropout = .33,
		use_second_order = False,
		token_loss = False,
		n_mlp_sec = 150,
		init_std = 0.25,
		factorize = True,
		use_sib = True,
		use_gp = True,
		use_cop = False,
		iterations = 3,
		binary = True,
		is_mst = False,
		rnn_layers: int = 3,
		lstm_dropout: float = 0.33,
		dropout: float = 0.0,
		word_dropout: float = 0.33,
		locked_dropout: float = 0.5,
		pickle_module: str = "pickle",
		interpolation: float = 0.5,
		factorize_interpolation: float = 0.025,
		config = None,
		use_decoder_timer = True,
		debug = False,
		target_languages = 1,
		word_map = None,
		char_map = None,
		relearn_embeddings = False,
		distill_arc: bool = False,
		distill_rel: bool = False,
		distill_crf: bool = False,
		distill_posterior: bool = False,
		distill_prob: bool = False,
		distill_factorize: bool = False,
		crf_attention: bool = False,
		temperature: float = 1,
		diagonal: bool = False,
		is_srl: bool = False,
	):
		"""
		Initializes a SequenceTagger
		:param hidden_size: number of hidden states in RNN
		:param embeddings: word embeddings used in tagger
		:param tag_dictionary: dictionary of tags you want to predict
		:param tag_type: string identifier for tag type
		:param use_crf: if True use CRF decoder, else project directly to tag space
		:param use_rnn: if True use RNN layer, otherwise use word embeddings directly
		:param rnn_layers: number of RNN layers
		:param dropout: dropout probability
		:param word_dropout: word dropout probability
		:param locked_dropout: locked dropout probability
		:param distill_crf: CRF information distillation
		:param crf_attention: use CRF distillation weights
		:param biaf_attention: use bilinear attention for word-KD distillation
		"""

		super(SemanticDependencyParser, self).__init__()
		self.debug = False
		self.biaf_attention = False
		self.token_level_attention = False
		self.use_language_attention = False
		self.use_language_vector = False
		self.use_crf = use_crf
		self.use_decoder_timer = False
		self.sentence_level_loss = False
		self.train_initial_hidden_state = train_initial_hidden_state
		#add interpolation for target loss and distillation loss
		self.token_loss = token_loss

		self.interpolation = interpolation
		self.debug = debug
		self.use_rnn = use_rnn
		self.hidden_size = hidden_size

		self.rnn_layers: int = rnn_layers
		self.embeddings = embeddings
		self.config = config
		self.punct = punct 
		self.punct_list = ['``', "''", ':', ',', '.', 'PU', 'PUNCT']
		self.tree = tree
		self.is_mst = is_mst
		self.is_srl = is_srl
		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)

		self.word_map = word_map
		self.char_map = char_map
		
		self.diagonal = diagonal

		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None

		# dropouts
		self.use_dropout: float = dropout
		self.use_word_dropout: float = word_dropout
		self.use_locked_dropout: float = locked_dropout

		self.pickle_module = pickle_module

		if dropout > 0.0:
			self.dropout = torch.nn.Dropout(dropout)

		if word_dropout > 0.0:
			self.word_dropout = flair.nn.WordDropout(word_dropout)

		if locked_dropout > 0.0:
			self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

		rnn_input_dim: int = self.embeddings.embedding_length

		self.relearn_embeddings: bool = relearn_embeddings

		if self.relearn_embeddings:
			self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

		self.bidirectional = True
		self.rnn_type = "LSTM"
		if not self.use_rnn:
			self.bidirectional = False
		# bidirectional LSTM on top of embedding layer
		num_directions = 1

		# hiddens
		self.n_mlp_arc = n_mlp_arc
		self.n_mlp_rel = n_mlp_rel
		self.mlp_dropout = mlp_dropout
		self.n_mlp_sec = n_mlp_sec
		self.init_std = init_std
		self.lstm_dropout = lstm_dropout
		self.factorize = factorize
		# Initialization of Biaffine Parser
		self.embed_dropout = IndependentDropout(p=word_dropout)
		if self.use_rnn:
			self.rnn = BiLSTM(input_size=rnn_input_dim,
							   hidden_size=hidden_size,
							   num_layers=self.nlayers,
							   dropout=self.lstm_dropout)
			self.lstm_dropout_func = SharedDropout(p=self.lstm_dropout)
			# num_directions = 2 if self.bidirectional else 1

			# if self.rnn_type in ["LSTM", "GRU"]:

			#   self.rnn = getattr(torch.nn, self.rnn_type)(
			#       rnn_input_dim,
			#       hidden_size,
			#       num_layers=self.nlayers,
			#       dropout=0.0 if self.nlayers == 1 else 0.5,
			#       bidirectional=True,
			#   )
			#   # Create initial hidden state and initialize it
			#   if self.train_initial_hidden_state:
			#       self.hs_initializer = torch.nn.init.xavier_normal_

			#       self.lstm_init_h = Parameter(
			#           torch.randn(self.nlayers * num_directions, self.hidden_size),
			#           requires_grad=True,
			#       )

			#       self.lstm_init_c = Parameter(
			#           torch.randn(self.nlayers * num_directions, self.hidden_size),
			#           requires_grad=True,
			#       )

			#       # TODO: Decide how to initialize the hidden state variables
			#       # self.hs_initializer(self.lstm_init_h)
			#       # self.hs_initializer(self.lstm_init_c)

			# final linear map to tag space
			mlp_input_hidden = hidden_size * 2
		else:
			mlp_input_hidden = rnn_input_dim

		# the MLP layers
		self.mlp_arc_h = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_arc,
							 dropout=mlp_dropout)
		self.mlp_arc_d = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_arc,
							 dropout=mlp_dropout)
		self.mlp_rel_h = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_rel,
							 dropout=mlp_dropout)
		self.mlp_rel_d = MLP(n_in=mlp_input_hidden,
							 n_hidden=n_mlp_rel,
							 dropout=mlp_dropout)
		# the Biaffine layers
		self.arc_attn = Biaffine(n_in=n_mlp_arc,
								 bias_x=True,
								 bias_y=False)
		self.rel_attn = Biaffine(n_in=n_mlp_rel,
								 n_out=self.tagset_size,
								 bias_x=True,
								 bias_y=True,
								 diagonal=self.diagonal,)
		self.binary = binary
		# the Second Order Parts
		self.use_second_order=use_second_order
		self.iterations=iterations
		self.use_sib = use_sib
		self.use_cop = use_cop
		self.use_gp = use_gp
		if self.use_second_order:
			if use_sib:
				self.mlp_sib_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_sib_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_sib = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_cop:
				self.mlp_cop_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_cop_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_cop = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_gp:
				self.mlp_gp_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_gp_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_gp_hd = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_gp = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
				

		# self.pad_index = pad_index
		# self.unk_index = unk_index
		self.rel_criterion = nn.CrossEntropyLoss()
		self.arc_criterion = nn.CrossEntropyLoss()

		if self.binary:
			self.rel_criterion = nn.CrossEntropyLoss(reduction='none')
			self.arc_criterion = nn.BCEWithLogitsLoss(reduction='none')

		self.to(flair.device)


	def _init_model_with_state_dict(state):
		use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
		use_word_dropout = (
			0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
		)
		use_locked_dropout = (
			0.0
			if not "use_locked_dropout" in state.keys()
			else state["use_locked_dropout"]
		)
		if 'biaf_attention' in state:
			biaf_attention = state['biaf_attention']
		else:
			biaf_attention = False
		if 'token_level_attention' in state:
			token_level_attention = state['token_level_attention']
		else:
			token_level_attention = False
		if 'teacher_hidden' in state:
			teacher_hidden = state['teacher_hidden']
		else:
			teacher_hidden = 256
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False

		model = SemanticDependencyParser(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_rnn=state["use_rnn"],
			tree=state["tree"],
			punct=state["punct"],
			train_initial_hidden_state=state["train_initial_hidden_state"],
			n_mlp_arc = state["n_mlp_arc"],
			n_mlp_rel = state["n_mlp_rel"],
			mlp_dropout = state["mlp_dropout"],
			token_loss = False if 'token_loss' not in state else state["token_loss"],
			use_second_order = state["use_second_order"],
			n_mlp_sec = state["n_mlp_sec"],
			init_std = state["init_std"],
			factorize = state["factorize"],
			use_sib = state["use_sib"],
			use_gp = state["use_gp"],
			use_cop = state["use_cop"],
			iterations = state["iterations"],
			is_mst = False if "is_mst" not in state else state["is_mst"],
			binary = state["binary"],
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			config=state['config'] if "config" in state else None,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			relearn_embeddings = True if 'relearn_embeddings' not in state else state['relearn_embeddings'],
			diagonal = False if 'diagonal' not in state else state['diagonal'],
		)
		model.load_state_dict(state["state_dict"])
		return model
	def _get_state_dict(self):
		model_state = {
			"state_dict": self.state_dict(),
			"embeddings": self.embeddings,
			"hidden_size": self.hidden_size,
			"tag_dictionary":self.tag_dictionary,
			"tag_type":self.tag_type,
			"use_crf": self.use_crf,
			"use_rnn":self.use_rnn,
			"tree":self.tree,
			"punct":self.punct,
			"train_initial_hidden_state": self.train_initial_hidden_state,
			"n_mlp_arc": self.n_mlp_arc,
			"n_mlp_rel": self.n_mlp_rel,
			"mlp_dropout": self.mlp_dropout,
			"token_loss": self.token_loss,
			"use_second_order": self.use_second_order,
			"n_mlp_sec": self.n_mlp_sec,
			"init_std": self.init_std,
			"factorize": self.factorize,
			"use_sib": self.use_sib,
			"use_gp": self.use_gp,
			"use_cop": self.use_cop,
			"iterations": self.iterations,
			"is_mst": self.is_mst,
			"binary": self.binary,
			"rnn_layers": self.rnn_layers,
			"dropout": self.use_dropout,
			"word_dropout": self.use_word_dropout,
			"locked_dropout": self.use_locked_dropout,
			"config": self.config,
			"word_map": self.word_map,
			"char_map": self.char_map,
			"relearn_embeddings": self.relearn_embeddings,
			"diagonal": self.diagonal,
		}
		return model_state
	def forward(self, sentences: List[Sentence]):
		self.zero_grad()

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		longest_token_sequence_in_batch: int = max(lengths)

		self.embeddings.embed(sentences)
		sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sentences.features],-1)
		if hasattr(self,'keep_embedding'):	
			sentence_tensor = [sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())]
			embedding_name = sorted(sentences.features.keys())[self.keep_embedding]
			if 'forward' in embedding_name or 'backward' in embedding_name:
				# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys()) if 'forward' in x or 'backward' in x],-1)
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if 'forward' not in x and 'backward' not in x:
						sentence_tensor[idx].fill_(0)
			else:
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if x != embedding_name:
						sentence_tensor[idx].fill_(0)
			sentence_tensor = torch.cat(sentence_tensor,-1)
		sentence_tensor = self.embed_dropout(sentence_tensor)[0]

		# if hasattr(sentences,'sentence_tensor'):
		#   sentence_tensor = getattr(sentences,'sentence_tensor').to(flair.device)

		#   if self.debug:
		#       pdb.set_trace()
		# else:

		#   self.embeddings.embed(sentences)
		#   sentence_tensor = torch.cat([sentences.features[x] for x in sentences.features],-1)
		#   # # initialize zero-padded word embeddings tensor
		#   # sentence_tensor = torch.zeros(
		#   #   [
		#   #       len(sentences),
		#   #       longest_token_sequence_in_batch,
		#   #       self.embeddings.embedding_length,
		#   #   ],
		#   #   dtype=torch.float,
		#   #   device=flair.device,
		#   # )

		#   # for s_id, sentence in enumerate(sentences):
		#   #   # fill values with word embeddings
		#   #   sentence_tensor[s_id][: len(sentence)] = torch.cat(
		#   #       [token.get_embedding().unsqueeze(0) for token in sentence], 0
		#   #   )
		#   flag=1
		#   for embedding in self.embeddings.embeddings:
		#       if embedding.static_embeddings==False:
		#           flag=0
		#           break

		#   
		#   if flag:
		#       setattr(sentences,'sentence_tensor',sentence_tensor.clone().cpu())
		#       # setattr(sentences,'sentence_tensor',sentence_tensor)
		# TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode
		# sentence_tensor = sentence_tensor.transpose_(0, 1)

		# --------------------------------------------------------------------
		# FF PART
		# --------------------------------------------------------------------
		# if self.use_dropout > 0.0:
		#   sentence_tensor = self.dropout(sentence_tensor)
		# if self.use_word_dropout > 0.0:
		#   sentence_tensor = self.word_dropout(sentence_tensor)
		# if self.use_locked_dropout > 0.0:
		#   sentence_tensor = self.locked_dropout(sentence_tensor)

		if self.relearn_embeddings:
			sentence_tensor = self.embedding2nn(sentence_tensor)
			# sentence_tensor = self.embedding2nn(sentence_tensor)

		if self.use_rnn:
			x = pack_padded_sequence(sentence_tensor, lengths, True, False)
			x, _ = self.rnn(x)
			sentence_tensor, _ = pad_packed_sequence(x, True, total_length=sentence_tensor.shape[1])
			sentence_tensor = self.lstm_dropout_func(sentence_tensor)
			
			# packed = torch.nn.utils.rnn.pack_padded_sequence(
			#   sentence_tensor, lengths, enforce_sorted=False
			# )

			# # if initial hidden state is trainable, use this state
			# if self.train_initial_hidden_state:
			#   initial_hidden_state = [
			#       self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
			#       self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
			#   ]
			#   rnn_output, hidden = self.rnn(packed, initial_hidden_state)
			# else:
			#   rnn_output, hidden = self.rnn(packed)

			# sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
			#   rnn_output, batch_first=True
			# )

			# if self.use_dropout > 0.0:
			#   sentence_tensor = self.dropout(sentence_tensor)
			# # word dropout only before LSTM - TODO: more experimentation needed
			# # if self.use_word_dropout > 0.0:
			# #     sentence_tensor = self.word_dropout(sentence_tensor)
			# if self.use_locked_dropout > 0.0:
			#   sentence_tensor = self.locked_dropout(sentence_tensor)
		# get the mask and lengths of given batch
		mask=self.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).cuda().type_as(sentence_tensor)
		self.mask=mask
		# mask = words.ne(self.pad_index)
		# lens = mask.sum(dim=1)

		# get outputs from embedding layers
		x = sentence_tensor

		# apply MLPs to the BiLSTM output states
		arc_h = self.mlp_arc_h(x)
		arc_d = self.mlp_arc_d(x)
		rel_h = self.mlp_rel_h(x)
		rel_d = self.mlp_rel_d(x)

		# get arc and rel scores from the bilinear attention
		# [batch_size, seq_len, seq_len]
		s_arc = self.arc_attn(arc_d, arc_h)
		# [batch_size, seq_len, seq_len, n_rels]
		s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

		# add second order using mean field variational inference
		if self.use_second_order:
			mask_unary, mask_sib, mask_cop, mask_gp = self.from_mask_to_3d_mask(mask)
			unary = mask_unary*s_arc
			arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
			layer_sib, layer_cop, layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
			s_arc = self.mean_field_variational_infernece(unary, layer_sib, layer_cop, layer_gp) 
		# set the scores that exceed the length of each sentence to -inf
		if not self.binary:
			s_arc.masked_fill_(~mask.unsqueeze(1).bool(), float(-1e9))
		return s_arc, s_rel

	def mean_field_variational_infernece(self, unary, layer_sib=None, layer_cop=None, layer_gp=None):
		layer_gp2 = layer_gp.permute(0,2,3,1)
		# modify from (dep, head) to (head, dep), in order to fit my code
		unary = unary.transpose(1,2)
		unary_potential = unary.clone()
		q_value = unary_potential.clone()
		for i in range(self.iterations):
			if self.binary:
				q_value=torch.sigmoid(q_value)
			else:
				q_value=F.softmax(q_value,1)
			if self.use_sib:
				second_temp_sib = torch.einsum('nac,nabc->nab', (q_value, layer_sib))
				#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a) 
				diag_sib1 = torch.diagonal(q_value,dim1=1,dim2=2).unsqueeze(-1) * torch.diagonal(layer_sib.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# (n x ma x mb x mc) -> (n x ma x mb)
				#Q(a,b)*p(a,b,b)
				diag_sib2 = q_value * torch.diagonal(layer_sib,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
			else:
				second_temp_sib=0

			if self.use_gp:
				second_temp_gp = torch.einsum('nbc,nabc->nab', (q_value, layer_gp))
				second_temp_gp2 = torch.einsum('nca,nabc->nab', (q_value, layer_gp2))
				#Q(b,a)*p(a,b,a)
				diag_gp1 = q_value.transpose(1,2) * torch.diagonal(layer_gp.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_gp2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_gp,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a)
				diag_gp21 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(-1) * torch.diagonal(layer_gp2.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,a)*p(a,b,b)
				diag_gp22 = q_value.transpose(1,2) * torch.diagonal(layer_gp2,dim1=-2,dim2=-1)

				second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
				#c->a->b
				second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
			else:
				second_temp_gp=second_temp_gp2=0

			if self.use_cop:
				second_temp_cop = torch.einsum('ncb,nabc->nab', (q_value, layer_cop))
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,b)*p(a,b,a)
				diag_cop1 = q_value * torch.diagonal(layer_cop.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# diag_cop1 = q_value * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,2,1,3])),perm=[0,2,1])
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_cop2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_cop,dim1=-2,dim2=-1)
				# diag_cop2 = tf.expand_dims(tf.linalg.diag_part(q_value),1) * tf.linalg.diag_part(layer_cop)
				second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
			else:
				second_temp_cop=0

			second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
			q_value = unary_potential + second_temp
		# transpose from (head, dep) to (dep, head)
		return q_value.transpose(1,2)

	def encode_second_order(self, memory_bank):

		if self.use_sib:
			edge_node_sib_h = self.mlp_sib_h(memory_bank)
			edge_node_sib_m = self.mlp_sib_d(memory_bank)
			arc_sib=(edge_node_sib_h, edge_node_sib_m)
		else:
			arc_sib=None

		if self.use_cop:
			edge_node_cop_h = self.mlp_cop_h(memory_bank)
			edge_node_cop_m = self.mlp_cop_d(memory_bank)
			arc_cop=(edge_node_cop_h, edge_node_cop_m)
		else:
			arc_cop=None

		if self.use_gp:
			edge_node_gp_h = self.mlp_gp_h(memory_bank)
			edge_node_gp_m = self.mlp_gp_d(memory_bank)
			edge_node_gp_hm = self.mlp_gp_hd(memory_bank)
			arc_gp=(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m)
		else:
			arc_gp=None

		return arc_sib, arc_cop, arc_gp

	def get_edge_second_order_node_scores(self, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp):

		if self.use_sib:
			edge_node_sib_h, edge_node_sib_m = arc_sib
			layer_sib = self.trilinear_sib(edge_node_sib_h, edge_node_sib_m, edge_node_sib_m) * mask_sib
			# keep (ma x mb x mc) -> (ma x mb x mb)
			#layer_sib = 0.5 * (layer_sib + layer_sib.transpose(3,2))
			one_mask=torch.ones(layer_sib.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_sib*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			
		else:
			layer_sib = None
		if self.use_cop:
			edge_node_cop_h, edge_node_cop_m = arc_cop
			layer_cop = self.trilinear_cop(edge_node_cop_h, edge_node_cop_m, edge_node_cop_h) * mask_cop
			# keep (ma x mb x mc) -> (ma x mb x ma)
			one_mask=torch.ones(layer_cop.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_cop=layer_cop.transpose(1,2)
			layer_cop = layer_cop-layer_cop*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_cop*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			layer_cop=layer_cop.transpose(1,2)
		else:
			layer_cop = None

		if self.use_gp:
			edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m = arc_gp
			layer_gp = self.trilinear_gp(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m) * mask_gp
		else:
			layer_gp = None
		
		return layer_sib,layer_cop,layer_gp

	def from_mask_to_3d_mask(self,token_weights):
		root_weights = token_weights.clone()
		root_weights[:,0] = 0
		token_weights3D = token_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		token_weights2D = root_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		# abc -> ab,ac
		#token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
		#abc -> ab,cb
		if self.use_cop:
			token_weights_cop = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * token_weights.unsqueeze(1).unsqueeze(1)
			token_weights_cop[:,0,:,0] = 0
		else:
			token_weights_cop=None
		#data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
		#abc -> ab, bc
		if self.use_gp:
			token_weights_gp = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_gp = None

		if self.use_sib:
			#abc -> ca, ab
			if self.use_gp:
				token_weights_sib = token_weights_gp.clone()
			else:
				token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_sib = None
		return token_weights3D, token_weights_sib, token_weights_cop, token_weights_gp



	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
	) -> torch.tensor:
		s_arc, s_rel = self.forward(data_points)
		# lengths = [len(sentence.tokens) for sentence in data_points]
		# longest_token_sequence_in_batch: int = max(lengths)

		# max_len = features.shape[1]
		# mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
		loss = self._calculate_loss(s_arc, s_rel, data_points, self.mask)
		return loss

	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))

	def _calculate_loss(
		self, arc_scores: torch.tensor, rel_scores: torch.tensor, sentences: List[Sentence], mask: torch.tensor, return_arc_rel = False,
	) -> float:
		if self.binary:
			root_mask = mask.clone()
			root_mask[:,0] = 0
			binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
			# arc_mat=
			if hasattr(sentences,self.tag_type+'_arc_tags'):
				arc_mat=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).float()
			else:
				arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).float()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rel_mat=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()
			else:
				rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			
			arc_loss = self.arc_criterion(arc_scores, arc_mat)
			rel_loss = self.rel_criterion(rel_scores.reshape(-1,self.tagset_size), rel_mat.reshape(-1))
			arc_loss = (arc_loss*binary_mask).sum()/binary_mask.sum()

			rel_mask = (rel_mat>0)*binary_mask
			num_rels=rel_mask.sum()
			if num_rels>0:
				rel_loss = (rel_loss*rel_mask.view(-1)).sum()/num_rels
			else:
				rel_loss = 0
			# rel_loss = (rel_loss*rel_mat.view(-1)).sum()/rel_mat.sum()
		else:
			if hasattr(sentences,self.tag_type+'_arc_tags'):
				arcs=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).long()
			else:
				arcs=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).long()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rels=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()
			else:
				rels=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			self.arcs=arcs
			self.rels=rels
			mask[:,0] = 0
			mask = mask.bool()
			gold_arcs = arcs[mask]
			rel_scores, rels = rel_scores[mask], rels[mask]
			rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]
			if self.use_crf:
				arc_loss, arc_probs = crf(arc_scores, mask, arcs)
				arc_loss = arc_loss/mask.sum()
				rel_loss = self.rel_criterion(rel_scores, rels)

				#=============================================================================================
				# dist=generate_tree(arc_scores,mask,is_mst=self.is_mst)
				# labels = dist.struct.to_parts(arcs[:,1:], lengths=mask.sum(-1)).type_as(arc_scores)
				# log_prob = dist.log_prob(labels)
				# if (log_prob>0).any():
					
				#   log_prob[torch.where(log_prob>0)]=0
				#   print("failed to get correct loss!")
				# if self.token_loss:
				#   arc_loss = - log_prob.sum()/mask.sum()
				# else:
				#   arc_loss = - log_prob.mean()
				
				# self.dist=dist
				
				# rel_loss = self.rel_criterion(rel_scores, rels)
				# if self.token_loss:
				#   rel_loss = rel_loss.mean()
				# else:
				#   rel_loss = rel_loss.sum()/len(sentences)

				# if self.debug:
				#   if rel_loss<0 or arc_loss<0:
				#       pdb.set_trace()
				#=============================================================================================
			else:
				arc_scores, arcs = arc_scores[mask], arcs[mask]
				arc_loss = self.arc_criterion(arc_scores, arcs)
			
				# rel_scores, rels = rel_scores[mask], rels[mask]
				# rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
				
				rel_loss = self.rel_criterion(rel_scores, rels)
		if return_arc_rel:
			return (arc_loss,rel_loss)
		loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)


		# score = torch.nn.functional.cross_entropy(features.view(-1,features.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)



		# if self.sentence_level_loss or self.use_crf:
		#   score = score.sum()/features.shape[0]
		# else:
		#   score = score.sum()/mask.sum()
			
		#   score = (1-self.posterior_interpolation) * score + self.posterior_interpolation * posterior_score
		return loss

	def evaluate(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
		prediction_mode: bool = False,
	) -> (Result, float):
		data_loader.assign_embeddings()
		with torch.no_grad():
			if self.binary:
				eval_loss = 0

				batch_no: int = 0

				# metric = Metric("Evaluation")
				# sentence_writer = open('temps/'+str(uid)+'_eval'+'.conllu','w')
				lines: List[str] = []
				utp = 0
				ufp = 0
				ufn = 0
				ltp = 0
				lfp = 0
				lfn = 0
				for batch in data_loader:
					batch_no += 1
					
					arc_scores, rel_scores = self.forward(batch)
					mask=self.mask
					root_mask = mask.clone()
					root_mask[:,0] = 0
					binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
					
					arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
					rel_predictions = (rel_scores.softmax(-1)*binary_mask.unsqueeze(-1)).argmax(-1)
					if not prediction_mode:
						arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in batch],0).float()
						rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in batch],0).long()
						loss = self._calculate_loss(arc_scores, rel_scores, batch, mask)
						if self.is_srl:
							# let the head selection fixed to the gold predicate only
							binary_mask[:,:,0] = arc_mat[:,:,0]
							arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
							

						# UF1
						true_positives = arc_predictions * arc_mat
						# (n x m x m) -> ()
						n_predictions = arc_predictions.sum()
						n_unlabeled_predictions = n_predictions
						n_targets = arc_mat.sum()
						n_unlabeled_targets = n_targets
						n_true_positives = true_positives.sum()
						# () - () -> ()
						n_false_positives = n_predictions - n_true_positives
						n_false_negatives = n_targets - n_true_positives
						# (n x m x m) -> (n)
						n_targets_per_sequence = arc_mat.sum([1,2])
						n_true_positives_per_sequence = true_positives.sum([1,2])
						# (n) x 2 -> ()
						n_correct_sequences = (n_true_positives_per_sequence==n_targets_per_sequence).sum()
						utp += n_true_positives
						ufp += n_false_positives
						ufn += n_false_negatives

						# LF1
						# (n x m x m) (*) (n x m x m) -> (n x m x m)
						true_positives = (rel_predictions == rel_mat) * arc_predictions
						correct_label_tokens = (rel_predictions == rel_mat) * arc_mat
						# (n x m x m) -> ()
						# n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
						# n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
						n_true_positives = true_positives.sum()
						n_correct_label_tokens = correct_label_tokens.sum()
						# () - () -> ()
						n_false_positives = n_unlabeled_predictions - n_true_positives
						n_false_negatives = n_unlabeled_targets - n_true_positives
						# (n x m x m) -> (n)
						n_targets_per_sequence = arc_mat.sum([1,2])
						n_true_positives_per_sequence = true_positives.sum([1,2])
						n_correct_label_tokens_per_sequence = correct_label_tokens.sum([1,2])
						# (n) x 2 -> ()
						n_correct_sequences = (n_true_positives_per_sequence == n_targets_per_sequence).sum()
						n_correct_label_sequences = ((n_correct_label_tokens_per_sequence == n_targets_per_sequence)).sum()
						ltp += n_true_positives
						lfp += n_false_positives
						lfn += n_false_negatives

						eval_loss += loss

					if out_path is not None:
						masked_arc_scores = arc_scores.masked_fill(~binary_mask.bool(), float(-1e9))
						# if self.target
						# lengths = [len(sentence.tokens) for sentence in batch]
						
						# temp_preds = eisner(arc_scores, mask)
						if not self.is_mst:
							temp_preds = eisner(arc_scores, root_mask.bool())
						for (sent_idx, sentence) in enumerate(batch):
							if self.is_mst:
								preds=MST_inference(torch.softmax(masked_arc_scores[sent_idx],-1).cpu().numpy(), len(sentence), binary_mask[sent_idx].cpu().numpy())
							else:
								preds=temp_preds[sent_idx]

							for token_idx, token in enumerate(sentence):
								if token_idx == 0:
									continue

								# append both to file for evaluation
								arc_heads = torch.where(arc_predictions[sent_idx,token_idx]>0)[0]
								if preds[token_idx] not in arc_heads:
									val=torch.zeros(1).type_as(arc_heads)
									val[0]=preds[token_idx].item()
									arc_heads=torch.cat([arc_heads,val],0)
								if len(arc_heads) == 0:
									arc_heads = masked_arc_scores[sent_idx,token_idx].argmax().unsqueeze(0)
								rel_index = rel_predictions[sent_idx,token_idx,arc_heads]
								rel_labels = [self.tag_dictionary.get_item_for_index(x) for x in rel_index]
								arc_list=[]
								for i, label in enumerate(rel_labels):
									if '+' in label:
										labels = label.split('+')
										for temp_label in labels:
											arc_list.append(str(arc_heads[i].item())+':'+temp_label)
									else:
										arc_list.append(str(arc_heads[i].item())+':'+label)
								eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
									token_idx,
									token.text,
									'X',
									'X',
									'X',
									'X=X',
									str(token_idx-1),
									'root' if token_idx-1==0 else 'det',
									'|'.join(arc_list),
									'X',
								)
								lines.append(eval_line)
							lines.append("\n")
				eval_loss /= batch_no
				UF1=self.compute_F1(utp,ufp,ufn)
				LF1=self.compute_F1(ltp,lfp,lfn)
				if out_path is not None:
					with open(out_path, "w", encoding="utf-8") as outfile:
						outfile.write("".join(lines))
				if prediction_mode:
					return None, None

				result = Result(
					main_score=LF1,
					log_line=f"\nUF1: {UF1} - LF1 {LF1}",
					log_header="PRECISION\tRECALL\tF1",
					detailed_results=f"\nUF1: {UF1} - LF1 {LF1}",
				)
			else:
				if prediction_mode:
					eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path,prediction_mode=prediction_mode)
					return eval_loss, metric
				else:   
					eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path)
				
				UAS=metric.uas
				LAS=metric.las
				result = Result(main_score=LAS,log_line=f"\nUAS: {UAS} - LAS {LAS}",log_header="PRECISION\tRECALL\tF1",detailed_results=f"\nUAS: {UAS} - LAS {LAS}",)
			return result, eval_loss
	def compute_F1(self, tp, fp, fn):
		precision = tp/(tp+fp + 1e-12)
		recall = tp/(tp+fn + 1e-12)
		return 2 * (precision * recall) / (precision + recall+ 1e-12)


	@torch.no_grad()
	def dependency_evaluate(self, loader, out_path=None, prediction_mode=False):
		# self.model.eval()

		loss, metric = 0, Metric()
		# total_start_time=time.time()
		# forward_time=0
		# loss_time=0
		# decode_time=0
		# punct_time=0
		lines=[]
		for batch in loader:
			forward_start=time.time()
			arc_scores, rel_scores = self.forward(batch)
			# forward_end=time.time()
			mask = self.mask
			if not prediction_mode:
				loss += self._calculate_loss(arc_scores, rel_scores, batch, mask)
			# loss_end=time.time()
			# forward_time+=forward_end-forward_start
			# loss_time+=loss_end-forward_end
			mask=mask.bool()
			# decode_start=time.time()
			arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
			# decode_end=time.time()
			# decode_time+=decode_end-decode_start
			# ignore all punctuation if not specified

			if not self.punct:
				for sent_id,sentence in enumerate(batch):
					for token_id, token in enumerate(sentence):
						upos=token.get_tag('upos').value
						xpos=token.get_tag('pos').value
						word=token.text
						if is_punctuation(word,upos,self.punct_list) or is_punctuation(word,upos,self.punct_list):
							mask[sent_id][token_id]=0
				# mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
			if out_path is not None:
				for (sent_idx, sentence) in enumerate(batch):
					for token_idx, token in enumerate(sentence):
						if token_idx == 0:
							continue

						# append both to file for evaluation
						eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
							token_idx,
							token.text,
							'X',
							'X',
							'X',
							'X',
							arc_preds[sent_idx,token_idx],
							self.tag_dictionary.get_item_for_index(rel_preds[sent_idx,token_idx]),
							'X',
							'X',
						)
						lines.append(eval_line)
					lines.append("\n")
				
			
			if not prediction_mode:
				# punct_end=time.time()
				# punct_time+=punct_end-decode_end
				metric(arc_preds, rel_preds, self.arcs, self.rels, mask)
		if out_path is not None:
			with open(out_path, "w", encoding="utf-8") as outfile:
				outfile.write("".join(lines))
		if prediction_mode:
			return None, None
		# total_end_time=time.time()
		# print(total_start_time-total_end_time)
		# print(forward_time)
		# print(punct_time)
		# print(decode_time)
		
		loss /= len(loader)

		return loss, metric

	def decode(self, arc_scores, rel_scores, mask):
		arc_preds = arc_scores.argmax(-1)
		bad = [not istree(sequence, not self.is_mst)
			   for sequence in arc_preds.tolist()]
		if self.tree and any(bad):

			arc_preds[bad] = eisner(arc_scores[bad], mask[bad])
			# if not hasattr(self,'dist') or self.is_mst:
			#   dist = generate_tree(arc_scores,mask,is_mst=False)
			# else:
			#   dist = self.dist
			# arc_preds=get_struct_predictions(dist)
			

			# deal with masking
			# if not (arc_preds*mask == result*mask).all():
			#   pdb.set_trace()

		rel_preds = rel_scores.argmax(-1)
		rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

		return arc_preds, rel_preds