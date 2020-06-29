import re
import codecs
#import six
import pdb
import json
import numpy as np
def count_file(file,target='train',is_file=False,max_len=999):
	if not is_file:
		filelist=os.listdir(file)
		for target_file in filelist:
			# pdb.set_trace()
			if '_' not in target_file and 'bio' in target_file:
				break

			if 'swp' in target_file:
				continue
			# if target in target_file:
			# 	break
			# if 'train' in target_file:
			#   if 'train_new' in target_file:
			#       # os.remove(os.path.join(file,target_file))
			#       pass
			#   else:
			#       break
		# pdb.set_trace()
		#if write:
		to_write=os.path.join(file,target_file)
	else:
		target_file=file
		to_write=file
	reader=open(to_write,'r')
	lines=reader.readlines()
	sentences=[]
	sentence=[]
	sent_length=[]
	pos_idx=0
	for line in lines:
		pos_idx+=1
		line=line.strip()
		if line:
			sentence.append(line)
		else:
			sent_length.append(len(sentence))
			sentences.append(sentence.copy())
			if len(sentence)>199:
				pdb.set_trace()
			sentence=[]

	if sentence != []:
		sent_length.append(len(sentence))
		sentences.append(sentence.copy())
		sentence=[]

	sent_length=np.array(sent_length)
	reader.close()
	try:
		print(target_file, sent_length.max(),len(sentences),(sent_length>max_len).sum())
	except:
		return False, False
		pdb.set_trace()
	return to_write,sentences

def remove_label(file,target='train',is_file=False,max_len=999):
	if not is_file:
		filelist=os.listdir(file)
		for target_file in filelist:
			# pdb.set_trace()
			if '_' not in target_file and 'bio' in target_file:
				break

			if 'swp' in target_file:
				continue
			# if target in target_file:
			# 	break
			# if 'train' in target_file:
			#   if 'train_new' in target_file:
			#       # os.remove(os.path.join(file,target_file))
			#       pass
			#   else:
			#       break
		# pdb.set_trace()
		#if write:
		to_write=os.path.join(file,target_file)
	else:
		target_file=file
		to_write=file
	reader=open(to_write,'r')
	lines=reader.readlines()
	sentences=[]
	sentence=[]
	sent_length=[]
	for line in lines:
		line=line.strip()
		if line:
			
			if not line.startswith('#'):
				fields=line.split('\t')	
				labels=fields[-2].split('|')
				new_labels = []
				for idx, label in enumerate(labels):
					label_fields=label.split(':')
					headid = label_fields[0]
					new_labels.append(headid+':'+'<unk>')
				fields[-2]='|'.join(new_labels)
				line = '\t'.join(fields)
			sentence.append(line)
		else:
			sent_length.append(len(sentence))
			sentences.append(sentence.copy())
			sentence=[]

	if sentence != []:
		sent_length.append(len(sentence))
		sentences.append(sentence.copy())
		sentence=[]

	sent_length=np.array(sent_length)
	reader.close()
	try:
		print(target_file, sent_length.max(),len(sentences),(sent_length>max_len).sum())
	except:
		return False, False
		pdb.set_trace()
	return to_write,sentences


def write_file(to_write,sentences,max_len=999):
	write_file=to_write
	writer=open(write_file,'w')
	remove_count=0
	for sentence in sentences:
		if len(sentence)>max_len:
			remove_count+=1
			continue
		for word in sentence:
			writer.write(word+'\n')
		writer.write('\n')
	writer.close()
	print(remove_count)



def count_sentence(conllu_file, first_set=False):
		#pdb.set_trace()
		if conllu_file.endswith('.zip'):
			open_func = zipfile.Zipfile
			kwargs = {}
		elif conllu_file.endswith('.gz'):
			open_func = gzip.open
			kwargs = {}
		elif conllu_file.endswith('.xz'):
			open_func = lzma.open
			kwargs = {'errors': 'ignore'}
		else:
			open_func = codecs.open
			kwargs = {'errors': 'ignore'}

		with open_func(conllu_file, 'rb') as f:
			reader = codecs.getreader('utf-8')(f, **kwargs)
			#writer = codecs.getwriter('utf-8')
			buff = []
			count=0
			length_count=0
			bufdict={}
			write_mode=False
			devdict={}
			loop_count=0
			copdict={}
			reldict={}
			currentline=0
			bufdict[currentline]=0
			for line in reader:
				line = line.strip()
				#pdb.set_trace()
				#if line:
				if not line:
					#pdb.set_trace()
					count+=1
					try:
						if bufdict[currentline]==5:
							#pdb.set_trace()
							pass
					except:
						pass
					currentline+=1
					bufdict[currentline]=0
				else:
					pass
					#'''
					bufdict[currentline]+=1

					sent=line.split('\t')
					
					#pdb.set_trace()
					#'''
				'''
				if line and not line.startswith('#'):
					if not re.match('[0-9]+[-.][0-9]+', line):
						buff.append(line.split('\t'))
						#buff.append(line.split())
				'''
			print(count)
			print(max(list(bufdict.values())))
			# pdb.set_trace()
		return


def write_sentence(conllu_file, first_set=False,maxlength=999,tb='ptb',extra_name='_modified'):
		#pdb.set_trace()
		if conllu_file.endswith('.zip'):
			open_func = zipfile.Zipfile
			kwargs = {}
		elif conllu_file.endswith('.gz'):
			open_func = gzip.open
			kwargs = {}
		elif conllu_file.endswith('.xz'):
			open_func = lzma.open
			kwargs = {'errors': 'ignore'}
		else:
			open_func = codecs.open
			kwargs = {'errors': 'ignore'}

		with open_func(conllu_file, 'rb') as f:
			reader = codecs.getreader('utf-8')(f, **kwargs)
			tags=conllu_file.split('.')
			#tags.insert(1,'en')
			#tags.insert(2,tb)
			
			tags[-2]=tags[-2]+extra_name
			fout = open('.'.join(tags), 'wb') 
			#pdb.set_trace()
			#fout = open('', 'wb') 
			#pdb.set_trace()
			writer = codecs.getwriter('utf-8')(fout, **kwargs)
			buff = []
			count=0
			bufdict={}
			write_mode=False
			devdict={}
			currentline=0
			bufdict[currentline]=0
			writecount=0
			for line in reader:
				line_ = line.strip()
				#pdb.set_trace()
				
				if not line_:
					count+=1
					if (not currentline==-1) and bufdict[currentline]<maxlength:
						#pdb.set_trace()
						writecount+=1
						#writer.write('#'+str(currentline)+'\n')
						for buf in buff:
							writer.write(buf)
						writer.write('\n')
					buff=[]
					currentline+=1
					bufdict[currentline]=0
				elif line_[0]=='#':
					continue
				else:
					#pdb.set_trace()
					try:
						bufdict[currentline]+=1
						data=line_.split('\t')
						if '.' in data[0]:
							continue
						deps=[]
						dependency=data[-2].split('|')
						for dep in dependency:
							depid=dep.split(':')[0]
							if '.' not in depid:
								deps.append(dep)
						data[-2]='|'.join(deps)
						#pdb.set_trace()
						if len(data)==8:
							data.append('_')
							data.append('_')

						# data[-2]=':'.join([data[-4],data[-3]])
						buff.append('\t'.join(data)+'\n')
					except:
						pdb.set_trace()

				'''
				if line and not line.startswith('#'):
					if not re.match('[0-9]+[-.][0-9]+', line):
						buff.append(line.split('\t'))
						#buff.append(line.split())
				'''
			if (not currentline==-1) and bufdict[currentline]<maxlength and len(buff)>0:
				#pdb.set_trace()
				writecount+=1
				#writer.write('#'+str(currentline)+'\n')
				for buf in buff:
					writer.write(buf)
				writer.write('\n')
			fout.close()
			print(writecount)
			# pdb.set_trace()
		return

def count_dependency(conllu_file, first_set=False):
		#pdb.set_trace()
		if conllu_file.endswith('.zip'):
			open_func = zipfile.Zipfile
			kwargs = {}
		elif conllu_file.endswith('.gz'):
			open_func = gzip.open
			kwargs = {}
		elif conllu_file.endswith('.xz'):
			open_func = lzma.open
			kwargs = {'errors': 'ignore'}
		else:
			open_func = codecs.open
			kwargs = {'errors': 'ignore'}

		with open_func(conllu_file, 'rb') as f:
			reader = codecs.getreader('utf-8')(f, **kwargs)
			#writer = codecs.getwriter('utf-8')
			buff = []
			count=0
			length_count=0
			bufdict={}
			write_mode=False
			devdict={}
			loop_count=0
			copdict={}
			reldict={}
			currentline=0
			bufdict[currentline]=0
			distance_dict={}
			for line in reader:
				line = line.strip()
				#pdb.set_trace()
				#if line:
				if not line:
					#pdb.set_trace()
					count+=1
					try:
						if bufdict[currentline]==5:
							#pdb.set_trace()
							pass
					except:
						pass
					currentline+=1
					bufdict[currentline]=0
				else:
					pass
					#'''
					bufdict[currentline]+=1

					sent=line.split('\t')
					# pdb.set_trace()
					distance=int(sent[6])-int(sent[0])
					if distance not in distance_dict:
						distance_dict[distance]=0
					distance_dict[distance]+=1
			print(count)
			res=[distance_dict[key] for key in distance_dict if abs(key)>10 ]
			print(max(list(bufdict.values())))
			# pdb.set_trace()
		return


def count_projective(conllu_file, first_set=False):
		#pdb.set_trace()
		if conllu_file.endswith('.zip'):
			open_func = zipfile.Zipfile
			kwargs = {}
		elif conllu_file.endswith('.gz'):
			open_func = gzip.open
			kwargs = {}
		elif conllu_file.endswith('.xz'):
			open_func = lzma.open
			kwargs = {'errors': 'ignore'}
		else:
			open_func = codecs.open
			kwargs = {'errors': 'ignore'}

		with open_func(conllu_file, 'rb') as f:
			reader = codecs.getreader('utf-8')(f, **kwargs)
			#writer = codecs.getwriter('utf-8')
			buff = []
			count=0
			length_count=0
			bufdict={}
			write_mode=False
			devdict={}
			loop_count=0
			copdict={}
			reldict={}
			currentline=0
			bufdict[currentline]=0
			sentences=[]
			sentence=[]
			for line in reader:
				line = line.strip()
				#pdb.set_trace()
				#if line:
				if not line:
					#pdb.set_trace()
					count+=1
					try:
						if bufdict[currentline]==5:
							#pdb.set_trace()
							pass
					except:
						pass
					currentline+=1
					bufdict[currentline]=0
					if sentence!=[]:
						sentences.append(sentence.copy())
						sentence=[]
				else:
					pass
					#'''
					bufdict[currentline]+=1
					if line[0]=='#':
						continue
					sent=line.split('\t')
					sentence.append(sent)
					# pdb.set_trace()
			proj=0
			nproj=0
			for sentence in sentences:
				flag=0
				for token1 in sentence:
					for token2 in sentence:
						# pdb.set_trace()
						start1=int(token1[0])
						start2=int(token2[0])
						end1=int(token1[6])
						end2=int(token2[6])
						x1=min(start1,end1)
						x2=min(start2,end2)
						y1=max(start1,end1)
						y2=max(start2,end2)
						# pdb.set_trace()
						# if (start1>start2 and end1>end2) or (start1<start2 and end1<end2):
						# 	nproj+=1
						# 	flag=1
						# if (end1>start2 and start1>end2) or (end1<start2 and start1<end2):
						# 	nproj+=1
						# 	flag=1
						if (x1<x2<y1<y2) or (x2<x1<y2<y1):
							nproj+=1
							flag=1
							break
					if flag:		
						break
				if flag==0:
					proj+=1

			print(proj)
			print(nproj)
			print(proj/(proj+nproj))
			# pdb.set_trace()
			# res=[distance_dict[key] for key in distance_dict if abs(key)>10 ]
			# print(max(list(bufdict.values())))
			# pdb.set_trace()
		return

def pf(sentbuf):
	for sent in sentbuf:
		print(sent)

def count_additional(conllu_file, first_set=False, write=False, extra_name='_modified'):
	#pdb.set_trace()
	if conllu_file.endswith('.zip'):
		open_func = zipfile.Zipfile
		kwargs = {}
	elif conllu_file.endswith('.gz'):
		open_func = gzip.open
		kwargs = {}
	elif conllu_file.endswith('.xz'):
		open_func = lzma.open
		kwargs = {'errors': 'ignore'}
	else:
		open_func = codecs.open
		kwargs = {'errors': 'ignore'}

	with open_func(conllu_file, 'rb') as f:
		reader = codecs.getreader('utf-8')(f, **kwargs)
		if write:
			tags=conllu_file.split('.')
			tags[-2]=tags[-2]+extra_name
			fout = open('.'.join(tags), 'wb') 
			writer = codecs.getwriter('utf-8')(fout, **kwargs)
		buff = []
		count=0
		length_count=0
		bufdict={}
		write_mode=False
		currentline=0
		bufdict[currentline]=0
		addition_flag=0
		extra_node=0
		total_add_token=0
		total_tokens=0
		appear_add_count=0
		bad_add_count=0
		sentbuf={'0':['0','root','_','_','_','_','_','_','_','_']}
		addbuf=[]
		reldict={}
		repeat_edge=0
		for line in reader:
			line = line.strip()
			#pdb.set_trace()
			#if line:
			if not line:
				#pdb.set_trace()
				if write:
					writer.write(line+'\n')
				count+=1
				try:
					if bufdict[currentline]==5:
						#pdb.set_trace()
						pass
				except:
					pass
				currentline+=1
				bufdict[currentline]=0
				if addition_flag:
					# pdb.set_trace()
					extra_node+=1
					addition_flag=0
					for node in addbuf:
						sent=sentbuf[node]
						dependency=sent[-2].split('|')
						if len(dependency)>1:
							parentid=dependency[0].split(':')[0]
							if sent[1]=='_':
								bad_add_count+=1
							else:
								for x in dependency:
									if 'conj' in x:
										parentid=x.split(':')[0]
									#xcomp	
							
						else:
							parentid=dependency[0].split(':')[0]
						parent=sentbuf[parentid]
						assert parentid==parent[0], 'parent not right!'
						if parent[1]==sent[1]:
							appear_add_count+=1
				total_tokens+=len(sentbuf)
				sentbuf={'0':['0','root','_','_','_','_','_','_','_','_']}
				addbuf=[]
			elif '#' in line:
				if write:
					writer.write(line+'\n')
				continue
			else:
				pass
				#'''
				bufdict[currentline]+=1

				sent=line.split('\t')
				# pdb.set_trace()
				dependency=sent[-2].split('|')
				if len(dependency)>1:
					headids=[x.split(':')[0] for x in dependency]

					if len(set(headids))<len(dependency):
						repeat_edge+=1
						rels=[':'.join(x.split(':')[1:]) for x in dependency]
						current_rel_dict={}
						for index, headid in enumerate(headids):
							if headid not in current_rel_dict:
								current_rel_dict[headid]=[]
							current_rel_dict[headid].append(rels[index])
						# pdb.set_trace()
						for headid in current_rel_dict:
							if len(current_rel_dict[headid])>1:
								newrel='+'.join(current_rel_dict[headid])
								if newrel not in reldict:
									reldict[newrel]=0
								reldict[newrel]+=1
								current_rel_dict[headid]=[newrel]
						write_dependency=[]
						for key in current_rel_dict:
							write_dependency.append(key+':'+current_rel_dict[key][0])
						sent[-2]='|'.join(write_dependency)
						# pdb.set_trace()





				if '.' in sent[0]:
					total_add_token+=1
					addition_flag=1
					addbuf.append(sent[0])
				sentbuf[sent[0]]=sent
				#pdb.set_trace()
				#'''
				if write:
					writer.write('\t'.join(sent)+'\n')
			'''
			if line and not line.startswith('#'):
				if not re.match('[0-9]+[-.][0-9]+', line):
					buff.append(line.split('\t'))
					#buff.append(line.split())
			'''
		if len(sentbuf)>1:
			#pdb.set_trace()
			count+=1
			try:
				if bufdict[currentline]==5:
					#pdb.set_trace()
					pass
			except:
				pass
			currentline+=1
			bufdict[currentline]=0
			if addition_flag:
				# pdb.set_trace()
				extra_node+=1
				addition_flag=0
				for node in addbuf:
					sent=sentbuf[node]
					dependency=sent[-2].split('|')
					if len(dependency)>1:
						pdb.set_trace()
					else:
						parentid=dependency[0].split(':')[0]
						try:
							parent=sentbuf[parentid]
						except:
							pdb.set_trace()
						assert parentid==parent[0], 'parent not right!'
						if parent[1]==sent[1]:
							appear_add_count+=1
		# print("Total Sents:", count)
		# print("Total additional Sents:", extra_node)
		# print("Total additional Sents Percentage:", extra_node/count)
		# print("Total additional appeared Tokens:",appear_add_count)
		# print("Total additional Bad Tokens:",bad_add_count)
		# print("Total additional Tokens:",total_add_token)
		# print("Total Tokens with Repeat Edges:",repeat_edge)
		# print("Total Tokens:",total_tokens)

		# print("Total additional appeared Tokens %:",appear_add_count/total_tokens)
		# print("Total additional Bad Tokens %:",bad_add_count/total_tokens)
		# print("Total additional Tokens %:",total_add_token/total_tokens)
		print(reldict)
		newlist=[]
		for key in reldict:
			vals=key.split('+')
			if set(vals) not in newlist:
				newlist.append(set(vals))
			else:
				pdb.set_trace()
		if write:
			writer.close()
		# pdb.set_trace()
	return

def back_conversion(conllu_file, first_set=False, write=False, empty_node=False, extra_name='_back'):
	#pdb.set_trace()
	if conllu_file.endswith('.zip'):
		open_func = zipfile.Zipfile
		kwargs = {}
	elif conllu_file.endswith('.gz'):
		open_func = gzip.open
		kwargs = {}
	elif conllu_file.endswith('.xz'):
		open_func = lzma.open
		kwargs = {'errors': 'ignore'}
	else:
		open_func = codecs.open
		kwargs = {'errors': 'ignore'}

	with open_func(conllu_file, 'rb') as f:
		reader = codecs.getreader('utf-8')(f, **kwargs)
		if write:
			tags=conllu_file.split('.')
			tags[-2]=tags[-2]+extra_name
			fout = open('.'.join(tags), 'wb') 
			writer = codecs.getwriter('utf-8')(fout, **kwargs)
		buff = []
		count=0
		length_count=0
		bufdict={}
		write_mode=False
		currentline=0
		bufdict[currentline]=0
		addition_flag=0
		extra_node=0
		total_add_token=0
		total_tokens=0
		appear_add_count=0
		bad_add_count=0
		sentbuf={'0':['0','root','_','_','_','_','_','_','_','_']}
		addbuf=[]
		reldict={}
		repeat_edge=0
		for line in reader:
			line = line.strip()
			#pdb.set_trace()
			#if line:

			node_deps=[]
			empty_node_deps=[]
			temp_sent=[]
			if not line:
				#pdb.set_trace()
				if write:
					writer.write(line+'\n')
				count+=1
				currentline+=1
				bufdict[currentline]=0
				
			elif '#' in line:
				if write:
					writer.write(line+'\n')
				continue
			else:
				pass
				#'''
				bufdict[currentline]+=1

				sent=line.split('\t')
				# pdb.set_trace()
				if '+' in sent[-2]:
					dependency=sent[-2].split('|')
					new_dependency=[]
					for dep in dependency:
						if '+' in dep:
							headid = dep.split(':')[0]
							rels = ':'.join(dep.split(':')[1:])
							rels = rels.split('+')
							for rel in rels:
								new_dependency.append(headid+':'+rel)
						else:
							new_dependency.append(dep)
					sent[-2]='|'.join(new_dependency)
				if empty_node:
					if '>' in sent[-2]:
						dependency=sent[-2].split('|')
						count=1
						for dep in dependency:
							if '>' in dep:
								headid = dep.split(':')[0]
								rels = ':'.join(dep.split(':')[1:])
								rels = rels.split('>')
								# start from the last relation, which is the label of current node, then iteratively create the empty nodes
								# if len(rels)>2:
								# 	pdb.set_trace()
								for index, rel in enumerate(rels[::-1]):
									current_head = sent[0]+'.'+str(count)
									if index == 0:
										node_deps.append(current_head+':'+rel)
										count+=1
									elif index == len(rels)-1:
										empty_node_deps.append(headid+':'+rel)
									else:
										empty_node_deps.append(current_head+':'+rel)
										count+=1
							else:
								node_deps.append(dep)
						sent[-2]='|'.join(node_deps)
						for index,temp_dep in enumerate(empty_node_deps):
							if '|' in temp_dep:
								pdb.set_trace()
							# pdb.set_trace()
							headid = dep.split(':')[0]
							rels = ':'.join(dep.split(':')[1:])
							temp_sent.append([sent[0]+'.'+str(index+1),'_','X','X','X','X',headid,rels,temp_dep,'X'])
				if '.' in sent[0]:
					total_add_token+=1
					addition_flag=1
					addbuf.append(sent[0])
				sentbuf[sent[0]]=sent
				#pdb.set_trace()
				#'''
				if write:
					writer.write('\t'.join(sent)+'\n')
					for temp in temp_sent:
						writer.write('\t'.join(temp)+'\n')
			'''
			if line and not line.startswith('#'):
				if not re.match('[0-9]+[-.][0-9]+', line):
					buff.append(line.split('\t'))
					#buff.append(line.split())
			'''
		# print("Total Sents:", count)
		# print("Total additional Sents:", extra_node)
		# print("Total additional Sents Percentage:", extra_node/count)
		# print("Total additional appeared Tokens:",appear_add_count)
		# print("Total additional Bad Tokens:",bad_add_count)
		# print("Total additional Tokens:",total_add_token)
		# print("Total Tokens with Repeat Edges:",repeat_edge)
		# print("Total Tokens:",total_tokens)

		# print("Total additional appeared Tokens %:",appear_add_count/total_tokens)
		# print("Total additional Bad Tokens %:",bad_add_count/total_tokens)
		# print("Total additional Tokens %:",total_add_token/total_tokens)
		print(reldict)
		if write:
			writer.close()
		# pdb.set_trace()
	return



#conllu_file='ptb/train.en.pas_modified.conllu'
# tb='ctb'
#conllu_file=tb+'/train.en.ctb_modified.conllu'
#conllu_file=tb+'/train.en.ctb_modified.conllu'
# conllu_file=tb+'/test.en.ctb.conllu'
# conllu_file='CTB5_YM/CTB5.1-devel.gp.conll'
# conllu_file='UD/test.cs_modified.conllu'
# conllu_file='ptb_3.3.0/train.conllu'
# conllu_file='UD_English-EWT/en_ewt-ud-train.conllu'
# conllu_file='UD_French/train.fr_modified.conllu'

# conllu_file='UD_Czech/train.cs_modified.conllu'
# conllu_file='UD_Czech/test.cs.conllu'
# count_dependency(conllu_file)
# count_sentence(conllu_file)

# count_projective(conllu_file)
# write_sentence(conllu_file,maxlength=999,tb='ctb',extra_name='_modified')
# write_sentence(conllu_file,maxlength=90,tb='ptb',extra_name='_modified')
'''
conllu_file=tb+'/dev.conllu'
#count_sentence(conllu_file)
write_sentence(conllu_file,maxlength=60,tb=conllu_file.split('/')[0])
conllu_file=tb+'/test.conllu'
#count_sentence(conllu_file)
write_sentence(conllu_file,maxlength=60,tb=conllu_file.split('/')[0])
#write_sentence(conllu_file,tb=conllu_file.split('/')[0])
'''
#'''
import os
# tar_dir='train-dev'
# tar_dir='shanghaitech_alibaba-sub0'

# tar_dir='test'
files=os.listdir(tar_dir)
# write=False
# # target='test'
# target='train'
target='train'

for file in files:
	# pdb.set_trace()
	filedir=os.path.join(tar_dir,file)
	if not os.path.isdir(filedir):
		continue
	if 'UD_' not in file:
		continue
	# if file == 'UD_Slovak':
	# 	print(1)
	# 	continue
	# print(file,end=' ')
	print(file)
	for filename in os.listdir(filedir):
		
		# if target not in filename:# or '_modified' not in filename:
		# 	continue
		# pdb.set_trace()
		'''
		if target in filename and 'conll' in filename and '_modified' not in filename:
			print(filename)
			# count_projective(os.path.join(filedir,filename))
			# count_additional(os.path.join(filedir,filename),write=True,extra_name='_modified')	
			#====================================================================================
			# to_write,sentences = count_file(os.path.join(filedir,filename),is_file = True)
			to_write,sentences = remove_label(os.path.join(filedir,filename),is_file = True)
			write_file(to_write,sentences)
			# dev_write=os.path.join(filedir,'dev.conllu')
			# test_write=os.path.join(filedir,'test.conllu')
			# dev_sents=sentences[:len(sentences)//2]
			# test_sents=sentences[len(sentences)//2:]
			# write_file(dev_write,dev_sents,max_len=999)
			# write_file(test_write,test_sents,max_len=999)
		'''
		if target in filename and 'conll' in filename and 'back' not in filename:
			# to_write,sentences = count_file(os.path.join(filedir,filename),is_file = True)
			# print(filename)
			# count_projective(os.path.join(filedir,filename))
			# count_additional(os.path.join(filedir,filename),write=True,extra_name='_modified')	
			# remove_label(os.path.join(filedir,filename),write=True,extra_name='')
			# ====================================================================================
			back_conversion(os.path.join(filedir,filename),write=True,empty_node=True,extra_name='_back')	

	# sets=os.listdir()
	# pdb.set_trace()
	# for dataset in sets:
	#   if '_modified' not in dataset:
	#       continue
	#   preprocessing(os.path.join(tar_dir,file,dataset))
	# pdb.set_trace()

#'''