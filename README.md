
# Structure-Level Knowledge Distillation for Multilingual NLP

The code is mainly for our ACL 2020 paper: [Structure-Level Knowledge Distillation For Multilingual Sequence Labeling](https://arxiv.org/abs/2006.01414)
A framework for training **unified multilingual models** with knowledge distillation, the code is mainly based on [flair version 0.4.3](https://github.com/flairNLP/flair) with a lot of modifications.
In this repo, we include the following attributes:

|Task |Monolingual|Multilingual|Finetuning|Knowledge Distillation|Notes|
|-----|-----------|------------|----------|----------------------|-------|
|Sequence Labeling|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|Structure-level knowledge distillation [(Wang et al., 2020)](https://arxiv.org/abs/2004.03846)|
|Dependency Parsing|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|State-of-the-Art Parser for [Enhanced Universal Dependencies](https://universaldependencies.org/iwpt20/data.html) in IWPT 2020 shared task [(Wang et al., 2020)](https://arxiv.org/abs/2006.01414) and State-of-the-Art Parser for Semantic Dependency Parsing [(Wang et al., 2019)](https://arxiv.org/pdf/1906.07880.pdf)|

---
## Training Sequence Labelers

### Requirements and Installation

The project is based on PyTorch 1.1+ and Python 3.6+. 

```
pip install -r requirements.txt
```

### Teacher Models

Let's train multilingual CoNLL named entity recognition (NER) model as an example. First we need to prepare the teacher models by downloading the pretrained teacher models on [google drive](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing) and put these models in `resources/taggers`. 

An alternative way is training the teacher models by yourself: 
```
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_1lr_256hidden_de_monolingual_crf_sentloss_10patience_baseline_nodev_ner0.yaml
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_1lr_256hidden_en_monolingual_crf_sentloss_10patience_baseline_nodev_ner0.yaml
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_1lr_256hidden_es_monolingual_crf_sentloss_10patience_baseline_nodev_ner1.yaml
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_1lr_256hidden_nl_monolingual_crf_sentloss_10patience_baseline_nodev_ner0.yaml
```

### Training the Multilingual Model without M-BERT finetuning

#### Knowledge Distillation

After all teacher models are ready, we can train the unified multilingual model with

**Posterior distillation**
To reproduce the accuracy in our paper, run:
```
python train_with_teacher.py --config config/multi_bert_300epoch_0.5anneal_2000batch_0.1lr_600hidden_multilingual_crf_sentloss_10patience_distill_fast_posterior_2.25temperature_old_relearn_nodev_fast_new_ner0.yaml
```

We also find that larger temperature can lead to better results:
```
python train_with_teacher.py --config config/multi_bert_300epoch_0.5anneal_2000batch_0.1lr_600hidden_multilingual_crf_sentloss_10patience_distill_fast_posterior_4temperature_old_relearn_nodev_fast_new_ner0.yaml
```

---

**Top-K distillation**
```
python train_with_teacher.py --config config/multi_bert_300epoch_0.5anneal_2000batch_0.1lr_600hidden_multilingual_crf_sentloss_10patience_distill_fast_1best_old_relearn_nodev_fast_new_ner0.yaml
```

---

**Top-WK distillation**
```
python train_with_teacher.py --config config/multi_bert_300epoch_0.5anneal_2000batch_0.1lr_600hidden_multilingual_crf_sentloss_10patience_distill_fast_1best_old_relearn_nodev_fast_new_ner0.yaml
```

---

### Training the Multilingual Model with M-BERT finetuning

#### Finetuning M-BERT **without** the CRF layer

Following the example of [transformers](https://github.com/huggingface/transformers/tree/master/examples/token-classification), we use a learning rate of `5e-5` for M-BERT finetuning, run:

```
python train_with_teacher.py --config config/multi_bert_10epoch_2000batch_0.00005lr_multilingual_nocrf_sentloss_baseline_fast_finetune_relearn_nodev_ner0.yaml
```

#### Finetuning M-BERT **with** the CRF layer

The key for finetuning M-BERT with the CRF layer is setting a larger learning rate for the transition table while the M-BERT layer with a small learning rate (`0.5` here), to train the model, run:

```
python train_with_teacher.py --config config/multi_bert_10epoch_2000batch_0.00005lr_10000lrrate_5decay_800hidden_multilingual_crf_sentloss_baseline_fast_finetune_relearn_nodev_ner0.yaml
```

---

**Posterior distillation**
To distill the posterior distribution with finetuning M-BERT model, run:
```
python train_with_teacher.py --config config_gen/multi_bert_10epoch_10anneal_2000batch_0.00005lr_10000lrrate_5decay_800hidden_multilingual_crf_sentloss_distill_posterior_4temperature_fast_finetune_relearn_nodev_ner1.yaml
```

---

#### Performance

Performance on CoNLL-02/03 NER with finetuning M-BERT are (average over 3 runs):


|Finetune|CRF|Knowledge Distillation|English|Dutch|Spanish|German|Average|
|-----|-----------|------------|----------|----------------------|-------|-------|-------|
|:heavy_check_mark:|:x:|:x:|91.09|90.34|87.88|82.59|87.97|
|:heavy_check_mark:|:heavy_check_mark:|:x:|91.47|90.97|88.15|82.80|88.35|
|:heavy_check_mark:|:heavy_check_mark:|Posterior|**91.63**|**91.38**|**88.78**|**83.21**|**88.75**|


<!-- #### Model Performance

To be updated -->

---

## Training Dependency Parsers

The dependency parsering module is based on the code of [parser](https://github.com/yzhangcs/parser), our parser is also able to parse the semantic dependency parsing [(Oepen et al., 2014)](aclweb.org/anthology/S14-2008) with second-order mean-field variational inference [(Wang et al., 2019)](https://www.aclweb.org/anthology/P19-1454).

### Multilingual Syntactic Dependency Parsing

For multilingal syntactic dependency parsing, we run on [Universal Dependencies](https://universaldependencies.org/) as an example:

```
python train_with_teacher.py --config config/multi_bert_1000epoch_0.5inter_3000batch_0.002lr_400hidden_multilingual_nocrf_fast_nodev_dependency0.yaml
```

Training the model with BERT finetuning:
```
python train_with_teacher.py --config config/multi_bert_10epoch_0.5inter_3000batch_0.00005lr_20lrrate_multilingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_nodev_dependency15.yaml
```

**Note**: The performance of Monolingual models have not been evaluated yet, if you want to train a monolingual model, please try the configuration of [parser](https://github.com/yzhangcs/parser).

### [Enhanced Universal Dependency (EUD)](https://universaldependencies.org/iwpt20/data.html) Parsing

To reproduce [our results](https://arxiv.org/abs/2004.03846) on EUD Parsing, we provide the [conversion scripts](https://universaldependencies.org/iwpt20/task_and_evaluation.html) for the official dataset. And we also provide our processed training/development/test set for the task. To train the model (here we take the Tamil dataset as an example), run (please refer to `config` for config files of other languages):

```
python train_with_teacher.py --config config/xlmr_word_origflair_1000epoch_0.1inter_2000batch_0.002lr_400hidden_ta_monolingual_nocrf_fast_2nd_unrel_250upsample_nodev_enhancedud27.yaml
```

As we described in the paper, we use the labeled F1 scores (originated from semantic dependency parsing) rather than ELAS for EUD training, therefore if you want to evaluate the ELAS score, first parse the graphs:

```
python train_with_teacher.py --config config/xlmr_word_origflair_1000epoch_0.1inter_2000batch_0.002lr_400hidden_ta_monolingual_nocrf_fast_2nd_unrel_250upsample_nodev_enhancedud27.yaml --parse --target_dir iwpt2020_test/ta --keep_order --batch_size 1000
```

Then evaluate the result by the official script: (Note that the official evaluation script does not check the connectivity, if you go strict process of official submission, please fix other [validation issues]() manually. But for the ELAS, the connectivity does not affect the result a lot.)

### Semantic Dependency Parsing (SDP)

The code for EUD parsing is also applicable for SDP parsing. We provide a PyTorch version of our [second-order SDP parser](https://arxiv.org/pdf/1906.07880.pdf) (For the [TensorFlow Version](https://github.com/wangxinyu0922/Second_Order_SDP)) here. However, we have not evaluate the performance on SDP datasets yet. You may need to modifiy some code and hyper-parameters to run on SDP datasets.

## Others

### Write Your Own Config File

We provide a detailed description of our config file in `config`.

### GPU Memory

We have update the code for better GPU utilization, therefore training a multilingual sequence labeling with knowledge distillation only needs 8\~9 GB for the GPU Memory now rather than 14\~15 GB reported in the paper.

### Faster Speed

We modified the code of flair for a signficantly faster training speed. For example, we update the `CharacterEmbeddings` class in `embeddings.py` to `FastCharacterEmbeddings` for significantly faster character embedding speed and the `WordEmbeddings` is updated to `FastWordEmbeddings` so that the word embeddings can be updated during training. For training sequence labelers, our code is more than 1.5 times faster than the origin version with word and character embeddings.

---

## Citing Us

### For Sequence Labelers

Please cite the following paper when training the multilingual sequence labeling models: 

```
@inproceedings{wang-etal-2020-structure,
    title = "Structure-Level Knowledge Distillation For Multilingual Sequence Labeling",
    author = "Wang, Xinyu  and
      Jiang, Yong  and
      Bach, Nguyen  and
      Wang, Tao  and
      Huang, Fei  and
      Tu, Kewei",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.304",
    pages = "3317--3330",
    abstract = "Multilingual sequence labeling is a task of predicting label sequences using a single unified model for multiple languages. Compared with relying on multiple monolingual models, using a multilingual model has the benefit of a smaller model size, easier in online serving, and generalizability to low-resource languages. However, current multilingual models still underperform individual monolingual models significantly due to model capacity limitations. In this paper, we propose to reduce the gap between monolingual models and the unified multilingual model by distilling the structural knowledge of several monolingual models (teachers) to the unified multilingual model (student). We propose two novel KD methods based on structure-level information: (1) approximately minimizes the distance between the student{'}s and the teachers{'} structure-level probability distributions, (2) aggregates the structure-level knowledge to local distributions and minimizes the distance between two local probability distributions. Our experiments on 4 multilingual tasks with 25 datasets show that our approaches outperform several strong baselines and have stronger zero-shot generalizability than both the baseline model and teacher models.",
}
```

### For Dependency Parsers

If you feel the second-order semantic dependency parser helpful, please cite:

```
@inproceedings{wang-etal-2019-second,
    title = "Second-Order Semantic Dependency Parsing with End-to-End Neural Networks",
    author = "Wang, Xinyu  and
      Huang, Jingxian  and
      Tu, Kewei",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1454",
    pages = "4609--4618",}

```

```
@inproceedings{Wan:Liu:Jia:19,
  author = {Wang, Xinyu and Liu, Yixian and Jia, Zixia
            and Jiang, Chengyue and Tu, Kewei},
  title = {{ShanghaiTech} at {MRP}~2019:
           {S}equence-to-Graph Transduction with Second-Order Edge Inference
           for Cross-Framework Meaning Representation Parsing},
  booktitle = CONLL:19:U,
  address = L:CONLL:19,
  pages = {\pages{--}{55}{65}},
  year = 2019
}
```

If run experiments on Enhanced Universal Dependencies, please cite:

```
@inproceedings{wang-etal-2020-enhanced,
    title = "Enhanced {U}niversal {D}ependency Parsing with Second-Order Inference and Mixture of Training Data",
    author = "Wang, Xinyu  and
      Jiang, Yong  and
      Tu, Kewei",
    booktitle = "Proceedings of the 16th International Conference on Parsing Technologies and the IWPT 2020 Shared Task on Parsing into Enhanced Universal Dependencies",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.iwpt-1.22",
    pages = "215--220",
    abstract = "This paper presents the system used in our submission to the \textit{IWPT 2020 Shared Task}. Our system is a graph-based parser with second-order inference. For the low-resource Tamil corpora, we specially mixed the training data of Tamil with other languages and significantly improved the performance of Tamil. Due to our misunderstanding of the submission requirements, we submitted graphs that are not connected, which makes our system only rank \textbf{6th} over 10 teams. However, after we fixed this problem, our system is 0.6 ELAS higher than the team that ranked \textbf{1st} in the official results.",
}
```

## Contact 

Please email your questions or comments to [Xinyu Wang](http://wangxinyu0922.github.io/).

