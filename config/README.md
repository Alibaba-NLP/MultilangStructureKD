# Configureation Files

---

The config files are based on yaml format.

* `targets`: The target task (for example, ner)
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
	* `Corpus`: The training corpora for the model, use `:` to split different corpora.
	* `teachers`: The teacher models for training, values are the config files and the values of these config files are the teaching corpora (split by `:`).
	* `tag_dictionary`: They tag dictionary for the task, this is important for multilingual knowledge distillation since all teachers and students should share the same tag dictionary. If the path does not exist, the code will generate a tag dictionary automaticly.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
* `FastSequenceTagger`: An example of `model`, which is a modified version of SequenceTagger class from flair. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters.
* `is_teacher_list`: Set to True in default.
* `trainer`: The trainer class.
* `ModelDistiller`: An example of `trainer`, the values are the parameters for the trainer.
* `train`: the parameters for the `train` function in `trainer` (for example, `ModelDistiller.train()`).
* `teacher_annealing`: Anneal the weight of distillation loss in training.
* `anneal_factor`: the anneal rate for the distillation.