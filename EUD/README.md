# Enhanced Universal Dependency Parsing

[Data](https://universaldependencies.org/iwpt20/data.html)
In our settings, we concat all treebanks for each language

## Preprocessing 

Modify `tar_dir` and `target` in `convert_EUD_to_collapsed.py`, then:

```
python convert_EUD_to_collapsed.py
```

## Postprocessing

move all output `.conll` file in to a certain directory and modify `tar_dir` in `back_conversion.py`, then:

```
python back_conversion.py
```

If you just want to evaluate the results offline without any other validation constraints, run:

```
python iwpt20_xud_eval.py $gold_file $system_file
```

If you want to use the official evaluation pipeline, run:
```
perl conllu-quick-fix.pl $system_file > $output_file
python validate.py --level 2 --lang $language $output_file
```
The `validate.py` may warn very few non-connected graphs in the file (usually for the low-resource Tamil outputs), I fixed this issue manually :). Then you can reformat the output files according to the [submission rules](https://universaldependencies.org/iwpt20/submission.html) and submit your results to the [submission site](https://quest.ms.mff.cuni.cz/sharedtask/)

