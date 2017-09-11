# Data

## Overview

This is a placeholder folder for keeping the datasets and models
trained by the Ovation framework. When you use this repository, this
folder will contain three folders, following the structure below:

```
 data/
  +- datasets/
  |   +- dataset_name1/
  |   +- dataset_name2/
  |   +- ...
  |   +- dataset_namen/
  +- experiments/
  |   +- experiment1/
  |   +- experiment2/
  |   +- ...
  |   +- experimentn/
  +- models/
      +- model1/
      +- model2/
      +- ...
      +- modeln/
```


## Datasets

Each dataset class in the `/datasets` folder expects the data to be
in a folder here. For example, the Gersen dataset expects the data to
be in `/data/datasets/gersen`, and the STS dataset expects the data to
be in `/data/datasets/sts`. When you call the constructor of these
classes, they will know where to look for their data.


## Experiments

Every time you use one of the template codes in `/templates`, they will
by default create a new folder here with the name of the experiment. If
you want them to create a folder somewhere else, you can use the
command line argument `--data_dir` to the template code. The name of
the folder to be created here can also be changed by using the command
line argument `--experiment_name`. For example, let's say you want to
run the `ner_seq2seq.py` experiment and want your
experiment data to be put in another folder, say `/data/blah` folder,
with the name `BLAH`, then you could do:

```sh
# Change directory to the root of the repository
cd /path/to/Ovation

# (maybe activate your virtual environment)

# Run ner_seq2seq.py
python templates/ner_seq2seq.py --data_dir=data/blah --experiment=BLAH
```

## Models





