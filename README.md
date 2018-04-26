# Named Entity Recognition with Tensorflow

This repository contains a NER implementation using Tensorflow (based on BiLSTM + CRF and character embeddings) that is based on the implementation by [Guillaume Genthial](https://github.com/guillaumegenthial/sequence_tagging). We have modified this implementation including its documentation. The major changes are listed below:


Mainly, we have done the following changes:
- convert from python 2 to python 3
- extract parameters from source code to a single config file
- create new script for testing new files
- create new script and modify source code for simple transfer learning
- support for several embeddings (GloVe, fasttext, word2vec)
- support to load all embeddings of a model
- support to dynamically load OOV embeddings during testing



Table of Content
================


  * [Task of Named Entity Recognition](#task-of-named-entity-recognition)
  * [Machine Learning Model](#machine-learning-model)
  * [Requirements](#requirements)
  * [Run an Existing Model](#run-an-existing-model)
  * [Train a New Model](#train-a-new-model)
  * [Citation](#citation)
  * [License](#license)
  







Task of Named Entity Recognition
================

The task of Named Entity Recognition (NER) is to predict the type of entity. Classical NER targets on the identification of locations (LOC), persons (PER), organization (ORG) and other (OTH). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```


Machine Learning Model
================

The model is similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf). A more detailed description can be found [here](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe, Word2Vec, FastText here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF


Run an Existing Model
================

Train a New Model
================




Requirements
================
To run the sourcecode you need to install the requirements from the [file](https://github.com/riedlma/sequence_tagging/blob/master/requirements.txt).
In addition, you need to build fastText manually, as described [here](https://github.com/facebookresearch/fastText/tree/master/python).





Citation
================
If you use this model cite the source code of [Guillaume Genthial](https://github.com/guillaumegenthial/sequence_tagging). If you use the German model and the extension, you can cite our paper:

```
@inproceedings{riedl18:_named_entit_recog_shoot_german,
  title = {A Named Entity Recognition Shootout for {German}},
  author = {Riedl, Martin and Padó, Sebastian},
  booktitle = {Proceedings of Annual Meeting of the Association for Computational Linguistics},
  series={ACL 2018},
  address = {Melbourne, Australia},
  note = {To appear},
  year = 2018
}

```


License
================

This project is licensed under the terms of the Apache 2.0 ASL license (as Tensorflow and derivatives). If used for research, citation would be appreciated.




## Getting started


1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

2. Build the training data, train and evaluate the model with
```
make run
```


## Details


Here is the breakdown of the commands executed in `make run`:

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/ner_model.py`

Training time on NVidia Tesla K80 is 110 seconds per epoch on CoNLL train set using characters embeddings and CRF.



## Training Data


The training data must be in the following format (identical to the CoNLL2003 dataset).

A default test file is provided to help you getting started.


```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence
```


Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
dev_filename = "data/coNLL/eng/eng.testa.iob"
test_filename = "data/coNLL/eng/eng.testb.iob"
train_filename = "data/coNLL/eng/eng.train.iob"
```






