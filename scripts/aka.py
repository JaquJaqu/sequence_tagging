#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:52:43 2017

@author: riedlmn
"""

import sys
#from model.ner_model import NERModel
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, get_same_word,load_vocab_rev

import json



    
config_file = sys.argv[1]
filename_test = sys.argv[2]

config = Config(config_file)

#print (config.dir_model)
#print(config.filename_trimmed)
    # build model
#model = NERModel(config)
#model.build()
#model.restore_session(config.dir_model)

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        if type(x[0]) == tuple:
            x = list(zip(*x))
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

# create dataset

test  = CoNLLDataset(filename_test, config.processing_word,
                     None, config.max_iter)
#model.predict_test(test)
idx2word = load_vocab_rev(config.filename_words)
idx2tag  = load_vocab_rev(config.filename_tags)
print (config.batch_size)
config.batch_size = 1
for words, labels in minibatches(test, config.batch_size):
    for lab, word in zip(labels, words):
        for i in range(len(word[1])):
            print (idx2word[word[1][i]])

