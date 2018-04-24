#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:52:43 2017

@author: riedlmn
"""

import sys
from model.ner_model import NERModel
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, get_same_word
#import json

    
config_file = sys.argv[1]
filename_test = sys.argv[2]

config = Config(config_file)

#print (config.dir_model)
#print(config.filename_trimmed)
    # build model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)


# create dataset

test  = CoNLLDataset(filename_test, config.processing_word,
                     None, config.max_iter)
print(test)
#model.predict_test(test)
model.predict_test(test)
