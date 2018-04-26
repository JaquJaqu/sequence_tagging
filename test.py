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
    get_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_embedding_vectors, get_processing_word, add_oov_words
#import json

    
config_file = sys.argv[1]
filename_test = sys.argv[2]

config = Config(config_file)

# build model
test  = CoNLLDataset(filename_test, config.processing_word,
                     None, config.max_iter)
#add OOV words to the model
add_oov_words(test,config)

model = NERModel(config)
model.build()
model.restore_session(config.dir_model)


# create dataset

model.predict_test(test)
