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
import json
import subprocess


def conv(n):
    n = n.decode("utf-8")
    n = n.replace(";","")
    n = n.replace(":","")
    n = n.replace("%","")
    return n

def readResults(f):
    cmd = ["sh","./evaluate_conll.sh",f]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in result.stdout.readlines():
        ls = line.strip().split()
        if b'accuracy:' in ls:
            p = conv(ls[3])
            r = conv(ls[5])
            f = conv(ls[7])
            return [p,r,f]

    
config_file = sys.argv[1]
filename_train2 = sys.argv[2]
#filename_test2 = sys.argv[3]
filename_dev2 = sys.argv[3]
#new_dir = sys.argv[5]

config = Config(config_file)

#print (config.dir_model)
#print(config.filename_trimmed)
    # build model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)


train2  = CoNLLDataset(filename_train2, config.processing_word,
                     config.processing_tag, config.max_iter)
dev2  = CoNLLDataset(filename_dev2, config.processing_word,
                     config.processing_tag, config.max_iter)

#test2  = CoNLLDataset(filename_test2, config.processing_word,
#                     None, config.max_iter)


model.train(train2,dev2)
# create dataset

#model.predict_test(test)
#model.predict_test(test2)



# =============================================================================
# config = Config(config_file)
# 
# #print (config.dir_model)
# #print(config.filename_trimmed)
#     # build model
# model = NERModel(config)
# model.build()
# model.restore_session(config.dir_model)
# 
# 
# # create dataset
# 
# test  = CoNLLDataset(filename_test2, config.processing_word,
#                      None, config.max_iter)
# #model.predict_test(test)
# model.predict_test(test)
# =============================================================================
