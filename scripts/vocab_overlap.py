import os
import sys

embeddings = {}
#embeddings["german_de_100"]= (100,"/projekte/ocex/martin/fastText/German_de_docs_skip_100.vec")
#embeddings["german_de_500"]= (500,"/projekte/ocex/martin/fastText/German_de_docs_skip_500.vec")
#embeddings["de_100"]= (500,"/projekte/ocex/martin/w2v/de_txt.s500.n15.skip")

#embeddings["wiki_500"] = (500,"/projekte/ocex/martin/NER/sequence_tagging3/de_tokenized_clean_w2v_skip_w5_n5_s500d.txt")


#embeddings["mr_wiki_de_300"]="/projekte/ocex/martin/fastText/de_wiki.txt.seg.10.lower.txt.300.bin"
#embeddings["mr_wiki_de_upper_300"]="/projekte/ocex/martin/fastText/de_wiki.txt.seg.10.txt.300.bin"
embeddings["facebook_de"]="/projekte/ocex/martin/embeddings/fasttext/wiki.de.vec"
embeddings["German_de_300"]="/projekte/ocex/martin/fastText/German_de_docs_skip_300.vec"


test = {}
test["germeval.test"]="/projekte/ocex/martin/NER/corpora/GermaNER-test.conll"
test["conll.test"] = "/projekte/ocex/martin/NER/corpora/new/utf8.deu.mr.testb"
test["lft.test"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.lft.mr.tok.test.bio"
test["onb.test"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.onb.mr.tok.test.bio"
#test["sbb.test"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.sbbmr.mr.tok.test.bio"

test["germeval.dev"]="/projekte/ocex/martin/NER/corpora/GermaNER-dev.conll"
test["conll.dev"] = "/projekte/ocex/martin/NER/corpora/new/utf8.deu.mr.testa"
test["lft.dev"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.lft.mr.tok.dev.bio"
test["onb.dev"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.onb.mr.tok.dev.bio"
#test["sbb.dev"]= "/projekte/ocex/martin/NER/corpora/tokenized/enp_DE.sbbmr.mr.tok.dev.bio"



def readVocab(f):
    ws = set()
    for l in open(f):
        ls = l.strip().split()
        
        #print(ls)
        if len(ls)==0:
            continue
        ws.add(ls[0])
    return ws

test_vocab = {}
for t in test:
    #print(t)
    test_vocab[t]=readVocab(test[t])

for e in embeddings:
    we = readVocab(embeddings[e])
    for t in test_vocab:
        wc = test_vocab[t]
        print("%s	%s	%d	%d	%d"%(e,t,len(we.intersection(wc)),len(wc),len(we)))
