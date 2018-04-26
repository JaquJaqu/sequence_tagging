import os
import sys


def readFile(f):
    words = set()
    for l in open(f):
        if len(l.strip())==0:
            continue
        ls = l.strip().split()
        words.add(ls[0])
    return words


f1 = readFile(sys.argv[1])
f2 = readFile(sys.argv[2])


print("F2-F1 %d"%(len(f2-f1)))
print("F1-F2 %d"%(len(f1-f2)))
print("F1&F2 %d"%(len(f1&f2)))

words = set()
avgs = 0.0
avgd = 0.0
num = 0
sent = ""
for l in open(sys.argv[2]):
    ls = l.strip().split()
    if len(l.strip())==0:
        print("%d/%d %s %s" %(len(words-f1),len(words),sent," ".join(list(words-f1))))
        num+=1
        avgs+=1.0*len(words-f1)/len(words)    
        avgd+=1.0*len(words)
        words = set()
        sent = ""
        continue
    sent +=" "+ls[0]
    words.add(ls[0])
print avgs/num
print avgd/num
print num
