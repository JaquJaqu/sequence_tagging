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
