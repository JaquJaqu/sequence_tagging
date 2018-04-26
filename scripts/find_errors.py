import os
import sys

for l in sys.stdin:
    if len(l.strip())==0:
    #    print("")
        continue
    ls = l.strip().split()
    if len(sys.argv)>1:
        ls = l.strip().replace("B-","I-").split()
    gold = ls[-1]
    for i in range(1,len(ls)-1):
        if ls[i]!=gold:
            print l.strip()
            break
