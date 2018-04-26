import os
import sys

i = 0
h = ""
cc = int(sys.argv[1])
for l in sys.stdin:
    i+=1
    ls = l.strip().split("\t")
    if i==1:
        h = [ls[0]]
    if i==cc:
        h.extend(ls[2:])
        
        print ("&\t".join(h)+"\\\\")
        i=0
    else:
        h.extend(ls[2:])

