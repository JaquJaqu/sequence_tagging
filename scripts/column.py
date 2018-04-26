import os
import sys


idxs = sys.argv[1].split(",")
idxs = [ int(i) for i in idxs]

for l in sys.stdin:
    if len(l.strip())==0:
        print( "")
        continue
    ls = l.strip().split()
    line = ""
    for i in idxs:
        line+=" "+ls[i]
    print line.strip()

