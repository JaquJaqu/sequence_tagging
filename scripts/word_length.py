import os
import sys

for l in sys.stdin:
    ls = l.strip().split()
    ls.insert(0,str(len(ls[0])))
    print( " ".join(ls))

