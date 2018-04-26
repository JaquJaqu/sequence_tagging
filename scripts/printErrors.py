import os
import sys

for l in sys.stdin:
    ls = l.strip().split()
    if len(ls)!=3 and len(ls)!=6:
        continue
    if ls[-1]!=ls[-2]:
        print (l.strip())
