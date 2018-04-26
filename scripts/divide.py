import os
import sys

i = int(sys.argv[1])
j = int(sys.argv[2])
for l in sys.stdin:
    ls  = l.strip().split()
    ls.append(str(float(ls[i])/float(ls[j])))
    print ("\t".join(ls))
