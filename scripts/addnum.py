import os
import sys
i = 1
for l in sys.stdin:
    if len(l.strip())==0:
        i=1
        print ("")
        continue
    print (str(i)+"\t"+l.strip())
    i=i+1
