import os
import sys

str_nums = sys.argv[1]

nums = [int(x) for x in str_nums.split(",")]

for l in sys.stdin:
    if len(l.strip())==0:
        print( l.strip())
        continue
    ls = l.strip().split()
    line = " ".join(ls[n] for n in nums)
    print (line)

