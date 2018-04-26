import os
import sys


prev_lab = "0"
leng = 0
for l in sys.stdin:
    if len(l.strip())==0:
        if prev_lab!="O":
            print (prev_lab+"\t"+str(leng))
            leng = 0
        continue
    ls = l.strip().split()
    lab = ls[-1]
    lab= lab.replace("B-","I-")
    if lab == "O":
        if not prev_lab =="O":
            print(prev_lab+"\t"+str(leng))
            leng = 0
    
    elif prev_lab !="O" and lab ==prev_lab:

            leng +=1
    elif prev_lab != "O" and lab != prev_lab:
            print (prev_lab+"\t"+str(leng))
            leng = 1
   # else:
   #     print(lab+"\t"+prev_lab)
    prev_lab = lab
