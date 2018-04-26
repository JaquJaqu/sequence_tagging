import os
import sys

def readFile(f):
    fs = []
    for l in open(f):
        
        ls = l.strip().split()
        if len(ls)>0:
            if  not "/GermaNER/" in os.path.dirname(os.path.abspath(f)):
                tmp = ls[-1]
                ls[-1]=ls[-2]
                ls[-2]=tmp
            else:
                #print (os.path.dirname(os.path.abspath(f)))
                #print (ls)
                ls = [ls[0],ls[3],ls[1]] 
        fs.append(ls)
    return fs

fss = []
for f in sys.argv[1:]:
    fss.append(readFile(f))

for i in range(0,len(fss[0])):
    line = []
    for k in range(0,len(fss)):
        fs = fss[k]
        if k==0:
            
            line=fs[i]
            if len(line)==0:
                break
        else:
            line.insert(len(line)-1,fs[i][1])
            #line.extend(fs[i])
    #$print(line)
    print(" ".join(line))
    line=[]    
