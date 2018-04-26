from subprocess import run, PIPE
import os
import sys

def conv(n):
    #n = n.decode("utf-8")
    n = n.replace(";","")
    n = n.replace(":","")
    n = n.replace("%","")
    return n

def readResults(p,g):
    inp = n""
    for i in range(0,len(p)):
        inp+=" ".join(["F",p[i],g[i]])+"\n"
    print (inp)

    cmd = ["sh","./evaluate_conll.sh"]
    result = run(cmd, stdout=PIPE,input=inp,encoding='ascii')
    #print (result)
    #print(result.stdout.split("\n"))
    for line in result.stdout.split("\n"):
        ls = line.strip().split()
        #print (ls)
        if 'accuracy:' in ls:
            #print(ls)

            p = conv(ls[3])
            r = conv(ls[5])
            f = conv(ls[7])
            return [p,r,f]

p = ["O","O","I-PER"]
g = ["O","O","I-LOC"]

print(readResults(p,g))
