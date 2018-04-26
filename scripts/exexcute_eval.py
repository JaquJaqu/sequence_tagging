import os
import sys
from subprocess import call
import subprocess

prefix = sys.argv[1]
file_suffix = sys.argv[2]
dd = "."
if len(sys.argv)>3:
    dd = sys.argv[3]
def removeFirstLines(f):
    i = 0
    for l in open(f):
        if (l.split())==3:
            return i
        i+=1
def conv(n):
    n = n.decode("utf-8")
    n = n.replace(";","")
    n = n.replace(":","")
    n = n.replace("%","")
    return n
def readResults(f):
    cmd = ["sh","./evaluate_conll.sh",f]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in result.stdout.readlines():
        ls = line.strip().split()
        if b'accuracy:' in ls:
            p = conv(ls[3])
            r = conv(ls[5])
            f = conv(ls[7])
            return [p,r,f]

results = {}
for d in os.listdir(dd):
     if not d.startswith(prefix):
         continue
     for f in os.listdir(dd+"/"+d+"/"):
        if not f.endswith(file_suffix):
            continue
        #print ("echo  "+os.path.join(d,f))
        res = readResults(os.path.join(dd,d,f))
        #print ("1"+f)
        if res==None:
            print ("No results: "+os.path.join(dd,d,f))
            continue
        [p,r,f1]=res
        #print("2"+f)
        print (d+"\t"+f+"\t"+p+"\t"+r+"\t"+f1)
        #icmd = ["sh","./evaluate_conll.sh",os.path.join(d,f)]
        #result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        #for line in result.stdout.readlines():
        #    print (line.strip())
        #res =  result.stdout
        #ress = res.split("\n")
        #for r in ress:
        #    print( r)
        #iprint (result.stdout)
        #print "cat " +os.path.join(d,f)+" | grep -v \"^[^ ]*$\"|grep -v \" .* .* \" | perl ../conlleval"  

