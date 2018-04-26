import os
import sys

cmds = []
ks = ""
for l in open(sys.argv[1]):
    if l.startswith("python transfer_learning"):
        if len(ks)>0:
            cmds.append(ks)
        ks = ""
    ks+=(l)
if len(ks)>0:
    cmds.append(ks)

print len(cmds)
fi = 1
k = int(len(cmds)/4.0+0.5)

tw = fi*k
fw = open(sys.argv[1]+".split"+str(fi),"w")
for i in range(0,len(cmds)):
    if i>tw:
        fi+=1
        tw = fi*k
        fw = open(sys.argv[1]+".split"+str(fi),"w")
    fw.write(cmds[i])

