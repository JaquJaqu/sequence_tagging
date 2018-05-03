import os
import sys
import urllib.request
import tarfile
import zipfile

stuttgart_server = "http://www2.ims.uni-stuttgart.de/data/ner_de/"
embeddings_folder ="embeddings"
model_folder = "models"
dest_embeddings_folder = embeddings_folder
dest_model_folder = "."

types = {}
types["GermEval"]=["Wiki","model_transfer_learning_conll2003_germeval_emb_wiki.tar.gz"]
types["CONLL2003"]=["Wiki","model_transfer_learning_germeval_conll2003_emb_wiki.tar.gz"]
types["ONB"]=["Europeana","model_transfer_learning_germeval_onb_emb_euro.tar.gz"]
types["LFT"]=["Wiki","model_transfer_learning_germeval_lft_emb_wiki.tar.gz"]

embeddings={}
embeddings["Wiki"]=["fasttext.wiki.de.bin.trimmed.npz","https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.zip"]
embeddings["Europeana"] =["fasttext.german.europeana.skip.300.bin.trimmed.npz","fasttext.german.europeana.skip.300.bin"]


def print_options():
    print ("python %s options"%(sys.argv[0]))
    print ("\nFollowing download options are possible:")
    print ("all         download all models and embeddings")
    print ("all_models  download all models")
    print ("all_embed   download all embeddings")

    for t in types:
        print (t+"\t\tdownload best model and embeddings for "+t)
    sys.exit(0)   

def uncompress(f,folder):
    if f.endswith("tar.gz"):
        print("Uncompress "+f)
        tar = tarfile.open(f, "r:gz")
        tar.extractall(folder)
        tar.close()
    elif f.endswith("zip"):
        print("Uncompress " +f)
        zip_ref = zipfile.ZipFile(f,"r")
        zip_ref.extractall(folder)
        zip_ref.close()
    elif f.endswith("gz"):
        print("ungz")

def download(name,f,folder,dest_folder):
    if len(f)==0:
        return
    if not "http" in f:
        f = stuttgart_server+"/"+folder+"/"+f
    if len(dest_folder)>0 and not os.path.exists(dest_folder):
        os.makedirs(folder)
    dest = dest_folder+"/"+f.split("/")[-1]
    if os.path.exists(dest):
        print("File already exists (%s)"%(dest)) 
        return
    print("Download %s (%s)"%(name,f))
    urllib.request.urlretrieve(f,dest)
    uncompress(dest,dest_folder)    

if len(sys.argv)<2:
    print ("No download option has been specified:")
    print_options()

type = sys.argv[1]

def downloadModels():
    print("Download Models")
    for t,[e,m] in types.items():
        download(t,m,model_folder,dest_model_folder)
def downloadEmbeddings():
    print("Download Embeddings")
    for e,[p,f] in embeddings.items():
        download(e,p,embeddings_folder,dest_embeddings_folder)
        download(e,f,embeddings_folder,dest_embeddings_folder)

if type =="all_embed":
    downloadEmbeddings()
    sys.exit(0)    
if type =="all_models":
    downloadModels()
    sys.exit(0)
if type =="all":
    downloadModels()
    downloadEmbeddings()
    sys.exit(0)
        
if not type in types:
    print("No valid tzpe has been specified")
    print_options()
else:
    fs = types[type]
    download("Model "+type,fs[1],model_folder,dest_model_folder)
    download("Embeddings "+fs[0],embeddings[fs[0]][0],embeddings_folder,dest_embeddings_folder)

 
