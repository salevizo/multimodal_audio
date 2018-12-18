import pickle
import csv
import os
from stat import *


import sys

def fileCreate(strNameFile, strData):
    try:
        f = open(strNameFile, "w")
        f.writelines(str(strData))
        f.close()
    except IOError:
        strNameFile = strNameFile.split(os.sep)[-1]
        f = open(strNameFile, "w")
        f.writelines(str(strData))
        f.close()
        print("file created: " + strNameFile + "\n")

def readTextFile(strNameFile):
    f = open(strNameFile, "r")
    print("file being read: " + strNameFile + "\n")
    if sys.version_info >=(3, 0):
        return f.read()
    else:
        return f.read().decode("windows-1252").encode('ascii', 'ignore') #----> python2

def walktree( repo_path,callback,folder=''):
    for f in os.listdir(repo_path+folder+"/subtitles"):
        pathname = os.path.join(repo_path+folder+"/subtitles", f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(pathname, callback)
        elif S_ISREG(mode):
            # It's a file, call the callback function
            callback(pathname,repo_path,folder)
        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)

def create_folders(repo_path,folder_path):
    os.chdir(repo_path)
    if (os.path.exists(folder_path))==False:
        #python understands octal so 0777 has to be 511 in decimal
        os.mkdir(folder_path)
        os.chmod(folder_path,511)
        print('Created ' + folder_path)
    else:
        print('Directory ' + folder_path +' exists.')

def read_url_and_ids(fn):
    urls=[]
    ids=[]
    with open(fn, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        print("CSV contains the following urls:")
        for i, line in enumerate(reader):
            if i>0:
                l=line[0].split(',')
                urls.append(l[1])
                ids.append(l[0])
                print("Id:" + l[0] + " URL:" +str(l[1]))
    return (urls,ids)

def read_csv(fn,folder=''):
    f=open(fn, "rt")
    reader=csv.reader(f)
    headers = next(reader, None)
    dataset = {}
    for h in headers:
        dataset[h] = []
    for row in reader:
        for h, v in zip(headers, row):            
            dataset[h].append(v)
    output = open(folder+fn.split('.')[0]+'.p', 'wb')
    pickle.dump(dataset, output)
    output.close()
    return dataset

