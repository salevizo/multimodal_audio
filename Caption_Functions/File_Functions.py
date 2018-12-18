import os, re, sys
import csv
from stat import *
import pickle


def create_folders(repo_path,folder_path):
    os.chdir(repo_path)
    if (os.path.exists(folder_path))==False:
        #python understands octal so 0777 has to be 511 in decimal
        os.mkdir(folder_path)
        os.chmod(folder_path,511)
        print('Created ' + folder_path)
    else:
        print('Directory ' + folder_path +' exists.')

def walktree(TopMostPath, callback,case):
    ret={}
    for f in os.listdir(TopMostPath):
        pathname = os.path.join(TopMostPath, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISREG(mode):
            if '.srt' in pathname:
             # It's a file, call the callback function
                print pathname
                if case is 'train':
                    (name,interval_segments,sentiment_segments)=callback(pathname,case)
                    print name
                    ret[int(name)]={}
                    ret[int(name)]['Id']=int(name)
                    ret[int(name)]['intervals']=interval_segments
                    ret[int(name)]['sentiments']=sentiment_segments
                else:
                    (name,interval_segments)=callback(pathname,case)
                    ret[int(name)]={}
                    ret[int(name)]['Id']=int(name)
                    ret[int(name)]['intervals']=interval_segments
        else:
             # Unknown file type, print a message
             print('Skipping %s' % pathname)
    return ret


def average(y):
    avg = float(sum(y))/len(y)
    return avg

def create_csv(repo_path,dic_ids,case):
    create_folders(repo_path,'polarity_csv')
    os.chdir(repo_path + '/polarity_csv')
    for dicts in dic_ids:
        new_csv='polarity_'+str(dic_ids[dicts]['Id']) +'.csv'
        if sys.version_info >= (3, 0):
            with open(new_csv, 'w') as f:
                #w -----------> python3
                writer = csv.writer(f)
                for j in range(len(dic_ids[dicts]['intervals'])):
                    if case is 'train':
                        context=(dic_ids[dicts]['intervals'][j],dic_ids[dicts]['sentiments'][j])
                    else:
                        context=(dic_ids[dicts]['intervals'][j])
                    writer.writerows([context])
        else:
             with open(new_csv, 'wb') as f:
                #wb -----------> python2
                writer = csv.writer(f)
                for j in range(len(dic_ids[dicts]['intervals'])):
                    if case is 'train':
                        context=(dic_ids[dicts]['intervals'][j],dic_ids[dicts]['sentiments'][j])
                    else:
                        context=(dic_ids[dicts]['intervals'][j])
                    writer.writerows([context])


def create_pickle(repo_path,dic_ids,case):
    create_folders(repo_path,'pickle_lists')
    os.chdir(repo_path + 'pickle_lists')
    for dicts in dic_ids.keys():
        context=[]
        pickle_name='polarity_'+str(dic_ids[dicts]['Id']) +'.p'
        for j in range(len(dic_ids[dicts]['intervals'])):
            #print(pickle_name+" with interval "+str(dic_ids[dicts]['intervals'][j])+" and sentiment "+str(dic_ids[dicts]['sentiments'][j]))
            if case is 'train':
                context.append((dic_ids[dicts]['intervals'][j],dic_ids[dicts]['sentiments'][j]))
            else:
                context.append((dic_ids[dicts]['intervals'][j]))
        pickle.dump( context, open( pickle_name, "wb" ) )
