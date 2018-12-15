import os
import csv
import os
import os, re, sys
from stat import *
import matplotlib.pyplot as plt
from pyAudioAnalysis.audioSegmentation import silenceRemoval as sR 
from pyAudioAnalysis.audioBasicIO import readAudioFile
import pysrt
import datetime
import pickle
from datetime import datetime
from sklearn.model_selection import KFold
import FtrainTest as ft
import shutil
import math
from shutil import copyfile
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import StratifiedShuffleSplit
import audioTrainTest_prj as aT



repo_path=''
pyaudioanalysis_path=''


def create_folders(repo_path):
    emotions=["positive","neutral","negative"]
    os.chdir(repo_path)
    if (os.path.exists("audio/test/"))==False:
        os.mkdir("audio/test")
        os.chmod("audio/test/",511)
    if (os.path.exists("audio/train/"))==False:
        os.mkdir("audio/train")
        os.chmod("audio/train/",511)
    for em_dir in emotions:
        if (os.path.exists("audio/test/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/test/"+em_dir)
            os.chmod("audio/test/"+em_dir,511)
            print('Created directory audio/test/'+em_dir)
        else:
            print('Directory audio/test/'+em_dir+' exists.')

        if (os.path.exists("audio/train/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/train/"+em_dir)
            os.chmod("audio/train/"+em_dir,511)
            print('Created directory audio/train/'+em_dir)
        else:
            print('Directory audio/train/'+em_dir+' exists.')

def create_folders_perID(repo_path,id):
    emotions=["positive","neutral","negative"]
    os.chdir(repo_path)
    if (os.path.exists("audio/"+id+"/"))==False:
        os.mkdir("audio/"+id+"/")
        os.chmod("audio/"+id+"/",511)
    for em_dir in emotions:
        if (os.path.exists("audio/"+id+"/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/"+id+"/"+em_dir)
            os.chmod("audio/"+id+"/"+em_dir,511)
            print('Created directory audio/' +id+ '/' +em_dir)
        else:
            print('Directory audio/' +id+ '/' +em_dir+' exists.')
        

def remove_folders(repo_path):
    train = repo_path + "/audio/train"
    test = repo_path + "/audio/test"
    print(test,train)
    for root, dirs, files in os.walk(train, topdown=False):
        for name in files:
            #print(os.path.join(root, name)) 
            os.remove(os.path.join(root, name))
        for name in dirs:
            #print(os.path.join(root, name))
            os.rmdir(os.path.join(root, name))
           

    for root, dirs, files in os.walk(test, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            os.remove(os.path.join(root, name))
            
        for name in dirs:
            #print(os.path.join(root, name))
            os.rmdir(os.path.join(root, name))






def retrieveSubs(subsPath,repo_path):
    os.chdir(repo_path)
    subtitles_pol_file=open(subsPath, 'rb')
    
    # Loading the Subtitle
    subtitles_pol = pickle.load(subtitles_pol_file)
    return subtitles_pol

def wavSegmentationFromSubs(relPath,subtitles,repo_path,audio_cnt,case):
    os.chdir(repo_path)
    audio_name= os.path.basename(os.path.normpath(relPath))
    dir_name=os.path.splitext(audio_name)[0]
    soundsPath=os.path.dirname(relPath)
    #load list
    countpos=countneg=countneu=0
    for i, val in enumerate(subtitles):
        if subtitles[i][1]==0:
            dir_name=case+"/neutral"
            countneu+=1
        elif subtitles[i][1]<0:
            dir_name=case+"/negative"
            countneg+=1
        else:
            dir_name=case+"/positive"
            countpos+=1
        filePath=repo_path+"/"+soundsPath+"/"+audio_name
        audio_number=audio_name.split(".")[0]
        #Problem with format of the time
        os.chdir(repo_path+"/"+soundsPath+"/"+dir_name)
        t1=str(subtitles[i][0][0]).replace(',','.')
        t2=str(subtitles[i][0][1]).replace(',','.')
        d1 = datetime.strptime(str(t1), "%H:%M:%S.%f")
        d2 = datetime.strptime(str(t2), "%H:%M:%S.%f")
        sec=(d2-d1).total_seconds()
        if os.path.isfile(str(audio_number)+"temp"+str(i)+".wav") is False:
            mstr="ffmpeg -i {} -ss {} -t {} {}temp{}.wav -loglevel panic -y".format(filePath, t1, sec,audio_number,i)
            print(mstr)
            os.system(mstr)
        else:
	        print(str(audio_number)+"temp"+str(i)+".wav already exists.")
        
        
    print('Done splitting wav file '+audio_name+'. Audio had '+str(countpos)+' positive segments, '+str(countneg)+ " negative segments and "+str(countneu)+"neutral segments. ")



def wavSegmentationFromSubs_perID(relPath,subtitles,repo_path,audio_cnt):
    os.chdir(repo_path)
    audio_name= os.path.basename(os.path.normpath(relPath))
    dir_name=os.path.splitext(audio_name)[0]
    soundsPath=os.path.dirname(relPath)
    #load list
    countpos=countneg=countneu=0
    for i, val in enumerate(subtitles):
        if subtitles[i][1]==0:
            dir_name=audio_cnt+"/neutral"
            countneu+=1
        elif subtitles[i][1]<0:
            dir_name=audio_cnt+"/negative"
            countneg+=1
        else:
            dir_name=audio_cnt+"/positive"
            countpos+=1
        filePath=repo_path+"/"+soundsPath+"/"+audio_name
        audio_number=audio_name.split(".")[0]
        #Problem with format of the time
        os.chdir(repo_path+"/"+soundsPath+"/"+dir_name)
        t1=str(subtitles[i][0][0]).replace(',','.')
        t2=str(subtitles[i][0][1]).replace(',','.')
        d1 = datetime.strptime(str(t1), "%H:%M:%S.%f")
        d2 = datetime.strptime(str(t2), "%H:%M:%S.%f")
        sec=(d2-d1).total_seconds()
        if os.path.isfile(str(audio_number)+"temp"+str(i)+".wav") is False:
            print("---------_>",os.getcwd())
            mstr="ffmpeg -i {} -ss {} -t {} {}temp{}.wav -loglevel panic -y".format(filePath, t1, sec,audio_number,i)
            print(mstr)
            os.system(mstr)
        else:
            print(str(audio_number)+"temp"+str(i)+".wav already exists.")
        
        
    print('Done splitting wav file '+audio_name+'. Audio had '+str(countpos)+' positive segments, '+str(countneg)+ " negative segments and "+str(countneu)+"neutral segments. ")
    

def sentiments(filepath):
    results=aT.fileRegression(filepath, pyaudioanalysis_path, "svm")
    return results


def searchVideo(TopMostPath,repo_path,case):
    '''
    Search for Video ID and copy/paste segments in CASE(Train or Test) Folder
    '''
    for f in os.listdir(TopMostPath):
        pathname = os.path.join(TopMostPath, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            searchVideo(pathname,repo_path,case)
        elif S_ISREG(mode):
            if "positive" in pathname:
                #print(pathname)
                os.chdir(repo_path)
                shutil.copy(pathname, repo_path+"/audio/"+case+"/positive")
            elif "neutral" in pathname:
                #print(pathname)
                os.chdir(repo_path)
                shutil.copy(pathname, "audio/"+case+"/neutral")
            elif "negative" in pathname:
                #print(pathname)
                os.chdir(repo_path)
                shutil.copy(pathname, repo_path+"/audio/"+case+"/negative")


        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)
           
        


def main(argv):

    global repo_path
    repo_path = str(os.getcwd())
    global pyaudioanalysis_path
    
    #pyaudioanalysis_path='/home/mscuser/pyaudio/pyAudioAnalysis/pyAudioAnalysis/data/speechEmotion/'
    #repo_path='/home/mscuser/multi/multimodal_audio'
    
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    #repo_path='/home/mscuser/multi/multimodal_audio'
    pyaudioanalysis_path=str(sys.argv[1])
    os.chdir(repo_path)

    '''load the pickle file that contains info about the dataset'''
    pkl_file = open('dataset_list.p', 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    
    create_folders(repo_path) ## create test/train folders
    subs=[]
    
    
    ##split for all IDs
    # for k in range(0,len(dataset["Pickle"][25:])):
    #     create_folders_perID(repo_path,str(k))  


    # Wav segmentation of all
    for k in range(0,len(dataset["Pickle"][25:])):
        create_folders_perID(repo_path,str(k))
        subtitles=retrieveSubs(dataset["Pickle"][k],repo_path)
        wavSegmentationFromSubs_perID(dataset["Audio"][k],subtitles,repo_path,str(k))

    ft.f(repo_path,dataset)



if __name__ == "__main__":
    main(sys.argv[1:])
