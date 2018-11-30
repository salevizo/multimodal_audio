from pyAudioAnalysis import audioTrainTest as aT
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


repo_path=''
pyaudioanalysis_path=''

def create_folders(repo_path):
    emotions=["positive","neutral","negative"]
    os.chdir(repo_path)
    for em_dir in emotions:
        if (os.path.exists("audio/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/"+em_dir)
            os.chmod("audio/"+em_dir,511)
            print 'Created directory audio/'+em_dir
        else:
            print 'Directory audio/'+em_dir+' exists.'
        

def retrieveSubs(subsPath,repo_path):
    os.chdir(repo_path)
    subtitles_pol_file=open(subsPath, 'rb')
    # Loading the Subtitle
    subtitles_pol = pickle.load(subtitles_pol_file)
    return subtitles_pol

def wavSegmentationFromSubs(relPath,subtitles,repo_path):
    os.chdir(repo_path)
    audio_name= os.path.basename(os.path.normpath(relPath))
    dir_name=os.path.splitext(audio_name)[0]
    soundsPath=os.path.dirname(relPath)
    #load list
    countpos=countneg=countneu=0
    for i, val in enumerate(subtitles):
        if subtitles[i][1]==0:
            dir_name="neutral"
            countneu+=1
        elif subtitles[i][1]<0:
            dir_name="negative"
            countneg+=1
        else:
            dir_name="positive"
            countpos+=1
        filePath=repo_path+"/"+soundsPath+"/"+audio_name
        if i==(len(subtitles)-1):
            break
        #Problem with format of the time
        os.chdir(repo_path+"/"+soundsPath+"/"+dir_name)
        t1=str(subtitles[i][0]).replace(',','.')
        t2=str(subtitles[i+1][0]).replace(',','.')
        d1 = datetime.strptime(str(t1), "%H:%M:%S.%f")
        d2 = datetime.strptime(str(t2), "%H:%M:%S.%f")
        sec=(d2-d1).total_seconds()
        mstr="ffmpeg -i {} -ss {} -t {} temp{}.wav -loglevel panic -y".format(filePath, t1, sec,i)
        #print mstr
        os.system("ffmpeg -i {} -ss {} -t {} temp{}.wav -loglevel panic -y".format(filePath, t1, sec,i))
        
    print 'Done splitting wav file '+audio_name+'. Audio had '+str(countpos)+' positive segments, '+str(countneg)+ " negative segments and "+str(countneu)+"neutral segments. "
    


def sentiments(filepath):
    results=aT.fileRegression(filepath, pyaudioanalysis_path, "svm")
    return results


        


def main(argv):

    global repo_path
    global pyaudioanalysis_path
    
    #pyaudioanalysis_path='/home/mscuser/pyaudio/pyAudioAnalysis/pyAudioAnalysis/data/speechEmotion/'
    #repo_path='/home/mscuser/multi/multimodal_audio'
    
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    #repo_path='/home/mscuser/multi/multimodal_audio'
    repo_path=str(sys.argv[1])
    pyaudioanalysis_path=str(sys.argv[2])
    os.chdir(repo_path)

    '''load the pickle file that conatins info about the dataset'''
    pkl_file = open('dataset_list.p', 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    
    create_folders(repo_path)
    
    subs=[]
    for k in range(0,len(dataset)):
        print dataset["Pickle"][k]
        subtitles=retrieveSubs(dataset["Pickle"][k],repo_path)
        wavSegmentationFromSubs(dataset["Audio"][k],subtitles,repo_path)
        break


    aT.featureAndTrain([repo_path+"/audio/positive",repo_path+"/audio/neutral",repo_path+"/audio/negative"],1.0,1.0,aT.shortTermWindow,aT.shortTermStep,"svm","svm5Classes")


if __name__ == "__main__":
    main(sys.argv[1:])
