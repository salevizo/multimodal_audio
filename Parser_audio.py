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

from datetime import datetime

dict={}
i=0


def wavSegmentationFromSubs(relPath,subtitles):
    audio_name= os.path.basename(os.path.normpath(relPath))
    dir_name=os.path.splitext(audio_name)[0]
    soundsPath=os.path.dirname(relPath)
    #os.getcwd()
    if (os.path.exists(soundsPath+"/"+dir_name))==False:
        #python understands octal so 0777 has to be 511 in decimal
        os.mkdir(soundsPath+"/"+dir_name)
        os.chmod(soundsPath+"/"+dir_name,511)
        print 'Created directory '+soundsPath+"/"+dir_name
    else:
        print 'Directory '+soundsPath+"/"+dir_name+' exists.'
    os.chdir(soundsPath+"/"+dir_name)
    #load list
    for i, val in enumerate(subtitles):
        d1 = datetime.strptime(str(val[2]), "%H:%M:%S.%f")
        d2 = datetime.strptime(str(val[3]), "%H:%M:%S.%f")
        sec=(d2-d1).total_seconds()
        os.system("ffmpeg -i {} -ss {} -t {} temp{}.wav " "-loglevel panic -y".format(soundsPath, val[2], sec,i))
        #os.system("aplay temp.wav")
        #a = raw_input()
    print 'Done splitting wav file '+audio_name+'.'

def sentiments(filepath):
    results=aT.fileRegression(filepath, "/home/mscuser/pyaudio/pyAudioAnalysis/pyAudioAnalysis/data/speechEmotion/", "svm")
    return results


def walktree(TopMostPath, callback):
    '''recursively descend the directory tree rooted at TopMostPath,
    calling the callback function for each regular file'''
    global dict
    for f in os.listdir(TopMostPath):
        pathname = os.path.join(TopMostPath, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(pathname, callback)
        elif S_ISREG(mode):
             # It's a file, call the callback function
             callback(pathname,f)
        else:
             # Unknown file type, print a message
             print 'Skipping %s' % pathname

def find_sentiment(file,f):
    global i
    if '.wav' in file:
        results=sentiments(file)
        dict[i]=((f,results))
        i=i+1
        


def main():
    #fs, x = readAudioFile(input_file)
    
    #FOR LOOP FOR ALL SUBTITLES
    
    input_srt="/home/mscuser/multi/multimodal_audio/subtitles/Travis.srt"
    # Loading the Subtitle
    subs = pysrt.open(input_srt)
    subtitles=[]
    len_subs=len(subs)
    for i in range(len_subs):
        sub = subs[i]
        # Subtitle text
        text = sub.text
        text_without_tags = sub.text_without_tags

        # Start and End time
        start = sub.start.to_time()
        if start.second==0 and start.microsecond==0:
            start=datetime.time(0,0,0,0)
        end = sub.end.to_time()
        subtitles.append([text,text_without_tags,start,end])
        print len(subtitles)
        print subtitles[i]
        
    print os.getcwd()
    
    #FOR LOOP FOR ALL   AUDIO FILE
    audioFile="/home/mscuser/multi/multimodal_audio/sounds/Travis.wav"
    wavSegmentationFromSubs(audioFile,subtitles)
    

    path = '/home/mscuser/multi/multimodal_audio'
    walktree(path, find_sentiment)
    print dict

if __name__ == "__main__":
    main()
