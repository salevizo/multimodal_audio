import csv
import os
import os, re, sys
from stat import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import audioSegmentation as aS

'''Convert vtt format to srt format'''

def convertContent(fileContents):
    replacement = re.sub(r'([\d]+)\.([\d]+)', r'\1,\2', fileContents)
    replacement = re.sub(r'WEBVTT\n\n', '', replacement)
    replacement = re.sub(r'^\d+\n', '', replacement)
    replacement = re.sub(r'\n\d+\n', '\n', replacement)
    return replacement

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
    return f.read().decode("windows-1252").encode('ascii', 'ignore')

def vtt_to_srt(strNameFile):
    fileContents = readTextFile(strNameFile)
    strData = ""
    strData = strData + convertContent(fileContents)
    strNameFile = strNameFile.replace(".vtt",".srt")
    print "Convert vtt files to srt format at path < /home/mscuser/multi/multimodal_audio/subtitles> "
    #print(strNameFile)
    fileCreate(strNameFile, strData)


def walktree(TopMostPath, callback):
    '''recursively descend the directory tree rooted at TopMostPath,
       calling the callback function for each regular file'''
    for f in os.listdir(TopMostPath):
        pathname = os.path.join(TopMostPath, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(pathname, callback)
        elif S_ISREG(mode):
            # It's a file, call the callback function
            callback(pathname)
        else:
            # Unknown file type, print a message
            print('Skipping %s' % pathname)

def convertVTTtoSRT(file):
    os.chdir('/home/mscuser/multi/multimodal_audio/subtitles')
    if '.vtt' in file:
        vtt_to_srt(file)


def main():
    '''urls of youtube videos'''
    urls=[]
    with open("dataset.csv", "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        print "CSV contains the following urls:"
        for i, line in enumerate(reader):
            print (line[0])
            urls.append(line[0])
    f=open("dataset.csv", "rt")
    reader=csv.reader(f)
    headers = next(reader, None)
    dataset = {}
    for h in headers:
        dataset[h] = []
    for row in reader:
        for h, v in zip(headers, row):
            dataset[h].append(v)
    print "-------------------------------------------------------------------------------------------------------------------"
    '''download available captions in english'''
    print "Vtt subtitles for the above urls will be downloaded at the path < /home/mscuser/multi/multimodal_audio/subtitles> "
    for url in urls:
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en ' + url
        os.chdir('/home/mscuser/multi/multimodal_audio/subtitles')
        os.system(strcaption)
    print "-------------------------------------------------------------------------------------------------------------------"
    print "mp3 file for the above urls will be downloaded at the path < /home/mscuser/multi/multimodal_audio/audio> "
    for url in urls:
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en ' + url
        os.chdir('/home/mscuser/multi/multimodal_audio/audio')
        strmp3='youtube-dl --extract-audio --audio-format mp3 ' + url
        os.system(strmp3)
   
    print "-------------------------------------------------------------------------------------------------------------------"
    ''' download  the audio, subtitles and convert them to srt format'''
    path = '/home/mscuser/multi/multimodal_audio/sounds'
    walktree(path, convertVTTtoSRT)
    
    '''Convert mp3 to wav
    call function: dirMp3toWav from pyaudioAnalysis with args:16000 -c 1'''
    cmd='python /home/cer/Desktop/multimodal/pyAudioAnalysis/pyAudioAnalysis/audioAnalysis.py dirMp3toWav -i /home/cer/Desktop/multimodal/multimodal_audio/sounds -r 16000 -c 1'
    os.system(cmd)


if __name__ == "__main__":
    main()











