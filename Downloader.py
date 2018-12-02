import csv
import os
import os, re, sys
from stat import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import audioSegmentation as aS
import pickle
import subprocess

repo_path=''


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
    if sys.version_info >=(3, 0):
        return f.read()
    else:
        return f.read().decode("windows-1252").encode('ascii', 'ignore') #----> python2

def vtt_to_srt(strNameFile):
    fileContents = readTextFile(strNameFile)
    strData = ""
    strData = strData + convertContent(fileContents)
    strNameFile = strNameFile.replace(".vtt",".srt")
    print("Convert vtt files to srt format at path " + str(repo_path) + "/subtitles> ")
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
    os.chdir(repo_path +'/subtitles')
    if '.vtt' in file:
        vtt_to_srt(file)


def create_folders(folder_path):
    os.chdir(repo_path)
    if (os.path.exists(folder_path))==False:
        #python understands octal so 0777 has to be 511 in decimal
        os.mkdir(folder_path)
        os.chmod(folder_path,511)
        print('Created ' + folder_path)
    else:
        print('Directory ' + folder_path +' exists.')


def main(argv):
    global repo_path
    if len(sys.argv)==2:
        print("Write in user specified path...")
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))
        #repo_path='/home/mscuser/multi/multimodal_audio'
        repo_path=str(sys.argv[1])
    else:
        repo_path = str(os.getcwd()) #able to work withou path . default write in the same directory
        print("Writing in the current path: ",repo_path)

    os.chdir(repo_path)
    '''urls of youtube videos'''
    urls=[]
    ids=[]
    with open("dataset.csv", "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        print("CSV contains the following urls:")
        for i, line in enumerate(reader):
            if i>0:
                l=line[0].split(',')
                urls.append(l[1])
                ids.append(l[0])
                print("Id:" + l[0] + " URL:" +str(l[1]))
    
    print("-------------------------------------------------------------------------------------------------------------------"     )
    f=open("dataset.csv", "rt")
    reader=csv.reader(f)
    headers = next(reader, None)
    dataset = {}
    for h in headers:
        dataset[h] = []
    for row in reader:
        for h, v in zip(headers, row):            
            dataset[h].append(v)
    print( "-------------------------dataset_list.p will contain: --------------------------------------"     )
    print(dataset)
    output = open('dataset_list.p', 'wb')
    pickle.dump(dataset, output)
    output.close()
    
    print("-------------------------------------------------------------------------------------------------------------------")
    '''download available captions in english inside subtitles with format 1.en.vtt, 2.en.vtt etc'''
    print("Vtt subtitles for the above urls will be downloaded at the path " +str(repo_path)+ "/subtitles> ")
    create_folders(repo_path +'/subtitles')
    for idx,url in enumerate(urls):
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en --output '+ids[idx]+".vtt " + url
        #print strcaption
        os.chdir(repo_path +'/subtitles')
        os.system(strcaption)
        
    print("-------------------------------------------------------------------------------------------------------------------")
    '''download available mp3 in english inside audio with format 1.mp3, 2.mp3t etc'''
    create_folders(repo_path +'/audio')
    print("mp3 file for the above urls will be downloaded at the path " + str(repo_path) + "/audio> ")
    for idx,url in enumerate(urls):
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en ' + url
        os.chdir(repo_path + '/audio')
        strmp3='youtube-dl --extract-audio --audio-format mp3  --output '+ids[idx]+".mp3 " + url
        #print strmp3
        os.system(strmp3)
   
    print("-------------------------------------------------------------------------------------------------------------------")

    path = repo_path+"/subtitles"
    walktree(path, convertVTTtoSRT)
    
    '''Convert mp3 to wav'''
    os.chdir(repo_path + '/audio')
    audio_files=os.listdir(repo_path + '/audio')
    print(audio_files)
    for mp3_ in audio_files:
        name=mp3_.split('.')
        subprocess.call(['ffmpeg', '-i', repo_path + "/audio/" + name[0] + ".mp3",repo_path + "/audio/" + name[0] + ".wav"])




if __name__ == "__main__":
    main(sys.argv[1:])











