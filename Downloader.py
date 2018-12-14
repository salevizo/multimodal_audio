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
    repo_path = str(os.getcwd())
    os.chdir(repo_path)
    
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
    
    print("Vtt subtitles for the above urls will be downloaded at the path " +str(repo_path)+ "/subtitles> ")
    create_folders(repo_path +'/subtitles')
    for idx,url in enumerate(urls):
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en --output '+ids[idx]+".vtt " + url
        #print strcaption
        if os.path.isfile(ids[idx]+".en.vtt")==False:
            os.chdir(repo_path +'/subtitles')
            os.system(strcaption)
        else:
            print(str(ids[idx])+".vtt already exists.")
        
    print("-------------------------------------------------------------------------------------------------------------------")

    create_folders(repo_path +'/audio')
    print("mp3 file for the above urls will be downloaded at the path " + str(repo_path) + "/audio> ")
    for idx,url in enumerate(urls):
        if os.path.isfile(ids[idx]+".wav")==False:
            strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en ' + url
            os.chdir(repo_path + '/audio')
            strmp3='youtube-dl --extract-audio --audio-format mp3  --output '+ids[idx]+".mp3 " + url
            #print strmp3
            os.system(strmp3)
        else:
            print("Wav already exists,no need to download "+str(ids[idx])+".mp3 ")
   
    print("-------------------------------------------------------------------------------------------------------------------")

    path = repo_path+"/subtitles"
    walktree(path, convertVTTtoSRT)
    
    os.chdir(repo_path + '/audio')
    audio_files=os.listdir(repo_path + '/audio')
    print(audio_files)
    for mp3_ in audio_files:
        if ".mp3" in mp3_ or ".wav" in mp3_:
            name=mp3_.split('.')
            if os.path.isfile(name[0]+".wav")==False:
                subprocess.call(['ffmpeg', '-i', repo_path + "/audio/" + name[0] + ".mp3",repo_path + "/audio/" + name[0] + ".wav"])
                if os.path.isfile(repo_path +"/audio/" + name[0] + ".mp3")==True:
                    os.remove(repo_path +"/audio/" + name[0] + ".mp3")
            else:
                print("File already exists,no need to convert to  "+str(name[0]) + ".wav")
        else:
            print("Encountered folder "+mp3_)



if __name__ == "__main__":
    main(sys.argv[1:])











