import Files_And_Dirs as fad
import os
import re
def convertContent(fileContents):
    replacement = re.sub(r'([\d]+)\.([\d]+)', r'\1,\2', fileContents)
    replacement = re.sub(r'WEBVTT\n\n', '', replacement)
    replacement = re.sub(r'^\d+\n', '', replacement)
    replacement = re.sub(r'\n\d+\n', '\n', replacement)
    return replacement

def vtt_to_srt(strNameFile,repo_path,folder=''):
    fileContents = fad.readTextFile(strNameFile)
    strData = ""
    strData = strData + convertContent(fileContents)
    strNameFile = strNameFile.replace(".vtt",".srt")
    #print("Convert vtt files to srt format at path " + str(repo_path) +folder+ "/subtitles> ")
    #print(strNameFile)
    fad.fileCreate(strNameFile, strData)

def convertVTTtoSRT(file,repo_path,folder=''):
    os.chdir(repo_path +folder+'/subtitles')
    if '.vtt' in file:
        vtt_to_srt(file,repo_path,folder)

def download_subs(repo_path,urls,ids,folder=''):
    print("Vtt subtitles for the above urls will be downloaded at the path " +str(repo_path)+ folder+"/subtitles ")
    fad.create_folders(repo_path,repo_path+folder+'subtitles')
    for idx,url in enumerate(urls):
        strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en --output '+ids[idx]+".vtt " + url
        #print strcaption
        if os.path.isfile(ids[idx]+".en.vtt")==False:
            os.chdir(repo_path +folder+'/subtitles')
            os.system(strcaption)
        else:
            print(str(ids[idx])+".vtt already exists.")
