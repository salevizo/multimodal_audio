import os
import shutil
from stat import *
#import datetime
from datetime import datetime
from decimal import Decimal
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

def datetime_to_secs(t):
    times=t.split(":")
    f=float(times[2])
    f=Decimal(f)
    return float(int(times[0])*3600+int(times[1])*60+round(f,3))

def wavSegmentationFromSubs_perID(folder,subtitles,repo_path,audio_cnt):
    soundsPath=repo_path+"/"+folder+"/audio"
    os.chdir(soundsPath)
    
    
    #load list
    countpos=countneg=countneu=0
    for i, val in enumerate(subtitles):
        dir_name=audio_cnt
        if folder=='train':
            if subtitles[i][1]==0:
                dir_name=dir_name+"/neutral"
                countneu+=1
            elif subtitles[i][1]<0:
                dir_name=dir_name+"/negative"
                countneg+=1
            else:
                dir_name=dir_name+"/positive"
                countpos+=1
        os.chdir(soundsPath+"/"+dir_name)
        if folder=='train':
            t1=str(subtitles[i][0][0]).replace(',','.')
            t2=str(subtitles[i][0][1]).replace(',','.')
        else:
            t1=str(subtitles[i][0]).replace(',','.')
            t2=str(subtitles[i][1]).replace(',','.')
        start=datetime_to_secs(t1)
        d1 = datetime.strptime(str(t1), "%H:%M:%S.%f")
        d2 = datetime.strptime(str(t2), "%H:%M:%S.%f")
        sec=(d2-d1).total_seconds()
        audio=repo_path+"/"+folder+"/audio/"+audio_cnt+'.wav'
        mstr="ffmpeg -i {} -ss {} -t {} {}temp{}.wav -loglevel panic -y".format(audio, start, sec,audio_cnt,i)
        print(mstr)
        if os.path.isfile(str(audio_cnt)+"temp"+str(i)+".wav") is False:
            
            
            print(mstr)
            os.system(mstr)
        else:
            print(str(audio_cnt)+"temp"+str(i)+".wav already exists.")
        
       
    print('Done splitting wav file '+audio_cnt+'. Audio had '+str(countpos)+' positive segments, '+str(countneg)+ " negative segments and "+str(countneu)+"neutral segments. ")
    

def sentiments(filepath):
    results=aT.fileRegression(filepath, pyaudioanalysis_path, "svm")
    return results
