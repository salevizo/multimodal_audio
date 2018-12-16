import Files_And_Dirs as fad
import os
import subprocess
def download_mp3(repo_path,urls,ids,test_folder=''):
    fad.create_folders(repo_path,repo_path+test_folder+'/audio')
    print("mp3 file for the above urls will be downloaded at the path " + str(repo_path) + test_folder+"audio> ")
    tmp_test=''
    if test_folder!='':
        tmp_test=test_folder+'/'
    for idx,url in enumerate(urls):
        if os.path.isfile(tmp_test+ids[idx]+".wav")==False:
            strcaption='youtube-dl --write-auto-sub --skip-download --sub-lang=en ' + url
            os.chdir(repo_path +test_folder+'/audio')
            strmp3='youtube-dl --extract-audio --audio-format mp3  --output '+ids[idx]+".mp3 " + url
            #print strmp3
            os.system(strmp3)
        else:
            print("Wav already exists,no need to download "+str(ids[idx])+".mp3 ")

def mp3_to_wav_and_remove_mp3(repo_path,audio_files,test_folder=''):
    for mp3_ in audio_files:
        print mp3_
        if ".mp3" in mp3_ or ".wav" in mp3_:
            name=mp3_.split('.')
            print os.getcwd()
            if os.path.isfile(test_folder+name[0]+".wav")==False:
                print mp3_
                print test_folder+name[0]+".wav"
                subprocess.call(['ffmpeg', '-i', repo_path +test_folder+ "/audio/" + name[0] + ".mp3",repo_path +test_folder+"/audio/" + name[0] + ".wav"])
                if os.path.isfile(repo_path +test_folder+"/audio/" + name[0] + ".mp3")==True:
                    os.remove(repo_path +test_folder+"/audio/" + name[0] + ".mp3")
            else:
                print("File already exists,no need to convert to  "+str(name[0]) + ".wav")
        else:
            print("Encountered folder "+mp3_)
