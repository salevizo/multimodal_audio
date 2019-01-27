import os
import sys
sys.path.append("Download_Functions")
import Files_And_Dirs as fad
import Download_Subtitles as d_subs
import Download_Audio as d_aud


def main(argv):
    global repo_path
    repo_path = str(os.getcwd())
    os.chdir(repo_path)
    fad.create_folders(repo_path,repo_path +'/train')
    fad.create_folders(repo_path,repo_path +'/test')
    fad.create_folders(repo_path,repo_path +'/train/subtitles')
    fad.create_folders(repo_path,repo_path +'/test/subtitles')
    fad.create_folders(repo_path,repo_path +'/train/audio')
    fad.create_folders(repo_path,repo_path +'/test/audio')
    fn_tr="dataset_train.csv"
    fn_te="dataset_test.csv"

    (urls_tr,ids_tr)=fad.read_url_and_ids(fn_tr)
    (urls_te,ids_te)=fad.read_url_and_ids(fn_te)
    (dataset_tr)=fad.read_csv(fn_tr,'train/')
    (dataset_te)=fad.read_csv(fn_te,'test/')
    
    
    print( "-------------------------dataset.p will contain: --------------------------------------"     )
    print(dataset_tr)
    print( "-------------------------test/dataset_test.p will contain: --------------------------------------"     )
    print(dataset_te)

    d_subs.download_subs(repo_path,urls_tr,ids_tr,'/train/')
    d_subs.download_subs(repo_path,urls_te,ids_te,'/test/')
        
    print("-------------------------------------------------------------------------------------------------------------------")
    d_aud.download_mp3(repo_path,urls_tr,ids_tr,'/train')
    d_aud.download_mp3(repo_path,urls_te,ids_te,'/test')

    fad.walktree(repo_path,d_subs.convertVTTtoSRT,'/train')
    fad.walktree(repo_path,d_subs.convertVTTtoSRT,'/test')
    
    audio_files_tr=os.listdir(repo_path + '/train'+'/audio')
    audio_files_te=os.listdir(repo_path + '/test'+'/audio')

    print('Training files:'+str(audio_files_tr))
    print('Test files:'+str(audio_files_te))
    os.chdir(repo_path+'/train/audio')
    d_aud.mp3_to_wav_and_remove_mp3(repo_path,audio_files_tr,'/train')
    os.chdir(repo_path+'/test/audio')
    d_aud.mp3_to_wav_and_remove_mp3(repo_path,audio_files_te,'/test')
    

    
if __name__ == "__main__":
    main(sys.argv[1:])











