import pickle
import numpy as np
import os
import sys
import nltk

nltk.download('stopwords')
nltk.download('punkt')
sys.path.append('Caption_Functions')
import File_Functions as ff
import Sentiment as sen




def main():

    repo_path = str(os.getcwd()) #able to work withou path . default write in the same directory
    print("Writing in the current path: ",repo_path)
    dict_tr=ff.walktree(repo_path+'/train/subtitles/', sen.sentiment,'train')
    dict_te=ff.walktree(repo_path+'/test/subtitles/', sen.sentiment,'test')
    ff.create_csv(repo_path+'/train/',dict_tr,'train')
    ff.create_csv(repo_path+'/test/',dict_te,'test')
    ff.create_pickle(repo_path+'/train/',dict_tr,'train')
    ff.create_pickle(repo_path+'/test/',dict_te,'test')
if __name__ == "__main__":
    main()

