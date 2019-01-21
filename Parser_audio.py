import csv
import os, re, sys
from pyAudioAnalysis.audioSegmentation import silenceRemoval as sR 
from pyAudioAnalysis.audioBasicIO import readAudioFile
import pickle
from sklearn.model_selection import KFold
import FtrainTest as ft
import math
from shutil import copyfile
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append('Audio_Functions')
import File_Functions as ff
import Parse_Functions as pf
import time

def main(argv):
    repo_path = str(os.getcwd())
    os.chdir(repo_path)

    '''load the pickle file that contains info about the dataset'''
    pkl_file_tr = open(repo_path+'/train/dataset.p', 'rb')
    dataset_tr = pickle.load(pkl_file_tr)
    pkl_file_tr.close()

    pkl_file_te = open(repo_path+'/test/dataset_test.p', 'rb')
    dataset_te = pickle.load(pkl_file_te)
    pkl_file_te.close()
    
    ff.create_folders(repo_path+'/train') ## create test/train folders
    ff.create_folders(repo_path+'/test')

    #split for all IDs
    for k in dataset_tr['Id']:
        ff.create_folders_perID(repo_path+'/train/',str(k),'train')  
    
    for k in dataset_te['Id']:
        ff.create_folders_perID(repo_path+'/test/',str(k),'test')
    # Wav segmentation of all
    for k in dataset_tr['Id']:
        p=[x for x in dataset_tr['Pickle'] if '_'+k+'.p' in x][0]
        id_=p.split('.')[0][-1]
        subtitles=ff.retrieveSubs(p,repo_path+'/train')
        pf.wavSegmentationFromSubs_perID('train',subtitles,repo_path,str(k))
        

    '''
    for k in dataset_te['Id']:
        p=[x for x in dataset_te['Pickle'] if '_'+k+'.p' in x][0]
        id_=p.split('.')[0][-1]
        subtitles=ff.retrieveSubs(p,repo_path+'/test')
        pf.wavSegmentationFromSubs_perID('test',subtitles,repo_path,str(k))
    #for the train set
    '''
    os.chdir(repo_path)
    #ft.f(repo_path+'/train',dataset_tr)  ## svm
    #ft.f_model(repo_path+'/train',dataset_tr) #blstm
    ft.final(repo_path+'/test')

if __name__ == "__main__":
    main(sys.argv[1:])
