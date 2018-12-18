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
import pickle
from datetime import datetime
from sklearn.model_selection import KFold
import FtrainTest as ft
import shutil
import math
from shutil import copyfile
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import StratifiedShuffleSplit
import audioTrainTest_prj as aT


def main(argv):
    repo_path = str(os.getcwd())
    pyaudioanalysis_path=str(sys.argv[1])
    os.chdir(repo_path)

    '''load the pickle file that contains info about the dataset'''
    pkl_file_tr = open('/train/dataset_list.p', 'rb')
    pkl_file_te = open('/test/dataset_list.p', 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    '''
    create_folders(repo_path) ## create test/train folders
    subs=[]
    
    
    #split for all IDs
    for k in range(0,len(dataset["Pickle"])):
        create_folders_perID(repo_path,str(k))  


    # Wav segmentation of all
    for k in range(0,len(dataset["Pickle"])):
        subtitles=retrieveSubs(dataset["Pickle"][k],repo_path)
        wavSegmentationFromSubs_perID(dataset["Audio"][k],subtitles,repo_path,str(k))

    ft.f(repo_path,dataset)
    '''


if __name__ == "__main__":
    main(sys.argv[1:])
