from pyAudioAnalysis import audioFeatureExtraction as aF
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import plotly
import plotly.graph_objs as go
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shutil
import sys
import Parser_audio as ap
import audioTrainTest_prj as aT
import pickle as cPickle
sys.path.append('Audio_Functions')
import File_Functions as ff
import Parse_Functions as pf



def writeTrainDataToARFF(model_name, features, classNames, feature_names):
    f = open(model_name + ".arff", 'w')
    f.write('@RELATION ' + model_name + '\n')
    for fn in feature_names:
        f.write('@ATTRIBUTE ' + fn + ' NUMERIC\n')
    f.write('@ATTRIBUTE class {')
    for c in range(len(classNames) - 1):
        f.write(classNames[c] + ',')
    f.write(classNames[-1] + '}\n\n')
    f.write('@DATA\n')
    for c, fe in enumerate(features):
        for i in range(fe.shape[0]):
            for j in range(fe.shape[1]):
                f.write("{0:f},".format(fe[i, j]))
            f.write(classNames[c] + "\n")
    f.close()


def compute_class_rec_pre_f1(c_mat):
    '''
    :param c_mat: the [n_class x n_class] confusion matrix
    :return: rec, pre and f1 for each class
    '''
    n_class = c_mat.shape[0]
    rec, pre, f1 = [], [], []
    for i in range(n_class):
        rec.append(float(c_mat[i, i]) / np.sum(c_mat[i, :]))
        pre.append(float(c_mat[i, i]) / np.sum(c_mat[:, i]))
        f1.append(2 * rec[-1] * pre[-1] / (rec[-1] + pre[-1]))
    return rec,  pre, f1


def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)



def plotly_classification_results(cm, class_names):
    heatmap = go.Heatmap(z=np.flip(cm, axis=0), x=class_names,
                         y=list(reversed(class_names)),
                         colorscale=[[0, '#4422ff'], [1, '#ff4422']],
                         name="confusin matrix", showscale=False)
    rec, pre, f1 = compute_class_rec_pre_f1(cm)
    mark_prop1 = dict(color='rgba(150, 180, 80, 0.5)',
                      line=dict(color='rgba(150, 180, 80, 1)', width=2))
    mark_prop2 = dict(color='rgba(140, 200, 120, 0.5)',
                      line=dict(color='rgba(140, 200, 120, 1)', width=2))
    mark_prop3 = dict(color='rgba(50, 150, 220, 0.5)',
                      line=dict(color='rgba(50, 150, 220, 1)', width=3))
    b1 = go.Bar(x=class_names,  y=rec, name="rec", marker=mark_prop1)
    b2 = go.Bar(x=class_names,  y=pre, name="pre", marker=mark_prop2)
    b3 = go.Bar(x=class_names,  y=f1, name="f1", marker=mark_prop3)
    figs = plotly.tools.make_subplots(rows=1, cols=2,
                                      subplot_titles=["Confusion matrix",
                                                      "Performance measures"])
    figs.append_trace(heatmap, 1, 1); figs.append_trace(b1, 1, 2)
    figs.append_trace(b2, 1, 2); figs.append_trace(b3, 1, 2)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)



def svm_train_evaluate(X, y,x_test,y_test,k_folds, C, use_regressor=False):
    '''
    :param X: Feature matrix
    :param y: Labels matrix
    :param k_folds: Number of folds
    :param C: SVM C param
    :param use_regressor: use svm regression for training (not nominal classes)
    :return: confusion matrix, average f1 measure and overall accuracy
    '''

    params_list ={}
    # normalize
    mean, std = X.mean(axis=0), np.std(X, axis=0)
    X = (X - mean) / std
    # k-fold evaluation:
    f1s, accs, count_cm = [], [], 0
    #for differenct values of c 

    if not use_regressor:
        cl = SVC(kernel='rbf', C=C)
    else:
        cl = SVR(kernel='rbf', C=C)
    ##fit_and_resample()
    cl.fit(X, y)
    y_pred = cl.predict(x_test)
    if use_regressor:
        y_pred = np.round(y_pred)
    # update aggregated confusion matrix:
    if count_cm == 0:
        cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    else:
        cm += (confusion_matrix(y_pred=y_pred, y_true=y_test))
    count_cm += 1
    #f1s.append(f1_score(y_pred=y_pred, y_true=y_test, average='micro'))
    #accs.append(accuracy_score(y_pred=y_pred, y_true=y_test))
    f1 = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print("KARIOLI -----> FOR C:",C,"F1: ",f1, "ACC:",acc)
    return cm, f1, acc







def featureAndTrain(list_of_dirs_train, list_of_dirs_test, mt_win, mt_step, st_win, st_step,classifier_type, model_name,C,compute_beat=False,k_folds=3):

	#feature extraction for train/test
    [features_train, classNames_train, filenames_train] = aF.dirsWavFeatureExtraction(list_of_dirs_train, mt_win,mt_step, st_win, st_step,compute_beat=compute_beat)

    [features_test, classNames_test, filenames_test] = aF.dirsWavFeatureExtraction(list_of_dirs_test, mt_win, mt_step,st_win, st_step,compute_beat=compute_beat)


    [x_test, y_test] = listOfFeatures2Matrix(features_test)

    ## for training SMOTE 
    [X_train, Y_train] = listOfFeatures2Matrix(features_train)
    sm = SMOTE(random_state=2)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)

    cm, acc, f1 = svm_train_evaluate(X_train, Y_train,x_test,y_test,k_folds, C)

   
    return cm,acc,f1



def f(repo_path,dataset):
    best_scores = []
    #find best params and crossvalidation
    classifier_par = numpy.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0])
    for C in classifier_par:
        e=0
        kfold = KFold(n_splits=3,shuffle=True)
        for train, test in kfold.split(np.array(dataset["Id"])):
            print("For C: ",C, "For Fold: ",e+1 )
            #create also train/test folders 
            ff.create_folders(repo_path)
            print("Train: ",train , "Test: ",test)
            for k in train:
                
                path= repo_path + "/audio/"+str(dataset['Id'][k])
                #copy video in CASE(train/test) folder
                pf.searchVideo(path,repo_path,"train")   
            for k in test:
                path= repo_path + "/audio/"+str(dataset['Id'][k])  
                pf.searchVideo(path,repo_path,"test") 

            cm,acc,f1 = featureAndTrain([repo_path+"/audio/train/positive",repo_path+"/audio/train/neutral",repo_path+"/audio/train/negative"],[repo_path+"/audio/test/positive",repo_path+"/audio/test/neutral",repo_path+"/audio/test/negative"],1.0,1.0,aT.shortTermWindow,aT.shortTermStep,"svm","svm5Classes",C)
            best_scores.append([C,cm,acc,f1])
            ff.remove_folders(repo_path)
    print(best_scores)

    ##find best f1 for optimal C
    best_f1= []
    for i in  range(len(best_scores)):
        best_f1.append(best_scores[i][2])

    m = max(best_f1)
    best_c=[i for i, j in enumerate(best_f1) if j == m]

    best_c=best_scores[best_c[0]]
    print("best Params ------->: ",best_c)
    ##visualise with the best score
    # visualize performance measures 
    #pos 1 is the cm matrix
    plotly_classification_results(best_c[1], ["positve", "neutral", "negative"]) 
    #print(acc, f1)


    ##normalize again this time all dataset this time and fit with the best params 
    ff.create_folders(repo_path)
    for k in dataset["Id"]:
        path= repo_path + "/audio/"+str(k)
        ap.searchVideo(path,repo_path,"train")


    [features_train, classNames_train, filenames_train] = aF.dirsWavFeatureExtraction([repo_path+"/audio/train/positive",repo_path+"/audio/train/neutral",repo_path+"/audio/train/negative"], 1.0,1.0,aT.shortTermWindow,aT.shortTermStep,compute_beat=False)
    [X, Y] = listOfFeatures2Matrix(features_train)


    sm = SMOTE(random_state=2)
    ##if fails here check number of instances from each class.smote has neighbours=5 as init parameter. So if a class has below 5 instances smote fails. Try put more instaces or change k
    x, y = sm.fit_sample(X, Y)

    MEAN, STD = x.mean(axis=0), np.std(x, axis=0)
    X = (x - MEAN) / STD
    cl = SVC(kernel='rbf', C=best_c[0])
    cl.fit(X, y)
    mt_win = 1.0
    mt_step= 1.0
    st_win = aT.shortTermWindow
    st_step=aT.shortTermStep
    compute_beat=False
    with open("svm5Classes", 'wb') as fid:
        cPickle.dump("svm", fid)
        fo = open("svm5Classes" + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames_train, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()

    ap.remove_folders(repo_path)
