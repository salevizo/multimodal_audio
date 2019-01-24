
import time
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
import Parser_audio as ap
import pickle as cPickle
import sys
sys.path.append("Audio_Functions")
import File_Functions as ff
import Parse_Functions as pf
import os
shortTermWindow = 0.050
shortTermStep = 0.050
eps = 0.00000001

import math

globalVar = 0
max_len = 1024
nb_features = 36

nb_attention_param = 256
attention_init_value = 1.0 / 256
nb_hidden_units = 512   # number of hidden layer units
dropout_rate = 0.5
nb_lstm_cells = 128
nb_classes = 3

masking_value = -100.0

frame_size = 0.025  # 25 msec segments
step = 0.01     # 10 msec time step


def normalizeFeatures(features):
    '''
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - features_norm:    list of NORMALIZED feature matrices
        - MEAN:        mean vector
        - STD:        std vector
    '''
    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)


def load_model(model_name, is_regression=False):
    '''
    This function loads an SVM model either for classification or training.
    ARGMUMENTS:
        - SVMmodel_name:     the path of the model to be loaded
        - is_regression:     a flag indigating whereas this model is regression or not
    '''
    try:
        fo = open(model_name + "MEANS", "rb")
    except IOerror:
        print("Load SVM model: Didn't find file")
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        if not is_regression:
            classNames = cPickle.load(fo)
        mt_win = cPickle.load(fo)
        mt_step = cPickle.load(fo)
        st_win = cPickle.load(fo)
        st_step = cPickle.load(fo)
        compute_beat = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    with open(model_name, 'rb') as fid:
        SVM = cPickle.load(fid)

    if is_regression:
        return (SVM, MEAN, STD, mt_win, mt_step, st_win, st_step, compute_beat)
    else:
        return (SVM, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step, compute_beat)



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
        if np.sum(c_mat[i, :])==0:
            rec.append(0)
        else:
            rec.append(float(c_mat[i, i]) / np.sum(c_mat[i, :]))
        if np.sum(c_mat[:, i])==0:
            pre.append(0)
        else:
            pre.append(float(c_mat[i, i]) / np.sum(c_mat[:, i]))
        if(rec[-1] + pre[-1])==0:
            f1.append(0)        
        else:
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
        size=0
        if numpy.array(f).ndim==1:
            size=1
        else:
            size=len(f)
        if i == 0:
            X = f
            Y = i * numpy.ones((size, 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((size, 1)))
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
    f1s, accs, count_cm = [], [], 0
    # probability=True
    if not use_regressor:
        cl = SVC(kernel='linear', C=C)
    else:
        cl = SVR(kernel='rbf', C=C)
    ##fit_and_resample()
    cl.fit(X, y)
    y_pred = cl.predict(x_test)
    if use_regressor:
        y_pred = np.round(y_pred)
    print("y_pred:",y_pred)
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







def featureAndTrain(list_of_dirs_train, list_of_dirs_test, mt_win, mt_step, st_win, st_step,classifier_type, model_name,C,model,compute_beat=False,k_folds=3):

    #feature extraction for train/test
    [features_train, classNames_train, filenames_train] = aF.dirsWavFeatureExtraction(list_of_dirs_train, mt_win,mt_step, st_win, st_step,compute_beat=compute_beat)

    [features_test, classNames_test, filenames_test] = aF.dirsWavFeatureExtraction(list_of_dirs_test, mt_win, mt_step,st_win, st_step,compute_beat=compute_beat)


  
    features2 = []
    for f in features_train:
        fTemp = []
        for i in range(f.shape[0]):
            temp = f[i, :]
            if (not numpy.isnan(temp).any()) and (not numpy.isinf(temp).any()):
                fTemp.append(temp.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        features2.append(numpy.array(fTemp))
    features_train = features2

    features3 = []
    for f in features_test:
        fTemp = []
        for i in range(f.shape[0]):
            temp = f[i, :]
            if (not numpy.isnan(temp).any()) and (not numpy.isinf(temp).any()):
                fTemp.append(temp.tolist())
            else:
                print("NaN Found! Feature vector not used for testing")
        features3.append(numpy.array(fTemp))
    features_test = features3

    # MEAN, STD = x.mean(axis=0), np.std(x, axis=0)
    # X = (x - MEAN) / STD
    (features_norm_train, MEAN, STD) = normalizeFeatures(features_train)
    n_classes_train = len(features_train)

    (features_norm_test, MEAN_test, STD_test) = normalizeFeatures(features_test)
    n_classes_test = len(features_test)

    [x_test, y_test] = listOfFeatures2Matrix(features_norm_test)
    

    ## for training SMOTE

    [X_train, Y_train] = listOfFeatures2Matrix(features_norm_train)
    print("Before OverSampling, counts of label 'positive': {}".format(sum(Y_train==1)))
    print("Before OverSampling, counts of label 'neutral': {} \n".format(sum(Y_train==0)))
    print("Before OverSampling, counts of label 'negative': {} \n".format(sum(Y_train==2)))

    sm = SMOTE(random_state=2,kind='svm')
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    print("A OverSampling, counts of label 'positive': {}".format(sum(Y_train==1)))
    print("A OverSampling, counts of label 'neutral': {} \n".format(sum(Y_train==0)))
    print("A OverSampling, counts of label 'negative': {} \n".format(sum(Y_train==2)))
    time.sleep(5)
    print("!="+str(features_train))
    print("x="+str(X_train))
    print("y="+str(Y_train))
    print("lx="+str(len(X_train)))
    print("lx="+str(len(Y_train)))
    # time.sleep(5)
    if model ==0:
        cm, acc, f1 = svm_train_evaluate(X_train, Y_train,x_test,y_test,k_folds, C)
        return cm,acc,f1
    else:
        create_model(X_train, Y_train,x_test,y_test)

def f_model(repo_path,dataset):
	from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
	from keras.layers.merge import dot
	from keras.models import Model, load_model
	from keras.utils import to_categorical
	from keras.callbacks import EarlyStopping
	from keras.callbacks import ModelCheckpoint
	from keras import optimizers
	from keras import backend as k
	from hyperas import optim
	from hyperopt import Trials, tpe
	from hyperopt import STATUS_OK
	from hyperas.distributions import choice
    perm = np.random.permutation(np.array(dataset["Id"]))
    train = perm[:(int(len(perm) * 0.7))]
    test = perm[(math.ceil(len(perm) * 0.3)):]
    ff.create_folders(repo_path)
    print("Train: ",train , "Test: ",test)
    for k in train:
        path= repo_path + "/audio/"+str(dataset['Id'][k])
        #copy video in CASE(train/test) folder
        pf.searchVideo(path,repo_path,"train")   
    for k in test:
        path= repo_path + "/audio/"+str(dataset['Id'][k])  
        pf.searchVideo(path,repo_path,"test")
    featureAndTrain([repo_path+"/audio/train/positive",repo_path+"/audio/train/neutral",repo_path+"/audio/train/negative"],[repo_path+"/audio/test/positive",repo_path+"/audio/test/neutral",repo_path+"/audio/test/negative"],1.0,1.0,shortTermWindow,shortTermStep,"svm","svm5Classes",C,1)



def f(repo_path,dataset):
    best_scores = []
    #find best params and crossvalidation
    classifier_par = numpy.array([0.01, 0.05, 0.1, 0.25]) #0.5, 1.0, 5.0, 10.0])
    e=0
    kfold = KFold(n_splits=2,shuffle=True)
    for train, test in kfold.split(np.array(dataset["Id"])):
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
        for C in classifier_par:
            print("For C: ",C, "For Fold: ",e )
            cm,acc,f1 = featureAndTrain([repo_path+"/audio/train/positive",repo_path+"/audio/train/neutral",repo_path+"/audio/train/negative"],[repo_path+"/audio/test/positive",repo_path+"/audio/test/neutral",repo_path+"/audio/test/negative"],1.0,1.0,shortTermWindow,shortTermStep,"svm","svm5Classes",C,0)
            best_scores.append([C,cm,acc,f1])
        e=e+1
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
    print(best_c[1])
    plotly_classification_results(best_c[1], ["positive", "neutral", "negative"]) 
    #print(acc, f1)


    ##normalize again this time all dataset this time and fit with the best params 
    ff.create_folders(repo_path)
    for k in dataset["Id"]:
        path= repo_path + "/audio/"+str(k)
        pf.searchVideo(path,repo_path,"train")


    [features, classNames, filenames] = aF.dirsWavFeatureExtraction([repo_path+"/audio/train/positive",repo_path+"/audio/train/neutral",repo_path+"/audio/train/negative"], 1.0,1.0,shortTermWindow,shortTermStep,compute_beat=False)
    [features_norm, MEAN, STD] = normalizeFeatures(features)        # normalize features
    # MEAN, STD = x.mean(axis=0), np.std(x, axis=0)
    # X = (x - MEAN) / STD

    os.chdir("../") ##one folder back
    MEAN = MEAN.tolist()
    STD = STD.tolist()
    featuresNew = features_norm
    [X, Y] = listOfFeatures2Matrix(featuresNew)
    
    ##if fails here check number of instances from each class.smote has neighbours=5 as init parameter. So if a class has below 5 instances smote fails. Try put more instaces or change k

    sm = SMOTE(random_state=2)
    # print("!="+str(features_train))
    # print("x="+str(X))
    # print("y="+str(Y))
    x, y = sm.fit_sample(X, Y)

   
    cl = SVC(kernel='linear', C=best_c[0])
    classifier=cl.fit(x, y)
    mt_win = 1.0
    mt_step = 1.0
    st_win = shortTermWindow
    st_step = shortTermStep
    compute_beat = False
    with open("svm3Classes", 'wb') as fid:
        cPickle.dump(classifier, fid)
    fo = open("svm3Classes" + "MEANS", "wb")
    cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    ff.remove_folders(repo_path)


def ff_one(repo_path, dataset):
    ## normalize each video id and then put train. Select each one from the corresponding position in dict 
    video_dict={}
    for i in range(np.array(dataset["Id"])):
        print(i)
        print([repo_path + "/audio/"+str(i) +"/positive", repo_path + "/audio/"+str(i) +"/neutral", repo_path +"/audio/"+str(i) +"/negative"])
        [features_train, classNames, filenames] = aF.dirsWavFeatureExtraction( [repo_path + "/audio/"+str(i) +"/positive", repo_path + "/audio/"+str(i) +"/neutral", repo_path +"/audio/"+str(i) +"/negative"], 1.0,
        1.0, shortTermWindow, shortTermStep, 0,compute_beat=False)
        print([features_train, classNames, filenames])
        [features_norm, MEAN, STD] = normalizeFeatures(features_train)        # normalize features
        video_dict[i] = [features_norm, MEAN, STD]


    for train, test in kfold.split(dataset["Pickle"][:25]):
        video_test = []
        video_train = []
        for t in train:
            video_train = np.concatenate((video_test,video_dict[t][0]),axis=0)
        for te in test:
            video_test = np.concatenate((video_test,video_dict[te][0]),axis=0)


        [X_train, Y_train] = listOfFeatures2Matrix(video_train)
        [x_test, y_test] = listOfFeatures2Matrix(video_test)

        print("Before OverSampling, counts of label 'positive': {}".format(sum(Y_train==1)))
        print("Before OverSampling, counts of label 'neutral': {} \n".format(sum(Y_train==0)))
        print("Before OverSampling, counts of label 'negative': {} \n".format(sum(Y_train==2)))

        sm = SMOTE(random_state=2,kind='svm')
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
        print("A OverSampling, counts of label 'positive': {}".format(sum(Y_train==1)))
        print("A OverSampling, counts of label 'neutral': {} \n".format(sum(Y_train==0)))
        print("A OverSampling, counts of label 'negative': {} \n".format(sum(Y_train==2)))


        # print("X:",X_train)
        # print("Y:",Y_train)
        cm, acc, f1 = svm_train_evaluate(X_train, Y_train, x_test, y_test, k_folds, C)

def ff_all(repo_path, dataset):
    #norm all and then..
    pass

def final(repo_path_to_test):
    ##load model
    [SVM, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step, compute_beat]=load_model("svm3Classes")
    ##load test,split X,y
    [features_test, classNames_test, filenames_test] = aF.dirsWavFeatureExtraction([repo_path_to_test+"/positive",repo_path_to_test+"/neutral",repo_path_to_test+"/negative"],1.0,1.0, shortTermWindow, shortTermStep,compute_beat=False)
    [x_test, y_test] = listOfFeatures2Matrix(features_test)
    SVM.predict(x_test)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    f1 = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    plotly_classification_results(cm, ["positive", "neutral", "negative"])
    print("FINAL -----> FOR C:",C,"F1: ",f1, "ACC:",acc)


def create_model(x_train, y_train, x_test, y_test):

    u_train = np.full((x_train.shape[0], nb_attention_param),
                      attention_init_value, dtype=np.float32)
    u_test = np.full((x_test.shape[0], nb_attention_param),
                     attention_init_value, dtype=np.float32)

    with k.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=(max_len, nb_features))
        x = Masking(mask_value=masking_value)(input_feature)
        x = Dense(nb_hidden_units, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(nb_hidden_units, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)

    with k.name_scope('AttentionLayer'):
        # Logistic regression for learning the attention parameters with a standalone feature as input
        input_attention = Input(shape=(nb_lstm_cells * 2,))
        u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)

        # To compute the final weights for the frames which sum to unity
        alpha = dot([u, y], axes=-1)  # inner prod.
        alpha = Activation('softmax')(alpha)

    with k.name_scope('WeightedPooling'):
        # Weighted pooling to get the utterance-level representation
        z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(nb_classes, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)

    choice_val = {{choice(['adam', 'rmsprop', 'sgd'])}}
    if choice_val == 'adam':
        optimizer = optimizers.Adam()
    elif choice_val == 'rmsprop':
        optimizer = optimizers.RMSprop()
    else:
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    globalVar += 1

    file_path = 'weights_blstm_hyperas_' + str(globalVar) + '.h5'
    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='auto'
        ),
        ModelCheckpoint(
            filepath=file_path,
            monitor='val_acc',
            save_best_only='True',
            verbose=1,
            mode='max'
        )
    ]

    hist = model.fit([u_train, x_train], y_train, batch_size=128, epochs={{choice([100, 150, 200])}}, verbose=2,
                     callbacks=callback_list, validation_data=([u_test, x_test], y_test))
    h = hist.history
    acc = np.asarray(h['acc'])
    loss = np.asarray(h['loss'])
    val_loss = np.asarray(h['val_loss'])
    val_acc = np.asarray(h['val_acc'])

    acc_and_loss = np.column_stack((acc, loss, val_acc, val_loss))
    save_file_blstm = 'blstm_run_' + str(globalVar) + '.txt'
    with open(save_file_blstm, 'w'):
        np.savetxt(save_file_blstm, acc_and_loss)

    score, accuracy = model.evaluate([u_test, x_test], y_test, batch_size=128, verbose=1)
    print("Final validation accuracy: %s" % accuracy)

    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
