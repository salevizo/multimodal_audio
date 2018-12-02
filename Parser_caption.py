import pysrt
from datetime import date, datetime, timedelta, time
from textblob import TextBlob
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os, re, sys
from stat import *
import re
import pickle
import numpy as np
import datetime
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
import sys
import spacy
from spacy.tokens import Doc




dict={}
i=interval_segments=sentiment_segments=0
repo_path=''




'''textblob sentiment analysis'''
def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity



'''VADER produces four sentiment metrics from these word ratings, 
which you can see below. The first three, positive, neutral and negative, 
represent the proportion of the text that falls into those categories. 
As you can see, our example sentence was rated as 45% positive, 
55% neutral and 0% negative. The final metric, the compound score, 
is the sum of all of the lexicon ratings (1.9 and 1.8 in this case) 
which have been standardised to range between -1 and 1. 
In this case, our example sentence has a rating of 0.69, which is pretty strongly positive.'''
def vader_sentiment(text):
    
    sid = SentimentIntensityAnalyzer()
    polarity=sid.polarity_scores(text)['compound']
    return polarity

"""Returns sentiment polarity is a value between -1.0 and +1.0  """

def pattern_sentiment(text):
  
    from pattern.en import sentiment
    polarity=sentiment(text)[0]
    return polarity



def get_sentiment(file):
    #exclude segmnets less than 2 secs, dont take in mind miliseconds
    start="00:00:02,000"
    start = datetime.datetime.strptime(start, '%H:%M:%S,%f')
    start = datetime.time(start.hour, start.minute,start.second)
    
    # Reading Subtitleget_sentiment
    subs = pysrt.open(file, encoding='iso-8859-1')
    n = len(subs)
    print("Initial segments for Srt file:" +file+ " are: " + str(n))
    # List to store the time periods
    intervals = []
    sentiments_text_blob = []
    sentiments_vader = []
    sentiments_pattern = []
    subs_len=-1
    pattern = re.compile("\[(.*?)\]|\-\[(.*?)\]")
    # Collect and combine all the text in each time interval
    for j in range(n):
        text = ""
        if (bool(pattern.match(subs[j].text_without_tags))==False):
            # Finding all subtitle text in the each time interval
            segment=subs[j].end-subs[j].start
            tocompare = datetime.datetime.strptime(str(segment), '%H:%M:%S,%f')
            tocompare = datetime.time(tocompare.hour, tocompare.minute, tocompare.second)
           
            if tocompare>=start:
                #print "end:" + str(subs[j].end)+ "strat:" +str(subs[j].start) + "segment:"+str(segment) + "segment.to_time:"+str(segment.to_time())
                intervals.append(subs[j].start + segment)
                subs_len=subs_len+1
                #for example [0,4],[4,6]->4,6
                if subs[j].end.to_time() <= (subs[j].start+ intervals[subs_len]).to_time():
                    text += subs[j].text_without_tags + " "
                else:
                    break
                # Sentiment Analysis with TextBlob
                sentiment_blob=textblob_sentiment(text)
                sentiments_text_blob.append(sentiment_blob)

                # Sentiment Analysis with Vader
                sentiment_vader=vader_sentiment(text)
                sentiments_vader.append(sentiment_vader)

                 # Sentiment Analysis with pattern
                sentiment_pattern=pattern_sentiment(text)
                sentiments_pattern.append(sentiment_pattern)
            #else:
                #print "exclude this segment. It duration is too small: " + str(segment)

    #find the avrage of the 2 different sentiment analysis
    avg_sentiments=[(a_i + b_i + c_i)/float(2) for a_i, b_i, c_i in zip(sentiments_vader, sentiments_text_blob,sentiments_pattern)]
    print("Segments for file: "+file+" after removing segments without text are: " + str(len(avg_sentiments)))
    print("------------------------------------------------------------------------------------------------")
    return (intervals, sentiments_text_blob)



# Utility to find average sentiment
def average(y):
    avg = float(sum(y))/len(y)
    return avg

def walktree(TopMostPath, callback):
    '''recursively descend the directory tree rooted at TopMostPath,
    calling the callback function for each regular file'''
    global dict
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
             
             
def sentiment(file): 
    global interval_segments, sentiment_segments
    #the srt files we need to train our model
    if '.srt' in file:
        #take the name of the file you extract sentiments
        name=str(file)
        name=name.split('/')
        name=name[-1]
        name=name.split('.')
        name=name[0]
        #print name
        interval_segments, sentiment_segments = get_sentiment(file)
        #DICTIONARY PATH,INTERVALS,SENTIMENTS
        dict[int(name)]={}
        dict[int(name)]['Id']=int(name)
        dict[int(name)]['intervals']=interval_segments
        dict[int(name)]['sentiments']=sentiment_segments
       
        
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
    if len(sys.argv)==2:
        print("Write in user specified path...")
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))
        #repo_path='/home/mscuser/multi/multimodal_audio'
        repo_path=str(sys.argv[1])
    else:
        repo_path = str(os.getcwd()) #able to work withou path . default write in the same directory
        print("Writing in the current path: ",repo_path)

    print(repo_path)
    '''vader lexicon'''
    nltk.downloader.download('vader_lexicon')
    
    '''Remove segments at which only tags are writtesn as caption text'''

    walktree(repo_path, sentiment)


                
                
    '''Make a csv for each video that contains: segment interval, pollarity for all segments of the video. The name of the file is polarity follwed by videoId. eg:polarity0'''
    create_folders(repo_path +'/polarity_csv')
    os.chdir(repo_path + '/polarity_csv')
    for dicts in range(len(dict)):
        new_csv='polarity_'+str(dict[dicts+1]['Id']) +'.csv'
        if sys.version_info >= (3, 0):
            with open(new_csv, 'w') as f:
                #w -----------> python3
                writer = csv.writer(f)
                for j in range(len(dict[dicts+1]['intervals'])):
                    context=(dict[dicts+1]['intervals'][j],dict[dicts+1]['sentiments'][j])
                    writer.writerows([context])
        else:
             with open(new_csv, 'wb') as f:
                #wb -----------> python2
                writer = csv.writer(f)
                for j in range(len(dict[dicts+1]['intervals'])):
                    context=(dict[dicts+1]['intervals'][j],dict[dicts+1]['sentiments'][j])
                    writer.writerows([context])
                    
    '''Write to pickle file'''
    create_folders(repo_path +'/pickle_lists')
    os.chdir(repo_path + '/pickle_lists')
    for dicts in range(len(dict)):
        context=[]
        pickle_name='polarity_'+str(dict[dicts+1]['Id']) +'.p'
        for j in range(len(dict[dicts+1]['intervals'])):
            context.append((dict[dicts+1]['intervals'][j],dict[dicts+1]['sentiments'][j]))
        pickle.dump( context, open( pickle_name, "wb" ) )
    
  
    
    

if __name__ == "__main__":
   
    main(sys.argv[1:])
 
