import pysrt
from datetime import date, datetime, timedelta, time
import pysrt
from textblob import TextBlob
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
import os

import csv
import os, re, sys
from stat import *

import numpy as np
import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

from nltk.corpus import sentiwordnet as swn

import re

import pickle


#dict={id, path, , intervals, sentiments}
dict={}
i=interval_segments=sentiment_segments=0




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




def get_sentiment(file):
    # Reading Subtitleget_sentiment
    subs = pysrt.open(file, encoding='iso-8859-1')
    n = len(subs)
    print "Initial segments for Srt file:" +file+ "are: " + str(n)
    # List to store the time periods
    intervals = []
    sentiments_text_blob = []
    sentiments_vader = []
    subs_len=-1
    pattern = re.compile("\[(.*?)\]|\-\[(.*?)\]")
    # Collect and combine all the text in each time interval
    for j in range(n):
        text = ""
        if (bool(pattern.match(subs[j].text_without_tags))==False):
            # Finding all subtitle text in the each time interval
            segment=subs[j].end-subs[j].start
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
        

    #find the avrage of the 2 different sentiment analysis
    avg_sentiments=[(a_i + b_i)/float(2) for a_i, b_i in zip(sentiments_vader, sentiments_text_blob)]
    print "Segments for file: "+file+" after removing segments without text are:" + str(len(avg_sentiments))
    print "------------------------------------------------------------------------------------------------"
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
             print 'Skipping %s' % pathname
             
             
def sentiment(file): 
    global i 
    global interval_segments, sentiment_segments
    #the srt files we need to train our model
    if '.srt' in file:
        interval_segments, sentiment_segments = get_sentiment(file)
        #DICTIONARY PATH,INTERVALS,SENTIMENTS
        dict[i]={}
        dict[i]['Id']=i
        dict[i]['path']=file
        dict[i]['intervals']=interval_segments
        dict[i]['sentiments']=sentiment_segments
        i=i+1
        



def main():
    '''vader lexicon'''
    nltk.downloader.download('vader_lexicon')
    
    '''Remove segments at which only tags are writtesn as caption text'''
   

    path = '/home/mscuser/multi/multimodal_audio'
    walktree(path, sentiment)

    ''''Make a csv for all videos that contains:videoid, path, list of interval segments, list of sentiments'''
    for i in range(len(dict)):
        with open('captions_polarity.csv', 'wb') as f:
            writer = csv.writer(f)
            for key, value in dict.items():
                writer.writerow([key, value])
                
                
    '''Make a csv for each video that contains: segment interval, pollarity for all segments of the video. The name of the file is polarity follwed by videoId. eg:polarity0'''
    for dicts in range(len(dict)):
        new_csv='polarity'+str(dict[dicts]['Id']) +'.csv'
        with open(new_csv, 'wb') as f:
            writer = csv.writer(f)
            for j in range(len(dict[dicts]['intervals'])):
                context=(dict[dicts]['intervals'][j],dict[dicts]['sentiments'][j])
                writer.writerows([context])
                
    '''Write to pickle file'''
    favorite_color = { "lion": "yellow", "kitty": "red" }
    for dicts in range(len(dict)):
        context=[]
        pickle_name='polarity'+str(dict[dicts]['Id']) +'.p'
        for j in range(len(dict[dicts]['intervals'])):
            context.append((dict[dicts]['intervals'][j],dict[dicts]['sentiments'][j]))
        pickle.dump( context, open( pickle_name, "wb" ) )
    
  
    
    

if __name__ == "__main__":
    main()
