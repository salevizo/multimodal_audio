import pysrt
import datetime
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import spacy
from spacy.tokens import Doc
import re
import math

def sentiment(file,case):
    name=str(file)
    name=name.split('/')
    name=name[-1]
    name=name.split('.')
    name=name[0]
    #print name
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(str(case)+"-"+str(name))
    interval_segments, sentiment_segments = get_sentiment(file,case)
    print(str(case)+"-"+str(name))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #DICTIONARY PATH,INTERVALS,SENTIMENTS
    if case is 'train':
        return (int(name),interval_segments,sentiment_segments)
    else:
        return (int(name),interval_segments)

def vader_sentiment(text):
    
    sid = SentimentIntensityAnalyzer()
    polarity=sid.polarity_scores(text)['compound']
    return polarity

"""Returns sentiment polarity is a value between -1.0 and +1.0  """



def pattern_sentiment(text):
  
    from pattern.en import sentiment
    polarity=sentiment(text)[0]
    return polarity

'''textblob sentiment analysis'''
def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity


def get_sentiment(file,case):
    #exclude segmnets less than 2 secs, dont take in mind miliseconds
    start="00:00:02,000"
    start = datetime.datetime.strptime(start, '%H:%M:%S,%f')
    
    start = datetime.time(start.hour, start.minute,start.second,start.microsecond)
    
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
    stop_words = set(stopwords.words('english'))
    for j in range(n):
        text = ""
        if (bool(pattern.match(subs[j].text_without_tags))==False):
            segment=subs[j].end-subs[j].start
            # Finding all subtitle text in the each time interva
            tocompare = datetime.datetime.strptime(str(segment), '%H:%M:%S,%f')
            tocompare = datetime.time(tocompare.hour, tocompare.minute, tocompare.second,tocompare.microsecond)
            if tocompare>=start:
                if case is 'train':
                    interval=subs[j].start + segment
                    subs_len=subs_len+1
                    if subs[j].end.to_time() <= (subs[j].start+ subs[j].start + segment).to_time():
                        text += subs[j].text_without_tags + " "
                    else:
                        break
                    word_tokens = word_tokenize(text) 
                    filtered_sentence = []
                    filtered_sentence = [w for w in word_tokens if not w in stop_words]
                    if len(filtered_sentence)>=4:
                        text_filtered=' '.join(filtered_sentence)

                        sentiment_blob=textblob_sentiment(text_filtered)
                        sentiment_vader=vader_sentiment(text_filtered)
                        sentiment_pattern=pattern_sentiment(text_filtered)
                        
                        if ((sentiment_blob>0 and sentiment_vader>0 and sentiment_pattern>0) or (sentiment_blob<0 and sentiment_vader<0 and sentiment_pattern<0) or (sentiment_blob==0 and sentiment_vader==0 and sentiment_pattern==0)):
                            avg_polarity=(sentiment_blob + sentiment_vader + sentiment_pattern)/float(3)
                            avg_polarity=math.fabs(avg_polarity)
                            if (avg_polarity>0.25 or avg_polarity==0):
                                sentiments_text_blob.append(sentiment_blob)
                                sentiments_vader.append(sentiment_vader)
                                sentiments_pattern.append(sentiment_pattern)
                                print("TRAIN:"+str(subs[j].start)+"-"+str(subs[j].end))
                                intervals.append([subs[j].start,subs[j].end])
                                #print ("sentiment_blob: " + str(sentiment_blob) + " sentiment_vader: " + str(sentiment_vader) + " sentiment_pattern:" + str(sentiment_pattern) +" for text "+ text_filtered +".")
                else:
                    interval=subs[j].start + segment
                    subs_len=subs_len+1
                    if subs[j].end.to_time() <= (subs[j].start+ subs[j].start + segment).to_time():
                        text += subs[j].text_without_tags + " "
                    else:
                        break
                    word_tokens = word_tokenize(text) 
                    filtered_sentence = []
                    filtered_sentence = [w for w in word_tokens if not w in stop_words]
                    if len(filtered_sentence)>=4:
                        print("FOR TEST:",filtered_sentence)
                        print("TEST:"+str(subs[j].start)+"-"+str(subs[j].end))
                        intervals.append([subs[j].start,subs[j].end])
    if case is 'train':
        avg_sentiments=[(a_i + b_i + c_i)/float(3) for a_i, b_i, c_i in zip(sentiments_vader, sentiments_text_blob,sentiments_pattern)]
    else:
        avg_sentiments=None
   
    #print("Segments for file: "+file+" after removing segments without text are: " + str(len(avg_sentiments)))
    print("------------------------------------------------------------------------------------------------")
    return (intervals, avg_sentiments)
