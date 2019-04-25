import praw
import time
import matplotlib.pyplot as plt
import numpy as np
from psaw import PushshiftAPI
import nltk
import datetime as dt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag          
from nltk.stem import WordNetLemmatizer
import pandas as pd

def grab_data(subreddit, reddit):
    """ Grabs at most 400000 submissions from subreddit for 2 year period
    
    Parameters
    ----------
    subreddit : string
        the subreddit of interest
    
    reddit : Reddit Instance 
        an instance of the PRAW Reddit class
    Returns
    ----------
    list of submissions
    """
    api = PushshiftAPI(reddit)
    
    start_epoch = int(dt.datetime(2017, 3, 30).timestamp())
    end_epoch = int(dt.datetime(2019, 3, 30).timestamp())

    submissions = list(api.search_submissions(before=end_epoch, 
                                after=start_epoch,
                                subreddit=subreddit,
                                limit=40000))
                                
    print(f'grabbed {len(submissions)} submissions')
    np.save(subreddit+'_submissions', submissions)
    return submissions
    
def grab_comments(submissions):
    """ Grabs list of comments and replies and all text from list of submissions.
        Appends data to list if text associated with submission is greater than 200 characters.
    
    Parameters
    ----------
    submissions : list
        list of submissions
    
    Returns
    ----------
    list of comments
    """
    all_comments = []
    for ind, sub in enumerate(submissions):
        if ind%2000 == 0:
            print(f'{ind} out of {len(submissions)}')
            time.sleep(2)
        if ('[removed]' not in sub.selftext) and ('[deleted]' not in sub.selftext):
            clist = sub.title.lower() + ' ' + sub.selftext.lower()
        else:
            clist = sub.title
        comments = sub.comments
        comments.replace_more()
        comments = comments.list()
        clist = clist +\
        ' '.join([item.body.lower() for item in comments])
        if len(clist)>200:
            all_comments.append(clist)
       
    np.save(submissions[0].subreddit.display_name+'_comments', all_comments)
        
    return all_comments
    

def text_process(text):
    """ process text data by removing punctuation and tokenizing words within 2-25 characters
    
    Parameters
    ----------
    text : string
        string to process
    Returns
    ----------
    tokenized string
    """
    # Remove Punctuation
    nopunc = [char for char in text if char not in string.punctuation +'’”']
    nopunc = ''.join(nopunc)
    # Remove Stopwords
    return [word for word in nopunc.split() if (len(word)>2) & (len(word)<25)]

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
            word_tokenize(articles)
            return [
                        self.wnl.lemmatize(token,self.get_wordnet_pos(tag)) 
                        for token, tag in pos_tag(text_process(articles))
                        if not tag.startswith('V')]
        
    def get_wordnet_pos(self, treebank_tag):
        
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN        
        
        
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def topic_df(model, feature_names):
    df = pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        col_name = "Topic " + str(topic_idx)
        words = [feature_names[i] for i in topic.argsort()[::-1]]
        df[col_name] = words
    return df
