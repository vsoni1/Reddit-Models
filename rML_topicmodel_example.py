import numpy as np
import pandas as pd
from random import shuffle
import praw
import reddit_tools
from psaw import PushshiftAPI
import datetime as dt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle

""" Return a topic model for r/MachineLearning

This script saves an LDA topic model for the MachineLearning subreddit.
The example here can be easily reworked into a pipeline for topic modelling
other Reddit communities. 

Result is a pickle of the model in the current directory.

"""

reddit = praw.Reddit(client_id='fAT8FETUwusAZg',
                      client_secret='JWdrt4geIOwH1pmSeiOqWxhSlzA',
                      user_agent='test')
					  
submissions = reddit_tools.grab_data('MachineLearning', reddit)
all_comments = reddit_tools.grab_comments(submissions)
shuffle(all_comments)

stop_words = np.load('stop_words.npy')
add_stop = [
                'good','work','time','machine',
                'learning','thing','lot','problem',
                'bad','doesnt','model'
            ]
            
lt = reddit_tools.LemmaTokenizer()
new_stop = list(stop_words)+add_stop
new_stop = lt(' '.join(new_stop))+add_stop

tf_vect = CountVectorizer(tokenizer=reddit_tools.LemmaTokenizer(),
                                strip_accents='unicode', 
                                stop_words=set(new_stop), 
                                lowercase=True,
                                min_df=20,max_df=.75,) 

search_params = {'lda__n_components': [5,6,7,8,9,10],
                'lda__learning_method':['online'], 'lda__learning_offset':[25]
                }

lda = LatentDirichletAllocation()

lda_pipe = Pipeline([
    ('vect', tf_vect),
    ('lda', lda)
])

model = GridSearchCV(lda_pipe, param_grid=search_params, verbose=10, n_jobs=-1)

model.fit(all_comments)

lda = model.best_estimator_

filename = 'final_model'
pickle.dump(lda, open(filename, 'wb'))

res_df = topic_df(lda.named_steps['lda'], lda.named_steps['vect'].get_feature_names())

print(res_df.head(10))
