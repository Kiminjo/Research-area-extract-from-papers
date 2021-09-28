# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 01:13:17 2021

@author: rubby
"""

import os
import nltk
import pickle
import gensim
import numpy as np
import pandas as pd
from gensim import corpora
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.matutils import cossim
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

filepath = 'data/clean_data.pickle' # path of preprocessed data

def text_preprocessing(data) :
    lemma = WordNetLemmatizer()
    clean_data = []
    tokens = []
    if (os.path.isfile(filepath)) :
        clean_data = load_clean()
    else :
        for doc in data :
            tokens.append(nltk.regexp_tokenize(doc.lower(), '[A-Za-z]+'))                        
        for words in tokens :
            clean_words = []
            for word in words :
                if len(word) > 1:
                        if word not in stopwords.words('english') :
                            clean_words.append(lemma.lemmatize(word))                     
            clean_data.append(clean_words)
        print('Preprocessing complete')
        with open(filepath, 'wb') as lf :
            pickle.dump(clean_data, lf)
    return clean_data

def load_clean() :
    with open(filepath, 'rb') as lf :
        clean_data = pickle.load(lf)
            
    return clean_data

def calculate_similarity(tokenized_doc) :
    dictionary = corpora.Dictionary(tokenized_doc)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
    
    model = gensim.models.ldamodel.LdaModel(corpus, 
                                               num_topics = 40, 
                                               id2word = dictionary, 
                                               passes = 30, 
                                               iterations = 400, 
                                               random_state = 1004)
    
    possibility_vector = pd.DataFrame(model[corpus])
    
    for idx in range(len(possibility_vector.columns)) :
        possibility_vector[idx] = possibility_vector[idx].apply(lambda x : x[1])
    
    similarity_matrix = pd.DataFrame(cosine_similarity(possibility_vector, possibility_vector))
    
    return similarity_matrix
    
if __name__ == '__main__' :
    raw_data = pd.read_csv('data/2016-2021.csv')
    data = [row['Title'] + row['Abstract'] + str(row['Author Keywords']) + str(row['Index Keywords']) for _, row in raw_data.iterrows()]
    clean_data = text_preprocessing(data)
    similarity_matrix = calculate_similarity(clean_data)