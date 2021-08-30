import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle
import os
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from doc2vec import Doc_to_vec, TaggedDocument
from argparse import ArgumentParser
import warnings 
warnings.filterwarnings('ignore')

filepath = 'data/clean_data.pickle'

# 이 코드 쓰실 분은 여기만 수정하시면 됩니다.
def text_preprocessing(data) :
    clean_data = []
    tokens = []
    s_stopwords = load_stopwords()
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
                        ##and word not in s_stopwords
                            clean_words.append(word)                     
            clean_data.append(clean_words)
        with open(filepath, 'wb') as lf :
            pickle.dump(clean_data, lf)
    return [TaggedDocument(words=word, tags=[str(index)]) for index, word in enumerate(clean_data)]

def load_clean() :
    with open(filepath, 'rb') as lf :
        clean_data = pickle.load(lf)
            
    return clean_data

def load_stopwords() :
    list_file = open('scientificstopwords.txt', 'r').read().split('\n')
    return list_file

if __name__ == '__main__':

    parser = ArgumentParser(description='Doc2vec learning')
    parser.add_argument('--data', type=str, help='text data path')
    parser.add_argument('--parameter', type=str, help='hyperparameter path')
    parser.add_argument('--mode', type=str, default='both', help='Doc2vec type. Just train or embedding and both')
    parser.add_argument('--model', type=str, help='Call trained model')
    parser.add_argument('--save_model_name', type=str, help='Save model name')
    args = parser.parse_args()

    mode = args.mode
    params_dict = {'epochs' : None, 'embedding_size' : None, 'alpha' : None, 'min_alpha' : None, 'min_count' : None, 'dm' : None}

    with open(args.parameter, 'r') as f :
        params = [[z[0].strip(), z[1].strip()] for z in [param.split(':') for param in f.readlines()]]
        for param in params : 
            params_dict[param[0]] = float(param[1])

    epochs = params_dict['epochs'] 
    embedding_size = params_dict['embedding_size']
    alpha = params_dict['alpha']
    min_alpha = params_dict['min_alpha']
    min_count = params_dict['min_count']
    dm = params_dict['dm']

    ### Doc2vec abstarct embedding

    df = pd.read_csv(args.data)
    data = [row['Title'] + row['Abstract'] + str(row['Author Keywords']) + str(row['Index Keywords']) for idx, row in df.iterrows()]
    ##data = [row['Title'] + row['Abstract'] for idx, row in df.iterrows()]
    processed_data = text_preprocessing(data)
    
    '''
    clean_data = load_clean()
    clean = []
    
    for doc in clean_data :
        words = ''
        for word in doc :
            words += str(word) + ' '
        clean.append(words)
    '''
    
    if mode == 'both' :
        model = Doc_to_vec(document=processed_data, epochs=epochs, embedding_size=embedding_size, alpha=alpha, model_name=args.save_model_name)
        model.train()

        tokenized_text = [nltk.word_tokenize(doc.lower()) for doc in data]
        embedding_vectors = [model.doc2vec_model.infer_vector(text) for text in tokenized_text]

    elif mode == 'train' :
        model = Doc_to_vec(document=processed_data, epochs=epochs, embedding_size=embedding_size, alpha=alpha, model_name=args.save_model_name)
        model.train()
        exit()

    elif mode == 'embedding' :
        model = Doc2Vec.load(args.model)
        tokenized_text = [nltk.word_tokenize(doc.lower()) for doc in data]
        embedding_vectors = [model.infer_vector(text) for text in tokenized_text]

    similarity_matrix = cosine_similarity(embedding_vectors, embedding_vectors)
    print(similarity_matrix)
    df = pd.DataFrame(similarity_matrix)
    df.to_csv('network/210829_08.csv', index=True)

    print('Embedding matrix size : {}'.format(np.array(embedding_vectors).shape))
    print('Doc2vec embedding complete')

    ### Clustering

    kmeans = KMeans(n_clusters=30).fit(embedding_vectors)
    clusters = kmeans.labels_

    print('Clustering complete')


    ### Visualization using T-SNE

    two_dim_embedded_vectors = TSNE(n_components=2).fit_transform(embedding_vectors)

    fig, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(two_dim_embedded_vectors[:,0], two_dim_embedded_vectors[:,1], hue=clusters, palette='deep', ax=ax)
    plt.show()
