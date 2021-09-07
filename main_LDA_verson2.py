# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:40:13 2021

@author: 82109
"""
import numpy as np
import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import pickle

NUM_TOPICS = 30 # 토픽갯수
passes = 100 # 반복횟수
now = '210906_' # 오늘날짜

#pickle 파일 열기
with open("data\clean_data.pickle","rb") as fr:
    documents = pickle.load(fr)

#csv파일로 stopwords 만들기
df = pd.read_csv(r"stopwords\universal_scientific_stopwords.csv",header = None)
d = df[0].values
stop_words_universal =d.tolist()


#stop_words 추가하기
def apply_stop_words(tokenized_text):
    stop_words = stop_words_universal
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result    

tokenized_doc = apply_stop_words(documents) # 토큰화된 documents
dictionary = corpora.Dictionary(tokenized_doc) # tokenized 데이터를 통해 dictionary로 변환
corpus = [dictionary.doc2bow(text) for text in tokenized_doc] # 코퍼스 구성

# passes : 알고리즘 동작횟수, num_words : 토픽의 수
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics =NUM_TOPICS, id2word=dictionary, passes= passes, )

# topics = 30 , words = 10
topics = ldamodel.print_topics(num_topics= 30, num_words=10)

df = pd.DataFrame(topics)

def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,10), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)

topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']

           
#LDA 모델링, 테이블 결과 저장
modeling_name = 'LDA_output/LDA_verson2' + now + '_' + 'epochs = ' + str(int(passes))+ '_modeling.csv'
df.to_csv(modeling_name, index=True)

table_name = 'LDA_output/LDA_verson2' + now + '_' + 'epochs = ' + str(int(passes))+ '_table.csv'
topictable.to_csv(table_name, index=True)
