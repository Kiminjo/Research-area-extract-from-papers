# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:52:53 2021

@author: 82109
"""

#pickle로 만들어진 데이터 불러오기(기본불용어 + 특수문자 + lemmatisation)
import random
import pandas as pd
from collections import Counter
import pickle

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

documents = apply_stop_words(documents)

#토픽 갯수 / epoch 수 / csv파일 저장할 오늘 날짜
K=30
epochs = 200
now = '210906_'

# a list of Counters, one for each document
document_topic_counts = [Counter() for _ in documents]

# a list of Counters, one for each topic
topic_word_counts = [Counter() for _ in range(K)]

# a list of numbers, one for each topic
topic_counts = [0 for _ in range(K)]

document_lengths = list(map(len, documents))

#unique한 word의 수
distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

#문서의 갯수
D = len(documents)

#Topic weight 계산
def topic_weight(d, word, topic):
    """given a document and a word in that document,
    return the weight for the kth topic"""
    
    def p_topic_given_document(topic, d, alpha=0.1):
        """the fraction of words in document _d_
        that are assigned to _topic_ (plus some smoothing)"""
        return ((document_topic_counts[d][topic] + alpha) / (document_lengths[d] + K * alpha))

    def p_word_given_topic(word, topic, beta=0.1):
        """the fraction of words assigned to _topic_
        that equal _word_ (plus some smoothing)"""
        return ((topic_word_counts[topic][word] + beta) / (topic_counts[topic] + W * beta))
    
    
    return p_word_given_topic(word, topic) * p_topic_given_document(topic, d)

#topic weight이 가장 높은 토픽 샘플
def choose_new_topic(d, word):
    
    def sample_from(weights):
        """returns i with probability weights[i] / sum(weights)"""
        total = sum(weights)
        rnd = total * random.random() # uniform between 0 and total
        for i, p in enumerate(weights):
            rnd -= p # return the smallest i such that
            if rnd <= 0: 
                return i # weights[0] + ... + weights[i] >= rnd
        
    return sample_from([topic_weight(d, word, topic) for topic in range(K)])

#topic assigbment(각 문서의 단어 numbering)초기화
document_topics = [[random.randrange(K) for word in document] for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1 
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1
        
#Traning
for epoch in range(epochs): # repetition
    for d in range(D): # each documnet
        for i, (word, topic) in enumerate(zip(documents[d],document_topics[d])):
            
            # gibbs sampling: 특정 하나의 topic assignment z를 제거하고 나머지들(-z)의 조건부 확률  
            
            # remove this word / topic from the counts
            # so that it doesn't influence the weights
            document_topic_counts[d][topic] -= 1 # 문서별 토픽 갯수
            topic_word_counts[topic][word] -= 1 # 토픽별 단어 갯수
            topic_counts[topic] -= 1 # 토픽별 카운트
            document_lengths[d] -= 1 # 문서별 단어갯수
            
            # choose a new topic based on the weights
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
            
            # and now add it back to the counts
            document_topic_counts[d][new_topic] += 1 # 문서별 토픽 갯수
            topic_word_counts[new_topic][word] += 1 # 토픽별 단어 갯수
            topic_counts[new_topic] += 1 # 토픽별 카운트
            document_lengths[d] += 1 # 문서별 단어갯수
            
 #토픽 30개, 토픽당 단어 10개
df = pd.DataFrame(columns=['Topic'+str(i) for i in range(1,30)], index=['Top'+str(i) for i in range(1,10)])

for k, word_counts in enumerate(topic_word_counts):
    for ix, (word, count) in enumerate(word_counts.most_common(10)): # 각 토픽별로 top 10 단어
            df.loc['Top'+str(ix+1),'Topic'+str(k+1)] = word+'({})'.format(count)

#LDA 결과 저장
test_name = 'LDA_output/LDA_verson1' + now + '_' + 'epochs = ' + str(int(epochs)) + '.csv'
df.to_csv(test_name, index=True)


