import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
from nltk.stem import WordNetLemmatizer



# 영어 제외하고 특수문자 및 공백 제거
import re

def apply_re(tokenized_text):
    result= []
    for tok_list in tokenized_text:
        list_par = []
        for tok in tok_list:
            text = re.sub('[^a-zA-Z]',' ',tok).strip() # 영어 제외 다 제거.
            if(text != ''): # 빈리스트 제거.
                list_par.append(text)
        result.append(list_par)
    return result
    
# 불용어 사전 적용
from nltk.corpus import stopwords

def apply_stop_words(tokenized_text):
    stop_words = stopwords.words('english') 
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  

# 어간추출 사전 적용
import nltk
nltk.download('wordnet')
 
def apply_lemma(tokenized_text):
    lemma = WordNetLemmatizer()
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            tok_result.append(lemma.lemmatize(tok))
        result.append(tok_result)
    return result  

# scientific 불용어사전 적용
file_path =r"C:/Users/82109/GitHub/doc2vec/stopwords/universal_scientific_stopwords.csv"
def apply_scientificword(tokenized_text):
    scientifiword = pd.read_csv(file_path,header = None) #csv 파일일때만 사용
    stop_words = scientifiword[0].tolist()
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  

# data 불러오기
row_data = pd.read_csv(r"C:/Users/82109/GitHub/doc2vec/data/paper.csv")
paper_notnull = row_data.fillna("")
data = [row['Title'] + row['Abstract']+ row['Author Keywords'] +row['Index Keywords'] for idx, row in paper_notnull.iterrows()]
#토큰화
tokenized_text = [nltk.word_tokenize(doc.lower()) for doc in data]

# 순서 re -> stopword -> scientificword -> lemma
result_re = apply_re(tokenized_text)
result_stopword = apply_stop_words(result_re)
result_scientific = apply_scientificword(result_stopword)
result_lemma = apply_lemma(result_scientific)

# pickle 파일로 저장
with open('data/preprocessing_data(2812).pickle', 'wb') as f:
    pickle.dump(result_lemma, f, pickle.HIGHEST_PROTOCOL)