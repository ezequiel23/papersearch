import requests
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import collections
import re
import time
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps =PorterStemmer()
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from lista import lista,lista_h
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

def procesar_txt(id):
  with open(f'./archivos/{id}.txt', "r", encoding="utf-8") as f:
    text = f.read()
   # text = preprocess_text(text)
  return text

def tokensteam(tokens):
  newtokens=[]
  for x in tokens:
    newtokens.append(ps.stem(x))
  return newtokens

def del_by_len(tokens2):
  newtokens2=[]
  for token in tokens2:
      if len(token) > 4:
        newtokens2.append(token)
  return newtokens2



def read_corpus():
    textnew = []
    with open('corpus.data', 'r') as file:
        lines = file.readlines()
        for line in lines:
            textnew.append(line)
    print(len(textnew))  
    return textnew

tfidf=TfidfVectorizer(analyzer='word')
x = tfidf.fit_transform(read_corpus())
clf = MultinomialNB().fit(x, lista)  

def procesar_texto(lista):
    return clf,tfidf

# @lru_cache(maxsize=200)
# def read_prepro_corpus():
#     #tfidfimport = tfidf
#     # prepro_corpus_read = []
#     # with open('prepro_corpus.data', 'r') as file:
#     #     lines = file.readlines()
#     #     for line in lines:
#     #         prepro_corpus_read.append(line)
#     # #print(len(prepro_corpus_read))
#     # return prepro_corpus_read
#     df =pd.read_csv('prepro_corpus.csv',sep=';',usecols= ['id','texto'])
#     pre_data=[]
#     for id in lista_h:
#         #pre_data.append([id,tfidfimport.transform(df[df['id'] == id]['texto'])])
#         pre_data.append([id,df[df['id'] == id]['texto']])
#     return pre_data



# @lru_cache(maxsize=200)
# def find_text(id,corpus):
#     for x in corpus:
#         if x[0] == id:
#             return x[1]
@lru_cache(maxsize=200)
def texto_real_data(texto):
    textreturn=[]
    text = texto
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    tokens = tokensteam(tokens)
    text = " ".join(tokens)
    textreturn.append(text)
    datatrans = tfidf.transform(textreturn)
    return datatrans

@lru_cache(maxsize=200)
def test_real_data(id):
    textreturn=[]
    text = procesar_txt(id)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    tokens = tokensteam(tokens)
    text = " ".join(tokens)
    textreturn.append(text)
    datatrans = tfidf.transform(textreturn)
    return datatrans

@lru_cache(maxsize=200)
def procesar_archivo(id1,id2):
    return cosine_similarity(test_real_data(id1),test_real_data(id2))

@lru_cache(maxsize=200)
def procesar_uno_vs_all(id):
    lista_h = [19690065840,19930091025,19930091026,19930080815,19690065840,19930091025,19930091026,19930080815]
    results_h=[]
    test_fijo=test_real_data(id)
    @lru_cache(maxsize=200)
    def run_similarity(id):
        worker_time = time.time()
        #result = cosine_similarity(test_fijo,test_real_data(id))
        result = float(cosine_similarity(test_fijo,test_real_data(id))[0][0])
        print("--- %s worker time ---" % (time.time() - worker_time))
        #print(type(result))
        results_h.append({'id':id,'result':result})
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(run_similarity,lista_h)
    print("--- %s seconds ---" % (time.time() - start_time))
    result_final=[]
    result_final_result=[]
    
    for item in results_h:
        #print(item['result'])
        result_final_result.append(item['result'])
        if item['result'] > 0.1:
           result_final.append({'id':item['id'],'result':item['result']}) 
    return result_final






