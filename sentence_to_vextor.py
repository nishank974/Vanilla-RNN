# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:10:28 2017

@author: nishank
"""

#Import Statements
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim, logging, nltk,os, sys, numpy as np
from sklearn.manifold import TSNE
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import re
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from operator import is_not
from functools import partial


def preprocess():    
    data = open('/home/user/Downloads/data/combined.txt', 'r').read()
    data = data.lower()
    data=re.sub('[^a-zA-Z!@#$%^&*()?]'," ",data)
    data= sorted(list(set(data.split(" "))))
    return data
    
def convert_text():
    data=preprocess()
    df1 = pd.DataFrame({'text':data})
    np.savetxt('/home/user/Downloads/data/files from coe/dat_clean.txt', df1['text'].values, fmt='%s')
    return data


def wordvectors(vocab_filename,dat):
     #CREATING WORD VECTORS
    #training data
    sentences=[]
    # NOTE: flight.txt is the data file. Can use others as well
    sentences = gensim.models.word2vec.LineSentence(vocab_filename);    
    #dimension=size
    #sg=0 : CBOW model
    #sg=1 : skip-gram model
    model = gensim.models.Word2Vec(size=100, min_count=0, workers=4, sg=0)    
    model.build_vocab(sentences) #sentencefeeder puts out sentences as lists of strings    
    model.save("newmodel")

    #FORMING SENTENCE VECTORS

    vec_final=[]
    final=[]
    removed=[]
    for i in dat:
        try:
            vec_final.append(list(model[i]))
            final.append(i)
        except KeyError:
            removed.append(i)
            
    return [model,final,vec_final]


def dataframe(train_filename):
    xl = pd.ExcelFile(train_filename)  
    return xl.parse("sheet1")
     

#==============================================================================
# def sentencevectors(model,final,vec_final,train_filename):    
#     count_vectorizer = CountVectorizer(vocabulary=final)
#     df= dataframe(train_filename)
#     p=list(df['data'])
#     test_set = [p[0+i]+p[1+i]+p[2+i] for i in xrange(0,2998,3)] 
#     freq_term_matrix = count_vectorizer.transform(test_set)
#     tfidf = TfidfTransformer(norm="l2")
#     tfidf.fit(freq_term_matrix)
#     tf_idf_matrix = tfidf.transform(freq_term_matrix)
#     p= tf_idf_matrix.todense()
#     result = p*np.array(vec_final)
#     X = np.array(result)
#     
#     y=list(df['reply'])
#     y =[x for x in y if str(x)!='nan']
#     xx=list(X)
#     #ans = pd.DataFrame({'input':xx,'output':y})
#     
#     return xx,y
 
#==============================================================================

def sentencevectors(model,final,vec_final,train_filename):    
     count_vectorizer = CountVectorizer(vocabulary=final)
     df= dataframe(train_filename)
     p=list(df['data'])
#     test_set = [p[0+i]+p[1+i]+p[2+i] for i in xrange(0,2998,3)] 
     q=list(df['reply'])
     
     test_set=[]
     k=0
     while(k<(len(df))):
     
         i=k
         while(str(q[i]) == 'nan'):
             i += 1
        
         pe=''
         for j in xrange(k,i+1):
             pe += p[j]
         test_set.append(pe)
         
         k=i+1
         

     freq_term_matrix = count_vectorizer.transform(test_set)
     tfidf = TfidfTransformer(norm="l2")
     tfidf.fit(freq_term_matrix)
     tf_idf_matrix = tfidf.transform(freq_term_matrix)
     p= tf_idf_matrix.todense()
     result = p*np.array(vec_final)
     X = np.array(result)
     
     y=list(df['reply'])
     y =[x for x in y if str(x)!='nan']
     xx=list(X)
     #ans = pd.DataFrame({'input':xx,'output':y})
     return xx,y

'''
#For plotting sentence vectors
     t1=[]
     for i in xrange(len(test_set)):
         t1.append(X[i]) 
 
     t1=np.array(t1)
     t2=["online","book","flight","baggage","booking"]
     At1 = TSNE(n_components=2, random_state=0)
     np.set_printoptions(suppress=True)
     At1.fit_transform(t1)
     plt.scatter(t1[:,0],t1[:,1])
     for label, xt1, yt2 in zip(t2,t1[:,0],t1[:,1]):
         plt.annotate(label,xy=(xt1,yt2),xytext=(0,0),textcoords='offset points')
     plt.show()
'''
     
     







    
###################33    
data=convert_text()

model,final,vec_final=wordvectors('/home/user/Downloads/data/files from coe/dat_clean.txt',data)

train_data_excel = '/home/user/Downloads/data/files from coe/train20.xlsx'


X,y=sentencevectors(model,final,vec_final,train_data_excel)
X=[X[i].reshape(100,1) for i in range(len(X)) ]
for i in range(0, len(y)):
    y[i] = "<s> " + y[i] + " </s>" 
#y = list(set(y))
Y=[]
for i in range(len(y)):
    Y.extend(y[i].split(" "))

#out = list([model[x] for x in y])






