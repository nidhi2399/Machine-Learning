#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

df = pd.read_csv("C:\\Users\\Hp\\Desktop\\IMDB Dataset.csv")


# In[6]:


df.head()


# In[3]:


df.tail(5)


# In[3]:


df["review"][5]


# In[4]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()#lancaster #snowball #regexp: singing =s

def stemmer_tokenize(text):
    return[porter.stem(word) for word in text.split()] #word_tokenize()


# In[5]:


stemmer_tokenize("We love coding.hence we keep coding and learning")


# In[6]:


#Vectorising text data #stanfordnlp

from sklearn.feature_extraction.text import TfidfVectorizer #countvectorizer

tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, tokenizer = stemmer_tokenize, use_idf = True)


# In[7]:


Y=df.sentiment.values
X=tfidf.fit_transform(df.review)


# In[9]:


#document classification/sentiment analyasis using logistic regression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 1, test_size= 0.5, shuffle = False)

import pickle

from sklearn.linear_model import LogisticRegressionCV

logitCV = LogisticRegressionCV (cv = 5, scoring ='accuracy', max_iter = 300).fit(X_train, Y_train)

saved_model = open('saved_model.sav','wb') #create a file

pickle.dump(logitCV, saved_model)

saved_model.close()


# In[10]:


#run saved model
filename = 'saved_model.sav'

saved_logitCV = pickle.load(open(filename,'rb'))

saved_logitCV.score(X_test,Y_test)


# In[ ]:




