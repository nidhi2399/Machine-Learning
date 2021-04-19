#!/usr/bin/env python
# coding: utf-8

# In[13]:


#20MAI0020- Nidhi Ghuble


# In[14]:


import pandas as pd
df= pd.read_csv("C:\\Users\\Hp\\Desktop\\student dataset.csv")


# In[15]:


print(df)


# '''The above dataset shows information regarding  usage of social media applications by students and weather they 
# are addicted to social media or not. 
# In the above dataset the time spent on each app is recorded in hours and addiction of a user is determined based
# on total usage time of all apps. '''

# In[16]:


df.head()


# In[17]:


#Vectorising text data

from sklearn.feature_extraction.text import TfidfVectorizer #countvectorizer

tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, use_idf = True)


# In[18]:


Y=df.Addiction.values

X=tfidf.fit_transform(df.usage_more_than_3)


# In[19]:


#document classification/sentiment analyasis using logistic regression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 1, test_size= 0.5, shuffle = False)

import pickle

from sklearn.linear_model import LogisticRegressionCV

logitCV = LogisticRegressionCV (cv = 5, scoring ='accuracy', max_iter = 150).fit(X_train, Y_train)

new_model = open('new_model.sav','wb') #create a file

pickle.dump(logitCV, new_model)

new_model.close()


# In[20]:


#run saved model
filename = 'new_model.sav'

saved_logitCV = pickle.load(open(filename,'rb'))

saved_logitCV.score(X_test,Y_test)


# In[ ]:




