#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


#load the data
data =pd.read_csv('C:\\Users\\Hp\\Desktop\\crx.data', header= None)


# In[6]:


#A1 A2 A3...A16
varnames = ['A'+ str(s) for s in range(1,17)]


# In[7]:


#add column names
data.columns = varnames


# In[8]:


data.head() #get glimps


# In[9]:


data.head(20)


# In[10]:


#replace ? with np.nan
data = data.replace('?',np.nan)


# In[11]:


data.info()


# In[12]:


#recast
data['A2'] = data['A2'].astype('float')
data['A14'] = data['A14'].astype('float')


# In[13]:


data['A16'] =data['A16'].map({'+':1, '-':0})


# In[14]:


data['A16']


# In[15]:


#find categorial variables
cat_columns = [c for c in data.columns if data[c].dtypes == 'O']
data[cat_columns].head()


# In[16]:


num_columns = [c for c in data.columns if data[c].dtypes != 'O']
data[cat_columns].head()


# In[17]:


data = pd.read_csv('C:\\Users\\Hp\\Desktop\\loan.csv')


# In[18]:


data.head()


# In[19]:


#continuous
data['disbursed_amount'].unique()


# In[20]:


#discrete
data['number_open_accounts'].unique()


# In[21]:


data['target'].unique()


# In[22]:


data['time_employed'].unique()


# In[23]:


data['householder'].value_counts()


# In[24]:


data[['date_issued','date_last_payment']].dtypes


# In[25]:


data['date_issued_dt'] = pd.to_datetime(data['date_issued'])


# In[26]:


data.head()


# In[31]:


data['month'] = data['date_issued_dt'].dt.month
data.head()


# In[27]:


#year
data['year'] = data['date_issued_dt'].dt.year


# In[28]:


data.head()


# In[29]:


#Matplotlib
from matplotlib import pyplot as plt


# In[187]:


ages = [25, 26, 27, 28, 29, 30]
dev_salary = [35000, 36000, 37000, 39000, 41000, 42000]

py_salary =[38000, 39000, 43000, 45000, 47000, 50000]


# In[188]:


plt.title('salary of developers')
plt.xlabel('ages')
plt.ylabel('median salary')

plt.plot(ages, py_salary)
plt.plot(ages, dev_salary)

plt.legend(['All Dev Sal', 'Python Dev Sal'])


# In[189]:


import sklearn.datasets
import numpy as np


# In[190]:


cancer_ds = sklearn.datasets.load_breast_cancer()


# In[191]:


X = cancer_ds.data
Y = cancer_ds.target


# In[192]:


print(X)
print(Y)


# In[193]:


print(X.shape, Y.shape) #gives number of rows/columns


# In[194]:


import pandas as pd
data = pd.DataFrame(cancer_ds.data, columns = cancer_ds.feature_names)


# In[195]:


data['class'] = cancer_ds.target


# In[196]:


data.head()


# In[197]:


data.describe()


# In[198]:


print(data['class'].value_counts)


# In[199]:


print(cancer_ds.target_names)


# In[200]:


data.groupby('mean area').mean()


# In[201]:


from sklearn.model_selection import train_test_split


# In[202]:


X = data.drop('class', axis=1)
Y = data['class']


# In[203]:


#to split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, stratify = Y, random_state = 1)


# In[204]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[205]:


print(Y.mean(),Y_train.mean(),Y_test.mean())


# In[206]:


print(X.mean(),X_train.mean(),X_test.mean())


# In[212]:


#binarization
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1,0])
#type(X_binarised_train)
X_binarised_train = X_binarised_train.values
X_binarised_test = X_binarised_test.values
type(X_binarised_train)


# In[243]:


print(X_binarised_train)
print(X_binarised_test)


# In[244]:


b=10  #any possible values for b

from random import randint
i=randint(0,X_binarised_train.shape[0])
if(np.sum(X_binarised_train[i,:])>=b):
    print('Model prediction is Malignant')
else:
    print('Model prediction is Bening')
    
    
if(Y_train[i]==1):
    print('Actual outcome is Malignant')
else:
    print('Actual outcome is Bening')


# In[242]:


#train
b=10
i=100
accurate_rows = 0

for X,y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x)>=b)
    accurate_rows += (y == y_pred)   
print('b=',b,'accurate rows-', accurate_rows,'accuracy=', accurate_rows/X_binarised_train.shape[0])


# In[237]:


#test
from random import randint

b=10
i=100
accurate_rows = 0
for X,y in zip(X_binarised_test, Y_test):
    y_pred = (np.sum(x)>=b)
    accurate_rows+=(y==y_pred)   
print('b=',b,';','accurate rows=',format(accurate_rows),'accuracy=', accurate_rows/X_binarised_test.shape[0])


# In[255]:


#train
for b in range (X_binarised_train.shape[1]+1):
    accurate_rows = 0

    for x,y in zip(X_binarised_train, Y_train):
         y_pred = (np.sum(x)>=b)
    accurate_rows += (y == y_pred)
    
    print(b,'accurate rows-', accurate_rows,'accuracy-', accurate_rows/X_binarised_train.shape[0])


# In[232]:


#test

for b in range (X_binarised_test.shape[1]+1):
    accurate_rows = 0

    for x,y in zip(X_binarised_test, Y_test):
         y_pred = (np.sum(x)>=b)
    accurate_rows += (y == y_pred)
    
    print(b,'accurate rows-', accurate_rows,'accuracy-', accurate_rows/X_binarised_test.shape[0])


# In[221]:


#train
b=27
i=100
accurate_rows = 0

for X,y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x)>=b)
    accurate_rows += (y == y_pred)   
print('b=',b,'-','accurate rows=',accurate_rows,'accuracy=', accurate_rows/X_binarised_train.shape[0])


# In[222]:


#train
b=28
i=100
accurate_rows = 0

for X,y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x)>=b)
    accurate_rows += (y == y_pred)  
print('b=',b,'-','accurate rows=',accurate_rows,'accuracy=', accurate_rows/X_binarised_train.shape[0])


# In[157]:


#test
from random import randint

b=28
i=100
accurate_rows = 0
for X,y in zip(X_binarised_test, Y_test):
    y_pred = (np.sum(x)>=b)
    accurate_rows+=(y==y_pred)   
print('b=',b,';','accurate rows=',format(accurate_rows),'accuracy=', accurate_rows/X_binarised_test.shape[0])


# In[230]:


#test
from random import randint

b=27
i=100
accurate_rows = 0
for X,y in zip(X_binarised_test, Y_test):
    y_pred = (np.sum(x)>=b)
    accurate_rows+=(y==y_pred)   
print('b=',b,';','accurate rows=',format(accurate_rows),'accuracy=', accurate_rows/X_binarised_test.shape[0])


# In[ ]:




