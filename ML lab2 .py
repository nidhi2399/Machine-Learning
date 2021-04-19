#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Registration no. = 20MAI0020


# In[114]:


#import libraries
import pandas as pd
import numpy as np


# In[115]:


#load the data
data =pd.read_csv('C:\\Users\\Hp\\Desktop\\crx.data', header= None)


# In[116]:


#A1 A2 A3...A16
varnames = ['col'+ str(s) for s in range(1,17)]


# In[117]:


#add column names
data.columns = varnames


# In[118]:


data.head()


# In[119]:


data.tail()


# In[120]:


#print last 10 columns
data.tail(10)


# In[121]:


#replace ? with np.nan(not a number)
data = data.replace('?',np.nan)


# In[122]:


#display
data.info()


# In[123]:


#datatypes are as float64, int64, object


# In[124]:


#recasting col2 and col14 to its correct type
data['col2'] = data['col2'].astype('float')
data['col14'] = data['col14'].astype('float')


# In[125]:


#replacing '+' and '-' values in col16 with 'P' and 'N' respectively
data['col16'] =data['col16'].map({'+':'P', '-':'N'})


# In[126]:


#display col16
data['col16']


# In[127]:


#display number of variables of type objects
cat_columns = [c for c in data.columns if data[c].dtypes == 'O']
data[cat_columns].head()


# In[128]:


#dataset= loan.csv
data = pd.read_csv('C:\\Users\\Hp\\Desktop\\loan.csv')


# In[129]:


data.head()


# In[130]:


#calculating mean
data['disbursed_amount'].mean()


# In[131]:


#calculating mean
data['interest'].mean()


# In[132]:


#number of discrete variables
print(data.market.value_counts())


# In[133]:


#display unique values
data['number_open_accounts'].unique()


# In[134]:


data['customer_id'].unique()


# In[138]:


data[['date_issued','date_last_payment']].dtypes


# In[140]:


data['date_issued_dt'] = pd.to_datetime(data['date_issued'])
data.head()


# In[153]:


#find months
data['month'] = data['date_issued_dt'].dt.month
data.head()


# In[160]:


#months with most of loan issued date
data.groupby(["month"])["date_issued"].count()


# In[161]:


#above output displays months along with the number of loans issued per month


# In[162]:


#teachers who are owners
data.loc[(data['employment'] == 'Teacher') & (data['householder'] == 'OWNER'),['employment','householder']]


# In[163]:


#count of teachers who are owners
x=data.loc[(data['employment'] == 'Teacher') & (data['householder'] == 'OWNER'),['employment','householder']]
x.value_counts()


# In[164]:


#count of teachers who are owners is 69.


# In[165]:


#employment of customers who mostly rent
data.loc[data['householder'] == 'RENT',['employment','householder']]


# In[ ]:




