#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#NIDHI GHUBLE 20MAI0020


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np

df=pd.read_csv('housing.csv')
df.head(20)


# 1. Nearest available mode of commute

# In[3]:


df['Nearest'].mode()


# 2. Find the count of properties that are Apartment Type and Have metros near to it

# In[4]:


Residential=df[df['Type']=='Apartment']
Residential_Metro=Residential[Residential['Nearest']=='Metro_Rail']
Residential_Metro.shape[0]


# 3. Mapping 'Yes':0 'No':1

# In[5]:


df['Parking']=df['Parking'].map({'Yes':1,'No':0})
df['Furnished']=df['Furnished'].map({'Yes':1,'No':0})
df['Occupany_certificate']=df['Occupany_certificate'].map({'Yes':1,'No':0})
df['Security']=df['Security'].map({'Yes':1,'No':0})

#Converting into 0 and 1 gives more scope of playing with the features.


# In[6]:


df.head()


# 4. Mapping the numerical values for Nearest.

# In[7]:


uniques=df['Nearest'].unique()
d={}
k=0
for i in uniques:
    if i not in d:
        d[i]=k
        k+=1
df['Nearest']=df['Nearest'].map(d)


# In[8]:


print("The mapping is: ",d)
#Converting into numerical values gives more scope of playing with the features.


# In[9]:


df.head()


# 5. appartments those are 2bhk 

# In[10]:


df.loc[df['Type'] == 'Apartment',['Type','BHK']]


# 6. Mapping the numerical values for Type

# In[11]:


uniques=list(df['Type'].unique())
d={}
k=0
for i in uniques:
    if i not in d:
        d[i]=k
        k+=1
df['Type']=df['Type'].map(d)


# In[12]:


print("The mapping is: ",d)
#Converting into numerical values gives more scope of playing with the features.


# In[13]:


df.head()


# 7. automatic construction of new features from raw data
# Creating a state column from pincode data

# In[14]:


state=[]
for i in range(df.shape[0]):
  
  if(df['Pincode'][i].astype(str).startswith('5')):
    state.append("Andhra Pradesh")
   
  elif(df['Pincode'][i].astype(str).startswith('6')):
    state.append("TamilNadu")
  
  elif(df['Pincode'][i].astype(str).startswith('7')):   
    state.append("West Bengal")
  
  elif(df['Pincode'][i].astype(str).startswith('1')):
    state.append("Delhi")
  
  elif(df['Pincode'][i].astype(str).startswith('2')):
    state.append("Uttar Pradesh")

  elif(df['Pincode'][i].astype(str).startswith('4')):  
    state.append("Maharastra")

df["State"]=state
df.head()


# 8. finding the mean of area col

# In[15]:


df['Area'].mean() 


# 9. Since price is normally distributed StandardScaler is used to normalize the values

# In[16]:


import seaborn as sns
sns.distplot(df["Price"],kde = True,bins = 10)


# 10. Min Max Scaling of Prices

# In[17]:



from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()

df["Price"] = scaler.fit_transform(df["Price"].values.reshape(-1,1))
df["Area"] = scaler.fit_transform(df["Area"].values.reshape(-1,1))


# In[18]:


df.head()


# https://drive.google.com/file/d/1aWzbbj4vgxo6lVSxDCPmhSvE8vLfZAIr/view

# In[ ]:


#video link - https://drive.google.com/file/d/1aWzbbj4vgxo6lVSxDCPmhSvE8vLfZAIr/view

