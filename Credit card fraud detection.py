#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


df=pd.read_csv(r"C:\Users\Akash\Downloads\CC.csv")
df


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[16]:


df.Amount.describe()


# In[11]:


#distribution of legit and fraud trnsaction 
df['Class'].value_counts()


# In[12]:


legit=df[df.Class == 0]
legit


# In[13]:


fraud=df[df.Class == 1]
fraud


# In[14]:


legit.shape


# In[15]:


fraud.shape


# In[22]:


legit.describe()


# In[21]:


fraud.Amount.describe()


# In[23]:


df.groupby('Class').mean()


# In[24]:


#build a sample datset containing similar distribution of normal and fraud transactions


# In[25]:


legit_sample = legit.sample(n=492)


# In[29]:


#concetating two datasets
new_data_set=pd.concat([legit_sample,fraud],axis=0)
new_data_set


# In[30]:


new_data_set.head()


# In[32]:


new_data_set['Class'].value_counts()


# In[34]:


new_data_set.groupby('Class').mean()


# splitting the data into features and target

# In[36]:


X=new_data_set.drop('Class',axis=1)
X


# In[37]:


Y=new_data_set['Class']
Y


# In[39]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=5)


# In[40]:


X_train.shape


# In[41]:


Y_train.shape


# In[43]:


x_test.shape


# In[44]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[48]:


prediction_on_training_data=model.predict(X_train)
accuracy_score_train=accuracy_score(Y_train,prediction_on_training_data)
accuracy_score_train


# In[50]:


prediction_on_testing_data=model.predict(x_test)
accuracy_score_test=accuracy_score(y_test,prediction_on_testing_data)
accuracy_score_test


# In[51]:


#just entertainment


# In[56]:


X=df.drop('Class',axis=1)
X


# In[57]:


Y=df['Class']
Y


# In[58]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=5)


# In[59]:


X_train.shape


# In[60]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[61]:


prediction_on_training_data=model.predict(X_train)
accuracy_score_train=accuracy_score(Y_train,prediction_on_training_data)
accuracy_score_train


# In[62]:


prediction_on_testing_data=model.predict(x_test)
accuracy_score_test=accuracy_score(y_test,prediction_on_testing_data)
accuracy_score_test


# In[ ]:




