#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.head())


# In[12]:


print(data)


# In[13]:


print(data.shape)


# In[15]:


print(data.describe())


# In[16]:


y=data.quality
x=data.drop('quality',axis=1)


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=123,stratify=y)


# In[23]:


scaler = preprocessing.StandardScaler().fit(x_train)
x_trained_scaled = scaler.transform(x_train)
print(x_trained_scaled.mean(axis=0))
print(x_trained_scaled.std(axis=0))


# In[24]:


x_test_scaled = scaler.transform(x_test)
print(x_test_scaled.mean(axis=0))
print(x_test_scaled.std(axis=0))


# In[25]:


pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))


# In[26]:


print(pipeline.get_params())


# In[36]:


hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth' : [None, 5, 3, 1]}


# In[37]:


clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(x_train, y_train)


# In[38]:


print(clf.refit)


# In[41]:


y_pred = clf.predict(x_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# In[42]:


joblib.dump(clf, 'rf_regressor.pk1')


# In[ ]:





# In[ ]:




