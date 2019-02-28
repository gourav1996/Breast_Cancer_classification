
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
'''
print(cancer.keys())
print(cancer['DESCR'])
print(cancer['feature_names'])
print(cancer['target'])
'''
print(cancer['target_names'])
cancer['data'].shape

# Convert the data into pandas dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns=np.append(cancer['feature_names'],['target']))

#df_cancer.head(4)
'''
# Visulization the data
sns.pairplot(df_cancer,vars=['mean radius','mean texture','mean perimeter','mean area',
 'mean smoothness'])
'''
# Visulization the data with respect to target
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean perimeter','mean area',
 'mean smoothness'])

sns.countplot(df_cancer['target'])

# Correlation 
plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)

# Model Training
X = df_cancer.drop(['target'], axis=1)
X.head()

y = df_cancer['target']
y.tail()

# Making training and testing data by spliting the dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # For SVM classification 
from sklearn.metrics import classification_report, confusion_matrix

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

# Normalazation 
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/ range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/ range_test

svc_model = SVC()
'''
svc_model.fit(X_train_scaled,y_train)
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm , annot=True)

print(classification_report(y_test, y_predict))
'''
# Improving the model
# Tunning the SVM Parameters
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid,refit=True, verbose= 4)
grid.fit(X_train_scaled,y_train)

grid.best_params_

y_predict = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm ,annot=True)

print(classification_report(y_test, y_predict))

