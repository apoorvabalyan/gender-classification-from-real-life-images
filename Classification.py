#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Scaling 
def scaling(x_train):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    return x_train

#Classification
def clf(x_train,x_test,y_train,y_test):
    #LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_lda = lda.predict(x_test)
    print('Lda:',metrics.accuracy_score(y_test,y_lda))
    
    #QDA

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    y_qda = qda.predict(x_test)
    print('Qda:',metrics.accuracy_score(y_test,y_qda))
    
    #Logisitic Regression
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    y_predicted = clf.predict(x_test)
    print("LR:",metrics.accuracy_score(y_test,y_predicted))

    # SVM
    model = svm.SVC(kernel = 'linear')
    model.fit(x_train,y_train)
    y_pred1 = model.predict(x_test)
    print("SVM Linear:",metrics.accuracy_score(y_test,y_pred1))
    
    model_s = svm.SVC(kernel = 'rbf')
    model_s.fit(x_train,y_train)
    y_preds = model.predict(x_test)
    print("SVM RBF:",metrics.accuracy_score(y_test,y_preds))
    
    #Naive Bayes
    model1 = GaussianNB()
    model1.fit(x_train,y_train)
    y_pred2 = model1.predict(x_test)
    print("NB:",metrics.accuracy_score(y_test,y_pred2))

    #KNN
    model2 = neighbors.KNeighborsClassifier()
    model2.fit(x_train,y_train)
    y_pred3 = model2.predict(x_test)
    print("KNN:",metrics.accuracy_score(y_test,y_pred3))

    #Decision Tree
    model3 = tree.DecisionTreeClassifier()
    model3.fit(x_train,y_train)
    y_pred4 = model3.predict(x_test)
    print("DT:",metrics.accuracy_score(y_test,y_pred4))

path = 'C:/Users/Apoorva/Desktop/data.csv'
data = pd.read_csv(path)

x = data.iloc[:,0:data.shape[1]-1].values
y = data.iloc[:,data.shape[1]-1].values
x = scaling(x)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.25)
pca = decomposition.PCA(n_components = 75)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
clf(x_train,x_test,y_train,y_test)

    

