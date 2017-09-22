#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 19:34:44 2017

@author: vidhya
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import Perceptron
from sklearn import metrics


bc = pd.read_csv('/Users/vikram/Downloads/data.csv')

bc = bc.drop("Unnamed: 32", axis=1)

tmp = bc.copy() 
bc = bc.drop('id', axis=1)
tmp['diagnosis'] = bc['diagnosis'].apply(lambda diagnosis: 0 if diagnosis == "B" else 1)

bc = bc.drop('diagnosis', axis=1)
target = tmp['diagnosis']

name = list(bc)
name.append('diagnosis')

### STANDARDIZATION ###
bc_std = (bc - bc.mean()) / (bc.std())

### NORMALIZATION ###
bc_norm = (bc - bc.min()) / (bc.max() - bc.min())

### LOG-NORMALIZATION ###
bc_log = bc.apply(np.log2)


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
X = bc_std.ix[:,0:30]
y = target

class_names = list(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# compute a cross-validated score.
svc = SVC(kernel="linear")

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(4),scoring='accuracy', verbose=0)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

name = bc_std.columns
name = name[0:30] 

print( 'selected attributes are:')

selectedFeatures = []

c=0
for i in np.arange(rfecv.support_.size):
    if rfecv.support_[i]==True :
        print('%f \t %s' % (rfecv.grid_scores_[i],name[i]))
        selectedFeatures.append(name[i])
        c=c+1

XafterRFECV = rfecv.transform(X)

bc_std_sel = pd.DataFrame(XafterRFECV)

bc_std_sel.columns = selectedFeatures


m_eval = pd.DataFrame(columns = ['method','trainscore','testscore','False Alarm','Miss','Accuracy'])

def addeval(method, train, test, tpos, tneg,acc):
    global m_eval
    d = pd.DataFrame([[method, train, test, tpos, tneg,acc]],columns = ['method','trainscore','testscore','False Alarm','Miss','Accuracy'])
    m_eval = m_eval.append(d)

svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(X,y)
svc_pred = svc.fit(X_train, y_train).predict(X_test)
print(svc.score(X_train,y_train), svc.score(X_test, y_test))

mtrx = confusion_matrix(y_test,svc_pred)
print(mtrx)
FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('SVM',svc.score(X_train,y_train), svc.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(svc_pred, y_test)*100))
print("Accuracy SVM: %s" % "{0:.3%}".format(metrics.accuracy_score(svc_pred, y_test)))

###Logistic Regression
lr = LogisticRegression(penalty = 'l2', dual = True)
lr_pred = lr.fit(X_train, y_train).predict(X_test)
print(lr.score(X_train,y_train), lr.score(X_test, y_test),lr.score(X_train,y_train)-lr.score(X_test,y_test))

mtrx = confusion_matrix(y_test,lr_pred)
print(mtrx)
FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('Log Reg',lr.score(X_train,y_train), lr.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(lr_pred, y_test)*100))
print("Accuracy Log Reg: %s" % "{0:.3%}".format(metrics.accuracy_score(lr_pred, y_test)))

#Decision Tree 
dt_clf = DecisionTreeClassifier(random_state=42,max_depth=1)
dt_pred = dt_clf.fit(X_train, y_train).predict(X_test)
print(dt_clf.score(X_train,y_train), dt_clf.score(X_test,y_test), dt_clf.score(X_train,y_train)-dt_clf.score(X_test,y_test))
mtrx = confusion_matrix(y_test,dt_pred)
print(mtrx)

FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('Decision Tree',dt_clf.score(X_train,y_train), dt_clf.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(dt_pred, y_test)*100))
print("Accuracy Decision Tree: %s" % "{0:.3%}".format(metrics.accuracy_score(dt_pred, y_test)))


#Gauss Naive Bayes
gauss = GaussianNB()
gauss_pred = gauss.fit(X_train, y_train).predict(X_test)
print(gauss.score(X_train,y_train), gauss.score(X_test,y_test), gauss.score(X_train,y_train)-gauss.score(X_test,y_test))

mtrx = confusion_matrix(y_test,gauss_pred)
print(mtrx)
FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('Gauss NB',gauss.score(X_train,y_train), gauss.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(gauss_pred, y_test)*100))
print("Accuracy Gauss: %s" % "{0:.3%}".format(metrics.accuracy_score(gauss_pred, y_test)))

#Perceptron

perc = Perceptron(alpha = 1, penalty = None,fit_intercept = False)
perc_pred = perc.fit(X_train, y_train).predict(X_test)
print(perc.score(X_train,y_train), perc.score(X_test,y_test), perc.score(X_train,y_train)-perc.score(X_test,y_test))

mtrx = confusion_matrix(y_test,perc_pred)
print(mtrx)

FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('Perceptron',perc.score(X_train,y_train), perc.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(perc_pred, y_test)*100))
print("Accuracy Perceptron: %s" % "{0:.3%}".format(metrics.accuracy_score(perc_pred, y_test)))

#Adaboost
ens = AdaBoostClassifier()
ens_pred = ens.fit(X_train, y_train).predict(X_test)
print(ens.score(X_train,y_train), ens.score(X_test,y_test), ens.score(X_train,y_train)-ens.score(X_test,y_test))

mtrx = confusion_matrix(y_test,ens_pred)
print(mtrx)
FAvalue=mtrx[1,0]/(mtrx[1,0]+mtrx[1,1])
missvalue=mtrx[0,1]/(mtrx[0,1]+mtrx[0,0])
addeval('Adaboost',ens.score(X_train,y_train), ens.score(X_test, y_test),FAvalue,missvalue,(metrics.accuracy_score(ens_pred, y_test)*100))
print("Accuracy Adaboost: %s" % "{0:.3%}".format(metrics.accuracy_score(ens_pred, y_test)))


print(m_eval)

mm1_eval = pd.melt(m_eval[['method','False Alarm','Miss']], "method", var_name="Measurement")

p = sns.pointplot(x="method", y="value", hue="Measurement", data=mm1_eval)
labs = list(m_eval['method'])
p.set_xticklabels(labs, rotation=90);

