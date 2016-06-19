# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 21:49:44 2016

@author: abulin
"""
from __future__ import division 
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

samples = []
samples_trafic = []
responses = []
clf = []
results = []
kmeans = []
sum_ = 0
num_ = 0
tt = 2961*0.8

with open('didi_train_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        samples.append(row)

#with open('didi_traffic.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for trafic in spamreader:
#        trafic = trafic[0].split(',')
#        trafic = [int(i)/50 for i in trafic ]    
#        samples_trafic.append(trafic)

with open('didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)

with open('kmeans.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for k in spamreader:
        k = int(k)
        kmeans.append(k)
        
responses = np.array(responses)
samples = np.array(samples)
#samples_trafic = np.array(samples_trafic)
#samples = np.c_[samples,samples_trafic]
responses_train = responses[1:2000:,]
samples_train = samples[1:2000:,]

responses_test = responses[2001:2961:,]
samples_test = samples[2001:2961:,]
     
responses = np.array(responses)
samples = np.array(samples)


#clf = joblib.load('RandomF.pkl')
#responses_train = responses[2001:2961:,]
#samples_train = samples[2001:2961:,]
#responses = responses[1:2000:,]
#samples = samples[1:2000:,]
#clf = RandomForestClassifier(n_estimators=30)
#clf.fit (samples, responses)
#joblib.dump(clf,'RandomF.pkl', compress = 3)
#predict_result = clf.predict(samples_train)
#a = np.abs(responses_train-predict_result)
#b = responses_train-predict_result
#sum_ = 0
#num_ = 0
#for i in range(predict_result.shape[0]):
#    for j in range(predict_result.shape[1]):
#        if (responses_train[i,j] != 0):
#            sum_ = sum_ + a[i,j]/responses_train[i,j]
#            num_ = num_ + 1
#print 'eva: %f, ls: %f\n' %(sum_/num_, np.sqrt(np.sum(b*b))/(predict_result.shape[0]*predict_result.shape[1]))
