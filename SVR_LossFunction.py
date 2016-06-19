# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:09:16 2016

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
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import linear_model
import copy
import pandas as pd
from Kmean_dis import KmeansPOI
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from collections import Counter

sample = []
samples = []
samples_trafic = []
samples_train = []
samples_test = []
responses = []
clf = []
coef = []
results = []
Results = []
weights = []
over_04_list = []
over_04_value = []
equal = []
equals = []
equal_value = []
equal_values = []
sum_ = 0
num_ = 0
tt = 2369
LF = 1000

with open('didi_train_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        sample.append(row)

#with open('didi_traffic.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for trafic in spamreader:
#        trafic = trafic[0].split(',')
#        trafic = [int(i)/100 for i in trafic ]    
#        samples_trafic.append(trafic)

with open('didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)

with open('best_weights.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ float(i) for i in lable]
        weights.append(lable)
        
#with open('coef_binary_weights.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for lable in spamreader:
#        lable = lable[0].split(',')
#        lable = [ float(i) for i in lable]
#        coef.append(lable)

for k in range(0,66):
    sample_tem = copy.deepcopy(sample)
    for m in range(0,2961):
        for n in range(0,199):
            sample_tem[m][n] = sample[m][n]*weights[n][k]#*coef[n][k]
    sample_tem = np.array(sample_tem)
    samples.append(sample_tem)
    sample_tem =[]
    
responses = np.array(responses)

#for z in range (0,66):
#    table = pd.DataFrame(samples[z])
#    table.to_csv('Trainset_weight'+str(z)+'.csv',index = False, header = False)

#samples_trafic = np.array(samples_trafic)
#samples = np.c_[samples,samples_trafic]
#pca = PCA(n_components=100)
#for i in range(0,66):
#    samples[i] = pca.fit(samples[i]).transform(samples[i])

responses_train = responses[1:tt:,]
for x in range(0,66):
    samples_train.append(samples[x][1:tt:,])

responses_test = responses[tt+1:2961:,]
for x in range(0,66):
    samples_test.append(samples[x][tt+1:2961:,])

clf = RandomForestRegressor(n_estimators =200, criterion = 'MAPE')
clf.fit (samples_train[0], responses_train[:,0])
