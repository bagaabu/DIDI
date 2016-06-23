# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:52:47 2016

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
from modify import modify
from sklearn.cross_validation import LabelShuffleSplit

sample = []
samples = []
samples_trafic = []
samples_train = []
samples_test = []
responses = []
responses_test = []
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

with open('.data/didi_train_data_w_t.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        sample.append(row)

with open('.data/didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)

with open('.data/best_weights10.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ float(i) for i in lable]
        weights.append(lable)
        
for k in range(0,66):
    sample_tem = copy.deepcopy(sample)
    for m in range(0,2961):
        for n in range(0,475):
            sample_tem[m][n] = sample[m][n]*weights[n][k]#*coef[n][k]
    sample_tem = np.array(sample_tem)
#    TEM = sample_tem[0:2369:,]
    TEM = sample_tem[591:2960:,]
    samples.append(TEM)
#    TEM_R = sample_tem[2370:2960:,]
    TEM_R = sample_tem[0:590:,]
    samples_test.append(TEM_R)
    sample_tem =[]
#RT = responses[0:2369]
RT = responses[591:2960]
Responses = np.array(RT)
#RT = responses[2370:2960]
RT = responses[0:590]
responses_test = np.array(RT)

ind=[7,1]     
Mo =[]
Mos = []
Mo_tem = []
for t in range (0,1):    
    locals()['clf' + str(0)]  = svm.SVR(C=ind[0],kernel='rbf', epsilon = ind[1],gamma =0.001)  
#    locals()['clf' + str(0)] = RandomForestRegressor(n_estimators = 1500, max_features=0.5,min_samples_leaf= 50, n_jobs =  - 1, verbose=1)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(samples[0], Responses[:,0], test_size=0.2, random_state=t)
    locals()['clf' + str(0)].fit (X_train,y_train)
    results = locals()['clf' + str(0)].predict(X_test)
    Response = y_test
    print 'dis'+str(0+1)+'rounds:'+str(t)
    mo = modify(results, y_test)
    Mo_tem = np.append(Mo_tem,mo)
Mo = np.mean(Mo_tem)
Mo_tem =[]
Mos = np.append(Mos,Mo)
results_m = [max(i+Mo,1) for i in results]

for i in range(1,66):
    for e in range(0,1):        
        locals()['clf' + str(i)]  = svm.SVR(C=ind[0],kernel='rbf', epsilon = ind[1],gamma =0.001)
#        locals()['clf' + str(i)] = RandomForestRegressor(n_estimators = 1500, max_features=0.5,min_samples_leaf= 50, n_jobs =  - 1, verbose=1)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(samples[i], Responses[:,i], test_size=0.2, random_state=e)
        locals()['clf' + str(i)].fit (X_train,y_train)
        locals()['result' + str(i)]= locals()['clf' + str(i)].predict(X_test)
        print 'dis'+str(i+1)+'rounds:'+str(e)
        mo = modify(locals()['result' + str(i)],y_test)
        Mo_tem = np.append(Mo_tem,mo)
    Mo = np.mean(Mo_tem)
    Mo_tem =[]
    Mos = np.append(Mos,Mo)
    results_m = np.c_[results_m,[ max(b+Mo,1) for b in locals()['result' + str(i)]]]
    results = np.c_[results,locals()['result' + str(i)]]    
    Response = np.c_[Response,y_test]

a = np.abs(Response-results)
b = Response-results
sum_ = 0
num_ = 0
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        if (Response[i,j] != 0):
            sum_ = sum_ + a[i,j]/Response[i,j]
            num_ = num_ + 1
print 'Origian-eva: %f, ls: %f\n' %(sum_/num_, np.sqrt(np.sum(b*b))/(results.shape[0]*results.shape[1]))

a = np.abs(Response-results_m)
b = Response-results_m
sum_ = 0
num_ = 0
for i in range(results_m.shape[0]):
    for j in range(results_m.shape[1]):
        if (Response[i,j] != 0):
            sum_ = sum_ + a[i,j]/Response[i,j]
            num_ = num_ + 1
print 'Modify-eva: %f, ls: %f\n' %(sum_/num_, np.sqrt(np.sum(b*b))/(results_m.shape[0]*results_m.shape[1]))
##############################################################################################################
RESULT = []
locals()['result' + str(0)]= locals()['clf' + str(0)].predict(samples_test[0])
Res = responses_test[:,0]
locals()['result' + str(0)] = [ max(B+Mos[0],1) for B in locals()['result' + str(0)]]
RESULT = locals()['result' + str(0)]
a = np.abs(Res-locals()['result' + str(0)])
b = Res-locals()['result' + str(0)]
sum_ = 0
num_ = 0
for i in range(Res.shape[0]):
    if (Res[i] != 0):
        sum_ = sum_ + a[i]/Res[i]
        num_ = num_ + 1
outputs = sum_/num_
#print 'Dist'+str(0+1)+'eva: %f' %(sum_/num_)
print '%f' %(outputs)
for I in range (1,66):
    locals()['result' + str(I)]= locals()['clf' + str(I)].predict(samples_test[I])
    Res = responses_test[:,I]
    locals()['result' + str(I)] = [ max(B+Mos[I],1) for B in locals()['result' + str(I)]]
    RESULT = np.c_[RESULT,locals()['result' + str(I)]]
    a = np.abs(Res-locals()['result' + str(I)])
    b = Res-locals()['result' + str(I)]
    sum_ = 0
    num_ = 0
    for i in range(Res.shape[0]):
        if (Res[i] != 0):
            sum_ = sum_ + a[i]/Res[i]
            num_ = num_ + 1
    output = sum_/num_
    outputs = np.append(outputs,output)
#    print 'Dist'+str(I+1)+'eva: %f' %(sum_/num_)
    print '%f' %(output)    
#a = np.abs(responses_test-RESULT)
#b = responses_test-RESULT
#sum_ = 0
#num_ = 0
#for i in range(responses_test.shape[0]):
#    if (responses_test[:,0][i] != 0):
#        sum_ = sum_ + a[i]/responses_test[:,0][i]
#        num_ = num_ + 1
#print 'finaleva: %f' %(sum_/num_)
a = np.abs(responses_test-RESULT)
b = responses_test-RESULT
sum_ = 0
num_ = 0
for i in range(responses_test.shape[0]):
    for j in range(responses_test.shape[1]):
        if (responses_test[i,j] != 0):
            sum_ = sum_ + a[i,j]/responses_test[i,j]
            num_ = num_ + 1
print 'finaleva: %f' %(sum_/num_)
