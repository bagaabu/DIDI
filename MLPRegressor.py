# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 21:24:35 2016

@author: abulin
"""
from __future__ import division 
import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

samples = []
responses = []
clf = []
results = []
sum_ = 0
num_ = 0
tt = 2961*0.8

with open('didi_train_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        samples.append(row)
with open('didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)
     
responses = np.array(responses)
samples = np.array(samples)
responses_train = responses[1:tt:,]
samples_train = samples[1:tt:,]

responses_test = responses[tt+1:2961:,]
samples_test = samples[tt+1:2961:,]

clf = MLPRegressor()
clf.fit(samples_train,responses_train)
joblib.dump(clf,'nueru.pkl', compress = 3)
