# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:24:15 2016

@author: abulin
"""
from __future__ import division 
import csv
import threading
import time
import numpy as np
from threading import Thread,Semaphore
from parafind import paratest
import copy

class myThread (threading.Thread):
    def __init__(self,C,D,traset,trares,tesset,testres,name):
        threading.Thread.__init__(self)
        self.C = C
        self.D = D
        self.traset = traset
        self.trares = trares
        self.tesset = tesset
        self.testres = testres
        self.name = name
    def run(self):
        print 'start'+str(self.name)
        a = paratest(self.C,self.D,self.traset,self.trares,self.tesset,self.testres,self.name)
        print 'end' +str(self.name)
        return (a)
        
sample = []
samples = []
samples_trafic = []
samples_train = []
samples_test = []
responses = []
clf = []
results = []
Results = []
weights = []
clatrain = []
clatest = []
sum_ = 0
num_ = 0
tt = 2369
LF = 1000

with open('./data/didi_train_data.csv', 'rb') as csvfile:
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

with open('./data/didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)

with open('./data/best_weights.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ float(i) for i in lable]
        weights.append(lable)

for k in range(0,66):
    sample_tem = copy.deepcopy(sample)
    for m in range(0,2961):
        for n in range(0,199):
            sample_tem[m][n] = sample[m][n]*weights[n][k]
    sample_tem = np.array(sample_tem)
    samples.append(sample_tem)
    sample_tem =[]
      
responses = np.array(responses)

#pca = PCA(n_components=100)
#samples = pca.fit(samples).transform(samples)
responses_train = responses[1:tt:,]
for x in range(0,66):
    samples_train.append(samples[x][1:tt:,])

responses_test = responses[tt+1:2961:,]
for x in range(0,66):
    samples_test.append(samples[x][tt+1:2961:,])

for x in range(0,66):
    clatrain.append(samples_train[x][0:1000:,])
claresp = responses_train[0:1000:,]

for x in range(0,66):
    clatest.append(samples_train[x][1001:2001:,])
claresptest = responses_train[1001:2001:,]

#C_range1 = np.linspace(0.00001,3,4)
C_range1 = np.logspace(-5, 1, 6)
degree_range1 = np.linspace(0.1,1,9)
#C_range2 = np.linspace(3,4,4)
C_range2 = np.logspace(-8, -4, 6)
degree_range2 = np.linspace(0.1,1,9)

thread1 = myThread(C_range1,degree_range1,clatrain,claresp,clatest,claresptest,1)
thread2 = myThread(C_range2,degree_range2,clatrain,claresp,clatest,claresptest,2)
thread1.start()
thread2.start()

