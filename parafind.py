# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:26:23 2016

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

def paratest(C,D,traset,trares,tesset,testres,name):   
    clatrain=traset
    claresp=trares
    clatest=tesset
    claresptest=testres
    LF = 1000
    C_range = C
    degree_range = D
    for degree in degree_range:
        for C in C_range:
            locals()['clf' + str(0)] = svm.SVR(C = C, epsilon = degree)
            locals()['clf' + str(0)].fit(clatrain[0],claresp[:,0])
            temr = locals()['clf' + str(0)].predict(clatest[0])
            for n in range (1,66):
                locals()['clf' + str(n)] = svm.SVR(C = C, epsilon = degree)
                locals()['clf' + str(n)].fit(clatrain[n],claresp[:,n])
                locals()['result' + str(n)] = locals()['clf' + str(n)].predict(clatest[n])
                temr = np.c_[temr,locals()['result' + str(n)]]            
            a = np.abs(claresptest-temr)
            b = claresptest-temr
            sum_ = 0
            num_ = 0
            for i in range(claresptest.shape[0]):
                for j in range(claresptest.shape[1]):
                    if (claresptest[i,j] != 0):
                        sum_ = sum_ + a[i,j]/claresptest[i,j]
                        num_ = num_ + 1
            print 'eva: %f'%(sum_/num_)+'__'+str(C)+'__'+str(degree)+'__'+str(name)
            F = (sum_/num_)
            if F<LF:
                ind = [C,degree]
                LF = F
    return (ind,F)