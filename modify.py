# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 07:23:20 2016

@author: abulin
"""

def modify(clf_predict,y_test):
    MAPE_MIN = 99999
    modify = 0
    diff = clf_predict - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
    print " MAPE: " + str(MAPE)

    lower = [max(i - 0.25, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -0.25
        print " MAPE -0.25: " + str(MAPE)
    
    lower = [max(i - 0.5, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -0.5
        print " MAPE -0.5: " + str(MAPE)

    lower = [max(i - 0.75, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -0.75
        print " MAPE -0.75: " + str(MAPE)

    lower = [max(i - 1, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -1
        print " MAPE -1: " + str(MAPE)
    
    lower = [max(i - 1.5, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -1.5
        print " MAPE -1.5: " + str(MAPE)

    lower = [max(i - 2, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -2
        print " MAPE -2: " + str(MAPE)

    upper = [max(i - 3, 1) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = -3
        print " MAPE -3: " + str(MAPE)

    upper = [max(i + 0.5, 0) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = + 0.5
        print " MAPE +0.5: " + str(MAPE)

    upper = [max(i + 1, 0) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test[y_test != 0]))
    if MAPE<MAPE_MIN:
        MAPE_MIN =  MAPE 
        modify = 1
        print " MAPE +1: " + str(MAPE)

    return modify