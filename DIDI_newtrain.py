# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:41:37 2016

@author: abulin
"""

import csv

weather = []

with open('.data/Weather_table.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        weather.append(row)

with open('.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        weather.append(row)

