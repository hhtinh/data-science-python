"""
@ Filename:       TreeRegression_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-13   
@ Update Date:    2019-05-15
@ Description:    Implement TreeRegression_TEST
"""

import matplotlib.pyplot as plt
from sklearn import linear_model

from TreeRegression import *
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import time

def plot(real_label, regression_label):
    # test_label = np.expand_dims(test_label, axis=1)
    plot1 = plt.plot(regression_label, 'r*', label='Regression values')
    plot2 = plt.plot(real_label, 'b', label='Real values')
    plt.xlabel('X ')
    plt.ylabel('Y')
    plt.legend(loc=3)
    plt.title('Tree Regression')
    plt.show()


trainData = np.array(pd.read_table('../dataset5/train.txt', header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table('../dataset5/test.txt', header=None, encoding='gb2312', delim_whitespace=True))
trainLabel = trainData[:, -1]
trainData = np.delete(trainData, -1, axis=1)
testLabel = testData[:, -1]
testData = np.delete(testData, -1, axis=1)


time_start1 = time.time()
clf1 =  DecisionTreeRegressor()
clf1.fit(trainData, trainLabel)
regression_label = clf1.predict(testData)
time_end1 = time.time()
plot(testLabel, regression_label)
print("Runtime of Sklearn-tree regression:", time_end1-time_start1)

time_start2 = time.time()
clf2 = treeRegression()
clf2.train(trainData, trainLabel)
regression_label2 = clf2.predict(testData)
time_end2 = time.time()
plot(testLabel,regression_label2 )
print("Runtime of self-tree regression:", time_end2-time_start2)
