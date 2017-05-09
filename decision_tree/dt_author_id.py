#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Set up classifier
clf = tree.DecisionTreeClassifier(min_samples_split=40)
train_st = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-train_st, 3), "s"

# Make some predictions
test_st = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-test_st, 3), "s"

# Grade the accuracy
acc = accuracy_score(labels_test, pred)
print acc
#########################################################


