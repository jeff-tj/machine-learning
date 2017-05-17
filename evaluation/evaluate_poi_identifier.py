#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
# create training-testing split - copied from validation_poi.py
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=.3, random_state=42)

# create and fit classifier - copied from validation_poi.py
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "predicted poi = ", sum(pred)
print "test size = ", len(pred)
print "acc if all 0s pred = ", 1. - sum(labels_test)/len(labels_test)

# calc true positives
def true_pos(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print "Lists not of same size"
        return
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
    return tp

print "true positives = ", true_pos(labels_test, pred)

# calculate some test benchmarks
from sklearn import metrics
print "precision = ", metrics.precision_score(labels_test, pred)
print "recall = ", metrics.recall_score(labels_test, pred)

# example data set
print "***** metrics on udacity example *****"
ex_pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
ex_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print "true pos = ", true_pos(ex_labels, ex_pred)
# calc true negatives
def true_neg(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print "Lists not of same size"
        return
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    return tn
print "true neg = ", true_neg(ex_labels, ex_pred)

# calc false negatives
def false_neg(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print "Lists not of same size"
        return
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return fn
print "false neg = ", false_neg(ex_labels, ex_pred)

# calc false positives
def false_pos(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print "Lists not of same size"
        return
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp
print "false pos = ", false_pos(ex_labels, ex_pred)

print "precision = ", metrics.precision_score(ex_labels, ex_pred)
print "recall = ", metrics.recall_score(ex_labels, ex_pred)