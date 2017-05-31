#!/usr/bin/python

import pickle
import numpy
from time import time
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn import tree
import numpy as np
# setup decision tree classifier
start_t = time()
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "run time: ", round(time() - start_t, 3), "s"
print "acc (training) = ", clf.score(features_train, labels_train)
print "acc (test) = ", clf.score(features_test, labels_test)

# extract the most important feature
importance = clf.feature_importances_
imp_high_n = np.argsort(np.asarray(importance))[-11:]
#imp_high_val = importance[imp_high_n]
#print imp_high_n
#print imp_high_val
feature_list = vectorizer.get_feature_names()
n = len(imp_high_n)
for i in imp_high_n:
    print n, ": ", importance[i], " ", feature_list[i], ", index: ", i
    n -= 1