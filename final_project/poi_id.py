#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn import tree

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you"ll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Add a full list of the features available to us
financial_features = ["salary",
                      "deferral_payments",
                      "total_payments",
                      "loan_advances",
                      "bonus",
                      "restricted_stock_deferred",
                      "deferred_income",
                      "total_stock_value",
                      "expenses",
                      "exercised_stock_options",
                      "other",
                      "long_term_incentive",
                      "restricted_stock",
                      "director_fees"]
email_features = ["to_messages",
                  "email_address",
                  "from_poi_to_this_person",
                  "from_messages",
                  "from_this_person_to_poi",
                  "shared_receipt_with_poi"]
features_list = ["poi",
                 "bonus",
                 "salary",
                 "from_this_person_to_poi_ratio"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Feature that expresses emails to/from poi as a ratio of total emails
# Where there is no feature we save "NaN" as per featureFormat
for person in my_dataset.keys():
    person_data = my_dataset[person]
    # Create ratio for received messages
    if person_data["to_messages"] != 0 and person_data["to_messages"] != "NaN":
        if person_data["from_poi_to_this_person"] != "NaN":
            person_data["from_poi_to_this_person_ratio"] = (
                person_data["from_poi_to_this_person"] / float(person_data["to_messages"]))
        else:
            person_data["from_poi_to_this_person_ratio"] = "NaN"
    else:
        person_data["from_poi_to_this_person_ratio"] = "NaN"
    # Create ratio for from messages
    if person_data["from_messages"] != 0 and person_data["from_messages"] != "NaN":
        if person_data["from_this_person_to_poi"] != "NaN":
            person_data["from_this_person_to_poi_ratio"] = (
                person_data["from_this_person_to_poi"] / float(person_data["from_messages"]))
        else:
            person_data["from_this_person_to_poi_ratio"] = "NaN"
    else:
        person_data["from_this_person_to_poi_ratio"] = "NaN"

# Correct errors in "restricted_stock_deferred"
error_dict = {"BELFER ROBERT": "-44093",
              "BHATNAGAR SANJAY": "-2604490"}
for person in error_dict.keys():
    my_dataset[person]["restricted_stock_deferred"] = error_dict[person]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you"ll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=15)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)