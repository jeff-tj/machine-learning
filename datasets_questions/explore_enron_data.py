#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np
import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Number of elements per entry
#print len(enron_data[enron_data.keys()[0]])

# Print the number of persons of interest in email and finance db (E+F)
n = 0
for _, sub_dict in enron_data.iteritems():
    if sub_dict["poi"] == 1:
        n += 1
print "number of poi:", n

# Total number of poi names
poi_names = []
with open("../final_project/poi_names.txt") as poi_txt:
    for line in poi_txt:
        if line[0] == "(":
            poi_names.append(line[4:][:-1])
#print poi_names
print "total poi", len(poi_names)

# Query the database
def query_enron_db(query_name, query_fld):
    name_str = query_name.split(" ")
    if len(name_str) == 2:
        first_name = name_str[0].upper()
        last_name = name_str[1].upper()
        query_name_caps = last_name + " " + first_name
    elif len(name_str) == 3:
        first_name = name_str[0].upper()
        mid_name = name_str[1].upper()
        last_name = name_str[2].upper()
        query_name_caps = last_name + " " + first_name + " " + mid_name
    return enron_data[query_name_caps][query_fld]

# Qn 1
query_name = "James Prentice"
query_fld = "total_stock_value"
print query_name, ":", query_fld, ":", query_enron_db(query_name, query_fld)

# Qn 2
query_name = "Wesley Colwell"
query_fld = "total_stock_value"
print query_name, query_enron_db(query_name, "from_this_person_to_poi")
#print enron_data["COLWELL WESLEY"]

# Qn 3
query_name = "Jeffrey K Skilling"
query_fld = "exercised_stock_options"
print query_name, query_enron_db(query_name, query_fld)

# Qn Follow the Money
qry_names_ls = ["Jeffrey K Skilling", "Andrew S Fastow", "Kenneth L Lay"]
for name in qry_names_ls:
    print name, query_enron_db(name, "total_payments")

# Qn Dealing with Unfilled
qry_flds_ls = ["email_address", "salary"]
for flds in qry_flds_ls:
    count = 0
    for name in enron_data.keys():
        if enron_data[name][flds] != "NaN":
            count += 1
    print flds, count

# dict to array conversion
print enron_data[enron_data.keys()[0]].keys()
array_flds = ["poi", "total_payments"]
ef_array = featureFormat(enron_data, array_flds, remove_all_zeroes=False)
#print len(featureFormat(enron_data, array_flds, remove_all_zeroes=False))
#print len(ef_array)
print len(ef_array[ef_array[:, 0] == 1])
poi_array = ef_array[ef_array[:, 0] == 1]
print len(poi_array[poi_array[:, 1] != 0])
print poi_array

print len(ef_array)
print len(ef_array[ef_array[:,1] == 0])