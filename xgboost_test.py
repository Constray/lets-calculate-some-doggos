"""
This module contains code to experemint on xgboost algorhytm
"""
import dogreader as dr
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# indexed of classes to filter
filtered_classes = [1, 2, 3, 4, 5, 6]

# Proportion of train/test data split
test_split = 0.20

# Read from data set
data_set = dr.read_data_set()
# data_set = data_set.loc[data_set['Primary status'].isin(filtered_classes)]

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=test_split)

classifier = XGBClassifier().fit(x_train, y_train)
y_result = classifier.predict(x_test)
print("============================")
train_accuracy = metrics.accuracy_score(y_train, classifier.predict(x_train))
print("Train set Accuracy: ", train_accuracy)
test_accuracy = metrics.accuracy_score(y_test, y_result)
print("Test set Accuracy: ", test_accuracy)
print("============================\n")