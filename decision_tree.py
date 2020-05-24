"""
This module contains code to experement on decision tree algorhytm
"""
import numpy as np
import dogreader as dr
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics

# indexed of classes to filter
filtered_classes = [1, 2, 3, 4, 5, 6]

# Proportion of train/test data split
test_split_sizes = [0.20]

# Read from data set
data_set = dr.read_data_set()
# data_set = data_set.loc[data_set['Primary status'].isin(filtered_classes)]

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

split_train_accuracy = []
split_test_accuracy = []
classifiers = []

for split in test_split_sizes:
    # Split data to train and test groups (test group size is 0.25)
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=split)

    # Train Model and Predict
    classifier = DecisionTreeClassifier().fit(x_train, y_train)
    y_result = classifier.predict(x_test)
    y_train_result = classifier.predict(x_train)
    test_accuracy = metrics.accuracy_score(y_test,y_result)
    train_accuracy = metrics.accuracy_score(y_train,y_train_result)
    print("============================")
    print("Train set Accuracy: ", train_accuracy)
    print("Test set Accuracy: ", test_accuracy)
    print("============================\n")
    classifiers.append(classifier)
    split_train_accuracy.append(train_accuracy)
    split_test_accuracy.append(test_accuracy)

best_index = split_test_accuracy.index(max(split_test_accuracy))

best_classifier = classifiers[best_index]

# Draw tree
dot_data = export_graphviz(best_classifier,
                           out_file=None,
                           feature_names=dr.independent_var_columns,
                           class_names=dr.target_classes,
                           filled=True,
                           rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('dogs_tree_image', view=True)

plt.title("Accuracy of decision tree classification for different train/test split")
plt.xlabel("Train group size (%)")
plt.ylabel("Accuracy (%)")
plt.plot(np.multiply(test_split_sizes, 100),
         np.multiply(split_test_accuracy, 100), 'b-')
plt.legend(loc='upper right')
plt.show()
