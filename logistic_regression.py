"""
This module contains code to experiment on logistic regression algorithm
"""
import numpy as np
import dogreader as dr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# indexed of classes to filter
# filtered_classes = [1, 2, 3, 4, 5, 6]
filtered_classes = [1, 2]

# Proportion of train/test data split
test_split_sizes = [0.20]

# Read from data set
data_set = dr.read_data_set()
data_set = data_set.loc[data_set['Primary status']
                   .isin(filtered_classes)]

split_train_accuracy = []
split_test_accuracy = []

x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

for split in test_split_sizes:
    # Split data to train and test groups (test group size is 0.25)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                        y, train_size=split)

    reduced_class_names = ['guide', 'unsuitable']

    # Train Model and Predict
    classifier = LogisticRegression(C=1e5, solver='lbfgs',
                                    multi_class='multinomial')
    classifier.fit(x_train, y_train)
    y_result = classifier.predict(x_test)
    y_train_result = classifier.predict(x_train)
    test_accuracy = metrics.accuracy_score(y_test, y_result)
    train_accuracy = metrics.accuracy_score(y_train, y_train_result)
    print("============================")
    print("Train set Accuracy: ", train_accuracy)
    print("Test set Accuracy: ", test_accuracy)
    print("============================\n")
    split_train_accuracy.append(train_accuracy)
    split_test_accuracy.append(test_accuracy)

plt.title("Accuracy of logistic regression classification for different train/test split")
plt.xlabel("train group size (%)")
plt.ylabel("Accuracy (%)")
plt.plot(np.multiply(test_split_sizes, 100),
         np.multiply(split_test_accuracy, 100), 'r--', label='Test set')
plt.plot(np.multiply(test_split_sizes, 100),
         np.multiply(split_train_accuracy, 100), 'b-', label='Train set')
plt.legend(loc='upper right')
plt.show()


