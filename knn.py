"""
This module contains code to experiment on KNN algorithm
"""
import numpy as np
import dogreader as dr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, preprocessing

# indexed of classes to filter
filtered_classes = [1, 2, 3, 4, 5, 6]

# Proportion of train/test data split
test_split_sizes = [0.20]

# Read from data set
data_set = dr.read_data_set()
data_set = data_set.loc[data_set['Primary status']
                   .isin(filtered_classes)]

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

# Scaling values
x = preprocessing.StandardScaler()\
                 .fit(x).transform(x.astype(float))

# Avarge accuracy across different K
avarge_split_train_accuracy = []
avarge_split_test_accuracy = []

for split in test_split_sizes:
    print("====================================")
    print("Accuracy for split: ", split)
    # Split data to train and test groups
    x_train, x_test, y_train, y_test = train_test_split(x,
                                       y, train_size=split)
    test_set_accuracy = []
    train_set_accuracy = []
    # Train Model and Predict for different K
    for k in range(3, 10):
        classifier = KNeighborsClassifier(n_neighbors=k)\
                     .fit(x_train, y_train)
        y_result = classifier.predict(x_test)
        print("============================")
        print("Accuracy for k: ", k)
        train_accuracy = metrics.accuracy_score(y_train,
                         classifier.predict(x_train))
        train_set_accuracy.append(train_accuracy)
        print("Train set Accuracy: ", train_accuracy)
        test_accuracy = metrics.accuracy_score(y_test, y_result)
        test_set_accuracy.append(test_accuracy)
        print("Test set Accuracy: ", test_accuracy)
        print("============================\n")
    # Add to avarge accuraty for split
    avarge_split_test_accuracy.append(np.max(test_set_accuracy))
    avarge_split_train_accuracy.append(np.max(train_set_accuracy))
    # Create plot for different K
    plt.title("Accuracy of classification for different K, Split: "
              + split.__str__())
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy (%)")
    plt.plot(range(3, 10), np.multiply(test_set_accuracy, 100),
             'r--', label='Test set')
    plt.plot(range(3, 10), np.multiply(train_set_accuracy, 100),
             'b-', label='Train set')
    plt.legend(loc='upper right')
    plt.show()
    print("====================================")
# Create plot for different splits
plt.title("Accuracy of KNN classification for different train/test split")
plt.xlabel("Train group size (%)")
plt.ylabel("Accuracy (%)")
plt.plot(np.multiply(test_split_sizes, 100),
         np.multiply(avarge_split_test_accuracy, 100),
         'r--', label='Test set')
plt.plot(np.multiply(test_split_sizes, 100),
         np.multiply(avarge_split_train_accuracy, 100),
         'b-', label='Train set')
plt.legend(loc='upper right')
plt.show()