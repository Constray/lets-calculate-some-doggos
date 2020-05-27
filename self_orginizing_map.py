"""
This module contains code to experiment on SOM
"""
import numpy as np
import dogreader as dr
from somtf import SOM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

# Indices of classes to filter
# filtered_classes = [1, 2, 3, 4, 5, 6]
filtered_classes = [1, 2]

# Sizes of SOM which are checked
sizes = [[1, filtered_classes.__len__()], [5,5], [10,5], [10,10]]

# Proportion of train/test data split
test_split = 0.20

# Read from data set
data_set = dr.read_data_set()
data_set = data_set.loc[data_set['Primary status'].isin(filtered_classes)]

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=test_split)

sc = MinMaxScaler(feature_range = (0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
flat_y_test = [item for sublist in y_test for item in sublist]
str_sizes = list(map(lambda size: str(size[0]) + 'x' + str(size[1]), sizes))

accuracies = []
for size in sizes:
    som = SOM(x=size[0], y=size[1], input_dim=49, learning_rate=0.5, num_iter=150, radius=1.0)
    som.train(x_train, y_train)
    y_result = som.predict(x_test)
    print("============================")
    test_accuracy = metrics.accuracy_score(flat_y_test, y_result)
    accuracies.append(test_accuracy)
    print("Test set Accuracy: ", test_accuracy)
    print("============================\n")
# Create plot for different sizes
plt.title("Accuracy of SOM classification with different map size")
plt.xlabel("Size of map")
plt.ylabel("Accuracy (%)")
plt.plot(str_sizes, np.multiply(accuracies, 100),
         'r-', label='Test set')
plt.legend(loc='upper right')
plt.show()
print("====================================")
