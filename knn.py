import numpy as np
import dogreader as dr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Read from data set
data_set = dr.read_data_set()

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

# Split data to train and test groups (test group size is 0.25)
x_train, x_test, y_train, y_test = train_test_split(x, y)

test_set_accuracy = []
train_set_accuracy = []
# Train Model and Predict for different K
for k in range(3, 7):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    y_result = classifier.predict(x_test)
    print("============================")
    print("Accuracy for k: ", k)
    train_accuracy = metrics.accuracy_score(y_train, classifier.predict(x_train))
    train_set_accuracy.append(train_accuracy)
    print("Train set Accuracy: ", train_accuracy)
    test_accuracy = metrics.accuracy_score(y_test, y_result)
    test_set_accuracy.append(test_accuracy)
    print("Test set Accuracy: ", test_accuracy)
    print("============================\n")

plt.title("Accuracy of classification for different K")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy (%)")
plt.plot(range(3, 7), np.multiply(test_set_accuracy, 100), 'r--', label='Test set')
plt.plot(range(3, 7), np.multiply(train_set_accuracy, 100), 'b-', label='Train set')
plt.legend(loc='upper right')
plt.show()