import dogreader as dr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

filtered_classes = [1,2]

data_set = dr.read_data_set()
data_set = data_set.loc[data_set['Primary status'].isin(filtered_classes)]

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

# Split data to train and test groups (test group size is 0.25)
x_train, x_test, y_train, y_test = train_test_split(x, y)

reduced_class_names = ['guide', 'unsuitable']

# Train Model and Predict
classifier = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
classifier.fit(x_train, y_train)
y_result = classifier.predict(x_test)
print("============================")
print("Train set Accuracy: ", metrics.accuracy_score(y_train, classifier.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_result))
print("============================\n")

