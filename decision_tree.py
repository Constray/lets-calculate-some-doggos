import dogreader as dr
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics

data_set = dr.read_data_set()

# Convert data frame to arrays
x = data_set[dr.independent_var_columns].values
y = data_set[dr.dependent_var_columns].values

# Split data to train and test groups (test group size is 0.25)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train Model and Predict
classifier = DecisionTreeClassifier().fit(x_train, y_train)
y_result = classifier.predict(x_test)
print("============================")
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_result))
print("============================\n")

# Draw tree
dot_data = export_graphviz(classifier,
                           out_file=None,
                           feature_names=dr.independent_var_columns,
                           class_names=dr.target_classes,
                           filled=True,
                           rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('dogs_tree_image', view=True)
