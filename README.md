# Id3trees

This is a python package that provides the ID3 Decision tree for
classifying and predicting data.


After installing the package in order to use it in the code, Please do
the following.

```python

from Id3trees import tree

# train_data : A ``pandas`` DataFrame training data on which the ID3 tree
# classification needs to be done
# train_data_class : A  ``pandas`` DataFrame consisting of a 1D series of 
# training class data
tree = tree.MyDecisionTreeClassifier()
tree.fit(train_data, train_data_class)

# test_data : A ``pandas`` DataFrame containing the testing data
# returns : An array of predicted class for the given
tree.predict(test_data)

# print the tree after fitting the value
tree.print()

```