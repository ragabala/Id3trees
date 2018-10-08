"""
This is an python package that implements the ID3 decision tree.

This package supports multi-way split and uses entropy information
for splitting.

This package also supports in-build tree ``tree_print``.
"""

import numpy
import pandas


class DecisionTree:
    """DataStructure for storing out the decision tree."""
    def __init__(self, attrib_name, tabs):
        self.attrib_name = attrib_name
        # Children would be edge_value : Decision Tree
        self.children = {}
        self.class_ = None
        self.tabs = tabs

    def __str__(self):
        return '['+self.attrib_name+'] --> '+DecisionTree.tree_print(self.children, self.tabs)

    __repr__ = __str__

    @staticmethod
    def tree_print(_dict, tabs):
        """Pretty print a tree with tabs and new lines."""
        _tabs = '\n' + ' ' * tabs
        return _tabs + (_tabs.join('{}: {}'.format(k, v) for k, v in _dict.items()) + '}')


class Leaf:
    """Leaf Nodes for the tree that has class set"""

    def __init__(self, class_):
        """ Create an object for Leaf class."""
        self.class_ = class_

    def __str__(self):
        """ String repr for Leaf."""
        return str(self.class_)

    __repr__ = __str__


class MyDecisionTreeClassifier:
    """Classifier class that classifies and predicts data."""
    def __init__(self):
        self.data = None
        self.class_data = None
        self.root = None

    def fit(self, data, class_):
        """Fit the training data to form the decision tree."""
        self.data = pandas.DataFrame(data)
        self.data['Class'] = class_
        self.root = self.build_tree(data, 0)
        return self

    def predict(self, df_test_data):
        """Predict the class in the test data with the built tree"""
        predict = []
        for _, row in df_test_data.iterrows():
            temp = self.root
            columns = row.axes[0].tolist()
            columns.remove('Class')

            while temp.class_ is None:
                for _attrib in columns:
                    # we parse through the sub tree
                    attrib_val = row[_attrib]
                    if _attrib == temp.attrib_name:
                        for key in temp.children:
                            # edge for the node
                            if key == attrib_val:
                                temp = temp.children[key]
                                break
                        if temp.class_ is not None:
                            break
            predict.append(temp.class_)
        return predict

    def build_tree(self, rows, level):
        """Build a decision tree with the pandas rows."""

        class_ = MyDecisionTreeClassifier.is_all_same_class(rows)
        if rows.shape[0] == 0 or class_ is not None:
            return Leaf(class_)

        _best_attrib = MyDecisionTreeClassifier.best_attrib(rows)
        decision_tree = DecisionTree(_best_attrib, len(_best_attrib) + level + 5)
        # for all the labels of the best_attrib, we should create
        # childrens
        for _label in rows[_best_attrib].unique().tolist():
            sub_rows = rows[rows[_best_attrib] == _label]
            sub_rows = sub_rows.drop([_best_attrib], axis=1)
            decision_tree.children[_label] = self.build_tree(
                sub_rows, decision_tree.tabs + len(_label) + 3
            )
        return decision_tree

    def print_tree(self):
        """ Prints the decision tree with proper indentations."""
        if self.root is None:
            raise Exception("Tree is empty. Please fit the training data to visualize")
        print(self.root)

    @staticmethod
    def is_all_same_class(rows):
        """Checks whether all the records have the same class"""
        distinct_classes = rows['Class'].unique().tolist()
        return distinct_classes[0] if len(distinct_classes) == 1 else None

    @staticmethod
    def best_attrib(rows):
        """This finds the best attribute for tree split."""
        attribs = rows.columns.values.tolist()
        classes = rows['Class'].unique().tolist()
        # removing class
        attribs.remove('Class')
        min_entropy = 100  # some random large value for minimalizing
        min_attrib = None

        for attrib in attribs:
            total_entropy = 0

            for _label in rows[attrib].unique().tolist():
                rows_label = rows[rows[attrib] == _label]
                label_count = rows_label.shape[0]
                label_prob = label_count / rows.shape[0]
                entropy = 0
                for _class in classes:  # Yes or No
                    rows_classes = rows_label[rows_label['Class'] == _class]
                    class_count = rows_classes.shape[0]
                    prob = class_count / label_count
                    entropy -= 0 if prob == 0 else prob * numpy.log2(prob)
                total_entropy += label_prob * entropy
            if total_entropy < min_entropy:
                min_entropy = total_entropy
                min_attrib = attrib
        return min_attrib
