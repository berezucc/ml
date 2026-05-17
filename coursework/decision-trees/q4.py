# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Define the training data from Table 1
# x1 and x2 are the features, and y is the target class.
X = [[0, 0], [0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 2], [0, 2], [1, 2]]
y = ['T', 'T', 'T', 'T', 'F', 'F', 'T', 'T', 'F', 'F']

# Create the DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the decision tree classifier
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=['x1', 'x2'], class_names=['F', 'T'], filled=True)
plt.title('Decision Tree')
plt.show()
