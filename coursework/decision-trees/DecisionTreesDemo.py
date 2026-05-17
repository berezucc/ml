import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the wine.csv data
data = pd.read_csv(r"C:\Users\elodi\Desktop\wine.csv", header = None)

# Split the data for training and for testing
train, test = train_test_split(data, test_size = 0.3)

# Extract the target class
yTrain = train.iloc[:,0]
yTest = test.iloc[:,0]

# Extract the data attributes
xTrain = train.iloc[:,1:]
xTest = test.iloc[:,1:]

# Create decision tree classifier
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 3)

# Train the classifier
clf.fit(xTrain, yTrain)

# Apply the decision tree to classify the test data
yPred = clf.predict(xTest)

# Computer the accuracy of the classifier on the test data
print("The accuracy of the classifier is", accuracy_score(yTest, yPred))

