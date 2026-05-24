from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

############## FOR EVERYONE ##############
# Please note that the blanks are here to guide you for this first assignment, but the blanks are  
# in no way representative of the number of commands/ parameters or length of what should be inputted.

### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
# Write code that fetches the breast cancer wisconsin dataset. 
# Hint: https://scikit-learn.org/stable/datasets/toy_dataset.html
# Hint: Make sure the data features and associated target class are returned instead of a "Bunch object".
X, y = datasets.load_breast_cancer(return_X_y=True) #(4 points) 

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are", len(X), "instances described by", len(X[0]), "features.") #(4 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addtion, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y, random_state = 42)  #(4 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split = 6)  #(4 points) 
clf.fit(X_train, y_train)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test)  #(4 points) 

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is', accuracy_score(y_test, predC))  #(1 point) 

# Visualize the tree created
# _ = tree.plot_tree(clf,filled=True, fontsize=12)  # set the font size the 18 (5 points) 

### PART 2.1 ###
# Visualize the training and test accuracies as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
trainAccuracy = []  #(1 point) 
testAccuracy = [] #(1 point) 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = range(1,16) #(1 point) 
for depth in depthOptions: #(1 point) 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split = 6, max_depth=depth) #(1 point) 
    # Decision tree training
    cltree = cltree.fit(X_train, y_train) #(1 point) 
    # Label predictions on training set 
    y_predTrain = cltree.predict(X_train) #(1 point) 
    # Label predictions on test set 
    y_predTest = cltree.predict(X_test) #(1 point) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) #(1 point) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) #(1 point) 

# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
plt.plot(depthOptions,trainAccuracy,'rv-',depthOptions,testAccuracy,'bo--') #(3 points) 
plt.legend(['Training Accuracy','Test Accuracy']) # add a legend for the training accuracy and test accuracy (1 point) 
plt.xlabel('Tree Depth') # name the horizontal axis 'Tree Depth' (1 point) 
plt.ylabel('Classifier Accuracy') # name the horizontal axis 'Classifier Accuracy' (1 point) 

# Fill out the following blanks: #(4 points (2 points per blank)) 
""" 
According to the test error, the best model to select is when the maximum depth is equal to ____, approximately. 
But, we should not use select the hyperparamters of our model using the test data, because _____.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Define the parameters to be optimized: the max depth of the tree and the minimum number of samples to split a node
parameters = {'max_depth':range(1,15), 'min_samples_split': range(2,20,2)} #(6 points)
# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters) #(6 points)
clf.fit(X=X_train, y=y_train) #(4 points)
tree_model = clf.best_estimator_ #(4 points)
print("The maximum depth of the tree is", clf.best_params_['max_depth'], 
      'and the minimum number of samples required to split a node is', clf.best_params_['min_samples_split']) #(6 points)

# The best model is tree_model. Visualize that decision tree (tree_model).
_ = tree.plot_tree(tree_model,filled=True, fontsize=12) #(4 points)

# # Fill out the following blank: #(2 points)
# """ 
# This method for tuning the hyperparameters of our model is acceptable, because ________. 
# """

# # Explain below was is tenfold Stratified cross-validation?  #(4 points)
# """Dividing the dataset into ten equal parts, using 9 parts for training and 1 part for testing, 
# and then measure the accuracy of the resulting model which is based on 9/10 of the data and 
# tested on 1/10 of the data. Repeating this step ten times so that each instance is used once for testing. 
# Finally averaging or counting the overall accuracy.
# """

# ### PART 3 ###
import pandas as pd
import math

# Read the data
data = pd.read_csv(r"C:\Users\elodi\Dropbox\TMU\Teaching\CPS803-8318F24\Assignment1\A1F2024.csv")

# Extract the target class
y = data.iloc[:,2]
x = data.iloc[:,:2]
# Extract the data attributes
x1 = data.iloc[:,1].values.reshape(-1, 1)
x2 = data.iloc[:,2].values.reshape(-1, 1)

# Create decision tree classifier
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)

# Train the classifier using the training attributes and class
# Choose dataTrain or colorTrain or sizeTrain or actTrain or ageTrain based on what we are interested to compute
clf = clf.fit(x, y)
tree.plot_tree(clf,filled=True, fontsize=12)

# Question 1
p1= 6/10
p2 = 4/10
entropyB = -p1*math.log(p1,2) - p2*math.log(p2,2)

# Question 3
# Option x1
p11= 3/5
p12 = 2/5
p21= 4/5
p22 = 1/5
entropyA1 = -p11*math.log(p11,2) - p12*math.log(p12,2)
entropyA2 = -p21*math.log(p21,2) - p22*math.log(p22,2)
gainX1 = entropyB - 5/10*entropyA1 - 5/10*entropyA2

# Option x2, 0 & 1,2
p11= 0/4
p12 = 4/4
p21= 4/6
p22 = 2/6
entropyA1 = -0 - p12*math.log(p12,2)
entropyA2 = -p21*math.log(p21,2) - p22*math.log(p22,2)
gainX2_0_12 = entropyB - 4/10*entropyA1 - 6/10*entropyA2
entropyB2 = entropyA2 # Saving this for next question

# Option x2, 0,1 & 2
p11= 5/7
p12 = 2/7
p21= 1/3
p22 = 2/3
entropyA1 = -p11*math.log(p11,2) - p12*math.log(p12,2)
entropyA2 = -p21*math.log(p21,2) - p22*math.log(p22,2)
gainX2_01_2 = entropyB - 7/10*entropyA1 - 3/10*entropyA2

# Question 4
# Second split: Option x1:
p11= 3/3
p12 = 0/3
p21= 2/3
p22 = 1/3
entropyA1 = 0
entropyA2 = -p21*math.log(p21,2) - p22*math.log(p22,2)
gain2X1 = entropyB2 - 3/6*entropyA1 - 3/6*entropyA2
# print(gain2X1)

# Second split: Option x2:
p11= 2/3
p12 = 1/3
p21= 2/3
p22 = 1/3
entropyA1 = -p11*math.log(p11,2) - p12*math.log(p12,2)
entropyA2 = -p21*math.log(p21,2) - p22*math.log(p22,2)
gain2X2 = entropyB2 - 3/6*entropyA1 - 3/6*entropyA2
# print(gain2X2)