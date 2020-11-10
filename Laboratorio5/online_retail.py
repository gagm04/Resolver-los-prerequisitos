#import necessary libraries
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib as plt

# Read in iris data set
iris = pd.read_csv("/home/sheynnie/Escritorio/online_retail_price1")
# Add column names, features as indepent variables
iris.columns = ['Quantity','Price','StockCode']

# Split data into 
# features and target
a = iris.iloc[:, 0:2]# first four columns of data frame with all rows
b = iris.iloc[:, 2:] # last column of data frame (species) with all rows


#import libraries
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import train_test_split

# Train, test split, a is X as matrix, b as y the vector
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size=0.20, random_state=0)

#from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Build Linear Support Vector Classifier
#fit method to train the algorithm on the training data passed as parameter
#clf = LinearSVC()
clf = SVC(kernel='rbf')
clf.fit(a_train, b_train.values.ravel())

# Make predictions on test set
predictions = clf.predict(a_test)

from sklearn.metrics import accuracy_score

# Assess model accuracy
result = accuracy_score(b_test, predictions, normalize=True)

#evaluating the algorithm
from sklearn.metrics import classification_report #confusion_matrix
#print(confusion_matrix(b_test, predictions))
print(classification_report(b_test, predictions))

print(result)