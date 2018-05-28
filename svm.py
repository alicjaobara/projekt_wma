#!/usr/bin/env python3

# Importing the libraries
# math
import numpy as np
# plot
# import matplotlib.pyplot as plt
# datasets managing
import pandas as pd

# Importing the dataset (HOG)
datasetHOG = pd.read_csv('/home/alicja/gnu/projekt_wma/data/hogFeatures/hog_A.txt')
X = datasetHOG.iloc[:, -1].values
"""
y = datasetHOG.iloc[:, ].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""