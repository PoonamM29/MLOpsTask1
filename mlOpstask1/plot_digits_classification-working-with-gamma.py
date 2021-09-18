
"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.




def digitsClassifier(data,gamma=0.001):
    #print("\n\ndata shape:", data.shape)
    #print("train-test split is:", 1-test_size,":",test_size, "\n\n")

    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    # split data into 70% train and 30% (test + val) subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    
    # split test into test(15%) and val(15%)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    accuracy_on_test= round(accuracy_score(y_test, predicted), 4)  
    f1_on_test = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)

    # Predict the value of the digit on the train subset
    predicted = clf.predict(X_train)
    accuracy_on_train= round(accuracy_score(y_train, predicted), 4)  
    f1_on_train = round(f1_score(y_train, predicted, average='macro', zero_division=0), 4)

    # Predict the value of the digit on the val subset
    predicted = clf.predict(X_val)
    accuracy_on_val= round(accuracy_score(y_val, predicted), 4)  
    f1_on_val = round(f1_score(y_val, predicted, average='macro', zero_division=0), 4)
    

    return [[accuracy_on_train,f1_on_train],[accuracy_on_test,f1_on_test],[accuracy_on_val,f1_on_val]]
    

# checking for different gamma values

data_org = digits.images
best_gamma=0
max_accuracy=0
for gamma in [0.5,0.01,1,0.001,0.0001,0.000005]:
    ans = []
    print(f"for gamma = {gamma}")
    ans.append(digitsClassifier(data_org, gamma=gamma))
    for a in ans:
        print("Asccuracy score on train set ",end=" ")
        print(a[0][0])
        print("Asccuracy score on test set ",end=" ")
        print(a[1][0])
        print("Asccuracy score on val set ",end=" ")
        print(a[2][0])
        print()

        if a[2][0]>max_accuracy:
          best_gamma=gamma
          max_accuracy=a[2][0]

print("Maximum accuracy is found at gamma value= ",best_gamma)
