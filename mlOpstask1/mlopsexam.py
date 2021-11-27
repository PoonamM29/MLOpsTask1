import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def mysplit(data, target):
  test_size=0.15
  valid_size=0.15
  X_train, X_test_valid, y_train, y_test_valid = train_test_split(data, target, test_size=test_size + valid_size, shuffle=False)
  X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,y_test_valid,test_size=valid_size / (test_size + valid_size),shuffle=False)
  return X_train, y_train, X_test, y_test, X_valid, y_valid


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

param_grid = {'C': [0.1, 10],
              'gamma': [1, 0.01],
              'kernel': ['linear', 'poly']}
for c_r in param_grid['C']:
  for gamma in param_grid['gamma']:
    for k in param_grid['kernel']:
      for i in range(3):
        X_train, y_train, X_test, y_test, X_valid, y_valid = mysplit(data, digits.target)
        clf = svm.SVC(C=c_r,gamma=gamma, kernel=k)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        valid_acc = clf.score(X_valid, y_valid)
        test_acc = clf.score(X_test, y_test)
        print("RUn",i+1)
        print("C"," gamma","  kernel"," train_acc","  valid_acc","  test_acc")
        if i==0:
          print(c_r,"   ",gamma,"   ",k,"   ",train_acc,"   ",valid_acc,"   ",test_acc)
          print()
        elif i==1:
          print(c_r,"   ",gamma,"   ",k,"   ",train_acc,"   ",valid_acc,"   ",test_acc)
          print()
        else:
          print(c_r,"   ",gamma,"   ",k,"   ",train_acc,"   ",valid_acc,"   ",test_acc)
          print()

