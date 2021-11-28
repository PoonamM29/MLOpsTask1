import os
import pandas as pd
import matplotlib.pyplot as plt
from util import test, preprocess,createsplitwithsuffle,run_classification_experiment,mytrain

from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

print("=============================\nClassifying Handwritten Digits")
print("=============================")


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_split = [10,20,30,40,50,60,70,80,90,100]

s_cols = ['Train Data (%)', 'gamma', 'SVMTestAcc', 'SVMValAcc', 'SVMF1Score']
svm_output = pd.DataFrame(data = [], columns=s_cols)
gammas = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]

d_cols = ['Train Data (%)', 'MaxDepth', 'DecTestAcc',  'DecValAcc', 'DecF1Score']
dt_output = pd.DataFrame(data = [], columns=d_cols)
depths = [6,8,10,12,14,16]


def train_dec(x_train, y_train, x_val, y_val, x_test, y_test, depth, cmd=False, td=None):
    dec = DecisionTreeClassifier(max_depth=depth)
    t_ac,val_ac,predicted,f1=mytrain(dec,x_train,y_train,x_test, y_test,x_val, y_val)
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'DT Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
    return t_ac, val_ac, f1

def train_svm(x_train, y_train, x_val, y_val, x_test, y_test, gamma, cmd=False, td = None):
    clf = svm.SVC(gamma=gamma)
    t_ac,val_ac,predicted,f1=mytrain(clf,x_train,y_train,x_test, y_test,x_val, y_val)
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'SVM Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
    return t_ac, val_ac, f1
test_size=0.1
valid_size=0.1
resized_images = preprocess(digits.images,1)
resized_images = np.array(resized_images)
data = resized_images.reshape((n_samples, -1))
x_train, x_test,x_val,y_train,y_test,y_val = createsplitwithsuffle(data, digits.target, test_size, valid_size)
for gamma in gammas:
  for tr in train_split:
    sp = int(tr/100 * len(x_train))
    n_train = x_train[:sp]
    n_ytrain = y_train[:sp]
    if gamma == 0.001:
      st_ac, sval_ac, sf1 = train_svm(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma, True, tr)
    else:
      st_ac, sval_ac, sf1 = train_svm(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma)
    out = pd.DataFrame(data = [[tr, gamma, st_ac, sval_ac, sf1]],columns = s_cols)
    svm_output = svm_output.append(out, ignore_index=True)
    
for depth in depths:
  for tr in train_split:
    sp = int(tr/100 * len(x_train))
    n_train = x_train[:sp]
    n_ytrain = y_train[:sp]
    if depth == 12:
      t_ac, val_ac, f1 = train_dec(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth, True, tr)
    else:
      t_ac, val_ac, f1 = train_dec(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth)
    out = pd.DataFrame(data = [[tr, depth, t_ac, val_ac, f1]],
    columns = d_cols)
    dt_output = dt_output.append(out, ignore_index=True)

print("SVM Training Output- ")
print(svm_output)
svm_output.to_csv("ML_Ops\svm_output.csv")

print("Decision Tree Training Output- ")
print(dt_output)
dt_output.to_csv("ML_Ops\dt_output.csv")


