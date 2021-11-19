from joblib import dump, load
from sklearn import datasets
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
targets = digits.target
def test_digit_correct_0():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i] != 0:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==0
    assert predicteddepth==0

def test_digit_correct_1():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=1:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==1
    assert predicteddepth==1

def test_digit_correct_2():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=2:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==2
    assert predicteddepth==2


def test_digit_correct_3():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=3:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==3
    assert predicteddepth==3



def test_digit_correct_4():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=4:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==4
    assert predicteddepth==4



def test_digit_correct_5():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=5:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==5
    assert predicteddepth==5


def test_digit_correct_6():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=6:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==6
    assert predicteddepth==6


def test_digit_correct_7():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=7:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==7
    assert predicteddepth==7


def test_digit_correct_8():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=8:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==8
    assert predicteddepth==8



def test_digit_correct_9():
    best_model_path='./bestmodelforgamma/model.joblib'
    svmmodel=load(best_model_path)
    best_model_path='./bestmodelfordepth/model.joblib'
    depthmodel=load(best_model_path)
    i=0
    while targets[i]!=9:
        i+=1
    
    image=data[i].reshape(1,-1)
    predictedsvm=svmmodel.predict(image)
    predicteddepth=depthmodel.predict(image)

    assert predictedsvm==9
    assert predicteddepth==9

