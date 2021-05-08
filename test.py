import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from utils import somCluster
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

startTime = time.time()

# Loading Dataset and preprocessing

x = pd.read_csv('DataSet/creditCard_X.csv')
y = pd.read_csv('DataSet/creditCard_Y.csv')
print('Initial Data: ', x.shape[0])
columns = list(x.columns)
X = []
Y = []
nFeatures = x.shape[1]
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=10)
SVM = SVC(kernel='rbf', C=10e4, gamma=10e-9,  cache_size=2000, decision_function_shape='ovo')
SVM.fit(x_train, y_train)
yPredict = SVM.predict(x)
ac_svm = accuracy_score(y, yPredict) * 100
print('SVM Accuracy', ac_svm)
print('Runtime:', time.time() - startTime)