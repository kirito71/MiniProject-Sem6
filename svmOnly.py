import time

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Loading Dataset and preprocessing

x = pd.read_csv('DataSet/indianPines_X.csv')
y = pd.read_csv('DataSet/indianPines_Y.csv')
startTime = time.time()

# Dimensionality Reduction

if x.shape[1] > 40:
    x = KernelPCA(n_components=30, eigen_solver='arpack').fit_transform(x)
    print('Reduction Done')


sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=10)
print('Initial Training Data: ', x_train.shape[0])
nClasses = len(y.value_counts())

# nFeatures = x.shape[1]
# columns = [i for i in range(nFeatures)]


# Training SVM
SVM = SVC(kernel='rbf', gamma='scale', cache_size=2000, decision_function_shape='ovr')
SVM.fit(x_train, np.ravel(y_train))
yPredict = SVM.predict(x_test)
ac_svm = accuracy_score(y_test, yPredict) * 100
print('SVM Accuracy', ac_svm)
print('Runtime:', time.time() - startTime)
