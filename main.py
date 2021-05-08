import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from utils import somCluster

startTime = time.time()

# Loading Dataset and preprocessing

x = pd.read_csv('DataSet/creditCard_X.csv')
y = pd.read_csv('DataSet/creditCard_Y.csv')
print('Initial Data: ', x.shape[0])
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)
nClasses = len(y.value_counts())

# Dimensionality Reduction
# x = KernelPCA(n_components=30, eigen_solver='arpack').fit_transform(x)
# x = TruncatedSVD(n_components=15, algorithm='arpack').fit_transform(x)
print('Reduction Done')
nFeatures = x.shape[1]
columns = [i for i in range(nFeatures)]

# Splitting the data set and SOM

df = []
for i in range(nClasses):
    df.append(x[y['class'] == i])

X = []
Y = []
for i in range(nClasses):
    if i == 0:
        X = somCluster(df[i], nFeatures)
        Y = [i] * len(X)
    else:
        tp = somCluster(df[i], nFeatures)
        X = np.concatenate((X, tp), axis=0)
        Y = np.concatenate((Y, [i] * len(tp)), axis=0)

# Dataset without Outliers
X = pd.DataFrame(X, columns=columns)
Y = pd.DataFrame(Y, columns=['class'])
# Shape after Removing Outliers
print('After Outlier Reduction: ', X.shape[0])

# K-MEANS HOMOGENEOUS ##########################################################
df = pd.concat([X, Y], axis=1, join='inner')
df = df.sample(frac=1).reset_index(drop=True)
queue = [df]
final = []
while len(queue) > 0:
    cluster = queue.pop(0)
    if len(cluster['class'].value_counts().keys()) == 1:  # If homogeneous
        final.append(list(cluster.mean()))
    else:
        classCentroid = []
        classes = list(cluster['class'].value_counts().keys())
        for i in classes:
            tp = cluster[cluster['class'] == i]
            centroid = list(tp.drop('class', axis=1).mean())
            classCentroid.append(centroid)
        kMeans = KMeans(n_clusters=len(classes), init=np.array(classCentroid), n_init=1, max_iter=500)
        label = kMeans.fit_predict(cluster.drop('class', axis=1))
        for i in range(len(np.unique(label))):
            queue.append(cluster[label == i])


# Final Reduced Dataset
print('Final Training Dataset length: ', len(final))
final = np.array(final)
# print(final)
x_train = final[:, :-1]
y_train = final[:, -1]

# Training SVM
SVM = SVC(kernel='rbf', gamma='scale',  cache_size=2000, decision_function_shape='ovr')
SVM.fit(x_train, y_train)
yPredict = SVM.predict(x)
ac_svm = accuracy_score(y, yPredict) * 100
print('SVM Accuracy', ac_svm)
print('Runtime:', time.time() - startTime)
