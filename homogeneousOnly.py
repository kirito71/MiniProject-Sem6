import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
x, x_test, y, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=10)
print('Initial Training Data: ', x.shape[0])
nClasses = len(y.value_counts())


nFeatures = x.shape[1]
columns = [i for i in range(nFeatures)]

X = pd.DataFrame(x, columns=columns)
Y = pd.DataFrame(y, columns=['class'])

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
yPredict = SVM.predict(x_test)
ac_svm = accuracy_score(y_test, yPredict) * 100
print('SVM Accuracy', ac_svm)
print('Runtime:', time.time() - startTime)
