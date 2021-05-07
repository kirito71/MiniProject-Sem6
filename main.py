import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from utils import somCluster
from sklearn.cluster import KMeans

startTime = time.time()

# Loading Dataset and preprocessing

x = pd.read_csv('DataSet/indianPines_X.csv')
y = pd.read_csv('DataSet/indianPines_Y.csv')
print('Initial Data: ', x.shape[0])
columns = list(x.columns)
X = []
Y = []
nFeatures = x.shape[1]
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)
nClasses = len(y.value_counts())

# Splitting the data set and SOM

df = []
for i in range(nClasses):
    df.append(x[y['class'] == i])

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
# p = list(df['class'].value_counts().keys())
# print(p)
while len(queue) > 0:
    cluster = queue.pop(0)
    if len(cluster['class'].value_counts()) == 1:  # If homogeneous
        final.append(list(cluster.mean()))
    else:
        classCentroid = []
        classes = list(cluster['class'].value_counts().keys())
        for i in classes:
            centroid = list(cluster[cluster['class'] == i].mean())
            classCentroid.append(centroid)
        kMeans = KMeans(n_clusters=len(classes), init=np.array(classCentroid), n_init=1, max_iter=500)
        label = kMeans.fit_predict(cluster)
        for i in range(len(classes)):
            queue.append(cluster[label == i])

# Final Reduced Dataset
print('Final Training Dataset length: ', len(final))


# Training SVM
print('Runtime:', time.time() - startTime)
