import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from utils import somCluster

startTime = time.time()

# Loading Dataset and preprocessing

x = pd.read_csv('DataSet/letter-recognition.csv')
# y = pd.read_csv('DataSet/indianPines_Y.csv')

y = np.array(x[x.columns[0]])
y = pd.DataFrame(data=y, columns=['class'])


def convert(x1):
    return ord(x1) - ord('A')


y['class'] = y['class'].apply(convert)

x = x[x.columns[1:]]

x.to_csv('DataSet/letter-recognition_X.csv', index=False)
y.to_csv('DataSet/letter-recognition_Y.csv', index=False)
