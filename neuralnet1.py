import sklearn
import pandas as pd
import numpy as np
import os
import pickle
import random
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
print(os.getcwd())
X = pd.read_csv('train2.csv', index_col= None, engine = 'python') # laatste kolomnaam blanco (edge18?) --> onbekend
print(X)
X.describe()

y = X[['ACCELERATION','BRAKE','STEERING']]
X.drop(['ACCELERATION','BRAKE','STEERING'], axis = 1, inplace = True)

meanX = np.mean(X)
meany = np.mean(y)
sdX = np.std(X)
sdy = np.std(y)

scaledX = (X - meanX)/sdX

scaledy = (y - meany)/sdy

model = MLPRegressor(hidden_layer_sizes=(12,12))
model.fit(scaledX, scaledy)

hoi = scaledX.iloc[4].values.reshape(1,-1)
print(type(hoi))
print((hoi.shape))
prediction = model.predict(hoi)
prediction = pd.DataFrame(prediction)
prediction.columns = ['ACCELERATION','BRAKE','STEERING']
print(prediction*sdy+meany)


joblib.dump(model, 'NN1.pkl')
