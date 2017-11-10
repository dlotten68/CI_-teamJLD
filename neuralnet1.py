import sklearn
import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
print(os.getcwd())
X = pd.read_csv('/Users/David/aalborg.CSV', index_col= None) # laatste kolomnaam blanco (edge18?) --> onbekend

X.describe()
X = X[:-1]

y = X[['ACCELERATION','BRAKE','STEERING']]
X.drop(['ACCELERATION','BRAKE','STEERING'], axis = 1, inplace = True)

meanX = np.mean(X)
meany = np.mean(y)
sdX = np.std(X)
sdy = np.std(y)

scaledX = (X - meanX)/sdX
scaledy = (y - meany)/sdy

model = MLPRegressor(hidden_layer_sizes=(30,30,30))
model.fit(scaledX, scaledy)
prediction = model.predict(scaledX.iloc[3])
prediction = pd.DataFrame(prediction)
prediction.columns = ['ACCELERATION','BRAKE','STEERING']
prediction*sdy+meany
