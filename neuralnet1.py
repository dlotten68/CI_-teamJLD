import sklearn
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
print(os.getcwd())
X = pd.read_csv('aalborg.csv', index_col= None, skipfooter = 1, engine = 'python') # laatste kolomnaam blanco (edge18?) --> onbekend
#print(X)
X.describe()

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
hoi = scaledX.iloc[4].values.reshape(1,-1)
prediction = model.predict(hoi)
prediction = pd.DataFrame(prediction)
prediction.columns = ['ACCELERATION','BRAKE','STEERING']
prediction*sdy+meany

NN1 = pickle.dumps(model)
joblib.dump(NN1, 'NN1.pkl')
