import numpy as np
import pandas as pd
from keras.models import load_model
from pickle import load

rf = pd.read_csv('Benchmark/Results/predictions_RF.csv')
dnn = pd.read_csv('Benchmark/Results/predictions_DNN.csv')
xgb = pd.read_csv('Benchmark/Results/predictions_XGB.csv') 

hybrid = rf
hybrid['DNN'] = dnn['DNN']
hybrid['XGB'] = xgb['XGB']

train_labels = hybrid.pop('exp')
x_train = hybrid.to_numpy()
y_train = train_labels.to_numpy()

models = list()          
for i in range(1):
    for j in range(10):
        filename = 'Models/models_Hybrid/model_' + str(i+1) + '_' + str(j+1)
        model = load_model("%s.h5" %filename)
        models.append(model)

ytrain = [model.predict(x_train) for model in models]
ytrain = np.squeeze(np.array(ytrain))

outcomes = pd.DataFrame(np.mean(ytrain, axis=0))
pred = pd.DataFrame()
pred['exp'] = train_labels
pred['Hybrid'] = outcomes
pred.to_csv('Benchmark/Results/predictions_Hybrid.csv', index=None)