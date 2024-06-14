import numpy as np
import pandas as pd
import joblib
from pickle import load

dataset = pd.read_csv("Benchmark/Testset/CASF2016_testset.csv",low_memory=False) 
          
train_labels = dataset.pop('pkd')

x_train_raw = dataset.to_numpy()
y_train = train_labels.to_numpy()

scaler = load(open('Models/scaler.pkl', 'rb'))
x_train = scaler.transform(x_train_raw)


models = list()          
for i in range(1):
    for j in range(10):
        filename = 'Models/models_XGB/model_' + str(i+1) + '_' + str(j+1)
        model = joblib.load("%s.joblib" %filename)
        models.append(model)

ytrain = [model.predict(x_train) for model in models]
ytrain = np.squeeze(np.array(ytrain))

outcomes = pd.DataFrame(np.mean(ytrain, axis=0))
pred = pd.DataFrame()
pred['exp'] = train_labels
pred['XGB'] = outcomes
pred.to_csv('Benchmark/Results/predictions_XGB.csv', index=None)