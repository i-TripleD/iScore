import numpy as np
import pandas as pd
from keras.models import load_model
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
        filename = 'Models/models_DNN/model_' + str(i+1) + '_' + str(j+1)
        model = load_model("%s.h5" %filename)
        models.append(model)

ytrain = [model.predict(x_train) for model in models]
ytrain = np.squeeze(np.array(ytrain))

outcomes = pd.DataFrame(np.mean(ytrain, axis=0))
pred = pd.DataFrame()
pred['exp'] = train_labels
pred['DNN'] = outcomes
pred.to_csv('Benchmark/Results/predictions_DNN.csv', index=None)