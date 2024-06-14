import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as sp
import csv
from keras.models import load_model
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
import os

rf = pd.read_csv('Models/models_RF/combined_RF.csv')
dnn = pd.read_csv('Models/models_DNN/combined_DNN.csv')
xgb = pd.read_csv('Models/models_XGB/combined_XGB.csv')

hybrid = rf
hybrid['DNN'] = dnn['DNN']
hybrid['XGB'] = xgb['XGB']

train_labels = hybrid.pop('exp')
x_train = hybrid.to_numpy()
y_train = train_labels.to_numpy()

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=8000, decay_rate=0.3, staircase=False)

MAEt = []      #for the first metric (MAE)
MSEt = []     #for the first metric (RMSE)
St = []        #for the first metric (Spearman)
Pt = []        #for the first metric (Pearson)
folds = 10     #define the number of kfolds

for i in range(1):
    i+=1
    
    if i==1: 
        kf = KFold (folds, shuffle = False)     #kfold set up 
    else:
        kf = KFold (folds, shuffle = True, random_state=i)     #kfold set up

    fold = 0
    
    for train, test in kf.split(x_train, y_train):     #train
        fold+=1
        print(f"Fold #{i}_{fold}")


        DNN_model = tf.keras.Sequential()
        DNN_model.add(tf.keras.layers.Dense(100, input_shape=(3,), activation='relu'))
        DNN_model.add(tf.keras.layers.Dense(50, activation='relu'))
        DNN_model.add(tf.keras.layers.Dense(10, activation='relu'))
        DNN_model.add(tf.keras.layers.Dense(1, activation='linear', ))

        DNN_model.compile(optimizer=tf.optimizers.Adam(lr_schedule), loss='mean_absolute_error', metrics=['mse'])

        history = DNN_model.fit(x_train[train], y_train[train], epochs=100, batch_size=20, verbose=1, validation_data=(x_train[test], y_train[test]))
        
        pred = DNN_model.predict(x_train[test])
        pred = np.squeeze(np.array(pred))

        filename  = 'Models/models_hybrid/model_' + str(i) + '_' + str(fold) + '.h5'
        DNN_model.save(filename )
        print('Saved: %s' % filename )
        
        pd.concat([pd.DataFrame(y_train[test]), pd.DataFrame(pred)], axis=1, ignore_index=True).to_csv("%s.csv" %filename, index=None, header=None)


        scores = DNN_model.evaluate(x_train[test], y_train[test], verbose=0)

        print("%s: %.2f" % (DNN_model.metrics_names[0], scores[0]))
        MAEt.append(scores[0])
        
        print("%s: %.2f" % (DNN_model.metrics_names[1], scores[1]))
        MSEt.append(scores[1])
        
        corr1, _ = spearmanr(y_train[test], pred)
        print('Spearmans correlation: %.3f' % corr1)
        St.append(corr1)

        corr2, _ = pearsonr(y_train[test], pred)
        print('Pearsons correlation: %.3f' % corr2)
        Pt.append(corr2)
        print('----------------------------------------------------')

    
print("Average_MAE = %.2f (+/- %.2f)" % (np.mean( MAEt), np.std( MAEt)))
print("Average_MSE = %.2f (+/- %.2f)" % (np.mean(MSEt), np.std(MSEt)))
print("Average_Spearmans = %.2f (+/- %.2f)" % (np.mean(St), np.std(St))) 
print("Average_Pearsons = %.2f (+/- %.2f)" % (np.mean(Pt), np.std(Pt)))

def combine_csv_files(directory, output_file):
    df_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, header=None)
            df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = ['exp', 'Hybrid']
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV created at {output_file}")

directory = 'Models/models_Hybrid'  
output_file = 'Models/models_Hybrid/combined_Hybrid.csv'  
combine_csv_files(directory, output_file)