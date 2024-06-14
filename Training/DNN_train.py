import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import os


train_dataset = pd.read_csv("Training/Data/iScore_Training-set.csv",low_memory=False)            
train_labels = train_dataset.pop('PKd')    

x_train_raw = train_dataset.to_numpy()
y_train = train_labels.to_numpy()

scaler = MinMaxScaler().fit(x_train_raw)
dump(scaler, open('Models/scaler.pkl', 'wb'))
x_train = scaler.transform(x_train_raw)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=8000, decay_rate=0.3, staircase=False)


MAEt = []      #for the first metric (MAE)
MSEt = []      #for the first metric (RMSE)
St = []        #for the first metric (Spearman)
Pt = []        #for the first metric (Pearson)
folds = 10     #number of k-folds XV

for i in range(1):
    i+=1
    kf = KFold (folds, shuffle=True, random_state=i)    
    fold = 0
    
    for train, test in kf.split(x_train, y_train): 
        fold+=1
        print(f"Fold #{i}_{fold}")

        
        DNN_model = tf.keras.Sequential()
        DNN_model.add(tf.keras.layers.Dense(350, input_shape=(122,), activation='relu', kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(250, activation='relu',kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(150, activation='relu',kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(50, activation='relu',kernel_initializer='HeNormal'))
        DNN_model.add(tf.keras.layers.Dense(1, activation='linear',kernel_initializer='HeNormal'))

        DNN_model.compile(optimizer=tf.optimizers.Adam(lr_schedule), loss='mean_absolute_error', metrics=['mse'])
        history = DNN_model.fit(x_train[train], y_train[train], epochs=100, verbose=1, validation_data=(x_train[test], y_train[test]))
        
        pred = DNN_model.predict(x_train[test])
        pred = np.squeeze(np.array(pred))

        filename  = 'Models/models_DNN/model_' + str(i) + '_' + str(fold) + '.h5'
        DNN_model.save(filename)
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
    combined_df.columns = ['exp', 'DNN']
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV created at {output_file}")

directory = 'Models/models_DNN'  
output_file = 'Models/models_DNN/combined_DNN.csv'  
combine_csv_files(directory, output_file)


