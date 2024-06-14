import numpy as np
import pandas as pd
import scipy as sp
import csv
import joblib
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
from pickle import load
import xgboost
from xgboost import XGBRegressor
import os


train_dataset = pd.read_csv("Training/Data/iScore_Training-set.csv",low_memory=False)         
train_labels = train_dataset.pop('PKd')

x_train_raw = train_dataset.to_numpy()
y_train = train_labels.to_numpy()

scaler = load(open('Models/scaler.pkl', 'rb'))
x_train = scaler.transform(x_train_raw)


MAEt = []      #for the first metric (MAE)
RMSEt = []     #for the first metric (RMSE)
R2t = []       #for the first metric (R2)
St = []        #for the first metric (Spearman)
Pt = []        #for the first metric (Pearson)
folds = 10     #define the number of kfolds

for i in range(1):
    i+=1
    kf = KFold (folds, shuffle=True, random_state=i)     
    fold = 0
    
    for train, test in kf.split(x_train, y_train):     
        fold+=1
        print(f"Fold #{i}_{fold}")

        XGBmodel = XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.01, gamma=1, subsample=0.7, colsample_bytree=1.0, n_jobs=-1) 
        XGBmodel.fit(x_train[train], y_train[train])

        pred = XGBmodel.predict(x_train[test])


        filename  = 'Models/models_XGB/model_' + str(i) + '_' + str(fold) + '.joblib'
        joblib.dump(XGBmodel, filename)
        print('Saved: %s' %filename )
        
        pd.concat([pd.DataFrame(y_train[test]), pd.DataFrame(pred)], axis=1, ignore_index=True).to_csv("%s.csv" %filename, index=None, header=None)


        R2_test = XGBmodel.score(x_train[test], y_train[test])
        print('R2 = %.3f' % R2_test)
        R2t.append(R2_test)

        corr1, _ = spearmanr(pred, y_train[test])
        print('Spearmans correlation: %.3f' % corr1)
        St.append(corr1)

        corr2, _ = pearsonr(pred, y_train[test])
        print('Pearsons correlation: %.3f' % corr2)
        Pt.append(corr2)

        MAE = mae(pred, y_train[test])
        print('MAE: %3f' % MAE)
        MAEt.append(MAE)

        RMSE = rmse(pred, y_train[test], squared=False)
        print('RMSE: %3f' % RMSE)
        RMSEt.append(RMSE)
        print('----------------------------------------------------')

print("Average_R2 = %.2f (+/- %.2f)" % (np.mean(R2t), np.std(R2t)))
print("Average_Spearmans = %.2f (+/- %.2f)" % (np.mean(St), np.std(St))) 
print("Average_Pearsons = %.2f (+/- %.2f)" % (np.mean(Pt), np.std(Pt)))
print("Average_MAE = %.2f (+/- %.2f)" % (np.mean(MAEt), np.std(MAEt))) 
print("Average_RMSE = %.2f (+/- %.2f)" % (np.mean(RMSEt), np.std(RMSEt))) 

def combine_csv_files(directory, output_file):
    df_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, header=None)
            df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = ['exp', 'XGB']
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV created at {output_file}")

directory = 'Models/models_XGB'  
output_file = 'Models/models_XGB/combined_XGB.csv'  
combine_csv_files(directory, output_file)