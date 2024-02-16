#!/usr/bin/env python

"""
Code to produce the the random forest regressor 99th percentile canopy height model for Landsat 8 sensors. This code uses the module klepto to serialise the model.
The Landsat 8 imagery used to calibrate the model was corrected to landsat like reflectance so the model can be applied to Landsat sensors 5 and 7 
It returns a number of stats on the model fit and band importance score.

Author: Grant Staben
Date: 30/01/2019
modified: 07/04/2019
"""



# import the required modules
from __future__ import print_function, division
import pandas as pd
import numpy as np
import pdb
from sklearn.ensemble import RandomForestRegressor as rfr
import klepto

def rfrmodel():

    """
    function to read in csv file, fit the random forest regressor model,out put the importance score, model fit statistic and rfr model.
    """
    df = pd.read_csv('C:/Users/grants/code/old_growth/sen2_l8/l8_combined/l8_model_dev/combined_training_data_l8_int_version_n148000.csv', header=0)
    df2 = df.dropna()
    # read in the selected predictor variables
        
    xdata1=df2[['CVId', 'NDVId', 'ratio52a', 'ratio63d', 'ratio63a', 'ratio54d',
       'psB5a', 'psB6d', 'GNDVId', 'NDIIa', 'ratio43a', 'ratio52d',
       'ratio53d', 'psB2a', 'ratio53a', 'CVIa', 'ratio62d', 'psB3d']].astype('int32') 
    X_1 = xdata1
    print (xdata1.dtypes)
    # read in the mean chm heights derived from lidar and convert it into a format read by sklearn random forest model
    ydata1 = df2[['perc99_1']].astype('float32')
    print (ydata1.dtypes)
    ydata2 = ydata1.values
    y_1 = ydata2.ravel()

    print (y_1.shape)
    print (X_1.shape)
    
    #pdb.set_trace()
    
     # set the paramaters for the random forest regressor model
    rfrModel_1 = rfr(n_estimators=512, oob_score=True,  max_features = 'log2', min_samples_split=1,n_jobs=-1) 

     # fit the model to the training data 
    rfrLCHM = rfrModel_1.fit(X_1, y_1)
     
    # calcualte the feature importance scores
    feature_importance = rfrModel_1.feature_importances_    
    fi = enumerate(rfrModel_1.feature_importances_)
    cols = xdata1.columns 
    fiResult2 = [(value,cols[i]) for (i,value) in fi]
    
    fiResult = np.array(fiResult2)
    score = fiResult[:,0]
    band = fiResult[:,1]

    # calculate stats for the fit model
    fmr3 = format(rfrLCHM.score(X_1, y_1), '.2f')
    print ('Fitted model r2 =' ,  format(rfrLCHM.score(X_1, y_1), '.2f'))
    fmrMSE3 = format(np.mean((y_1 - rfrLCHM.predict(X_1))**2), '.2f')
    print ('Fitted model mse =', format(np.mean((y_1 - rfrLCHM.predict(X_1))**2), '.2f'))
    print ('n =', len(y_1))
     
    fmr4 = float(fmr3)
    fmMSE4 = float(fmrMSE3)
    
     
    return (score,band, fmr4,fmMSE4,rfrLCHM)

def importance_score(score,band):
    """
   function to sort and convert importance score data into a pandas df
   """
    # concatenate lists and transpose 
    fit_score = np.vstack([score, band]).T
    
    # create an empty pandas dataframe to append the stats in each csv files.
    score = pd.DataFrame(fit_score, columns=['score','band'])
    # sort importance score in decending order base on score value
    score = score.sort_values(by = 'score',ascending=False)
    
    return (score)


def mainRoutine():

    """
    run the main routine outputs the model fit stats and band importance scores
    and a seralises the model using the klepto library - I have prevosly used pickle however I have run into memory issues 
    when I moved to python 3 
    
    """
    score,band, fmr4, fmMSE4,rfrLCHM = rfrmodel()
    
    df = importance_score(score,band)
    
    df.to_csv('D:/A_L8_model_dev/final_l8_models/99/ImportanceScore_model_l8_99perc_20190407_512_n18_int32v.csv')
    
    stats = np.vstack([fmr4,fmMSE4]).T
    
    df2 = pd.DataFrame(stats, columns=['r2','mse'])
    
    df2.to_csv('D:/A_L8_model_dev/final_l8_models/99/Model_stats_model_l8_99perc_20190407_512_n18_int32v.csv')
    
    d = klepto.archives.file_archive('D:/A_L8_model_dev/final_l8_models/99/rfrLCHMcpickle_l8_99perc_20190407_512_n18_int32v', serialized=True)
    d['model'] = rfrLCHM
    
    d.dump()
    d.clear()
    

if __name__ == "__main__":
    mainRoutine()