#!/usr/bin/env python

"""
Code to produce the cPickle file for the random forest regressor 99th percentile canopy height model for Sentinel-2 sensors at 10m spatial resolution. 

It returns a number of stats on the model fit and band importance score and serialises the model using the klepto module.

Author: Grant Staben
Date: 26/01/2019
modified 19/02/2019
"""



# import the required modules
from __future__ import print_function, division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
import pickle
import klepto

def rfrmodel():

    """
    function to read in csv file, fit the random forest regressor model,out put the importance score, model fit statistic and rfr model.
    """
    df = pd.read_csv('C:/Users/grants/code/old_growth/sen2_l8/zonal_stats/s2_10m_combined/sent2_training_data_10m_f.csv', header=0)
    df2 = df.dropna()
    # read in the selected predictor variables
        
    xdata1=df2[['NDRE1d', 'ratio63a', 'ratio62a', 'CVIa', 'GNDVIa', 'NDRE2a','CVId', 'ratio95d', 'psB9d', 'ratio98a', 
	'ratio62d','ratio54d','ratio92a', 'psB3d', 'ratio105a', 'SAVIa', 'psB10a', 'NDIId','psB2a', 'ratio103d', 'ratio107d', 'ratio103a', 
	'ratio109d', 'NDGId']].astype('int32') 
    X_1 = xdata1
    # read in the mean chm heights derived from lidar and convert it into a format read by sklearn random forest model
    ydata1 = df2[['perc99_1']].astype('float32')
    ydata2 = ydata1.values
    y_1 = ydata2.ravel()

    print (y_1.shape)
    print (X_1.shape)

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
    and a seralises the model using the klepto library - I have prevosly used pickle however I have run into memory issures when I moved to python 3 """
    score,band, fmr4, fmMSE4,rfrLCHM = rfrmodel()
    
    df = importance_score(score,band)
    
    df.to_csv('G:/SENTINEL-2_10m_DATA/s2_10m_final_models/h99_10m/ImportanceScore_model_s2_10m_99perc_20190219_512_n24.csv')
    
    stats = np.vstack([fmr4,fmMSE4]).T
    
    df2 = pd.DataFrame(stats, columns=['r2','mse'])
    
    df2.to_csv('G:/SENTINEL-2_10m_DATA/s2_10m_final_models/h99_10m/Model_stats_model_s2_10m_99perc_20190219_512_n24.csv')
    

    d = klepto.archives.file_archive('G:/SENTINEL-2_10m_DATA/s2_10m_final_models/h99_10m/rfrLCHM_s2_10m_99perc_20190219_512_n24', serialized=True)
    d['model'] = rfrLCHM
    
    d.dump()
    d.clear()
    

if __name__ == "__main__":
    mainRoutine()