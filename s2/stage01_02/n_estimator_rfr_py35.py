#!/usr/bin/env python

"""
Script to execute sklearn random forest regressor code using new e_estimator values
The code runs the rfr model multiple times which outputs stats and feature importance scores. 
The average accuracy statistics written out for each n_estimator to a csv file.

author: Grant Staben
email: gwstaben@gmail.com

Date created: 27/05/2016

"""

# import the requried modules
from __future__ import print_function, division
import sys
import os
import argparse
import pdb
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn import cross_validation
from sklearn.metrics import explained_variance_score


# command arguments 
def getCmdargs():

    p = argparse.ArgumentParser()

    p.add_argument("--datafile", help="Input data file")
    p.add_argument("--nskipcols", default=2, type=int,
        help="Skip the first <nskipcols> columns in the data file (default=%(default)s)")
    p.add_argument("--outDir", help="directory to output the csv file with the results")
    p.add_argument("--itera",default=20, type=int, help="number of iterations of the model (default=%(default)s)")
    #p.add_argument("--n_estimators", type=int, default=512,help="Number of estimators in random forest models (default=%(default)s)")
    p.add_argument("--y_var",default='chm', type=str, help="variable being predicted (default=%(default)s)")
    p.add_argument("--njobs", default=1, type=int, help="This defines the number of cores used default is 1 core, dont use -1 on the qld system as it uses all the cores avaialble and people get upset (default=%(default)s)")
        
    cmdargs = p.parse_args()
    
    # if there is no image list the script will terminate
    if cmdargs.datafile is None:

        p.print_help()

        sys.exit()

    return cmdargs



def createVarlist(cmdargs):
    
    """
    function to read in the datafile and create a list of all the predctor variables
    after the first run of the iterate model.
    """
    
    df = pd.read_csv(cmdargs.datafile, header=0)# reads in the datafile 
        
    varlist = df[df.columns[cmdargs.nskipcols:]]# reads in the predictor variables
    varlist2 = list(varlist) # creates a list of variabls
    
    Orig_lenVarlist = len(varlist2)
 
    df2 = pd.DataFrame(varlist2,columns=['band'])# creates a pandas df with the list of predictor variables and a column header 
    df2.to_csv('varlist.csv') # outputs the new csv file with the list of predictor variables
    #pdb.set_trace()
    
    return Orig_lenVarlist
    

def RFRmodel(cmdargs,n_est):
    
    """
    function to execute the script IterateRFR_chm_model_argVarTrim.py  
    """
    predV = pd.read_csv('varlist.csv', header=0)
    varlist = predV['band'].tolist()
            
    print (len(varlist))
   
    df = pd.read_csv(cmdargs.datafile, header=0)
    df1 = df.dropna()# added to remove any nan values
    
    xdata1 = df1[df1.columns[cmdargs.nskipcols:]].astype('float32')
    ydata1 = df1[cmdargs.y_var].astype('float32')
     
    # split data into train and tes, the user needs to define the variables 
    ydata2 = ydata1.values
    ydata = ydata2.ravel()
    X_1, X_2, y_1, y_2 = cross_validation.train_test_split(xdata1, ydata, train_size=0.9)   #random_state=20
         
    print (X_1.shape, y_1.shape)
    print  (X_2.shape, y_2.shape)
    
    rfrModel_1 = rfr(n_estimators=n_est, oob_score=True,  max_features = 'log2', min_samples_split=1, n_jobs= cmdargs.njobs) # n_jobs=-1 dont use n_jobs on the qld system people get upset!!
        
    rfrLCHM = rfrModel_1.fit(X_1, y_1)

    #feature_importance = rfrModel_1.feature_importances_
    fi = enumerate(rfrModel_1.feature_importances_)
    cols = xdata1.columns 
    fiResult2 = [(value,cols[i]) for (i,value) in fi]
    
    # string identifying the n_estimator to be written to the output csv files 
    lenP = str(n_est)
    
    fiResult = np.array(fiResult2)
    score = fiResult[:,0]
    band = fiResult[:,1]

    fmr3 = format(rfrLCHM.score(X_1, y_1), '.2f')
    print ('Fitted model r2 =' ,  format(rfrLCHM.score(X_1, y_1), '.2f'))
    fmrMSE3 = format(np.mean((y_1 - rfrLCHM.predict(X_1))**2), '.2f')
    print ('Fitted model mse =', format(np.mean((y_1 - rfrLCHM.predict(X_1))**2), '.2f'))
    print ('n =', len(y_1))
     
    # predict current model on test data
    y2_predict = rfrLCHM.predict(X_2)
    print ('Predicted data r2 =', rfrLCHM.score(X_2, y_2))
    tmr3 = rfrLCHM.score(X_2, y_2)
    print ('MSE =', format(np.mean((y_2 - rfrLCHM.predict(X_2))** 2), '.3f'))
    print ('RMSE =', format(np.sqrt(np.mean((y2_predict - y_2) ** 2)), '.3f'))
    tmRMSE3 = format(np.sqrt(np.mean((y2_predict - y_2) ** 2)), '.3f')
    print ('explained_var =',format(explained_variance_score(y_2, y2_predict),  '.3f'))
    print ('bias =' , format(np.mean(y_2) - np.mean(y2_predict), '.3f'))
    print ('n =' , len(y_2))
    
    fmr4 = float(fmr3)
    fmMSE4 = float(fmrMSE3)
    tmr4 = float(tmr3)
    tmRMSE4 = float(tmRMSE3)
     
    return (score,band, fmr4, fmMSE4, tmr4, tmRMSE4, y2_predict, y_2, lenP)

def mainRoutine():
    # runs the main routine iterates the RFR model and outputs the results in csv format
    
    cmdargs = getCmdargs() 
    
    # read in the datafile and creates the original list of variables to start with
    Orig_lenVarlist = createVarlist(cmdargs)
    
    n_esti = [512] #[2,4,8,16,32,64,128,256,512,1024,2048,4096]   #[
    # empty list to put all the summary training stats in
    total_results = [] 
    
    # create a header file and append it to the top of the total_results list
    header = ['n_pred','T_rmseMean','T_rmse50th','T_rmse25th','T_rmse75th','T_r2Mean','T_r2_50th','T_r2_25th','T_r2_75th']
    total_results.append(header)
    
    for i in n_esti:
        n_est = i
        print (n_est)
        
        #pdb.set_trace() 
        # create some empty list to append the result
        fi_s2 = []
        fi_b2 = []
        fmr2 = []
        fmrMSE = []
        tmr2 = []
        tmRMSE =[]
        y2_pred = []
        y_ob = []
        
        # need to define how many times to run the model by defining the range
        for i in range(cmdargs.itera):
            print (i+1)
            # appends the out put of the last run of the model to the lists
            score, band, fmr4, fmMSE4, tmr4, tmRMSE4, y2_predict, y_2, lenP = RFRmodel(cmdargs, n_est)
            fi_s2.append(score)
            fi_b2.append(band)
            fmr2.append(fmr4)
            fmrMSE.append(fmMSE4)
            tmr2.append(tmr4)
            tmRMSE.append(tmRMSE4)
            y2_pred.append(y2_predict)
            y_ob.append(y_2)
        
        # concatenate lists and transpose 
        fiscore2 = np.vstack([fi_s2, fi_b2])
        predOb = np.vstack([y2_pred, y_ob]).T  
    
    
        fiscore3 = fiscore2[0:cmdargs.itera + 1].T
        print (fiscore3)
        # put feature importance data into a pandas df and calculate the mean and export it for reference later
        df = pd.DataFrame(fiscore3)
        df2 = df[df.columns[0:cmdargs.itera]].astype('float64')# takes the number of iterations from the cmdargs function to determine the number of columns to select and calculate the mean score
        df2['meanScore'] = df2.mean(axis=1)
        df2['perc50th'] = df2.quantile(.50,axis=1)
        df2['perc25th'] = df2.quantile(.25,axis=1)
        df2['perc75th'] = df2.quantile(.75,axis=1)
        Scores = df2[['meanScore','perc50th','perc25th','perc75th']]
    
        band = df[df.columns[-1:]]# get a list of the predictor variable names to concatenate with their corresponding importance score
        results = pd.concat([band, Scores],axis=1)
        results.columns = ['band','meanScore','perc50th','perc25th','perc75th'] # header names for the columns]
        results_sort = results.sort_values(by='meanScore',ascending=False)
        results_sort.to_csv(cmdargs.outDir + 'FeatureImportance_Band_Score_n'+ lenP +'.csv') # write out the feature importance results for the current run of the model
    
          
        # calculate the mean rmse for the training data 
        errorStats = np.vstack([tmr2,tmRMSE]).T
    
        columns = ['trainR2','trainRMSE']
        df_r = pd.DataFrame(errorStats, columns=columns, dtype='float32')
        T_rmseMean = df_r['trainRMSE'].mean()
        T_rmse50th = df_r['trainRMSE'].quantile(.50)
        T_rmse25th = df_r['trainRMSE'].quantile(.25)
        T_rmse75th = df_r['trainRMSE'].quantile(.75)
    
        T_r2Mean = df_r['trainR2'].mean()
        T_r2_50th = df_r['trainR2'].quantile(.50)
        T_r2_25th = df_r['trainR2'].quantile(.25)
        T_r2_75th = df_r['trainR2'].quantile(.75)
    
    
        rmseRt = np.vstack([lenP,T_rmseMean,T_rmse50th,T_rmse25th,T_rmse75th,T_r2Mean,T_r2_50th,T_r2_25th,T_r2_75th]).T 
    
        rmseRn = [lenP,T_rmseMean,T_rmse50th,T_rmse25th,T_rmse75th,T_r2Mean,T_r2_50th,T_r2_25th,T_r2_75th] 
    
        # put the results into a numpy array and transpose it to enable them to go int a pandas df
        rmserC = ['n_pred','T_rmseMean','T_rmse50th','T_rmse25th','T_rmse75th','T_r2Mean','T_r2_50th','T_r2_25th','T_r2_75th']
    
        df_rmseR = pd.DataFrame(rmseRt, columns=rmserC)
    
        df_rmseR.to_csv(cmdargs.outDir + 'Train_r2_rmse_n'+ lenP +'.csv')
    
        # write out the individual results for each itereaton in each run just in case we what to check things later   
   
       
        predictOb = csv.writer(open(cmdargs.outDir + 'PredictedANDobserved_n' + lenP +'.csv', "w"))
        predictOb.writerows(predOb)
    
        fis = csv.writer(open(cmdargs.outDir + 'FeatureImportance_Score_n' + lenP + '.csv', "w"))
        fis.writerows(fi_s2)
    
        fib = csv.writer(open(cmdargs.outDir + 'FeatureImportance_band_n' + lenP + '.csv', "w"))
        fib.writerows(fi_b2)
    
        statResults = fmr2, fmrMSE ,tmr2, tmRMSE
        stats = csv.writer(open(cmdargs.outDir + 'IterateStats_n'+ lenP + '.csv', "w"))
        stats.writerows(statResults)
      
         
        # join the elements in each of the lists row by row 
        total_results.append(rmseRn) 
    
        
    # read out the final resuts to a csv file   
        
    w_stats = csv.writer(open('SummaryTrainRMSE.csv', "w"))
    w_stats.writerows(total_results)
    
                            

if __name__ == "__main__":
    mainRoutine()

    
    
