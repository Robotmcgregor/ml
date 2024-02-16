#!/usr/bin/env python
"""
this script applys the h99 model to a  list of landsat imagery 
Author: Grant Staben
Date: 15/16/2019
"""
from __future__ import print_function, division

# import the requried modules
import sys
import os
import argparse
import pdb
import pandas as pd
import csv


# command arguments 
def getCmdargs():

    p = argparse.ArgumentParser()

    p.add_argument("-s","--imglist", help="read in the list of imagery to apply the model")
    p.add_argument("-o","--outdir", help="path to the directory to save out the h99 imagery")
    
    cmdargs = p.parse_args()
    
    # if there is no image list the script will terminate
    if cmdargs.imglist is None:

        p.print_help()

        sys.exit()

    return cmdargs


def applyModel(imglist,outdir):
    
    """
    produce the H99 tree height product by reading the image list and script.
    """
    
    # open the list of imagery and read it into memory
    df = pd.read_csv(imglist,header=0)
    
    for index, row in df.iterrows():
        annual = (str(row['an']))
        dry = (str(row['dr']))

        fileN = (str(row['an']))
        
        outdir = outdir
        
        #print (fileN)
        print (outdir+fileN)         
 
        # call and run the vegetation index scripts 
        cmd = "apply_rfr_99percCanopyHeight_model_landsat8_in32bit_version_NoData.py --reffile_a %s --reffile_d %s --outdir %s"% (annual,dry,outdir) 
            
        os.system(cmd)
         
           
# calls in the command arguments and applyModel function.        
def mainRoutine():
    
    cmdargs = getCmdargs()
    
    applyModel(cmdargs.imglist,cmdargs.outdir)
  
if __name__ == "__main__":
    mainRoutine()
    
