#!/usr/bin/env python
"""
converts a list of landsat-8 dbg image to landsat-7 like image using Neils correction factors.
this script reads in the convertL8_toL5_sref.py script

Author: Grant Staben
Date: 06/05/2017

"""

# import the requried modules
import sys
import os
import argparse
import pdb
import csv


# command arguments 
def getCmdargs():

    p = argparse.ArgumentParser()

    p.add_argument("-s","--imglist", help="list of landsat-8 imagery to apply the conversion")
    
    p.add_argument("-i","--indir", help="directory containing the imagery")
    
    p.add_argument("-o","--outdir", help="output directory for the chm product")
    
     
    cmdargs = p.parse_args()
    
    # if there is no image list the script will terminate
    if cmdargs.imglist is None:

        p.print_help()

        sys.exit()

    return cmdargs


def applyModel(imglist,indir,outdir):
    
    """
    function to apply the test random forest canopy height model to a list of Landsat5 imagery
    """
    
    # open the list of imagery and read it into memory
    with open(imglist, "r") as imagerylist:
        
        # loop throught the list of imagery and strip out the name of the input image and create the output file name with a new extention.          
        l8conv = []
	
	for image in imagerylist:
            imageS = image.rstrip()
            #inp = imageS
	    inp = indir + imageS 
            outp = outdir + imageS[:24] + '_dbg_zstdmask_c.img'
            l8conv.append(outp)
            # call and run the apply_rfrModel_Landsat5.py script 
            cmd = "/scratch/rsc5/grant/working/scripts/comp/convertL8_toL5_sref.py --sfcref %s --outfile %s" % (inp, outp) 
            
            os.system(cmd)
            
            # print out the file name of the processed image
            print outp + ' ' + 'is' + ' ' + 'complete'
	
	with open('l8conv_list.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for file in l8conv:
                writer.writerow([file])
            
	
            
# calls in the command arguments and applyModel function.        
def mainRoutine():
    
    cmdargs = getCmdargs()
    
    applyModel(cmdargs.imglist,cmdargs.indir,cmdargs.outdir)
    
if __name__ == "__main__":
    mainRoutine()