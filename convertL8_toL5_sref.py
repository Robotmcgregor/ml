#!/usr/bin/env python
"""
converts a landsat-8 dbg image to landsat-7 like image using Neils correction factors.

Author: Grant Staben
Date: 05/05/2017
"""
from __future__ import print_function, division

import sys
import os
import argparse
import numpy
from osgeo import gdal
from rios import applier
from rios import fileinfo
#import qvf
#from rsc.utils import DNscaling
#from rsc.utils import history

def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser(description="""
        converts a Landsat-8 image to landsat-5 surface reflectance like values 
    """)
    p.add_argument("--sfcref", help="Name of input Landsat-8 file")
    p.add_argument("--outfile", help="Name of output image file")
    cmdargs = p.parse_args()
    
    if cmdargs.sfcref is None:
        p.print_help()
        sys.exit()
    
    return cmdargs


def mainRoutine():
    """
    Main routine
    """
    cmdargs = getCmdargs()
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    controls = applier.ApplierControls()
    otherargs = applier.OtherInputs()
    
    infiles.sfcref = cmdargs.sfcref
    outfiles.outimg = cmdargs.outfile
    ref10info = fileinfo.ImageInfo(cmdargs.sfcref)
    
    otherargs.refnull = ref10info.nodataval[0]
    # assign the null data value
    otherargs.outNull = 32767 
    controls.setReferenceImage(cmdargs.sfcref)
    controls.setStatsIgnore(otherargs.refnull)
    
    applier.apply(doConvert, infiles, outfiles, otherargs, controls=controls)
    
    
    #copyScaling(cmdargs, ref10info)
    #addHistory(cmdargs)

def doConvert(info, inputs, outputs, otherargs):
    """
    apply the corrections to the landsat-8 imagery
    """
    # define the mask area taken from the input image
    mask = inputs.sfcref[0]==otherargs.refnull    
    
    psB1 = inputs.sfcref[1]
    psB2 = inputs.sfcref[2]
    psB3 = inputs.sfcref[3]
    psB4 = inputs.sfcref[4]
    psB5 = inputs.sfcref[5]
    psB6 = inputs.sfcref[6]
    
    psB1fp = psB1.astype(numpy.float32)
    psB2fp = psB2.astype(numpy.float32)
    psB3fp = psB3.astype(numpy.float32)
    psB4fp = psB4.astype(numpy.float32)
    psB5fp = psB5.astype(numpy.float32)
    psB6fp = psB6.astype(numpy.float32)
    
    
    psB1f = (psB1fp /10000.0).astype(numpy.float32)
    psB2f = (psB2fp /10000.0).astype(numpy.float32)
    psB3f = (psB3fp /10000.0).astype(numpy.float32)
    psB4f = (psB4fp /10000.0).astype(numpy.float32)
    psB5f = (psB5fp /10000.0).astype(numpy.float32)
    psB6f = (psB6fp /10000.0).astype(numpy.float32)
    
    # apply Neils l7 correction factors 
    
    psB1fc = 0.97470*psB1f + 0.00041
    psB2fc = 0.99779*psB2f + 0.00289
    psB3fc = 1.00446*psB3f + 0.00274
    psB4fc = 0.98906*psB4f + 0.00004
    psB5fc = 0.99467*psB5f + 0.00256
    psB6fc = 1.02551*psB6f + -0.00327
       
    # convert the corrected floating point data back to interger 
    psb1 = (psB1fc*10000).astype(numpy.int16)
    psb1[mask] = otherargs.outNull
    psb2 = (psB2fc*10000).astype(numpy.int16)
    psb2[mask] = otherargs.outNull
    psb3 = (psB3fc*10000).astype(numpy.int16)
    psb3[mask] = otherargs.outNull
    psb4 = (psB4fc*10000).astype(numpy.int16)
    psb4[mask] = otherargs.outNull
    psb5 = (psB5fc*10000).astype(numpy.int16)
    psb5[mask] = otherargs.outNull
    psb6 = (psB6fc*10000).astype(numpy.int16)
    psb6[mask] = otherargs.outNull
    
    output = numpy.array([psb1,psb2,psb3,psb4,psb5,psb6])#dtype=numpy.int16)
    outputs.outimg = output


def copyScaling(cmdargs,ref10info): 
    """
    Set layer names. 
    """
    ref10scale = DNscaling.getDNscaleStackFromFile(cmdargs.sfcref)
    scaleList = ref10scale[:] 
    outScale = DNscaling.DNscaleStack(scaleList)
    DNscaling.writeDNscaleStackToFile(cmdargs.outfile, outScale)
    # Set explicit layer names 1 to 10
    layerNames = ['B1','B2','B3','B4','B5','B6']
    ds = gdal.Open(cmdargs.outfile, gdal.GA_Update)
    for i in range(6):
        band = ds.GetRasterBand(i+1)
        band.SetDescription(layerNames[i])
   
    del ds

def addHistory(cmdargs):
    """
    Add processing history
    """
    parents = [cmdargs.sfcref]
    opt = {}
    opt['DESCRIPTION'] = """Landsat-8 has been converted to look like landsat-7 reflectance values using Neils corection factors.  
    """
    history.insertMetadataFilename(cmdargs.outfile, parents, opt)

if __name__ == "__main__":
    mainRoutine()
