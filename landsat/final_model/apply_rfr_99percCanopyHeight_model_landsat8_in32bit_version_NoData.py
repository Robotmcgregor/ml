#!/usr/bin/env python

"""
Created on Mon Apr 11 05:59:43 2016
Modified on 07/04/2019
@author: grants

Apply a canopy height random forest model 99 percentile height n18 predictor variables to Landsat dbi composites.

"""
from __future__ import print_function, division
import sys
import os
import argparse
import pandas as pd
import numpy
from osgeo import gdal
from rios import applier, fileinfo
import pdb
import klepto
from sklearn.preprocessing import Imputer

def getCmdargs():
    """
    Get command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("--reffile_a", help="Input the annual surface reflectance file")
    
    p.add_argument("--reffile_d", help="Input the dry season surface reflectance file")
    
    p.add_argument("--outdir", help="path to output the new raster layer")
    
    p.add_argument("--picklefile", default="rfrLCHMcpickle_l8_99perc_20190407_512_n18_int32v",help="Input pickle file (default is %(default)s)")
    cmdargs = p.parse_args()
    
    if cmdargs.reffile_a is None:
        p.print_help()
        sys.exit()
        
    return cmdargs


def main():
    """
    Main routine
    
    """
    cmdargs = getCmdargs()
    controls = applier.ApplierControls()
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    
    infiles.sfcref_a = cmdargs.reffile_a
    
    infiles.sfcref_d = cmdargs.reffile_d
       
    imageExt = infiles.sfcref_d
         
    controls.setReferenceImage(infiles.sfcref_d) 
  
    # temporary output of the model 
    outfile = 'temp_model_output_to_delete.img'
    outfiles.hgt = outfile
    
    d = klepto.archives.file_archive(cmdargs.picklefile, serialized=True)
    d.load('model')
    
    smodel = d['model']
    
    otherargs.rf = smodel 

    # null value for the input surface reflectance composities
    otherargs.refnull =  32767
    
    applier.apply(doModel, infiles, outfiles, otherargs,controls=controls)
    
    
def doModel(info, inputs, outputs, otherargs):
    """
    Called from RIOS.    
    Apply the random forest model for canopy height
    
    """
    # define the regions with valid data
    nonNullmask = (inputs.sfcref_d[0] != otherargs.refnull)
    
 
    # get the shape of the annual image and convert it to the shape of a single band  
    a_imgshape = inputs.sfcref_a.shape
    # convert the tuple to a list to convert the 6 bands to represent 1 band and then convert it back to a tuple
    list_imgshape = list(a_imgshape)
    list_imgshape[0] = 1
    imgShape = tuple(list_imgshape)

    # input the predictor variables 
    psB1a = inputs.sfcref_a[0][nonNullmask]
    psB2a = (inputs.sfcref_a[1][nonNullmask]) #.astype(numpy.int32)
    psB3a = (inputs.sfcref_a[2][nonNullmask])#.astype(numpy.int32)
    psB4a = (inputs.sfcref_a[3][nonNullmask])#.astype(numpy.int32)
    psB5a = (inputs.sfcref_a[4][nonNullmask])#.astype(numpy.int32)
    psB6a = (inputs.sfcref_a[5][nonNullmask])#.astype(numpy.int32)
    
    psB1d = (inputs.sfcref_d[0][nonNullmask])#.astype(numpy.int32)
    psB2d = (inputs.sfcref_d[1][nonNullmask])#.astype(numpy.int32)
    psB3d = (inputs.sfcref_d[2][nonNullmask])#.astype(numpy.int32)
    psB4d = (inputs.sfcref_d[3][nonNullmask])#.astype(numpy.int32)
    psB5d = (inputs.sfcref_d[4][nonNullmask])#.astype(numpy.int32)
    psB6d = (inputs.sfcref_d[5][nonNullmask])#.astype(numpy.int32)
    
       
    # convert the reflectance data to floating point and calculate the band ratios and veg indicies calculations   
    
       
    psB1fa = (psB1a*0.0001)+0.0
    psB2fa = (psB2a*0.0001)+0.0
    psB3fa = (psB3a*0.0001)+0.0
    psB4fa = (psB4a*0.0001)+0.0
    psB5fa = (psB5a*0.0001)+0.0
    psB6fa = (psB6a*0.0001)+0.0
    
  
    psB1fd = (psB1d*0.0001)+0.0
    psB2fd = (psB2d*0.0001)+0.0
    psB3fd = (psB3d*0.0001)+0.0
    psB4fd = (psB4d*0.0001)+0.0
    psB5fd = (psB5d*0.0001)+0.0
    psB6fd = (psB6d*0.0001)+0.0
    
    
    # ratio and veg index calculations and convert them to 32 bit interger format 

    ratio42a = numpy.int32(numpy.around(  (psB4a / psB2a)  *10**7))
    ratio43a = numpy.int32(numpy.around(  (psB4a / psB3a)  *10**7))
    ratio54a = numpy.int32(numpy.around(  (psB5a / psB4a)  *10**7)) 
    ratio53a = numpy.int32(numpy.around(  (psB5a / psB3a)  *10**7)) 
    
    ratio52a = numpy.int32(numpy.around(  (psB5a / psB2a)  *10**7)) 
    ratio65a = numpy.int32(numpy.around(  (psB6a / psB5a)  *10**7))
    
    ratio63a = numpy.int32(numpy.around(  (psB6a / psB3a)  *10**7)) 
    ratio62a = numpy.int32(numpy.around(  (psB6a / psB2a)  *10**7))
    ratio65a = numpy.int32(numpy.around(  (psB6a / psB5a)  *10**7))
    
    ratio32d = numpy.int32(numpy.around(  (psB3d / psB2d)  *10**7))
    ratio63d = numpy.int32(numpy.around(  (psB6d / psB3d)  *10**7))
    ratio54d = numpy.int32(numpy.around(  (psB5d / psB4d)  *10**7))
    ratio53d = numpy.int32(numpy.around(  (psB5d / psB3d)  *10**7))
    ratio52d = numpy.int32(numpy.around(  (psB5d / psB2d)  *10**7))
    ratio64d = numpy.int32(numpy.around(  (psB6d / psB4d)  *10**7))
    ratio62d = numpy.int32(numpy.around(  (psB6d / psB2d)  *10**7))
    ratio65d = numpy.int32(numpy.around(  (psB6d / psB5d)  *10**7))
    
    
    
    NDIId =  numpy.int32(numpy.around( ((psB4fd-psB5fd)/(psB4fd+psB5fd))  *10**7))
    
    CVId = numpy.int32(numpy.around(   ((psB4fd/psB2fd)*(psB3fd/psB2fd))  *10**7))
    
    NBRd = numpy.int32(numpy.around(   ((psB4fd-psB6fd)/(psB4fd+psB6fd))  *10**7))
    
    NDVId = numpy.int32(numpy.around(  ((psB4fd-psB3fd)/(psB4fd+psB3fd))*10**7))
    
    MSRd = numpy.int32(numpy.around(   (((psB4fd/psB3fd)-1)/(((numpy.sqrt(psB4fd/psB3fd))+1)))  *10**7)) ##
    
    GNDVId = numpy.int32(numpy.around(  ((psB4fd-psB2fd)/(psB4fd+psB2fd))   *10**7))
    
    SAVId = numpy.int32(numpy.around(   (((psB4fd-psB3fd)/(psB4fd+psB3fd+0.5))*(1.5))  *10**7))    
    
    GSAVId = numpy.int32(numpy.around(  (((psB4fd-psB2fd)/(psB4fd+psB2fd+0.5))*(1.5))   *10**7))
        
    MSRa = numpy.int32(numpy.around(   ((psB4fa/psB3fa)-1)/(((numpy.sqrt(psB4fa/psB3fa))+1))    *10**7))
    
    NDIIa =  numpy.int32(numpy.around(  ((psB4fa-psB5fa)/(psB4fa+psB5fa))  *10**7))
    
    NDGIa = numpy.int32(numpy.around(   ((psB2fa-psB3fa)/(psB2fa+psB3fa))   *10**7))   
    
    CVIa = numpy.int32(numpy.around(   ((psB4fa/psB2fa)*(psB3fa/psB2fa))   *10**7))
    
    GNDVIa = numpy.int32(numpy.around(  ((psB4fa-psB2fa)/(psB4fa+psB2fa))  *10**7))
    
    NDVIa = numpy.int32(numpy.around(   ((psB4fa-psB3fa)/(psB4fa+psB3fa))   *10**7))
    
    SAVIa = numpy.int32(numpy.around(   (((psB4fa-psB3fa)/(psB4fa+psB3fa+0.5))*(1.5))    *10**7))
    
    GSAVIa = numpy.int32(numpy.around(  (((psB4fa-psB2fa)/(psB4fa+psB2fa+0.5))*(1.5))    *10**7))
    
    # pass the variables into a numpy array and transform it to look like the pandas df 
    
   # 'CVId', 'NDVId', 'ratio52a', 'ratio63d', 'ratio63a', 'ratio54d','psB5a', 'psB6d', 'GNDVId', 'NDIIa', 'ratio43a', 'ratio52d',
# 'ratio53d', 'psB2a', 'ratio53a', 'CVIa', 'ratio62d', 'psB3d'
  
    allVars= numpy.vstack([CVId,NDVId,ratio52a,ratio63d,ratio63a,ratio54d,psB5a,psB6d,GNDVId,NDIIa,ratio43a,ratio52d,ratio53d,psB2a, ratio53a,CVIa,ratio62d,psB3d]).T
        
    
    # sets up the shape and dtype for the chm output  
     
    outputs.hgt = numpy.zeros(imgShape, dtype=numpy.float32) #.int16)
 
    # applies the rfr model to produce the chm layer
    
    if allVars.shape[0] > 0:
        # run check over the input data and replaces nan and infinity values
        allVars[numpy.isnan(allVars)] = 0.0
        allVars[numpy.isinf(allVars)] = 0.0
        
        hgt = otherargs.rf.predict(allVars)
        
       
        outputs.hgt[0][nonNullmask] = hgt

        
def apply_new_nodata():
    """
    This function applies a new nodata value to the model output
    it uses the nodata areas defined by the dry season composite.
    This step is nessecary as when the models is output 
    it can produce zero which is considered notdata when 
    the model is written out to an image and creates larger areas of nodata
    """
    cmdargs = getCmdargs()
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    
    # read in the dry season composite 
    infiles.sfcref_d = cmdargs.reffile_d 
    
    # this reads in the output from applying the model 
    infiles.model = 'temp_model_output_to_delete.img'
    
    # set up the output file name
    outpath = cmdargs.outdir + "/"
    inf_part1 = cmdargs.reffile_a[:-9]
    inf_part2 = inf_part1[-30:]
    inf_part3 = cmdargs.reffile_a[-6:]
    # path and file name for the modelled output
    nodata_file_name = outpath + inf_part2 + 'h99' + inf_part3
    print (nodata_file_name)
    outfiles.model_nodata = nodata_file_name

    applier.apply(defineNoData, infiles, outfiles, otherargs)
    
    # set the nodata value using gdal
    setNodataValue(nodata_file_name)
    
    # removes the temp file
    os.remove("temp_model_output_to_delete.img")
    # needed if script is run on a windows system
    os.remove("temp_model_output_to_delete.img.aux.xml")

def defineNoData(info, inputs, outputs, otherargs):
    
    """ this function reads in the modelled data an applies a nodata value '
    based on the nodata values in the dry season composite"""
    
    # read in the modelled data 
    model = inputs.model
    # read in the dry season composite which defines the nodata areas
    mask = inputs.sfcref_d.astype(numpy.float32)
    
    stack = numpy.vstack((model, mask))
    # select the first band of the seasonal composite and if 
    # they are true no data values change them to 0.0001
    stack[stack[1:2] == 32767.0] = 0     
    stack1 = stack
    # select out the modelled band from the stack to output as the new layer
    model_nodata = stack1[:1]
  
    outputs.model_nodata = model_nodata
    
def setNodataValue(nodata_file_name):
    
    """this function sets the new no data value so 
    when it is open in arcmap or qgis it recognises the new values"""
    
    newImg = gdal.Open(nodata_file_name, gdal.GA_Update)
    newImg.GetRasterBand(1).SetNoDataValue(0)

if __name__ == "__main__":
    main()    
    apply_new_nodata()  
