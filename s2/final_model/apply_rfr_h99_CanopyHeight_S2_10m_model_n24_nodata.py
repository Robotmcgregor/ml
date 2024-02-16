#!/usr/bin/env python

"""
19/02/2019
modified: 18/03/2019
Grant Staben 

Apply a canopy height random forest model 99 percentile height n24 predictor variables to Sentinel-2 10m seasonal composites (annual and Dry season). 
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
    
    p.add_argument("--kleptofile", default="D:/SENTINEL-2_10m_DATA/s2_10m_final_models/h99_10m/rfrLCHM_s2_10m_99perc_20190219_512_n24",help="Input klepto file (default is %(default)s)")
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
    
    d = klepto.archives.file_archive(cmdargs.kleptofile, serialized=True)
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
    nonNullmask = (inputs.sfcref_d[0] != otherargs.refnull)
    
    # get the shape of the annual image and convert it to the shape of a single band  
    a_imgshape = inputs.sfcref_a.shape
    # convert the tuple to a list to convert the 6 bands to represent 1 band and then convert it back to a tuple
    list_imgshape = list(a_imgshape)
    list_imgshape[0] = 1
    imgShape = tuple(list_imgshape)

# input the predictor variables 
    
    #psB1a = inputs.sfcref_a[0][nonNullmask]
    psB2a = inputs.sfcref_a[1][nonNullmask]
    psB3a = inputs.sfcref_a[2][nonNullmask]
    psB4a = inputs.sfcref_a[3][nonNullmask]
    psB5a = inputs.sfcref_a[4][nonNullmask]
    psB6a = inputs.sfcref_a[5][nonNullmask]
    psB7a = inputs.sfcref_a[6][nonNullmask]
    psB8a = inputs.sfcref_a[7][nonNullmask]
    psB9a = inputs.sfcref_a[8][nonNullmask]
    psB10a = inputs.sfcref_a[9][nonNullmask]
    
    #psB1d = inputs.sfcref_d[0][nonNullmask]
    psB2d = inputs.sfcref_d[1][nonNullmask]
    psB3d = inputs.sfcref_d[2][nonNullmask]
    psB4d = inputs.sfcref_d[3][nonNullmask]
    psB5d = inputs.sfcref_d[4][nonNullmask]
    psB6d = inputs.sfcref_d[5][nonNullmask]
    psB7d = inputs.sfcref_d[6][nonNullmask]
    psB8d = inputs.sfcref_d[7][nonNullmask]
    psB9d = inputs.sfcref_d[8][nonNullmask]
    psB10d = inputs.sfcref_d[9][nonNullmask]
    
   
    # convert the reflectance data to floating point and calculate the band ratios and veg indicies calculations
    # annual 
    #psB1fa = (psB1a*0.0001)+ 0.0 # blue 
    psB2fa = (psB2a*0.0001)+ 0.0 # green
    psB3fa = (psB3a*0.0001)+ 0.0 # red
    psB4fa = (psB4a*0.0001)+ 0.0 # nir

    psB5fa = (psB5a*0.0001)+ 0.0 # rededge 01
    psB6fa = (psB6a*0.0001)+ 0.0 # rededge 02
    psB7fa = (psB7a*0.0001)+ 0.0 # rededge 03
    psB8fa = (psB8a*0.0001)+ 0.0 # narrow band nir

    psB9fa = (psB9a*0.0001)+ 0.0 # swir1
    psB10fa = (psB10a*0.0001)+ 0.0 # swir2
    
    # dry season
    #psB1fd = (psB1d*0.0001)+ 0.0 # blue 
    psB2fd = (psB2d*0.0001)+ 0.0 # green
    psB3fd = (psB3d*0.0001)+ 0.0 # red
    psB4fd = (psB4d*0.0001)+ 0.0 # nir

    psB5fd = (psB5d*0.0001)+ 0.0 # rededge 01
    psB6fd = (psB6d*0.0001)+ 0.0 # rededge 02
    psB7fd = (psB7d*0.0001)+ 0.0 # rededge 03
    psB8fd = (psB8d*0.0001)+ 0.0 # narrow band nir

    psB9fd = (psB9d*0.0001)+ 0.0 # swir1
    psB10fd = (psB10d*0.0001)+ 0.0 # swir2
    
      
    # do ratio and veg index calculations
    
    NDRE1d = numpy.int32(numpy.around(   ((psB6fd-psB5fd)/(psB6fd+psB5fd))  *10**7))
    
    ratio63a = numpy.int32(numpy.around(  (psB6a / psB3a)  *10**7))
    
    CVIa = numpy.int32(numpy.around(   ((psB4fa/psB2fa)*(psB3fa/psB2fa))   *10**7))
    
    ratio62a = numpy.int32(numpy.around(  (psB6a / psB2a)  *10**7))
    
    #psB9d 
    
    CVId = numpy.int32(numpy.around(   ((psB4fd/psB2fd)*(psB3fd/psB2fd))   *10**7))
    
    GNDVIa = numpy.int32(numpy.around(  ((psB4fa-psB2fa)/(psB4fa+psB2fa))  *10**7))
    
    NDRE2a = numpy.int32(numpy.around(   ((psB7fa-psB5fa)/(psB7fa+psB5fa))  *10**7))
                  
    ratio95d = numpy.int32(numpy.around(  (psB9d / psB5d)  *10**7))
    
    ratio98a = numpy.int32(numpy.around(  (psB9a / psB8a)  *10**7))
    
    ratio62d = numpy.int32(numpy.around(  (psB6d / psB2d)  *10**7))
    
    #psB10a 
    
    ratio92a = numpy.int32(numpy.around(  (psB9a / psB2a)  *10**7))
    
    ratio105a = numpy.int32(numpy.around(  (psB10a / psB5a)  *10**7))
    
    ratio54d = numpy.int32(numpy.around(  (psB5d / psB4d)  *10**7))
    
    SAVIa =  numpy.int32(numpy.around(  (((psB4fa-psB3fa)/(psB4fa+psB3fa+0.5))*(1+0.5))  *10**7))
        
    NDIId = numpy.int32(numpy.around(  ((psB4fd-psB9fd)/(psB4fd+psB9fd))  *10**7))
    
    #psB3d, 
    
    ratio107d = numpy.int32(numpy.around(  (psB10d / psB7d)  *10**7))
    
    #psB2a, 
    
    ratio103d = numpy.int32(numpy.around(  (psB10d / psB3d)  *10**7))
    
    ratio103a = numpy.int32(numpy.around(  (psB10a / psB3a)  *10**7))
    
    ratio102a = numpy.int32(numpy.around(  (psB10a / psB2a)  *10**7))
    
    #psB5a,
    
    ratio102d = numpy.int32(numpy.around(  (psB10d / psB2d)  *10**7))
    
    ratio93a = numpy.int32(numpy.around(  (psB9a / psB3a)  *10**7))
        
    ratio53d = numpy.int32(numpy.around(  (psB5d / psB3d)  *10**7))
    
    ratio93d = numpy.int32(numpy.around(  (psB9d / psB3d)  *10**7))
    
    ratio109d = numpy.int32(numpy.around(  (psB10d / psB9d)  *10**7))
    
    GSAVIa = numpy.int32(numpy.around( (((psB4fa-psB2fa)/(psB4fa+psB2fa+0.5))*(1.5))   *10**7))
    
    NDGId = numpy.int32(numpy.around(   ((psB2fd-psB3fd)/(psB2fd+psB3fd))   *10**7))
    
    ratio109a = numpy.int32(numpy.around(  (psB10a / psB9a)  *10**7))
    
    ratio52d = numpy.int32(numpy.around(  (psB5d / psB2d)  *10**7))
    
    ratio53a = numpy.int32(numpy.around(  (psB5a / psB3a)  *10**7))
    
    ratio52a = numpy.int32(numpy.around(  (psB5a / psB2a)  *10**7))
    
    ratio86a = numpy.int32(numpy.around(  (psB8a / psB6a)  *10**7)) 
    
    GSAVId = numpy.int32(numpy.around( (((psB4fd-psB2fd)/(psB4fd+psB2fd+0.5))*(1.5))   *10**7))
    
    #psB6d, 
    
    #psB8a, 
    
    ratio86d = numpy.int32(numpy.around(  (psB8d / psB6d)  *10**7)) 
        
    ratio87d = numpy.int32(numpy.around(  (psB8d / psB7d)  *10**7)) 
    
    #psB6a, 
    
    #psB8d, 
    
    ratio87a = numpy.int32(numpy.around(  (psB8a / psB7a)  *10**7)) 
    
    ratio76a = numpy.int32(numpy.around(  (psB7a / psB6a)  *10**7)) 
    
    ratio76d = numpy.int32(numpy.around(  (psB7d / psB6d)  *10**7)) 
    
    ratio64d = numpy.int32(numpy.around(  (psB6d / psB4d)  *10**7)) 
    
    ratio84d = numpy.int32(numpy.around(  (psB8d / psB4d)  *10**7)) 
    
    ratio64a = numpy.int32(numpy.around(  (psB6a / psB4a)  *10**7)) 
    
    # pass the variables into a numpy array and transform it to look like the pandas df 


#'NDRE1d', 'ratio63a', 'ratio62a', 'CVIa', 'GNDVIa', 'NDRE2a','CVId', 'ratio95d', 'psB9d', 'ratio98a', 'ratio62d','ratio54d','ratio92a', 'psB3d', 'ratio105a', 'SAVIa', 'psB10a', 'NDIId','psB2a', 'ratio103d', 'ratio107d', 'ratio103a', 'ratio109d', 'NDGId'
    
    allVars= numpy.vstack([NDRE1d, ratio63a, ratio62a, CVIa, GNDVIa, NDRE2a,CVId, ratio95d, psB9d, ratio98a, ratio62d,ratio54d,ratio92a, psB3d, ratio105a, SAVIa, psB10a, NDIId,psB2a, ratio103d, ratio107d, ratio103a, ratio109d, NDGId]).T

   
    # sets up the shape and dtype for the chm output  
    outputs.hgt = numpy.zeros(imgShape, dtype=numpy.float32)
    
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
    
    # add 100 to all modelled values so valid zero tree height pixels are not masked
    
    model = model + 100
    
    # read in the dry season composite which defines the nodata areas
    mask = inputs.sfcref_d.astype(numpy.float32)
    
    stack = numpy.vstack((model, mask))
    # select the first band of the seasonal composite and if 
    # they are true no data values change them to 0
    
    # testing if the if it matters what value the nodata is 
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