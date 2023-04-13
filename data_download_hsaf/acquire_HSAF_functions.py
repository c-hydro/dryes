#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:54:57 2021

@author: al
"""

# Functions for HSAF downloading

import os
import glob


def h14cdoconverter(gribfile,outputfile,bbox,destfld):
    # CDO 1, regrid
    filein = gribfile
    fileout = os.path.join(destfld,'tmp_regrid.grib')
    cdocmd = 'cdo -R remapcon,r1600x800 -setgridtype,regular'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 2, convert grib to nc
    filein = fileout
    fileout = os.path.join(destfld,'tmp_nc.nc')
    cdocmd = 'cdo -f nc copy'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 3, cut on bounding box
    filein = fileout
    fileout = os.path.join(destfld,'tmp_bbox.nc')
    cdocmd = 'cdo -sellonlatbox,' + ','.join([str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3])])
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)

    # CDO 4, change name
    filein = fileout
    fileout = os.path.join(destfld,'tmp_name.nc')
    cdocmd = 'cdo chname,var40,layer1,var41,layer2,var42,layer3,var43,layer4,var200,qc'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 5, change longname layer1
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l1.nc')
    cdocmd = 'cdo setattribute,layer1@long_name=sm_first_layer_0-7cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)    
    
    # CDO 6, change units layer1
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l1un.nc')
    cdocmd = 'cdo setattribute,layer1@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)    

    # CDO 7, change longname layer2
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l2.nc')
    cdocmd = 'cdo setattribute,layer2@long_name=sm_first_layer_7-28cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
 
    # CDO 8, change units layer2
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l2un.nc')
    cdocmd = 'cdo setattribute,layer2@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 9, change longname layer3
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l3.nc')
    cdocmd = 'cdo setattribute,layer3@long_name=sm_first_layer_28-100cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 

    # CDO 10, change units layer3
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l3un.nc')
    cdocmd = 'cdo setattribute,layer3@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 11, change longname layer4
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l4.nc')
    cdocmd = 'cdo setattribute,layer4@long_name=sm_first_layer_100-289cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 12, change units layer4
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l4un.nc')
    cdocmd = 'cdo setattribute,layer4@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 12, change longname QC
    filein = fileout
    fileout = outputfile
    cdocmd = 'cdo setattribute,qc@long_name=quality_check'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # Clean
    fileList = glob.glob(destfld+'/tmp_*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)    
            
# h27, different grid, no quality control
def h27cdoconverter(gribfile,outputfile,bbox,destfld):
    # CDO 1, regrid
    filein = gribfile
    fileout = os.path.join(destfld,'tmp_regrid.grib')
    cdocmd = 'cdo -R remapcon,r2560x1280 -setgridtype,regular'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 2, convert grib to nc
    filein = fileout
    fileout = os.path.join(destfld,'tmp_nc.nc')
    cdocmd = 'cdo -f nc copy'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 3, cut on bounding box
    filein = fileout
    fileout = os.path.join(destfld,'tmp_bbox.nc')
    cdocmd = 'cdo -sellonlatbox,' + ','.join([str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3])])
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)

    # CDO 4, change name
    filein = fileout
    fileout = os.path.join(destfld,'tmp_name.nc')
    cdocmd = 'cdo chname,var40,layer1,var41,layer2,var42,layer3,var43,layer4'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)
    
    # CDO 5, change longname layer1
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l1.nc')
    cdocmd = 'cdo setattribute,layer1@long_name=sm_first_layer_0-7cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)    
    
    # CDO 6, change units layer1
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l1un.nc')
    cdocmd = 'cdo setattribute,layer1@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline)    

    # CDO 7, change longname layer2
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l2.nc')
    cdocmd = 'cdo setattribute,layer2@long_name=sm_first_layer_7-28cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
 
    # CDO 8, change units layer2
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l2un.nc')
    cdocmd = 'cdo setattribute,layer2@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 9, change longname layer3
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l3.nc')
    cdocmd = 'cdo setattribute,layer3@long_name=sm_first_layer_28-100cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 

    # CDO 10, change units layer3
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l3un.nc')
    cdocmd = 'cdo setattribute,layer3@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 11, change longname layer4
    filein = fileout
    fileout = os.path.join(destfld,'tmp_longname_l4.nc')
    cdocmd = 'cdo setattribute,layer4@long_name=sm_first_layer_100-289cm'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # CDO 12, change units layer4
    filein = fileout
    fileout = outputfile
    cdocmd = 'cdo setattribute,layer4@units=-'
    cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
    os.system(cdocmdline) 
    
    # Clean
    fileList = glob.glob(destfld+'/tmp_*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)   