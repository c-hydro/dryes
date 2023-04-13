# REGRID HSAF NETCDF (H14-H27)
# CONVERT to Geotiff
# v1: 2021-02-26
# v3: 2021-02-28, manage both h14 and h27 input files
# v4: 2021-10-07, mask on salares
# need for CDO with grib extension: export PATH="/usr/local/cdo_1_9/bin:$PATH"

__author__ = 'al'


import os
import datetime
import numpy as np
from rasterRegrid import rasterRegrid
import glob
import json
import sys
import getopt
from dateutil.relativedelta import *

## FIXED INPUTS
# start = 20210201
# end = 20210202
# varName = 'bol-h14_'
# outName = 'hsaf-SM_'
# origFld = '/Users/al/Documents/CIMA/HSAF/downloader/out'
# destFld = '/Users/al/Documents/CIMA/HSAF/out'
# referenceTif = 'ReferenceGrid_BOL.tif'
# method = 'nearest'
# clean_flag = 1

# Json name
jsoname = 'conf_HSAF_RZSMtif.json'

with open(jsoname) as jf:
    params = json.load(jf)
    origFld_s = params["origFld"]
    varName_s = params["input_name"]
    destFld = params["destFld"]
    outName = params["output_name"]
    switch_date = params["switch_date"]
    referenceTif = params["reference_grid"]
    mask_file = params["mask"]
    method = params["interp_method"]
    clean_flag = params["CLEAN_TEMP"]
    start = params["start_date"]
    end = params["end_date"]

# Get arguments
argv=sys.argv[1:]

### DATES
opts, a1Args = getopt.getopt(argv,"ht:e:",["help","timerange=","dateend="])

daysBefore = None
dateend = None

for opt, arg in opts:
    if opt in ('-h',"--help"):
        displayHelp();
        sys.exit()
    elif opt in ("-t", "--timerange"):
        print('Time range from command line')
        daysBefore = arg
    elif opt in ("-e", "--dateend"):
        print('Final Date from command line')
        dateend = arg

if dateend != None:
    oDateTo = datetime.datetime.strptime(dateend,"%Y%m%d")
else:
    oDateTo = datetime.datetime.strptime(str(end),'%Y%m%d')

if daysBefore !=None:
    oDateFrom = oDateTo + relativedelta(days=-int(daysBefore))
else:
    oDateFrom = datetime.datetime.strptime(str(start),'%Y%m%d')

oDate = oDateFrom

# H27 to H14
oDateSwitch = datetime.datetime.strptime(str(switch_date),'%Y%m%d')

while (oDate <= oDateTo):
    print(oDate.strftime("%Y-%m-%d"))
    # Generate Dates
    yr = oDate.strftime('%Y')
    mnt = oDate.strftime('%m')
    day = oDate.strftime('%d')
    jd = oDate.strftime('%j')
    hr = oDate.strftime('%H')

    # Input File
    if oDate < oDateSwitch:
        varName = varName_s[0] # H27
        origFld = origFld_s[0]
    else:
        varName = varName_s[1] # H14
        origFld = origFld_s[1]

    # Input Folder to search in
    origin = os.path.join(origFld, yr, mnt, day)

    origFile = varName + oDate.strftime('%Y%m%d') + '000000.nc'
    filein = os.path.join(origin,origFile)
    print('|----> ' + filein)

    if os.path.isfile(filein):
        # Generate Destination Folder to save to
        dest = os.path.join(destFld, yr, mnt, day)
        destFile = outName + oDate.strftime('%Y%m%d') + '000000.tif'
        if not os.path.exists(dest):
            print(dest)
            os.system('mkdir -p ' + dest)

        try:
            # Compute Root Zone Soil Moisture (weighted)
            fileout = os.path.join(dest,'tmp_rzsm.nc')
            cdocmd = """cdo -setunit,'-' -expr,'RZSM=layer1*0.07+layer2*0.21+layer3*0.72'"""
            cdocmdline = cdocmd + ' ' + filein + ' ' + fileout
            print(cdocmdline)
            os.system(cdocmdline)

            # convert to Geotiff
            filein = fileout
            fileout = os.path.join(dest,'tmp_rzsm.tif')
            gdalcmd = 'gdal_translate -of GTiff NETCDF:'
            gdalcmdline = gdalcmd + '"' + filein + '":RZSM ' + fileout
            print(gdalcmdline)
            os.system(gdalcmdline)

            # regrid to final resolution (reference grid)
            filein = fileout
            fileout = os.path.join(dest,'tmp_regrid.tif')
            rasterRegrid (filein ,referenceTif, fileout, method)

            # Mask
            filein = fileout
            fileout = os.path.join(dest,'tmp_regrid_mask.tif')
            cmdLine_mask = 'gdal_calc.py -A ' + filein + ' -B ' + mask_file + ' --outfile=' + fileout + ' --calc=\"A*(B>0)\" --NoDataValue=0'
            print(cmdLine_mask)
            os.system(cmdLine_mask)

            # COMPRESS
            filein = fileout
            fileout = os.path.join(dest,destFile)
            print(fileout)
            cmdLine_comp = 'gdal_translate -of GTiff -co \"COMPRESS=DEFLATE\" ' + filein + ' ' + fileout
            print(cmdLine_comp)
            os.system(cmdLine_comp)

            # CLEANER
            if clean_flag == 1:
                print('->| Clean grib files')
                fileList = glob.glob(dest+'/tmp_*')
                for filePath in fileList:
                    try:
                        os.remove(filePath)
                    except:
                        print("Error while deleting file : ", filePath)


        except:
            print("Error while computing RZSM : ", origFile)


    oDate = oDate + datetime.timedelta(days=1)








#rasterRegrid (sFileIn ,sFileMatch, sFileOut, method)
