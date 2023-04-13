# REGRID HSAF NETCDF (H14-H27)
# CONVERT to Geotiff
# v1: 2021-02-26

__author__ = 'al'


import os
import datetime
import numpy as np
import json
import shutil

# Json name
jsoname = 'conf_mask_h14.json'

with open(jsoname) as jf:
    params = json.load(jf)
    origFld = params["origFld"]
    varName = params["input_name"]
    destFld = params["destFld"]
    outName = params["output_name"]
    mask_file = params["mask"]
    start = params["start_date"]
    end = params["end_date"]

### DATES
oDateFrom = datetime.datetime.strptime(str(start),'%Y%m%d')
oDateTo = datetime.datetime.strptime(str(end),'%Y%m%d')
oDate = oDateFrom

while (oDate <= oDateTo):
    print(oDate.strftime("%Y-%m-%d"))
    # Generate Dates
    yr = oDate.strftime('%Y')
    mnt = oDate.strftime('%m')
    day = oDate.strftime('%d')
    jd = oDate.strftime('%j')
    hr = oDate.strftime('%H')

    # Input Folder to search in
    origin = os.path.join(origFld, yr, mnt, day)

    # Input File
    origFile = varName + oDate.strftime('%Y%m%d') + '000000.tif'
    filein = os.path.join(origin,origFile)
    print('| ' + filein)

    if os.path.isfile(filein):
        # Generate Destination Folder to save to
        dest = os.path.join(destFld, yr, mnt, day)
        destFile = outName + oDate.strftime('%Y%m%d') + '000000.tif'
        fileout = os.path.join(dest,destFile)
        tmp_file = os.path.join(dest,'tmp_file.tif')
        print('|----> '+fileout)
        if not os.path.exists(dest):
            print(dest)
            os.system('mkdir -p ' + dest)

        # Mask
        cmdLine_mask = 'gdal_calc.py -A ' + filein + ' -B ' + mask_file + ' --outfile=' + tmp_file + ' --calc=\"A*(B>0)\" --NoDataValue=0'
        print(cmdLine_mask)
        os.system(cmdLine_mask)

        # COMPRESS
        cmdLine_comp = 'gdal_translate -of GTiff -co \"COMPRESS=DEFLATE\" -co \"PREDICTOR=3\" -co \"TILED=YES\" ' + tmp_file + ' ' + fileout
        cmdLine_comp = 'gdal_translate -of GTiff -co \"COMPRESS=DEFLATE\" ' + tmp_file + ' ' + fileout
        print(cmdLine_comp)
        os.system(cmdLine_comp)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    oDate = oDate + datetime.timedelta(days=1)
