# DOWNLOAD H14 ARCHIVE
# CONVERT to NETCDF for Bolivia
# v1: 2021-02-23

__author__ = 'al'

    
import datetime
import string
import os
import subprocess
import ftplib
import time
import sys
import logging
import glob
import json
from acquire_HSAF_functions import h27cdoconverter

## FIXED INPUTS
HSAFproduct = 'h27'

# Json name
jsoname = 'conf_' + HSAFproduct + '.json'

with open(jsoname) as jf:
    params = json.load(jf)
    destFld = params["destFld"]        
    varName = params["output_name"]       
    bbox = params["bbox"]  
    FTP_URL = params["FTP_URL"]
    FTP_USR = params["FTP_USR"]
    FTP_PWD = params["FTP_PWD"]
    FTP_DIR = params["FTP_DIR"]
    LOG_DIR = params["LOG_DIR"]
    clean_flag = params["CLEAN_GRIB"]
    start = params["start_date"]
    end = params["end_date"]


### DATES
oDateFrom = datetime.datetime.strptime(str(start),'%Y%m%d')
oDateTo = datetime.datetime.strptime(str(end),'%Y%m%d')
oDate = oDateFrom

## LOG
LOG_FILENAME = varName[0] + 'cut_log.txt'
LOG_FILENAME_RUNNING = 'running_cut_log.txt'
LOG_FILENAME_DIR = os.path.join(LOG_DIR, LOG_FILENAME)
LOG_FILENAME_RUNNING_DIR = os.path.join(LOG_DIR, LOG_FILENAME_RUNNING)
logging.basicConfig(filename=LOG_FILENAME_DIR, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


# CORE
while (oDate <= oDateTo):
    try:
        print(oDate.strftime("%Y-%m-%d %H:%M:%S"))
        logging.info('| Searching H27 for date : ' + oDate.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate Dates
        yr = oDate.strftime('%Y')
        mnt = oDate.strftime('%m')
        day = oDate.strftime('%d')
        jd = oDate.strftime('%j')
        hr = oDate.strftime('%H')
        fdate = oDate.strftime('%Y%m%d00')
        yrmnt = oDate.strftime('%Y%m')
        
        # Generate Destination Folder to save to
        dest = os.path.join(destFld, yr, mnt, day)
        if not os.path.exists(dest):
            print(dest)
            logging.info('->| Destination: ' + dest)
            os.system('mkdir -p ' + dest)
          

        # Connect to FTP
        ftp = ftplib.FTP(FTP_URL, FTP_USR, FTP_PWD)
        ftpDateDir = 'SMDAS3_' + yr + '_T1279'
        ftpDir = os.path.join(FTP_DIR, ftpDateDir, yrmnt)
        ftp.cwd(ftpDir)
        filelist=ftp.nlst()

        for file in filelist:
            # Check date
            if file.endswith(fdate+'_T1279.grib'):   
                fnc = varName[0] + fdate + '0000.nc' 
                ffnc = os.path.join(dest,fnc)
                if not os.path.isfile(ffnc):
                    time.sleep(0.05)        
                    # download
                    try:         
                        dwnFile = os.path.join(dest, file)
                        logging.debug('->| Downloading: ' + file)
                        ftp.retrbinary('RETR ' + file , open(dwnFile, 'wb').write)
                        print('->|Downloaded: ' + file)
                        logging.info('->| Downloaded: ' + file)
                        
                        filein = os.path.join(dest,file)
                        
                        ##CDO operations
                        h27cdoconverter(filein,ffnc,bbox,dest)
                        print('->| Created:  ' + ffnc)
                        logging.info('->| Created:  ' + ffnc)
                        
                        # CLEANER
                        if clean_flag == 1:    
                            print('->| Clean grib files')
                            fileList = glob.glob(dest+'/*grib*')
                            for filePath in fileList:
                                try:
                                    os.remove(filePath)
                                except:
                                    print("Error while deleting file : ", filePath)
                                    logging.info('->| !Error while deleting file : ', filePath)
                        
                    except:
                        print('Error: File could not be downloaded or converted ' + file)
                        logging.info('->| !Error: File could not be downloaded or converted ' + file)
                        continue
                    
                else:
                    print('file already processed ' + fnc)
                    logging.info('->| file already processed ' + fnc)
                    
    except:
        print('Error: Connection to Ftp')
        logging.info('->| !Error: Connection to Ftp')
        #ftp.quit()
        
    oDate = oDate + datetime.timedelta(days=1)

# FINISH

print('Process ended with success')
logging.info('||||| Process ended with success')
logging.shutdown()


    

