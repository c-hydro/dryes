import os, sys, getopt
import json
import numpy as np
from geotiff import *
import datetime
from dateutil.relativedelta import *


def displayHelp():

    print ('\nCompute SSMI index')
    print ('Options:')
    print ('          -t | --timerange              months before end of calculation to override .json file')
    print ('          -e | --dateend                calculation end date (format: YYYYMM) to override .json file ')
    print ('          -h | --help                   display this help')

def calculate_fapar_result(FAPARPrefix, FAPARmonthlyFold, FAPARstatsFold,
                         indexPrefix, destFold, year, month, monthCount, min, max):
    # adjust folders and filenames

    dest_dir = os.path.join(destFold, year, month)
    os.system("mkdir -p "+dest_dir)

    destFileName = os.path.join(dest_dir, indexPrefix + "{:02d}".format(monthCount) +"-MODIS_"+ year+month+".tif")

    FAPARinpFileName = os.path.join(FAPARmonthlyFold, "{:d}-Month-Files".format(monthCount),FAPARPrefix +"_"+ year+month+ ".tif")
    FAPARstaFileName = os.path.join(FAPARstatsFold, "FAPAR-statistics-{:02d}months-Month".format(monthCount)+month+".tif")

    FAPAR, col, row, geoTrans, geoproj = readGeotiff (FAPARinpFileName)

    FAPAR[FAPAR>max]=np.nan
    FAPAR[FAPAR<min]=np.nan

    data3d, col, row, geoTrans, geoproj = readGeotiff (FAPARstaFileName)

    FAPARmean = data3d[:,:,0]
    FAPARstd = data3d[:,:,1]

    mask, col, row, geoTrans, geoproj=  readGeotiff("mask_bolivia.tif")

    FAPAR_anomaly = (FAPAR - FAPARmean ) / FAPARstd *  mask

    writeGeotiffSingleBand(destFileName, geoTrans, geoproj, FAPAR_anomaly,nodata=np.nan, BandName="FAPAR_anomaly")

    print(destFileName)

def main(argv):

    with open('FAPAR_config.json') as jf:
        params = json.load(jf)
        FAPARstatsFold = params["Stats_folder"]
        if FAPARstatsFold[-1] == os.path.sep:
            FAPARstatsFold = FAPARstatsFold[:-1]
        FAPARmonthlyFold = params["Monthly_folder"]
        if FAPARmonthlyFold[-1] == os.path.sep:
            FAPARmonthlyFold = FAPARmonthlyFold[:-1]
        indexFold = params["Index_folder"]
        if indexFold[-1] == os.path.sep:
            indexFold = indexFold[:-1]
        sDateFrom = params["StartDate"]
        sDateTo = params["EndDate"]
        indexPrefix = params["Index_prefix"]
        FAPARPrefix = params["FAPAR_prefix"]
        aggMonths = params["Agg_months"]
        min = params["Valid_min"]
        max = params["Valid_max"]

        oDateFrom = datetime.datetime.strptime(sDateFrom,"%Y%m")
        oDateTo = datetime.datetime.strptime(sDateTo,"%Y%m")

        # override

        opts, a1Args = getopt.getopt(argv,"ht:e:",["help","timerange=","dateend="])

        monthsBefore = None
        dateend = None

        for opt, arg in opts:
            if opt in ('-h',"--help"):
                displayHelp();
                sys.exit()
            elif opt in ("-t", "--timerange"):
                monthsBefore = arg
            elif opt in ("-e", "--dateend"):
                dateend = arg

        if dateend != None:
            oDateTo = datetime.datetime.strptime(dateend,"%Y%m")
        else:
            oDateTo = datetime.datetime.strptime(sDateTo,"%Y%m")

        if monthsBefore !=None:
            oDateFrom = oDateTo +relativedelta(months=-int(monthsBefore))
        else:
            oDateFrom = datetime.datetime.strptime(sDateFrom,"%Y%m")


        while (oDateFrom <= oDateTo):

            year=oDateFrom.strftime("%Y")
            month=oDateFrom.strftime("%m")

            for nmonths in aggMonths:

                fileName = os.path.join(indexFold,indexPrefix+"-Monthly-Files", "{:d}-Month-Files".format(nmonths),
                                        FAPARPrefix + "_" + oDateFrom.strftime("%Y%m")+".tif")

                if os.path.exists(fileName):

                    # compute fapar result
                    calculate_fapar_result(FAPARPrefix, FAPARmonthlyFold, FAPARstatsFold,
                             indexPrefix, indexFold, year, month, nmonths, min, max)

            oDateFrom = oDateFrom +relativedelta(months=+1)

if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
