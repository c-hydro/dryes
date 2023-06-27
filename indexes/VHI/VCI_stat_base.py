import os
import json
import datetime
#from shutil import  rmtree
import numpy as np
from geotiff import *
from dateutil.relativedelta import *


def get_all_dates(foldAdd, nameIndex):
    # remove path separator from end of folder address
    if foldAdd[-1] == os.path.sep:
        foldAdd = foldAdd[:-1]

    # get list of all files
    allFiles = [x[2] for x in os.walk(foldAdd)]

    # get dates from file names
    count = len(nameIndex)
    result = []
    for str in allFiles:
        if len(str) > 0:
            for stf in str:
                if nameIndex in stf:
                    date=stf.split('_')[1].split(".")[0]
                    #date = stf[count+1:count+9]
                     # print(date)
                    result.append(date)

    result.sort()
    return result


def get_date(dateStr):
    year = int(dateStr[0:4])
    month = int(dateStr[4:6])
    return year, month


def diff_date_str(curDate, refDate):
    dateVal = datetime.datetime.strptime(curDate, "%Y%m%d").date()
    refDateVal = datetime.datetime.strptime(refDate, "%Y%m%d").date()
    return (dateVal - refDateVal).days


def convert_to_monthly_data (VARfold, VARprefix, indexFold, monthCount, maskFile):

    dateList = get_all_dates(VARfold, VARprefix)
    dateList.sort()

    dateListMonths = [i[0:6] for i in dateList ]

    oDateFrom = datetime.datetime.strptime(dateList[0],"%Y%m%d")
    oDateTo =   datetime.datetime.strptime(dateList[-1],"%Y%m%d")

    #in case month was not completed
    oDateTo = oDateTo.replace(day=1) + relativedelta(months=+1) + relativedelta(days=-1)

    print(oDateFrom.strftime("%Y%m%d"))
    print(oDateTo.strftime("%Y%m%d"))

    #oDateFrom = oDateFrom + relativedelta(months=monthCount)
    oDate = oDateFrom
    while (oDate <= oDateTo):

        if oDate < oDateFrom + relativedelta(months=monthCount-1):
            # remove the first monthCount from dateList
            #dateList = [s for s in dateList if oDate.strftime("%Y%m") not in s]
            oDate = oDate +relativedelta(months=+1)
            continue

        if oDate.strftime("%Y%m")  in dateListMonths:

            dateInMonth = []

            for m in range(monthCount):

                for date in dateList:

                    oDateAggr=oDate + relativedelta(months = -m)

                    year = oDateAggr.strftime("%Y")
                    month = oDateAggr.strftime("%m")

                    if  year+month in date[0:6]:

                        dateInMonth.append(date)

            if len(dateInMonth)>0:

                calculate_monthly_average(indexFold, VARfold, VARprefix, dateInMonth, oDate.strftime("%Y"), oDate.strftime("%m"), monthCount, maskFile)

            #dateListMonths.append(oDate.strftime("%Y%m"))
        else:
            print ("Missing data for : "+ oDate.strftime("%Y%m"))

        oDate = oDate +relativedelta(months=+1)


def calculate_monthly_average(indexFold, VARfold, VARprefix, dateInMonth, year, month, monthCount, maskFile):

    mask , col, row, geoTrans, geoproj = readGeotiff (maskFile)

    data3d = np.zeros((row, col, len (dateInMonth)))

    for i, key in enumerate(dateInMonth):
        #daFold = os.path.join(foldAdd, key[0:4], key[4:6], key[6:8])
        fileName = os.path.join(VARfold, key[0:4], key[4:6], key[6:8],VARprefix + "_" + key + ".tif")
        data , col, row, geoTrans, geoproj = readGeotiff (fileName)
        #simple_plot_index(data,key)
        data3d [:,:,i] = data

    data_mean = np.nanmean(data3d,axis=2)

    BandName=VARprefix+"montly_mean"

    os.system("mkdir -p "+os.path.join(indexFold,str(monthCount)+"-Month-Files"))

    outFileName= os.path.join(indexFold,str(monthCount)+"-Month-Files", VARprefix + "_" +year+month+".tif".format(year, month))

    #simple_plot_index(data_mean)
    print (outFileName)

    writeGeotiffSingleBand(outFileName, geoTrans, geoproj, data_mean,nodata=np.nan, BandName=BandName)


def compute_extremes_inrange(foldAdd, fileNames, newFold, monthCount, month, vmin, vmax,indexPrefix):
    print("Create GeoTiff for {:d} months ending month {:02d}".
          format(monthCount, month))

    mask, col, row, geoTrans, geoproj=  readGeotiff("mask_bolivia.tif")

    data3d = np.zeros((row,col,len(fileNames)))

    for i, f in enumerate(fileNames):

        data , col, row, geoTrans, geoproj = readGeotiff (os.path.join(foldAdd,f))

        data3d [:,:,i]= data

    stats_param =np.zeros((row,col,2))* np.nan


    dataMin= np.nanmin(data3d,axis=2) * mask

    dataMin[dataMin<vmin]=np.nan

    dataMax= np.nanmax(data3d,axis=2) * mask

    dataMax[dataMax>vmax]=np.nan

    stats_param [:,:,0] = dataMin

    stats_param [:,:,1] = dataMax

    bandDescr = ["min","max"]

    name = "Statistics-"+ indexPrefix +"-{:02d}months-Month{:02d}.tif".format(monthCount, month)
    name = os.path.join(newFold, name)

    writeGeotiff(name, geoTrans, geoproj, stats_param,nodata=np.nan, BandNames=bandDescr, globalDescr="Min_Max_values")

def create_extremes_inrange(mainFoldAdd, newFold, nameIndex, monthCount, vmin, vmax,indexPrefix):
    dateList = get_all_dates(mainFoldAdd, nameIndex)
    dateList.sort()

#    newFold = os.path.join(mainFoldAdd, "Normal PDF")
    if not(os.path.isdir(newFold)):
        os.mkdir(newFold)

    for month in range(1, 13):
        fileNames = []
        for str in dateList:
            yy, mm = get_date(str)
            if mm == month:
                fileNames.append(nameIndex + "_" + str + ".tif")
        compute_extremes_inrange(mainFoldAdd, fileNames, newFold, monthCount, month, vmin, vmax,indexPrefix)


def main():
    with open('VHI_config.json') as jf:
        params = json.load(jf)

        # get NDVIFold tif files
        NDVIFold = params["NDVI_folder"]
        if NDVIFold[-1] == os.path.sep:
            NDVIFold = NDVIFold[:-1]
        NDVINameIndex = params["NDVI_prefix"]
        # NDVIDates = get_all_dates(NDVIFold, NDVINameIndex)

        monthlyFold = params["NDVI_Monthly_folder"]
        statsFold = params["NDVI_Stats_folder"]
        vmin = params["NDVI_valid_min"]
        vmax = params["NDVI_valid_max"]
        aggMonths = params["Agg_months"]
        maskFile = params["mask_file"]

        for nmonths in aggMonths:
        # monthly averages
            convert_to_monthly_data(NDVIFold, NDVINameIndex, monthlyFold, nmonths, maskFile)

        # compute min and max in given range
            monthsFold = os.path.join(monthlyFold, "{:d}-Month-Files".format(nmonths))
            create_extremes_inrange(monthsFold, statsFold, NDVINameIndex, nmonths, vmin, vmax,NDVINameIndex)





if __name__ == '__main__':
    main()
else:
    pass