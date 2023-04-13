import os
import json
import datetime
from shutil import rmtree
import string
import scipy.stats as stat
import numpy as np
from dateutil.relativedelta import *
from calendar import monthrange
from geotiff import *
from lmoments3 import distr

import matplotlib.pylab as plt
import matplotlib as mpl

def simple_plot_index(a,title="", save_img=False,save_dir='.'):
    fig, ax = plt.subplots()
    #cmap='RdBu'
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","orange","yellow","white","pink","violet", "#0f0f0f"])
    plt.title(title)
    im = plt.imshow(a, interpolation='none',cmap=cmap)
    plt.colormaps()
    # cbar = fig.colorbar(cax,ticks=[-1, 0, 1, 2, 10],orientation='vertical')
    cbar = fig.colorbar(im, orientation='vertical')
    #plt.clim(-3,3)
    # cbar.ax.set_yticklabels(['< -1', '0', 1, 2,'> 10'])# vertically oriented
    # colorbar
    #ax.format_coord = Formatter(im)
    if save_img:
        plt.savefig(os.path.join(save_dir,title+'.png'))
    else:
        plt.show()

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
    day = int(dateStr[6:8])
    return year, month, day

def diff_date_str(curDate, refDate):
    dateVal = datetime.datetime.strptime(curDate, "%Y%m%d").date()
    refDateVal = datetime.datetime.strptime(refDate, "%Y%m%d").date()
    return (dateVal - refDateVal).days


def convert_to_monthly_data(PrecFold, PrecNameIndex, PrecmonthlyFold, monthCount, maskFile):
    newFold = os.path.join(PrecmonthlyFold,"{:d}-Month-Files".format(monthCount))
    if os.path.isdir(newFold):
        rmtree(newFold)
    os.system ("mkdir -p " + newFold)
    dateList = get_all_dates(PrecFold, PrecNameIndex)
    dateList.sort()
    #print ([int(s[0:6]) for s in dateList])

    dateListMonths = [i[0:6] for i in dateList ]

    oDateFrom = datetime.datetime.strptime(dateList[0],"%Y%m%d")
    oDateTo =   datetime.datetime.strptime(dateList[-1],"%Y%m%d")

    #oDateFrom = oDateFrom + relativedelta(months=monthCount)
    oDate = oDateFrom
    while (oDate <= oDateTo):

        if oDate < oDateFrom + relativedelta(months=monthCount-1):
            # remove the first monthCount from dateList
            #dateList = [s for s in dateList if oDate.strftime("%Y%m") not in s]
            oDate = oDate +relativedelta(months=+1)
            continue
        year = int(oDate.strftime("%Y"))
        month = int(oDate.strftime("%m"))

        if oDate.strftime("%Y%m")  in dateListMonths:
            convert_to_month_average_data(PrecFold, PrecNameIndex, dateList, newFold, month, year, monthCount,maskFile)
            #dateListMonths.append(oDate.strftime("%Y%m"))
        else:
            print ("Missing data for : "+ oDate.strftime("%Y%m"))

        oDate = oDate +relativedelta(months=+1)


def convert_to_month_average_data(foldAdd, nameIndex, dateList, newFold,
                                  month, year, monthCount,maskFile):
    dateCountDic = dict()
    for i in range(monthCount):
        yy = year
        mm = month - i
        if (mm < 1):
            mm += 12
            yy -= 1
        totalDays = monthrange(yy, mm)[1]

        for dd in range(1, totalDays + 1):
            dateStr = "{:4d}{:02d}{:02d}".format(yy, mm, dd)
            if dateStr in dateList:
                if dateStr in dateCountDic:
                    dateCountDic[dateStr] += 1
                else:
                    dateCountDic[dateStr] = 1
            else:
                dateList.append(dateStr)
                dateList.sort()
                ind = dateList.index(dateStr)
                topDate, bottDate = str(), str()
                if ind == 0:
                    topDate = dateList[1]
                    bottDate = dateList[1]
                elif ind == len(dateList) - 1:
                    topDate = dateList[-2]
                    bottDate = dateList[-2]
                else:
                    topDate = dateList[ind + 1]
                    bottDate = dateList[ind - 1]
                dateList.remove(dateStr)
                refDate = bottDate
                diff = diff_date_str(dateStr, bottDate)
                if diff >= 8:
                    newDiff = diff_date_str(dateStr, topDate)
                    if newDiff < diff:
                        refDate = topDate
                if refDate in dateCountDic:
                    dateCountDic[refDate] += 1
                else:
                    dateCountDic[refDate] = 1
    calculate_weighted_averge_prec(foldAdd, nameIndex, dateCountDic, newFold, year, month, maskFile)
    
def calculate_weighted_averge_prec(foldAdd, nameIndex, dateCountDic, newFold, year, month, maskFile):

    data , col, row, geoTrans, geoproj = readGeotiff (maskFile)


    data3d = np.zeros((row, col, len (dateCountDic)))
    datasum = np.zeros((row, col ))
    for i, key in enumerate(dateCountDic):

        daFold = os.path.join(foldAdd, key[0:4], key[4:6], key[6:8])

        fileName = os.path.join(daFold, nameIndex + "_" + key + ".tif")

        data , col, row, geoTrans, geoproj = readGeotiff (fileName)

        #simple_plot_index(data)
        data3d [:,:,i] = data

    data3d[data3d<1]=0
    data_mean = np.nanmean(data3d,axis=2)

    outFileName= os.path.join(newFold, nameIndex + "_" +"{:4d}{:02d}.tif".format(year, month))

    #simple_plot_index(data_mean)
    print (outFileName)
    writeGeotiffSingleBand(outFileName, geoTrans, geoproj, data_mean,nodata=np.nan, BandName="Prec_accumulation")


def create_statistics(mainFoldAdd, newFold, nameIndex, monthCount, maskFile):
    dateList = get_all_dates(mainFoldAdd, nameIndex)
    dateList.sort()
#    newFold = os.path.join(mainFoldAdd, "Normal PDF")
    if not(os.path.isdir(newFold)):
        os.system ("mkdir -p "+newFold)

    for month in range(1, 13):
        fileNames = []
        for str in dateList:
            yy = int(str[0:4])
            mm = int(str[4:6])
            if mm == month:
                fileNames.append(nameIndex + "_" + str + ".tif")
        compute_statistics (mainFoldAdd, fileNames, newFold, monthCount, month,nameIndex, maskFile)


def compute_statistics(mainFoldAdd, fileNames, newFold, monthCount, month,nameIndex, maskFile):
    print("Create statistics GeoTiff for {:d} months from month {:02d}".
          format(monthCount, month))


    mask, col, row, geoTrans, geoproj=  readGeotiff(maskFile)

    #data , col, row, geoTrans, geoproj = readGeotiff ("GSMAP_20200501.tif")



    data3d = np.zeros((row,col,len(fileNames)))

    for i, f in enumerate(fileNames):

        data , col, row, geoTrans, geoproj = readGeotiff (os.path.join(mainFoldAdd,f))


        data3d [:,:,i]= data

    distr_param =np.zeros((row,col,4))*np.nan

    invalid_pixels = 0

    for i in range(row):
        for j in range(col):

            if sum(np.isnan(data3d[i,j,:])) ==len(fileNames):
                continue

            d1d = data3d[i,j,:]
            dpd = d1d[np.where(d1d > 0)]  # only non null values
            zeros = d1d[d1d==0]
            if len(dpd) < 4:
                print ("Not enough valid values in time series, impossible fitting!")
                continue
            
            fit_dict = distr.gam.lmom_fit(dpd)
            fit = (fit_dict['a'],fit_dict['loc'],fit_dict['scale'])

            #max_distance, p_value = stat.kstest(dpd,"gamma",args=fit)
            #if p_value < 0.5:
            #    invalid_pixels += 1

            #    continue
                ##print("Kolmogorov-Smirnov test for goodness of fit: "+str(round(p_value*100))+"%, max distance: "+str(max_distance))

            distr_param[i,j,0] = fit[0]
            distr_param[i,j,1] = fit[1]
            distr_param[i,j,2] = fit[2]
            #distr_param[i,j,3] = 1 - len(dpd) / len(d1d)    # zero probability
            distr_param[i,j,3] = len(zeros) / len(d1d)    # zero probability

    print ("Invalid pixel: " + str(round(invalid_pixels/(row*col)*100))+"%")
    name = "Statistics-"+ nameIndex +"-{:02d}months-Month{:02d}.tif".format(monthCount, month)
    fileStatsOut = os.path.join(newFold, name)

    writeGeotiff(fileStatsOut,geoTrans, geoproj,distr_param, nodata=np.nan, BandNames=list(fit_dict.keys()).append("P0")
                 ,globalDescr = "SPI_distr_param_c_loc_scale_P0")



def main():
    with open('SPI_config.json') as jf:
        params = json.load(jf)

        # get FAPARFold tif files

        PrecFold = params["Prec_folder"]
        if PrecFold[-1] == os.path.sep:
            PrecFold = PrecFold[:-1]

        IndexFold = params["Index_folder"]
        if IndexFold[-1] == os.path.sep:
            IndexFold = PrecFold[:-1]

        PrecNameIndex = params["Prec_prefix"]

        maskFile = params["mask_file"]

       # monthlyFold = params["Monthly_folder"]

        statsFold = params["Stats_folder"]

        aggMonths = params["Agg_months"]

        PrecmonthlyFold = os.path.join(IndexFold,PrecNameIndex+"-Monthly-Files")

        for nmonths in aggMonths:

            # monthly averages

            convert_to_monthly_data(PrecFold, PrecNameIndex, PrecmonthlyFold, nmonths, maskFile)

            monthsFold = os.path.join(PrecmonthlyFold, "{:d}-Month-Files".format(nmonths))

            create_statistics(monthsFold, statsFold, PrecNameIndex, nmonths, maskFile)

if __name__ == '__main__':
    main()
else:
    pass