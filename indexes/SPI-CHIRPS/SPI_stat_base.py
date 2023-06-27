import os, sys, getopt
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

def displayHelp():

    print ('\nOptions available:')
    print ('          -n | --nostats                do not update statistics ')
    print ('          -h | --help                   display this help')

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

    data3d[data3d<1]=0
    data_mean = np.nanmean(data3d,axis=2)
    BandName="Prec_accumulation"

    os.system("mkdir -p "+os.path.join(indexFold,str(monthCount)+"-Month-Files"))

    outFileName= os.path.join(indexFold,str(monthCount)+"-Month-Files", VARprefix + "_" +year+month+".tif".format(year, month))

    #simple_plot_index(data_mean)
    print (outFileName)

    writeGeotiffSingleBand(outFileName, geoTrans, geoproj, data_mean,nodata=np.nan, BandName=BandName)



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

            max_distance, p_value = stat.kstest(dpd,"gamma",args=fit)
            if p_value < 0.5:
                invalid_pixels += 1

                continue
                #print("Kolmogorov-Smirnov test for goodness of fit: "+str(round(p_value*100))+"%, max distance: "+str(max_distance))

            distr_param[i,j,0] = fit[0]
            distr_param[i,j,1] = fit[1]
            distr_param[i,j,2] = fit[2]
            distr_param[i,j,3] = len(zeros) / len(d1d)    # zero probability

    print ("Invalid pixel: " + str(round(invalid_pixels/(row*col)*100))+"%")
    name = "Statistics-"+ nameIndex +"-{:02d}months-Month{:02d}.tif".format(monthCount, month)
    fileStatsOut = os.path.join(newFold, name)

    writeGeotiff(fileStatsOut,geoTrans, geoproj,distr_param, nodata=np.nan, BandNames=list(fit_dict.keys()).append("P0")
                 ,globalDescr = "SPI_distr_param_c_loc_scale_P0")



def main(argv):
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

        statsFold = params["Stats_folder"]

        aggMonths = params["Agg_months"]

        PrecmonthlyFold = os.path.join(IndexFold,PrecNameIndex+"-Monthly-Files")

        stats = True

        opts, a1Args = getopt.getopt(argv,"hn",["help","nostats"])

        for opt, arg in opts:
            if opt in ("-n", "--nostats"):
                stats = False


        for nmonths in aggMonths:

            # monthly averages

            convert_to_monthly_data(PrecFold, PrecNameIndex, PrecmonthlyFold, nmonths, maskFile)

            monthsFold = os.path.join(PrecmonthlyFold, "{:d}-Month-Files".format(nmonths))

            if stats:

                create_statistics(monthsFold, statsFold, PrecNameIndex, nmonths, maskFile)

if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass