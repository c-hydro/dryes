import os, sys, getopt
import json
import datetime
from shutil import rmtree
import numpy as np
from dateutil.relativedelta import *
from geotiff import *
from lmoments3 import distr
import scipy.stats as stat

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
    #plt.clim(-1,1)
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


def convert_to_monthly_data (VARfold, VARprefix, indexFold, monthCount):

    dateList = get_all_dates(VARfold, VARprefix)
    dateList.sort()

    dateListMonths = [i[0:6] for i in dateList ]

    oDateFrom = datetime.datetime.strptime(dateList[0],"%Y%m%d")
    oDateTo =   datetime.datetime.strptime(dateList[-1],"%Y%m%d")

    #in case month was not completed
    oDateTo = oDateTo.replace(day=1) + relativedelta(months=+1) + relativedelta(days=-1)
    #oDateTo = oDateTo.replace(day=1) + relativedelta(days=-1)

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
        year = oDate.strftime("%Y")
        month = oDate.strftime("%m")

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

                calculate_monthly_average(indexFold, VARfold, VARprefix, dateInMonth, oDate.strftime("%Y"), oDate.strftime("%m"), monthCount)

            #dateListMonths.append(oDate.strftime("%Y%m"))
        else:
            print ("Missing data for : "+ oDate.strftime("%Y%m"))

        oDate = oDate +relativedelta(months=+1)


def calculate_monthly_average(indexFold, VARfold, VARprefix, dateInMonth, year, month, monthCount):

    maskSnow , col, row, geoTrans, geoproj = readGeotiff ("mask_bolivia.tif")

    #maskSnow [maskSnow==0] =np.nan

    data3d = np.zeros((row, col, len (dateInMonth)))

    for i, key in enumerate(dateInMonth):
        #daFold = os.path.join(foldAdd, key[0:4], key[4:6], key[6:8])
        fileName = os.path.join(VARfold, key[0:4], key[4:6], key[6:8],VARprefix + "_" + key + ".tif")
        data , col, row, geoTrans, geoproj = readGeotiff (fileName)
        #simple_plot_index(data,key)
        data3d [:,:,i] = data

    if VARprefix == "MODIS-PET":
        data_mean = np.nanmean(data3d,axis=2)/8
        BandName="Potential_Evap"
    else:
        data3d[data3d<1]=0 ### to remove noise from PERSIANN
        data_mean = np.nanmean(data3d,axis=2)
        BandName="Prec_accumulation"

    os.system("mkdir -p "+os.path.join(indexFold,str(monthCount)+"-Month-Files"))

    outFileName= os.path.join(indexFold,str(monthCount)+"-Month-Files", VARprefix + "_" +year+month+".tif".format(year, month))

    #simple_plot_index(data_mean)
    print (outFileName)

    writeGeotiffSingleBand(outFileName, geoTrans, geoproj, data_mean,nodata=np.nan, BandName=BandName)


def create_statistics(mainFoldAdd, newFold, nameIndex, monthCount):
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
        compute_statistics (mainFoldAdd, fileNames, newFold, monthCount, month)


def compute_statistics(foldAdd, fileNames, newFold, monthCount, monthStart):
    print("Create statistics GeoTiff for {:d} months from month {:02d}".
          format(monthCount, monthStart))

    mask, col, row, geoTrans, geoproj=  readGeotiff("mask_bolivia.tif")

    data3d = np.zeros((row,col,len(fileNames)))*np.nan

    for i, f in enumerate(fileNames):

        data , col, row, geoTrans, geoproj = readGeotiff (os.path.join(foldAdd,f))
        data3d [:,:,i]= data

    distr_param =np.zeros((row,col,3))*np.nan

    invalid_pixels = 0

    for i in range(row):
        for j in range(col):

            array = data3d[i,j,:]
            array = array[np.logical_not(np.isnan(array))]
            #fit = stat.genextreme.fit(data3d[i,j,:], loc=0, scale = 1)  #loc initial guess
            if len(array) < 4:
                continue
            fit_dict = distr.gev.lmom_fit(data3d[i,j,:])
            fit = (fit_dict['c'],fit_dict['loc'],fit_dict['scale'])
            #print (fit)
            max_distance, p_value = stat.kstest(array,"genextreme",args=fit)
            #print("Kolmogorov-Smirnov test for goodness of fit: "+str(round(p_value*100))+"%, max distance: "+str(max_distance))
            if p_value < 0.3:
                invalid_pixels += 1
                continue
#
            distr_param [i,j,0]  = fit[0]
            distr_param [i,j,1]  = fit[1]
            distr_param [i,j,2]  = fit[2]

    print ("Invalid pixels: " + str(round(invalid_pixels/(row*col)*100))+"%")

    name = "Statistics-Prec-PET-{:02d}months-Month{:02d}.tif".format(monthCount, monthStart)
    name = "Statistics-Prec-PET-{:02d}months-Month{:02d}.tif".format(monthCount, monthStart)
    fileStatsOut = os.path.join(newFold, name)

    writeGeotiff(fileStatsOut,geoTrans, geoproj,distr_param, nodata=np.nan, BandNames=list(fit_dict.keys()),globalDescr = "SPEI_distr_param_c_loc_scale")



def get_same_values(first, second):
    same = [f for f in first if f in second]
    return same

def convertPrec_to_monthly_data (PrecFold, PrecNameIndex, PrecmonthlyFold, nmonths):

    dateListPrec = get_all_dates(PrecFold, PrecNameIndex)

    oDateFrom = datetime.datetime.strptime(dateListPrec[0],"%Y%m%d")
    oDateTo =   datetime.datetime.strptime(dateListPrec[-1],"%Y%m%d")

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
            convert_to_month_average_data(foldAdd, nameIndex, dateList, newFold, month, year, monthCount)
            #dateListMonths.append(oDate.strftime("%Y%m"))
        else:
            print ("Missing data for : "+ oDate.strftime("%Y%m"))

        oDate = oDate +relativedelta(months=+1)

    return dateListMonths

    return dateListPrec

def computePrec_PET (dateListSPEI, Prec_PETmonthlyFold , PETmonthlyFold,  PrecmonthlyFold, PETNameIndex, PrecNameIndex, nmonths):

    for date in dateListSPEI:
        filePET = os.path.join (PETmonthlyFold, "{:d}-Month-Files".format(nmonths), PETNameIndex+ "_" + date + ".tif")
        filePrec = os.path.join (PrecmonthlyFold, "{:d}-Month-Files".format(nmonths), PrecNameIndex+ "_" + date + ".tif")

        PET , col, row, geoTrans, geoproj = readGeotiff (filePET)

        Prec , col, row, geoTrans, geoproj = readGeotiff (filePrec)

        Prec_PET = Prec - PET

        months_file_dir = os.path.join (Prec_PETmonthlyFold, "{:d}-Month-Files".format(nmonths))

        os.system("mkdir -p "+months_file_dir)

        filePrec_PET = os.path.join (months_file_dir, "Prec-PET"+ "_" + date + ".tif")

        writeGeotiffSingleBand (filePrec_PET, geoTrans, geoproj, Prec_PET,nodata=np.nan, BandName="Prec_accumulation")
        print("saving:  "+filePrec_PET)

    return


def main(argv):
    with open('SPEI_config.json') as jf:
        params = json.load(jf)

        # get FAPARFold tif files
        PETFold = params["PET_folder"]
        if PETFold[-1] == os.path.sep:
            PETFold = PETFold[:-1]

        PrecFold = params["Prec_folder"]
        if PrecFold[-1] == os.path.sep:
            PrecFold = PrecFold[:-1]

        IndexFold = params["Index_folder"]
        if PrecFold[-1] == os.path.sep:
            PrecFold = PrecFold[:-1]

        SPEINameIndex = params["Index_prefix"]
        PETNameIndex = params["PET_prefix"]
        PrecNameIndex = params["Prec_prefix"]

        PrecPETNameIndex = params["Prec-PET_prefix"]

        statsFold = params["Stats_folder"]

        aggMonths = params["Agg_months"]

        stats = True

        opts, a1Args = getopt.getopt(argv,"hn",["help","nostats"])

        for opt, arg in opts:
            if opt in ("-n", "--nostats"):
                stats = False

        PETmonthlyFold = os.path.join(IndexFold,PETNameIndex+"-Monthly-Files")
        PrecmonthlyFold = os.path.join(IndexFold,PrecNameIndex+"-Monthly-Files")
        PrecPETmonthlyFold = os.path.join (IndexFold,PrecPETNameIndex+"-Monthly-Files")

        for nmonths in aggMonths:

            # monthly averages

            convert_to_monthly_data(PETFold, PETNameIndex, PETmonthlyFold, nmonths)

            convert_to_monthly_data(PrecFold, PrecNameIndex, PrecmonthlyFold, nmonths)

            dateListPET = get_all_dates(os.path.join(PETmonthlyFold,str(nmonths)+"-Month-Files"),PETNameIndex)

            dateListPrec = get_all_dates(os.path.join(PrecmonthlyFold,str(nmonths)+"-Month-Files"),PrecNameIndex)

            dateListSPEI = get_same_values(dateListPET,dateListPrec)

            computePrec_PET (dateListSPEI, PrecPETmonthlyFold , PETmonthlyFold,  PrecmonthlyFold, PETNameIndex, PrecNameIndex, nmonths)

            monthsFold = os.path.join(PrecPETmonthlyFold, "{:d}-Month-Files".format(nmonths))

            if stats:

            	create_statistics(monthsFold, statsFold, PrecPETNameIndex, nmonths)

if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
