import os, sys, getopt
import json
import datetime
import scipy.stats as stat
import numpy as np
from dateutil.relativedelta import *
from calendar import monthrange
from geotiff import *
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

    #allFiles.sort()

    # get dates from file names
    count = len(nameIndex)
    result = []
    for str in allFiles:
        if len(str) > 0:
            for stf in str:
                if nameIndex in stf:
                    #if "200405" in stf:print(stf)
                    date=stf.split('_')[1].split(".")[0]
                    result.append(date)

    result.sort()
    return result


def get_same_values(first, second):
    same = [f for f in first if f in second]
    return same


def diff_date_str(curDate, refDate):
    dateVal = datetime.datetime.strptime(curDate, "%Y%m%d").date()
    refDateVal = datetime.datetime.strptime(refDate, "%Y%m%d").date()
    return (dateVal - refDateVal).days


def convert_to_monthly_data (snowFold, snowPrefix, indexFold, monthCount,indexPrefix):

    dateList = get_all_dates(snowFold, snowPrefix)
    dateList.sort()

    dateListMonths = [i[0:6] for i in dateList ]

    oDateFrom = datetime.datetime.strptime(dateList[0],"%Y%m%d000000")
    oDateTo =   datetime.datetime.strptime(dateList[-1],"%Y%m%d000000")

    #in case month was not completed
    #oDateTo = oDateTo.replace(day=1) + relativedelta(months=+1) + relativedelta(days=-1)
    oDateTo = oDateTo.replace(day=1) + relativedelta(days=-1)

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

                calculate_monthly_average(indexFold, snowFold, snowPrefix, dateInMonth, oDate.strftime("%Y"), oDate.strftime("%m"), indexPrefix, monthCount)

            #dateListMonths.append(oDate.strftime("%Y%m"))
        else:
            print ("Missing data for : "+ oDate.strftime("%Y%m"))

        oDate = oDate +relativedelta(months=+1)


def calculate_monthly_average(indexFold, PETfold, PETprefix, dateInMonth, year, month,indexPrefix,monthCount):

    maskSnow , col, row, geoTrans, geoproj = readGeotiff ("mask_bolivia.tif")

    #maskSnow [maskSnow==0] =np.nan

    data3d = np.zeros((row, col, len (dateInMonth)))

    for i, key in enumerate(dateInMonth):
        #daFold = os.path.join(foldAdd, key[0:4], key[4:6], key[6:8])
        fileName = os.path.join(PETfold, key[0:4], key[4:6], key[6:8],PETprefix + "_" + key + ".tif")
        data , col, row, geoTrans, geoproj = readGeotiff (fileName)
        #simple_plot_index(data,key)
        data3d [:,:,i] = data

    data_mean = np.nanmean(data3d,axis=2)

    outFileNamePET= os.path.join(indexFold, year, month, PETprefix +"{:02d}_".format(monthCount)+year+month+".tif")

    data_mean [data_mean>1]=np.nan


    os.system("mkdir -p "+os.path.join(indexFold, year,month))

    #simple_plot_index(data_mean,"NSDI  mean")

    print (outFileNamePET)

    writeGeotiff(outFileNamePET, geoTrans, geoproj, data_mean,nodata=np.nan, BandNames="Soil_moisture",globalDescr="NDSI500m_monthly")


def create_statistics(indexFold, statFold, geoFold, indexPrefix, monthCount,PETprefix):
    dateList = get_all_dates(indexFold, PETprefix+"{:02d}".format(monthCount))
    dateList.sort()
#    newFold = os.path.join(mainFoldAdd, "Normal PDF")
    if not(os.path.isdir(statFold)):
        os.system ("mkdir -p "+statFold)

    for month in range(1, 13):
        fileNames = []
        for date in dateList:
            yy = date[0:4]
            mm = date[4:6]
            if int(mm) == month:
                fileNames.append(os.path.join(indexFold,yy,mm,PETprefix+"{:02d}_".format(monthCount) + date + ".tif"))

        compute_statistics (indexFold, fileNames, statFold, geoFold, monthCount, month,indexPrefix)


def compute_statistics(indexFold, fileNames, statFold, geoFold, monthCount, month, indexPrefix):
    print("Create statistics GeoTiff for {:d} months from month {:02d}".
          format(monthCount, month))

    maskFC, col, row, geoTrans, geoproj = readGeotiff(os.path.join(geoFold, "geo_field_capacity.tif"))
    maskWP, col, row, geoTrans, geoproj = readGeotiff(os.path.join(geoFold,"geo_wilting_point.tif"))
    maskSnow , col, row, geoTrans, geoproj = readGeotiff ("mask_bolivia.tif")

    data3d = np.zeros((row,col,len(fileNames))) * np.nan

    for i, f in enumerate(fileNames):

        data , col, row, geoTrans, geoproj = readGeotiff (os.path.join(indexFold,f))

        data3d [:,:,i]= (data-maskFC)/(maskFC-maskWP)

        #simple_plot_index(data)

    stats =np.zeros((row,col,2))*np.nan

    nameParams  = [None] * 2

    #invalid_pixels = 0

    data_mean = np.nanmean(data3d, axis=2, dtype=np.float32)
    data_std = np.nanstd(data3d, axis=2, dtype=np.float32)

    stats[:,:,0] = data_mean
    stats[:,:,1] = data_std

    nameParams [0]= "mean"
    nameParams [1]= "std"

    name = "Statistics-"+ indexPrefix +"-{:02d}months-Month{:02d}.tif".format(monthCount, month)
    fileStatsOut = os.path.join(statFold, name)

    writeGeotiff(fileStatsOut,geoTrans, geoproj,stats, nodata=np.nan, BandNames=nameParams
                 ,globalDescr = "NDSI_distr_param_a_b_loc_scale_P0")


def main(argv):


    with open('SWDI_HSAF_config.json') as jf:
        params = json.load(jf)

        # get PET tif files
        soilmoistureFold = params["SoilMoisture_folder"]
        if soilmoistureFold[-1] == os.path.sep:
            soilmoistureFold = soilmoistureFold[:-1]

        soilmoisturePrefix = params["SoilMoisture_prefix"]

        indexFold = params["Index_folder"]
        if indexFold[-1] == os.path.sep:
            indexFold = indexFold[:-1]

        indexPrefix = params["Index_prefix"]

        statFold = params["Stats_folder"]
        if statFold[-1] == os.path.sep:
            statFold = statFold[:-1]

        geoFold = params["Geo_folder"]
        if geoFold[-1] == os.path.sep:
            geoFold = geoFold[:-1]

        aggMonths = params["Agg_months"]

        stats = True

        opts, a1Args = getopt.getopt(argv,"hn",["help","nostats"])

        for opt, arg in opts:
            if opt in ("-n", "--nostats"):
                stats = False

        parent = os.path.join(soilmoistureFold, os.pardir)

        soilmoistureParent = os.path.abspath(parent)

        """
        dateList = get_all_dates(soilmoistureParent, soilmoisturePrefix)

        dateList.sort()
        """
        sDateFrom = "20140101"  #dateList[0]
        sDateTo =   datetime.datetime.now().strftime("%Y%m%d")


        for monthCount in aggMonths:

                convert_to_monthly_data(soilmoistureFold, soilmoisturePrefix, indexFold, monthCount,indexPrefix)

                if stats:
                    create_statistics(indexFold, statFold, geoFold, indexPrefix, monthCount, soilmoisturePrefix)


if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
