import os, sys, getopt
import json
import scipy.stats as stat
import numpy as np
from geotiff import *
import datetime
from dateutil.relativedelta import *
from SWDI_HSAF_stat_base import calculate_monthly_average, get_all_dates

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
    plt.clim(-3,3)
    # cbar.ax.set_yticklabels(['< -1', '0', 1, 2,'> 10'])# vertically oriented
    # colorbar
    #ax.format_coord = Formatter(im)
    if save_img:
        plt.savefig(os.path.join(save_dir,title+'.png'))
    else:
        plt.show()


def displayHelp():

    print ('\nCompute SWDI index')
    print ('Options:')
    print ('          -t | --timerange              months before end of calculation to override .json file')
    print ('          -e | --dateend                calculation end date (format: YYYYMM) to override .json file ')
    print ('          -h | --help                   display this help')


def calculate_result(fileName, statsFileName, indexFold, geoFold, year, month, monthCount, outIndex,sourcePrefix):
    # adjust destination folder
    dest_dir = os.path.join(indexFold, year, month)
    os.system("mkdir -p "+dest_dir)

    maskFC, col, row, geoTrans, geoproj = readGeotiff(os.path.join(geoFold,"geo_field_capacity.tif"))
    maskWP, col, row, geoTrans, geoproj = readGeotiff(os.path.join(geoFold,"geo_wilting_point.tif"))
    data, col, row, geoTrans, geoproj = readGeotiff (fileName)
    data_ref = (data-maskFC)/(maskFC-maskWP)

    data3d, col, row, geoTrans, geoproj = readGeotiff (statsFileName)
    mean = data3d[:,:,0]
    std = data3d[:,:,1]

    SWDI = (data_ref - mean) / std

    #simple_plot_index(SSCI,"SSCI_"+year+month)

    outFileName = os.path.join(dest_dir, outIndex+"{:02d}".format(monthCount)+"-"+sourcePrefix+"_" +year+month +".tif")

    writeGeotiffSingleBand (outFileName, geoTrans, geoproj,SWDI, nodata=np.nan, BandName="Standardized_SSCI")

    print ("saving... " + outFileName)

def main(argv):

    with open('SWDI_HSAF_config.json') as jf:
        params = json.load(jf)

        statsFold = params["Stats_folder"]
        if statsFold[-1] == os.path.sep:
            statsFold = statsFold[:-1]

        soilmoistureFold = params["SoilMoisture_folder"]
        if soilmoistureFold[-1] == os.path.sep:
            soilmoistureFold = soilmoistureFold[:-1]

        indexFold = params["Index_folder"]
        if indexFold[-1] == os.path.sep:
            indexFold = indexFold[:-1]

        geoFold = params["Geo_folder"]
        if geoFold[-1] == os.path.sep:
            geoFold = geoFold[:-1]

        indexPrefix= params["Index_prefix"]

        index2Prefix= params["Index2_prefix"]

        soilmoisturePrefix = params["SoilMoisture_prefix"]

        sourcePrefix = params["Source_prefix"]


        aggMonths = params["Agg_months"]

        sDateFrom = params["StartDate"]
        sDateTo = params["EndDate"]


        print ('\n'+str(datetime.datetime.now())+' - Process started...\n')

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

            for monthCount in aggMonths:

                fileName = os.path.join(indexFold,year,month,soilmoisturePrefix+"{:02d}_".format(monthCount) + oDateFrom.strftime("%Y%m")+".tif")

                statsFileName = os.path.join(statsFold,"Statistics-"+indexPrefix+"-{:02d}months-Month".format(monthCount))+month+".tif"

                if os.path.exists(fileName):

                    calculate_result(fileName, statsFileName, indexFold, geoFold, year, month, monthCount, index2Prefix,sourcePrefix)

                else:

                    print ("Not enough data to compute: "+fileName)

            oDateFrom = oDateFrom +relativedelta(months=+1)


if __name__ == '__main__':

    argv=sys.argv[1:]
    main(argv)

else:
    pass
