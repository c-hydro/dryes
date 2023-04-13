import os, sys, getopt
import json
from shutil import copyfile
import scipy.stats as stat
import numpy as np
from geotiff import *
import datetime
from dateutil.relativedelta import *
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
    plt.clim(-2.5,2.5)
    # cbar.ax.set_yticklabels(['< -1', '0', 1, 2,'> 10'])# vertically oriented
    # colorbar
    #ax.format_coord = Formatter(im)
    if save_img:
        plt.savefig(os.path.join(save_dir,title+'.png'))
    else:
        plt.show()
def displayHelp():

    print ('\nCompute index')
    print ('Options:')
    print ('          -t | --timerange              months before end of calculation to override .json file')
    print ('          -e | --dateend                calculation end date (format: YYYYMM) to override .json file ')
    print ('          -h | --help                   display this help')


def calculate_result(fileName, statsFileName, destFold, year, month, monthCount, outIndex,PrecNameIndex,maskFile):
    # adjust destination folder
    dest_dir = os.path.join(destFold, year, month)
    os.system("mkdir -p "+dest_dir)

    # first copy file to new folderlo
    newFileName = os.path.join(dest_dir, PrecNameIndex+"-{:02d}_".format(monthCount)+year+month+".tif"
                               .format(monthCount))
    copyfile(fileName, newFileName)

    # estimate probability corresponding to our distribution and calculate ESI Index

    data, col, row, geoTrans, geoproj = readGeotiff (fileName)

    mask, col, row, geoTrans, geoproj=  readGeotiff (maskFile)

    #data , col, row, geoTrans, geoproj = readGeotiff ("GSMAP_20200501.tif")

    #mask = (mask+1)/(mask+1)

    #writeGeotiffSingleBand ("mask_haiti2.tif", geoTrans, geoproj, mask, nodata=np.nan, BandName="prova_mask")

    data = data * mask   # DEBUG !!!

    dataStat, col, row, geoTrans, geoproj = readGeotiff (statsFileName)
    shape = dataStat[:,:,0]
    loc = dataStat[:,:,1]
    scale = dataStat[:,:,2]
    P0 = dataStat[:,:,3]

    probVal = data * np.nan

    for i in range(row):
        for j in range(col):
            if data[i,j]>=0:
                probVal[i,j] = distr.gam.cdf(data[i,j], a=shape[i,j], loc=loc[i,j], scale=scale[i,j])
                probVal[i,j] = (1 - P0[i,j]) * probVal[i,j] + P0[i,j]
            elif data[i,j]<0:
                probVal[i,j] = P0[i,j]/2

    probVal [probVal==0] = 0.0000001
    probVal [probVal==1] = 0.9999999

    SPI = stat.norm.ppf(probVal, loc=0, scale=1)

    #simple_plot_index(SPI,"SPI_"+year+month)
    #simple_plot_index(probVal,"Prob_"+year+month)

    # create geotiff for spei output
    outFileName = os.path.join(dest_dir, outIndex+"{:02d}".format(monthCount)+"-"+PrecNameIndex+"_" +year+month +".tif")
   # ESI[data==1] = np.nan
    writeGeotiffSingleBand (outFileName, geoTrans, geoproj,SPI, nodata=np.nan, BandName="Standardized_SPI")
    print ("saving... " + outFileName)

def main(argv):

    print ('\n'+str(datetime.datetime.now())+' - Process started...\n')

    with open('SPI_config.json') as jf:
        params = json.load(jf)

        statsFold = params["Stats_folder"]
        if statsFold[-1] == os.path.sep:
            statsFold = statsFold[:-1]

        PrecFold = params["Prec_folder"]
        if PrecFold[-1] == os.path.sep:
            PrecFold = PrecFold[:-1]

        indexFold = params["Index_folder"]
        if indexFold[-1] == os.path.sep:
            indexFold = indexFold[:-1]

        PrecNameIndex = params["Prec_prefix"]

        aggMonths = params["Agg_months"]
        sDateFrom = params["StartDate"]
        sDateTo = params["EndDate"]
        indexPrefix= params["Index_prefix"]

        monthlyFolder = os.path.join(indexFold,PrecNameIndex+"-Monthly-Files")

        maskFile = params["mask_file"]

        #override

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

                fileName = os.path.join(monthlyFolder, "{:d}-Month-Files".format(monthCount),
                                        PrecNameIndex + "_" + oDateFrom.strftime("%Y%m")+".tif")

                statsFileName = os.path.join(statsFold,"Statistics-"+PrecNameIndex+"-{:02d}months-Month".format(monthCount))+month+".tif"

                if os.path.exists(fileName):
				
                    calculate_result(fileName, statsFileName, indexFold, year, month, monthCount, indexPrefix,PrecNameIndex,maskFile)

                else:

                    print ("Not enough data to compute: "+fileName)

            oDateFrom = oDateFrom +relativedelta(months=+1)

if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
