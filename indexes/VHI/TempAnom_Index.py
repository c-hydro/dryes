import os, sys, getopt
import json
import datetime
from dateutil.relativedelta import *
from geotiff import *
import numpy as np
from pathlib import Path
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



def calculate_vhi_result(fileNameLST, LSTPrefix, LSTstatsFold,
                         destFold, year, month, monthCount,maskFile):

    dest_dir = os.path.join(destFold, year, month)

    LST, col, row, geoTrans, geoproj = readGeotiff (fileNameLST)


    LSTstatsFile  = os.path.join(LSTstatsFold,"Statistics-"+LSTPrefix+"-{:02d}months-Month".format(monthCount)+month+".tif")

    data3dLST, col, row, geoTrans, geoproj = readGeotiff (LSTstatsFile)

    LSTmin = data3dLST[:,:,0]
    LSTmax = data3dLST[:,:,1]
    LSTmean = data3dLST[:,:,2]

    mask, col, row, geoTrans, geoproj=  readGeotiff(maskFile)

    TempAnom = (LST - LSTmean)

    products_dir = Path(destFold).parent

    TempAnom_dir = (os.path.join(products_dir,"TempAnom",year, month))
    os.system("mkdir -p "+TempAnom_dir)

    TempAnomFileName = os.path.join(TempAnom_dir, "TempAnom{:02d}".format(monthCount)+"-MODIS_" +year+month +".tif")

    writeGeotiffSingleBand (TempAnomFileName, geoTrans, geoproj,TempAnom, nodata=np.nan, BandName="VHI")

    print(TempAnomFileName)



def main(argv):


    with open('VHI_config.json') as jf:
        params = json.load(jf)

        LSTstatsFold = params["LST_Stats_folder"]
        if LSTstatsFold[-1] == os.path.sep:
            LSTstatsFold = LSTstatsFold[:-1]

        NDVIstatsFold = params["NDVI_Stats_folder"]
        if NDVIstatsFold[-1] == os.path.sep:
            NDVIstatsFold = NDVIstatsFold[:-1]

        LSTmonthlyFold = params["LST_Monthly_folder"]
        print('-------------------------------------_> '+LSTmonthlyFold)
        if LSTmonthlyFold[-1] == os.path.sep:
            LSTmonthlyFold = LSTmonthlyFold[:-1]

        NDVImonthlyFold = params["NDVI_Monthly_folder"]
        if NDVImonthlyFold[-1] == os.path.sep:
            NDVImonthlyFold = NDVImonthlyFold[:-1]

        indexFold = params["Index_folder"]
        if indexFold[-1] == os.path.sep:
            indexFold = indexFold[:-1]

        indexPrefix = params["Index_prefix"]
        NDVIPrefix = params["NDVI_prefix"]
        LSTPrefix = params["LST_prefix"]

        aggMonths = params["Agg_months"]

        maskFile = params["mask_file"]

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

                fileNameLST = os.path.join(LSTmonthlyFold,str(monthCount)+"-Month-Files".format(monthCount),LSTPrefix+"_"+year+month+".tif")


                if os.path.exists(fileNameLST):

                        calculate_vhi_result(fileNameLST, LSTPrefix, LSTstatsFold,
                              indexFold, year, month, monthCount, maskFile)

                else:
                    print ("Missing file : " + fileNameLST)

            oDateFrom = oDateFrom +relativedelta(months=+1)


if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
