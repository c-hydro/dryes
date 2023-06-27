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



def calculate_vhi_result(fileNameLST,fileNameNDVI, NDVIPrefix, LSTPrefix, NDVIstatsFold, LSTstatsFold,
                         indexPrefix, destFold, year, month, monthCount):

    dest_dir = os.path.join(destFold, year, month)
    os.system("mkdir -p "+dest_dir)

    LST, col, row, geoTrans, geoproj = readGeotiff (fileNameLST)

    NDVI, col, row, geoTrans, geoproj = readGeotiff (fileNameNDVI)

    LSTstatsFile  = os.path.join(LSTstatsFold,"Statistics-"+LSTPrefix+"-{:02d}months-Month".format(monthCount)+month+".tif")

    NDVIstatsFile  = os.path.join(NDVIstatsFold,"Statistics-"+NDVIPrefix+"-{:02d}months-Month".format(monthCount)+month+".tif")


    data3dLST, col, row, geoTrans, geoproj = readGeotiff (LSTstatsFile)

    LSTmin = data3dLST[:,:,0]
    LSTmax = data3dLST[:,:,1]
    LSTmean = data3dLST[:,:,2]

    data3dNDVI, col, row, geoTrans, geoproj = readGeotiff (NDVIstatsFile)

    NDVImin = data3dNDVI[:,:,0]
    NDVImax = data3dNDVI[:,:,1]

    mask, col, row, geoTrans, geoproj=  readGeotiff("mask_bolivia.tif")

    rangeNDVI = (NDVImax - NDVImin)
    rangeNDVI[rangeNDVI==0]=np.nan
    rangeLST = (LSTmax - LSTmin)
    rangeLST[rangeNDVI==0]=np.nan

    VCI = (NDVI - NDVImin) / rangeNDVI
    TCI = (LSTmax - LST) / rangeLST

    TempAnom = (LST - LSTmean)* mask

    VHI = 0.5 * ( VCI + TCI )* mask

    """
    simple_plot_index(VCI, "VCI{:02d}_".format(monthCount)+year+month )
    simple_plot_index(TCI, "TCI{:02d}_".format(monthCount)+year+month )
    simple_plot_index(VHI, "VCI{:02d}_".format(monthCount)+year+month )
    """

    # create geotiff for spei output
    VCIFileName = os.path.join(dest_dir, "VCI{:02d}".format(monthCount)+"-MODIS_" +year+month +".tif")
    TCIFileName = os.path.join(dest_dir, "TCI{:02d}".format(monthCount)+"-MODIS_" +year+month +".tif")
    VHIFileName = os.path.join(dest_dir, indexPrefix+"{:02d}".format(monthCount)+"-MODIS_" +year+month +".tif")


    writeGeotiffSingleBand (VCIFileName, geoTrans, geoproj,VCI, nodata=np.nan, BandName="VCI")
    writeGeotiffSingleBand (TCIFileName, geoTrans, geoproj,TCI, nodata=np.nan, BandName="TCI")
    writeGeotiffSingleBand (VHIFileName, geoTrans, geoproj,VHI, nodata=np.nan, BandName="VHI")

    products_dir = Path(destFold).parent

    TempAnom_dir = (os.path.join(products_dir,"TempAnom",year, month))
    os.system("mkdir -p "+TempAnom_dir)

    TempAnomFileName = os.path.join(TempAnom_dir, "TempAnom{:02d}".format(monthCount)+"-MODIS_" +year+month +".tif")

    writeGeotiffSingleBand (TempAnomFileName, geoTrans, geoproj,TempAnom, nodata=np.nan, BandName="VHI")

    print(VHIFileName)


    """


    # adjust folders and filenames
    desty = os.path.join(destFold, "{:04d}".format(year))
    if not(os.path.isdir(desty)):
        os.mkdir(desty)
    destm = os.path.join(desty, "{:02d}".format(month))
    if not(os.path.isdir(destm)):
        os.mkdir(destm)
    destFileName = os.path.join(destm, indexPrefix + "-{:02d}months".format(monthCount) + "_{:04d}_{:02d}.tif"
                                .format(year, month))
    NDVIinpFileName = os.path.join(NDVImonthlyFold, "{:d} Month Files".format(monthCount), NDVIPrefix +
                                   "_{:04d}_{:02d}_01.tif".format(year, month))
    LSTinpFileName = os.path.join(LSTmonthlyFold, "{:d} Month Files".format(monthCount), LSTPrefix +
                                  "_{:04d}_{:02d}_01.tif".format(year, month))
    NDVIstaFileName = os.path.join(NDVIstatsFold, "NDVI-extremes-{:02d}months-Month{:02d}.tif"
                                   .format(monthCount, month))
    LSTstaFileName = os.path.join(LSTstatsFold, "LST-extremes-{:02d}months-Month{:02d}.tif"
                                  .format(monthCount, month))

    # calculate VHI Index
    NDVIstaDataSet = gdal.Open(NDVIstaFileName)
    geoTrans = NDVIstaDataSet.GetGeoTransform()
    col, row = NDVIstaDataSet.GetRasterBand(1).ReadAsArray().shape
    LSTstaDataSet = gdal.Open(LSTstaFileName)
    NDVImin = NDVIstaDataSet.GetRasterBand(1).ReadAsArray()
    NDVImax = NDVIstaDataSet.GetRasterBand(2).ReadAsArray()
    LSTmin = LSTstaDataSet.GetRasterBand(1).ReadAsArray()
    LSTmax = LSTstaDataSet.GetRasterBand(2).ReadAsArray()

    NDVIdataSet = gdal.Open(NDVIinpFileName)
    NDVIdataArr = NDVIdataSet.GetRasterBand(1).ReadAsArray()
    LSTdataSet = gdal.Open(LSTinpFileName)
    LSTdataArr = LSTdataSet.GetRasterBand(1).ReadAsArray()

    vhiArr = []
    for i in range(col):
        vhiTemp = []
        for j in range(row):
            # NDVIdataArr[i][j] *= 0.0001
            # LSTdataArr[i][j] *= 0.02
            vhiVal = 0.5 * (NDVIdataArr[i][j] - NDVImin[i][j]) / (NDVImax[i][j] - NDVImin[i][j])
            vhiVal -= 0.5 * (LSTdataArr[i][j] - LSTmax[i][j]) / (LSTmax[i][j] - LSTmin[i][j])
            vhiTemp.append(vhiVal)
        vhiArr.append(vhiTemp)

    # create geotiff for vhi output
    vhiDataSet = gdal.GetDriverByName('GTiff').Create(destFileName, row, col, 1, gdal.GDT_Float32)
    vhiDataSet.SetGeoTransform(geoTrans)
    vhiDataSet.GetRasterBand(1).WriteArray(numpy.asarray(vhiArr))
    vhiDataSet.FlushCache()
    speiDataSet = None
    print(destFileName)
    """

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

                fileNameNDVI = os.path.join(NDVImonthlyFold,str(monthCount)+"-Month-Files".format(monthCount),NDVIPrefix+"_"+year+month+".tif")

                if os.path.exists(fileNameLST):

                    if os.path.exists(fileNameNDVI):

                        calculate_vhi_result(fileNameLST,fileNameNDVI,NDVIPrefix, LSTPrefix, NDVIstatsFold, LSTstatsFold,
                             indexPrefix, indexFold, year, month, monthCount)

                    else:
                        print ("Missing file : " + fileNameNDVI)
                else:
                    print ("Missing file : " + fileNameLST)

            oDateFrom = oDateFrom +relativedelta(months=+1)


if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
