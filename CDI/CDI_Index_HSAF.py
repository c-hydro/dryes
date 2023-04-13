__author__ = 'lauro'

from geotiff import *
import datetime
import json
import os, sys, getopt
from dateutil.relativedelta import *
import matplotlib.pylab as plt
import matplotlib as mpl
#from compressed_pickle import save, load
import numpy as np


def simple_plot_index(a,title="", min = -3,  max= 3,save_img=False,save_dir='.'):

    """simple plot of a matrix"""

    fig, ax = plt.subplots()
    #cmap='RdBu'
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","orange","yellow","white","pink","violet", "#0f0f0f"])
    plt.title(title)
    im = plt.imshow(a, interpolation='none',cmap=cmap)
    plt.colormaps()
    # cbar = fig.colorbar(cax,ticks=[-1, 0, 1, 2, 10],orientation='vertical')
    cbar = fig.colorbar(im, orientation='vertical')
    plt.clim(min,max)
    # cbar.ax.set_yticklabels(['< -1', '0', 1, 2,'> 10'])# vertically oriented
    # colorbar
    #ax.format_coord = Formatter(im)
    if save_img:
        os.system("mkdir -p "+save_dir)
        plt.savefig(os.path.join(save_dir,title+'.png'))
    else:
        plt.show()


def displayHelp():

    """display help"""

    print ('\nCompute index')
    print ('Options:')
    print ('          -t | --timerange              months before end of calculation to override .json file')
    print ('          -e | --dateend                calculation end date (format: YYYYMM) to override .json file ')
    print ('          -h | --help                   display this help')
    print ('          -j | --json                   .json parameter file')


def main (argv):

    print('\n' + str(datetime.datetime.now()) + ' - Process started.\n')

    opts, a1Args = getopt.getopt(argv,"ht:e:j:",["help","timerange=","dateend=","json="])

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
            elif opt in ("-j", "--json"):
                file_json = arg

    with open(file_json) as jf:

        params = json.load(jf)
        
        INDEX_METEO01= params["INDEX_METEO01"]
        accumulation_METEO01= params["accumulation_METEO01"]
        METEO_threshold01= params["METEO_threshold01"]

        INDEX_METEO03= params["INDEX_METEO03"]
        accumulation_METEO03= params["accumulation_METEO03"]
        METEO_threshold03= params["METEO_threshold03"]

        INDEX_HUM= params["INDEX_HUM"]
        accumulation_HUM= params["accumulation_HUM"]
        HUM_threshold= params["HUM_threshold"]

        INDEX_VEG= params["INDEX_VEG"]
        accumulation_VEG= params["accumulation_VEG"]
        VEG_threshold= params["VEG_threshold"]

        products_folder = params["Products_folder"]
        
        sm_source = params["Soilmoisture_source"]

        sDateFrom = params["StartDate"]
        sDateTo = params["EndDate"]

        if dateend != None:
            oDateTo = datetime.datetime.strptime(dateend,"%Y%m")
        else:
            oDateTo = datetime.datetime.strptime(sDateTo,"%Y%m")

        if monthsBefore !=None:
            oDateFrom = oDateTo +relativedelta(months=-int(monthsBefore))
        else:
            oDateFrom = datetime.datetime.strptime(sDateFrom,"%Y%m")

        compute_CDI (oDateFrom,oDateTo,products_folder, sm_source,
                     INDEX_METEO01, accumulation_METEO01, METEO_threshold01,
                     INDEX_METEO03, accumulation_METEO03, METEO_threshold03,
                     INDEX_HUM,     accumulation_HUM,     HUM_threshold,
                     INDEX_VEG,     accumulation_VEG,     VEG_threshold)

    print('\n' + str(datetime.datetime.now()) + ' - Process ended.\n')


def compute_CDI (oDateFrom,oDateTo,products_folder, sm_source,
                     INDEX_METEO01, accumulation_METEO01, METEO_threshold01,
                     INDEX_METEO03, accumulation_METEO03, METEO_threshold03,
                     INDEX_HUM,     accumulation_HUM,     HUM_threshold,
                     INDEX_VEG,     accumulation_VEG,     VEG_threshold):

    """ compute CDI and save into products folder """

    while (oDateFrom <= oDateTo):

        print (oDateFrom)

        ######## METEO 01

        if INDEX_METEO01=="SPI":
            METEO_prefix01= INDEX_METEO01 + accumulation_METEO01 + "-PERSIANN_"
        elif INDEX_METEO01=="SPEI":
            METEO_prefix01= INDEX_METEO01 + accumulation_METEO01 + "-PERSIANN-MODIS_"

        METEO_file01= os.path.join(products_folder,INDEX_METEO01,oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"),METEO_prefix01+oDateFrom.strftime("%Y%m")+".tif")

        METEO01, xsize, ysize, geotransform, geoproj = readGeotiff(METEO_file01)


        ######## METEO 03

        if INDEX_METEO03=="SPI":
            METEO_prefix03= INDEX_METEO03 + accumulation_METEO03 + "-PERSIANN_"
        elif INDEX_METEO03=="SPEI":
            METEO_prefix03= INDEX_METEO03 + accumulation_METEO03 + "-PERSIANN-MODIS_"

        METEO_file03= os.path.join(products_folder,INDEX_METEO03,oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"),METEO_prefix03+oDateFrom.strftime("%Y%m")+".tif")

        METEO03, xsize, ysize, geotransform, geoproj = readGeotiff(METEO_file03)

        ######## SOIL MOISTURE

        prefix_HUM= INDEX_HUM + accumulation_HUM + "-" + sm_source + "_"

        HUM_file= os.path.join(products_folder,INDEX_HUM,oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"),prefix_HUM+oDateFrom.strftime("%Y%m")+".tif")

        print(HUM_file)

        HUM, xsize, ysize, geotransform, geoproj = readGeotiff(HUM_file)


        ######## VEGETATION

        prefix_VEG= INDEX_VEG + accumulation_VEG + "-MODIS_" #FAPAR e VHI have the same

        VEG_file= os.path.join(products_folder,INDEX_VEG,oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"),prefix_VEG+oDateFrom.strftime("%Y%m")+".tif")

        VEG, col, row, geotransform, geoproj = readGeotiff(VEG_file)


        watch = (METEO03 < METEO_threshold03)*METEO03/METEO03+ (METEO01 < METEO_threshold01)*METEO01/METEO01

        #simple_plot_index(METEO01,"METEO01_"+oDateFrom.strftime("%Y%m"))
        #simple_plot_index(METEO03,"METEO03_"+oDateFrom.strftime("%Y%m"))
        #simple_plot_index(watch, "METEO01+METEO03_"+oDateFrom.strftime("%Y%m"))

        watch [watch>1] = 1

        warning = watch * (HUM < HUM_threshold) * 2 * HUM/HUM

        alert = watch * (VEG < VEG_threshold) * 3 * VEG/VEG

        data3d = np.zeros((row, col, 3))

        data3d [:,:,0] = watch
        data3d [:,:,1] = warning
        data3d [:,:,2] = alert

        combined = np.nanmax (data3d, axis=2)

        # PLOT combined
        save_img=False
        """
        fig, ax = plt.subplots()
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white","yellow","orange","red","violet"])
        im = plt.imshow(combined, interpolation='none',cmap=cmap)
        title = "CDI-"+INDEX_METEO01+"_"+oDateFrom.strftime("%Y%m")
        plt.title(title)
        cbar = fig.colorbar(im, orientation='vertical')
        plt.colormaps()
        plt.clim(0,4)
        if save_img:
            os.system ("mkdir -p "+png_dir)
            plt.savefig(os.path.join(png_dir,title+'.png'))
        else:
            plt.show()
        """

        # combined drought indicator

        combined_dir = products_folder+"/CDI/"+oDateFrom.strftime("%Y")+"/"+oDateFrom.strftime("%m")
        combined_file = combined_dir+"/CDI-" + sm_source + "-"+INDEX_METEO01+"_"+oDateFrom.strftime("%Y%m")+".tif"

        os.system("mkdir -p " + combined_dir)

        #SAVE GEOTIFF
        writeGeotiff(combined_file, geotransform, geoproj, combined, nodata=np.nan, BandNames="CDI ("+METEO_prefix01+"-based)", globalDescr="Combined Drought Index")

        print(combined_file)

        oDateFrom = oDateFrom +relativedelta(months=+1)
    print('Process ended with success')



if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass

