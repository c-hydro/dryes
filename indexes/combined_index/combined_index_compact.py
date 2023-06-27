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
    os.system("mkdir -p "+save_dir)
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

        INDEX= params["INDEX"]

        products_folder = params["Products_folder"]

        accumulation = params["accumulation"]

        sm_source = params["Soilmoisture_source"]

        threshold = params["threshold"]

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

        ################################################################

        #tmp_dir = "/Users/lauro/Downloads/drought/png/combined"
        ## SPI, SPEI
        ### SMI, SWDI, PDSI
        ### FAPAR, VHI, ESI
        # INDEX=    ["SPEI","SWDI","VHI", "VHI"]
        # accumulation = ["06",  "06", "02",  "01"]
        # threshold=     [-1.3,    -1,  0.2,  0.05]

        ################################################################


        compute_combined (oDateFrom,oDateTo,products_folder,INDEX,accumulation,threshold,sm_source)

    print('\n' + str(datetime.datetime.now()) + ' - Process ended.\n')



def compute_combined (oDateFrom,oDateTo,products_folder,INDEX,accumulation,threshold,sm_source):


    while (oDateFrom <= oDateTo):

        print (oDateFrom)

        indexes = "Combined"

        descr = "Combined"

        for i in range(0, len(INDEX)):

            if INDEX[i]=="SPI":
                prefix= INDEX[i] + accumulation[i] + "-PERSIANN_"
            elif INDEX[i]=="SPEI":
                prefix= INDEX[i] + accumulation[i] + "-PERSIANN-MODIS_"
            elif INDEX[i]=="SSMI":
                 prefix= INDEX[i] + accumulation[i]+ "-"+sm_source+"_"
            elif INDEX[i]=="SWDI":
                 prefix= INDEX[i] + accumulation[i]+ "-"+sm_source+"_"
            elif INDEX[i]=="VHI" or INDEX[i]=="FAPAR" or INDEX[i]=="ESI":
                 prefix= INDEX[i] + accumulation[i] + "-MODIS_"
            elif INDEX[i]=="PDSI":
                prefix= INDEX[i] +  "-PERSIANN-MODIS-SMAP_"

            fileIn= os.path.join(products_folder,INDEX[i],oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"),prefix+oDateFrom.strftime("%Y%m")+".tif")
            #print (fileIn)
            try:
                DATA, xsize, ysize, geotransform, geoproj = readGeotiff(fileIn)

            except OSError as err:
                print("OS error: {0}".format(err))
                continue

            if i==0:
                combined = DATA * 0.0

            DATA=(DATA<threshold[i])*1

            combined[np.isnan(combined)]=0

            combined = combined + (DATA)

            ## hierarchical order (intersection instead of union)
            # combined = combined * (DATA)
            if INDEX[i] == "SSMI" or INDEX[i] == "SWDI":
                indexes = indexes+"-"+INDEX[i]+sm_source+accumulation[i]
            else:
                indexes = indexes + "-" + INDEX[i] + accumulation[i]

            descr = descr+"|"+INDEX[i]+accumulation[i]+"tr"+str(threshold[i])

        """
        save_img=False

        png_dir = "/Users/lauro/Downloads/drought/png/Combined"

        fig, ax = plt.subplots()
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white","yellow","orange","red","violet"])
        im = plt.imshow(combined, interpolation='none',cmap=cmap)
        title = descr+"_"+oDateFrom.strftime("%Y%m")
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

        # apply mask
        mask, xsize, ysize, geotransform, geoproj = readGeotiff("mask_bolivia.tif")
        combined = combined * mask;

        # save combined drought indicator
        combined_dir = os.path.join(products_folder,"Combined",oDateFrom.strftime("%Y"),oDateFrom.strftime("%m"))

        os.system("mkdir -p "+combined_dir)

        combined_file = os.path.join(combined_dir,indexes+"_"+oDateFrom.strftime("%Y%m")+".tif")


        #SAVE GEOTIFF
        writeGeotiff(combined_file, geotransform, geoproj, combined, nodata=np.nan, BandNames=descr, globalDescr="Combined index")

        print('File saved in: '+combined_file)

        oDateFrom = oDateFrom +relativedelta(months=+1)


if __name__ == '__main__':
    argv=sys.argv[1:]
    main(argv)
else:
    pass
