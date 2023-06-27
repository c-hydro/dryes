__author__ = 'lauro'

from geotiff import *
import datetime
import os
from dateutil.relativedelta import *
import matplotlib.pylab as plt
import matplotlib as mpl
from compressed_pickle import save, load


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


def main (year,months,INDEX,accumulation,save_img):


        for i in range(0, len(INDEX)):

            fig,axes = plt.subplots(nrows = len(accumulation)*1, ncols = len(months), figsize=(50,50))
            plt.rcParams['axes.labelsize'] = 8

            for acc in range (0,len(accumulation)):

                for m in range(0,len(months)):


                        #for m in ("01","02","03","04","05","06","07","08","09","10","11","12"):

                        if INDEX[i]=="SPI":
                            prefix= INDEX[i] + accumulation[acc] + "-PERSIANN_"
                        elif INDEX[i]=="SPEI":
                            prefix= INDEX[i] + accumulation[acc] + "-PERSIANN-MODIS_"
                        elif INDEX[i]=="SMI":
                             prefix= INDEX[i] + accumulation[acc]+ "-SMAP_"
                        elif INDEX[i]=="SWDI":
                             prefix= INDEX[i] + accumulation[acc]+ "-SMAP_"
                        elif INDEX[i]=="VHI" or INDEX[i]=="FAPAR" or INDEX[i]=="ESI":
                             prefix= INDEX[i] + accumulation[acc] + "-MODIS_"
                        elif INDEX[i]=="PDSI":
                            prefix= INDEX[i] +  "-PERSIANN-MODIS-SMAP_"

                        fileIn= os.path.join(products_dir,INDEX[i],year,months[m],prefix+year+months[m]+".tif")

                        DATA, xsize, ysize, geotransform, geoproj = readGeotiff(fileIn)

                        mask,x,y,geot, geop= readGeotiff("mask_bolivia.tif")

                        DATA = DATA * mask

                        #simple_plot_index (DATA,prefix+oDateFrom.strftime("%Y%m"),save_img,tmp_dir)

                        for ax in axes.flatten():
                            ax.axis('off')

                        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","orange","yellow","white","pink","violet", "#0f0f0f"])

                        ax_row = acc
                        ax_col = m
                        ax=axes[ax_row,ax_col]
                        im=ax.imshow(DATA, interpolation='none',cmap=cmap)
                        #ax.set_title (INDEX[i]+" "+year+months[m],fontsize=9)
                        if INDEX[i]=="PDSI":
                            ax.set_title (INDEX[i]+" "+year+months[m],fontsize=6, fontweight="bold")
                        else:
                            ax.set_title (INDEX[i]+accumulation[acc]+" "+year+months[m],fontsize=6, fontweight="bold")

                        if INDEX[i]== "VHI":
                            im.set_clim(0, 1)
                        elif INDEX[i]== "PDSI":
                            im.set_clim(-4, 4)
                        else:
                            im.set_clim(-3, 3)

                        ax.tick_params(labelsize='xx-small')
                        if ax_col==len(months)-1:
                            if INDEX[i]== "VHI":
                                fig.colorbar(im, ax=ax,shrink=0.6,aspect = 5, ticks = [0,0.25,0.5,0.75,1])
                            elif INDEX[i]== "PDSI":
                                fig.colorbar(im, ax=ax,shrink=0.6,aspect = 5, ticks = [-4,-2,0,2,4])
                            else:
                                fig.colorbar(im, ax=ax,shrink=0.6,aspect = 5, ticks = [-3,-1.5,0,1.5,3])  #spacing='proportional'
                if INDEX[i]== "PDSI": break
            fig.suptitle(INDEX[i]+" "+year, fontsize=12)
            plt.show()

        print('Process ended with success')


if __name__ == "__main__":

    #################################################################

    tmp_dir = "/Users/lauro/Downloads/tmp"
    products_dir = "/Users/lauro/Downloads/products"

    #################################################################


#    INDEX        = ["SPI","SPEI","SMI","SWDI","PDSI","VHI", "FAPAR"]
    INDEX        = ["FAPAR"]
    accumulation = ["01",  "03", "06"]
    year         = "2014"
    months       = ["01","02","03","04","05","06","07","08","09","10","11","12"]

    #################################################################

    save_img= True

    main(year,months,INDEX,accumulation,save_img)
    print('\n' + str(datetime.datetime.now()) + ' - Process ended.\n')
