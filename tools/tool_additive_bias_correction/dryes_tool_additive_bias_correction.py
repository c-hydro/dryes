"""
DRYES Drought Metrics Tool - Tool to compute and apply mean monthly bias-correction factors.
Originally developed for HSAF soil moisture data
__date__ = '20230915'
__version__ = '1.0.0'
__author__ =
        'Anna Mapelli (anna.mapelli@cimafoundation.org)'
        'Nicola Testa (nicola.testa@cimafoundation.org)'
        'Francesco Avanzi (francesco.avanzi@cimafoundation.org)'

__library__ = 'dryes'

General command line:
python dryes_tool_additive_bias_correction.py

Version(s):
20230915 (1.0.0) --> First release

Note: this script computes layers of mean monthly differences between layer 1 and layer 2 as E[layer 1 - layer 2]. It
then corrects daily layer 2 data applying this equation: layer 2 corrected = layer 2 + E[layer 1 - layer 2].
This has implications in terms of the paths given in input below!
"""

# -------------------------------------------------------------------------------------
# Complete library
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import numpy as np  # gridded data
import xarray as xr
import os  # file management
from shutil import copyfile
import matplotlib.pyplot as plt
from dryes_tool_additive_bias_correction_utils_tiff import write_file_tiff
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# INPUT INFO!
path_trunk_layer1_for_bias_correction = ""
path_trunk_layer2_for_bias_correction = ""
Period_for_bias_correction = pd.date_range(start='1/1/2014', end='31/12/2018') # IMPORTANT: start date MUST be Jan 1 and end date MUSt be Dec  31
Flag_compute_bias_correction = False
heading_layer1 = ""
tail_layer1 = ""
heading_layer2 = ""
tail_layer2 = ""
format_dates_in_files = "%Y%m%d"

path_trunk_layer2_to_be_corrected = ""
Period_to_be_corrected = pd.date_range(start='1/1/1992', end='31/12/2013')
heading_layer2_bc = ""
tail_layer2_bc = ""
format_dates_in_files_bc = "%Y%m%d"
path_monthly_bc_layers = ''
heading_monthly_bc_layers = ""
tail_monthly_bc_layers = ""
format_dates_in_monthly_bc_layers = "%m"

enforce_range_post_bc = True
range_enforced_post_bc = [0,1]

grid = \
    xr.open_rasterio('')
grid = np.squeeze(grid)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Start of automatic procedures
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# check on dates for bias correction
if Period_for_bias_correction.dayofyear[0] > 1:
    raise Exception("Period_for_bias_correction MUST start on January 1!")
if Period_for_bias_correction.dayofyear[-1] < 365:
    raise Exception("Period_for_bias_correction MUST end on December 31!")
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Bias correction
if Flag_compute_bias_correction:

    # initialize counter of months and container arrays
    i_months = 0
    diff_month_ALL = \
        np.empty(shape=(grid.sizes.mapping['y'], grid.sizes.mapping['x'], len(np.unique(Period_for_bias_correction.year))*12))*np.nan
    months_ALL = np.empty(shape=(len(np.unique(Period_for_bias_correction.year))*12))*np.nan

    # now loop on dates to gather daily differences and then monthly average differences
    for i, date in enumerate(Period_for_bias_correction):

        # create candidate path for layer 1
        namefile_layer1 = heading_layer1 + date.strftime(format_dates_in_files) + tail_layer1
        path_this_day_layer1 = os.path.join(path_trunk_layer1_for_bias_correction, date.strftime("%Y/%m/%d"), namefile_layer1)

        # create candidate path for layer 2
        namefile_layer2 = heading_layer2 + date.strftime(format_dates_in_files) + tail_layer2
        path_this_day_layer2 = os.path.join(path_trunk_layer2_for_bias_correction, date.strftime("%Y/%m/%d"), namefile_layer2)

        # load both files
        if (os.path.isfile(path_this_day_layer1)) & (os.path.isfile(path_this_day_layer2)):
            with rasterio.open(path_this_day_layer1) as image_layer1:
                image_layer1_array = image_layer1.read()
                image_layer1_array = np.squeeze(image_layer1_array)

            with rasterio.open(path_this_day_layer2) as image_layer2:
                image_layer2_array = image_layer2.read()
                image_layer2_array = np.squeeze(image_layer2_array)

            print('Processed images for date ... ' + date.strftime("%Y%m%d"))

        else:
            print('WARNING: Image(s) for date ... ' + date.strftime("%Y%m%d") + 'NOT FOUND')

        # compute cumulative difference
        diff_this_day = image_layer1_array - image_layer2_array
        if i == 0:
            diff_this_month = diff_this_day
        else:
            diff_this_month = diff_this_month + diff_this_day

        # compute monthly difference if we are at the end of the month...
        if date.is_month_end:
            diff_this_month_mean = diff_this_month/date.daysinmonth

            diff_month_ALL[:,:,i_months] = diff_this_month_mean
            months_ALL[i_months] = date.month
            i_months = i_months + 1
            diff_this_month = np.zeros(np.shape(image_layer1_array))

    # loop on 12 months to compute avg difference
    months = np.arange(start=1, stop=13)
    for i, month in enumerate(months):

        # extract layers for this month
        diff_month_ALL_this_month = diff_month_ALL[:,:, months_ALL == month]

        # compute mean difference
        diff_tmp = np.nanmean(diff_month_ALL_this_month, axis=2)

        # Save to geotiff
        transform = from_origin(image_layer1.bounds[0], image_layer1.bounds[3], image_layer1.res[0], image_layer1.res[1])
        with rasterio.open('Bias correction for month ' + str(month), 'w', driver='GTiff',
                           height=diff_tmp.shape[0], width=diff_tmp.shape[1],
                           count=1, dtype=image_layer1_array.dtype,
                           crs=image_layer1.crs, transform=transform) as dst:
                dst.write((diff_tmp), 1)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Apply bias correction now
for i, date in enumerate(Period_to_be_corrected):

    namefile_layer2 = heading_layer2_bc + date.strftime(format_dates_in_files_bc) + tail_layer2_bc
    namefile_layer2_tmp_copy = 'pre_bias_correction_' + namefile_layer2
    path_this_day_layer2 = os.path.join(path_trunk_layer2_to_be_corrected, date.strftime("%Y/%m/%d"), namefile_layer2)
    path_this_day_layer2_tmp_copy = os.path.join(path_trunk_layer2_to_be_corrected, date.strftime("%Y/%m/%d"),
                                        namefile_layer2_tmp_copy)

    if (os.path.isfile(path_this_day_layer2)):

        print('Processing ' + path_this_day_layer2)

        # copy original layer to keep track of pre-bc values
        copyfile(path_this_day_layer2, path_this_day_layer2_tmp_copy)

        # open file
        with rasterio.open(path_this_day_layer2) as image_layer2:
            image_layer2_array = image_layer2.read()
            image_layer2_array = np.squeeze(image_layer2_array)

            # plt.figure()
            # plt.imshow(image_layer2_array)
            # plt.savefig('image_layer2_array.png')
            # plt.close()

        # open monthly bias-correction layer
        namefile_array_bc = heading_monthly_bc_layers + date.strftime(format_dates_in_monthly_bc_layers) + tail_monthly_bc_layers
        path_array_bc = os.path.join(path_monthly_bc_layers, namefile_array_bc)
        with rasterio.open(path_array_bc) as image_bc_this_month:
            image_bc_this_month_array = image_bc_this_month.read()
            image_bc_this_month_array = np.squeeze(image_bc_this_month_array)

            # plt.figure()
            # plt.imshow(image_bc_this_month_array)
            # plt.savefig('image_bc_this_month_array.png')
            # plt.close()

        # apply correction
        image_layer2_array_bc = image_layer2_array + image_bc_this_month_array
        #why this? image_bc_this_month_array were computed above as the mean of image_layer1_array - image_layer2_array,
        #while here we are correcting image_layer2_array. If the mean of those difference is positive for a given pixel,
        #this means that, for that month, image_layer1_array is generally higher than image_layer2_array. Thus we need
        #to SUM image_bc_this_month_array to image_layer2_array to make it closer to image_layer1_array. Viceversa, if
        #the difference is negative, this means that image_layer1_array is generally lower than image_layer2_array. So
        #we still need to sum as above (preserving the minus sign of course).

        # plt.figure()
        # plt.imshow(image_layer2_array_bc)
        # plt.savefig('image_layer2_array_bc.png')
        # plt.close()

        # enforce min max range
        if enforce_range_post_bc:
            image_layer2_array_bc[image_layer2_array_bc < range_enforced_post_bc[0]] = range_enforced_post_bc[0]
            image_layer2_array_bc[image_layer2_array_bc > range_enforced_post_bc[1]] = range_enforced_post_bc[1]

        # save
        transform = from_origin(image_layer2.bounds[0], image_layer2.bounds[3], image_layer2.res[0],
                                image_layer2.res[1])
        write_file_tiff(path_this_day_layer2, image_layer2_array_bc,
                        image_layer2_array_bc.shape[1], image_layer2_array_bc.shape[0],
                        transform, 'EPSG:4326')
# -------------------------------------------------------------------------------------


