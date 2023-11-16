from datetime import datetime
import os
import numpy as np
import xarray as xr

from DRYES.data_sources import copernicus_cds as cds
from DRYES.variables import DRYESVariable
from DRYES.time_aggregation import TimeAggregation, average_of_window

from DRYES.lib.time import TimeRange
from DRYES.lib.log import setup_logging

LOG_FILE = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/log.txt'
setup_logging(LOG_FILE)

GRID_FILE = '/home/drought/share/DRYES/data/Italia/static/ET/MCM_mask.tif'

OUT = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/output'
OUT_STATIC  = os.path.join(OUT, 'static')
OUT_DYNAMIC = os.path.join(OUT, '%Y/%m/%d')

ARCHIVE = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/archive'
ARCHIVE_PAR  = os.path.join(ARCHIVE, 'parameters')
ARCHIVE_DATA = os.path.join(ARCHIVE, 'data/%Y/%m/%d')
ARCHIVE_OUT  = os.path.join(ARCHIVE, 'maps/%Y/%m/%d')

# Make the variables needed to calculate the index
field_capacity = DRYESVariable.from_data_source(
    data_source = cds.EFASDownloader('field_capacity').average_soil_layers(save_before=True),
    variable = 'field_capacity_avg', type = 'static',
    destination = OUT_STATIC
)

wilting_point = DRYESVariable.from_data_source(
    data_source = cds.EFASDownloader('wilting_point').average_soil_layers(save_before=True),
    variable = 'wilting_point_avg', type = 'static',
    destination = OUT_STATIC
)

soil_moisture = DRYESVariable.from_data_source(
    data_source = cds.EFASDownloader('volumetric_soil_moisture').average_soil_layers(save_before=True),
    variable = 'volumetric_soil_moisture_avg', type = 'dynamic',
    destination = OUT_DYNAMIC
)

def calc_smi(sm:xr.DataArray, fc:xr.DataArray, wp:xr.DataArray) -> xr.DataArray:

    fc_d = fc.values
    wp_d = wp.values

    stacked = np.stack([fc_d, wp_d], axis=0)
    theta50 = np.median(stacked, axis=0)

    smi = sm.copy()
    smi.data = 1 - 1 / (1 + (sm.values / theta50)**(6))
    smi.name = 'smi'

    return smi

smi = DRYESVariable.from_other_variables(
    variables = {'sm' : soil_moisture, 'fc': field_capacity, 'wp' : wilting_point},
    function = calc_smi, 
    name = 'SMI', type = 'dynamic',
    destination = OUT_DYNAMIC
)

# define the time aggregation
time_agg = TimeAggregation(timesteps_per_year = 36).\
    add_aggregation(name = "Agg1dk", function = average_of_window(size = 1, unit = 'dekads'))

#field_capacity.gather(grid = Grid(grid_file), time_range = TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))
#soil_moisture.gather(grid = Grid(grid_file), time_range = TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))
#SMI.compute(time_range=TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))
smi_aggregated = time_agg(smi, TimeRange(datetime(1992, 1, 1), datetime(1993, 1, 1)))
breakpoint()

# Create the index object
# EDO_SMAnomaly = DRYESAnomaly(log_file = LOG_FILE)

# # # Set the data chain parameters
# # # TODO: what if the reference period changes (this is the case for EDO SMA anomaly)
# EDO_SMAnomaly.reference_period = TimeRange(datetime(1992, 1, 1), datetime(1993, 1, 1))
# EDO_SMAnomaly.timesteps_per_year = 36 
# EDO_SMAnomaly.time_aggregation

# # EDO_SMAnomaly.set_grid = Grid(grid_file)

# aggregate_time(variable = smi, time_range = TimeRange(datetime(1992, 1, 1), datetime(1993, 1, 1)),
#                timesteps_per_year   = EDO_SMAnomaly.timesteps_per_year, 
#                aggregation_function = EDO_SMAnomaly.time_aggregation)