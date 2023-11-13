from datetime import datetime
from DRYES.data_sources.cds import efas
import os
import numpy as np

from DRYES.data_processes import *
from DRYES.data_chains import DRYESDataChain, DRYESVariable

log_file = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/log.txt'

grid_file = '/home/drought/share/DRYES/data/Italia/static/ET/MCM_mask.tif'

output_path = '/home/drought/DRYES/analyses/luca_workspace/tests/CDS_SM_lisflood/Italia/output'
output_static = os.path.join(output_path, 'static')
output_dynamic = os.path.join(output_path, '%Y/%m/%d')

# Create the data chain
EDO_SMAnomaly = DRYESDataChain('EDO_SMAnomaly', log_file = log_file)

# # Set the data chain parameters
# # TODO: what if the reference period changes (this is the case for EDO SMA anomaly)
# EDO_SMAnomaly.set_reference_period = TimeRange(datetime(1992, 1, 1), datetime(1993, 1, 1))
# EDO_SMAnomaly.set_timestepping = 'dekads'
# EDO_SMAnomaly.set_timeagg = {
#     "Agg1d" : time_aggregation_mean(time = 1, unit = 'dekads'),
#     "Agg1m" : time_aggregation_mean(time = 1, unit = 'months'),
# }

# EDO_SMAnomaly.set_grid = Grid(grid_file)

# Make the variables available to the data chain
field_capacity = DRYESVariable.from_data_source(
    data_source = efas.EFASDownloader('field_capacity').average_soil_layers(save_before=True),
    variable = 'field_capacity_avg', type = 'static',
    destination = output_static
)

wilting_point = DRYESVariable.from_data_source(
    data_source = efas.EFASDownloader('wilting_point').average_soil_layers(save_before=True),
    variable = 'wilting_point_avg', type = 'static',
    destination = output_static
)

soil_moisture = DRYESVariable.from_data_source(
    data_source = efas.EFASDownloader('volumetric_soil_moisture').average_soil_layers(save_before=True),
    variable = 'volumetric_soil_moisture_avg', type = 'dynamic',
    destination = output_dynamic
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

SMI = DRYESVariable.from_other_variables(
    variables = {'sm' : soil_moisture, 'fc': field_capacity, 'wp' : wilting_point},
    function = calc_smi, 
    name = 'SMI', type = 'dynamic',
    destination = output_dynamic
)

field_capacity.gather(grid = Grid(grid_file), time_range = TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))
soil_moisture.gather(grid = Grid(grid_file), time_range = TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))
SMI.compute(time_range=TimeRange(datetime.datetime(1992,1,1), datetime.datetime(1993,1,1)))