import datetime
import xarray as xr
import zipfile
import os
import numpy as np

from typing import Optional
from tempfile import TemporaryDirectory

from . import CDSDownloader
from ...data_processes import Grid
from ...lib.log import log

class EFASDownloader(CDSDownloader):

    dataset = 'efas-historical'
    _timerange = (datetime.datetime(1992, 1, 1), datetime.datetime(2022, 12, 31))

    # list of available variables -> those not implementeed/tested yet are commented out
    # for each variable the list indicates [model_levels, time] -> time is False if the variable is static
    available_variables = {\
         'volumetric_soil_moisture' : ['soil_levels', True]\
        #,'snow_depth_water_equivalent' : ['surface_level', True]\
        ,'wilting_point' : ['soil_levels', False]\
        ,'river_discharge_in_the_last_6_hours' : ['surface_level', True]\
        ,'field_capacity' : ['soil_levels', False]\
        #,'elevation' : ['surface_level', False]\
        #,'upstream_area' : ['surface_level', False]\
        #,'soil_depth' : ['soil_levels', False]\
    }

    @classmethod
    def get_available_variables(cls) -> dict:
        available_variables = cls.available_variables
        return {v: {'model_levels': available_variables[v][0],\
                    'time': available_variables[v][1]}\
                for v in available_variables}

    def __init__(self, variable) -> None:
        """
        Initializes the EFASDownloader class.
        Available variables are:
        - snow depth water equivalent: "snow_depth_water_equivalent"
        - wilting point: "wilting_point"
        - volumetric soil moisture: "volumetric_soil_moisture"
        - river discharge in the last 6 hours: "river_discharge_in_the_last_6_hours"
        - field capacity: "field_capacity"
        - elevation: "elevation"
        - upstream area: "upstream_area"
        - soil depth: "soil_depth"
        """
        super().__init__()

        # stub in the request
        self._request = {'system_version': 'version_5_0'} # this is the operational version as of 2023-11-10

        # set the variable
        self.variable = variable
        if not self._varopts['time']:
            self.isstatic = True
        
        # list of preprocess algorithms to be applied to the data
        self._preprocess = []

    @property
    def variable(self):
        return self._variable
    
    @property
    def varopts(self):
        return self._varopts
    
    @property
    def request(self):
        return self._request
    
    def get_timerange(self) -> Optional[tuple[datetime.datetime, datetime.datetime]]:
        """
        Returns the timerange of the dataset.
        """
        if self._varopts['time']:
            return self._timerange
        else:
            return None

    @variable.setter
    def variable(self, variable: str):
        available_list = self.get_available_variables().keys()
        # check if the variable is available
        if variable not in available_list:
            msg = f'Variable {variable} is not available. Available variables are: '
            msg += ', '.join(available_list)
            raise ValueError(msg)
        
        # set the variable
        self._variable = variable

        # add the variable-specific parameters
        self._varopts = self.get_available_variables()[variable]
        self.request = self.update_request_from_variable(variable)

    @request.setter
    def request(self, request: dict):
        # if the request already exist, update it
        if self._request:
            self._request.update(request)
        else:
            self._request = request

    def update_request_from_variable(self, variable: str) -> dict:
        """
        Stubs in a request for the EFAS API based on the variable.
        """
        request = self.request
        request['variable'] = variable

        # each variable has its own model levels
        request['model_levels'] = self.varopts['model_levels']

        # if the variable is time-dependent, use grib as the format, which is faster and more compact
        if self.varopts['time']:
            request['format'] = 'grib.zip'
        else:
            request['format'] =  'netcdf4.zip',    

        # there are three soil levels, we want all of them
        if request['model_levels'] == 'soil_levels':
            request['soil_level'] = [1,2,3]

        return request

    def download(self, output: str, time: Optional[datetime.datetime] = None) -> None:
        """
        Downloads EFAS data from the CDS API. This will work for a single day and a single variable. 
        """
        dataset = self.dataset
        request = self.request

        # if the variable is time-dependent, add the time to the request
        if self.varopts['time']:
            if time:
                request['hyear'] = str(time.year)
                request['hmonth'] = str(time.month)
                request['hday'] = str(time.day)
                # dowload all the times available
                request['time'] = ['00:00', '06:00', '12:00', '18:00']
            else:
                raise ValueError('The variable is time-dependent, please specify a time')
        else:
            if time:
                raise Warning('The variable is not time-dependent, the time will be ignored')

        super().download(dataset, request, output)

    def get_data(self, grid: Grid, time: Optional[datetime.datetime] = None) -> xr.Dataset:
        """
        Get EFAS data from the CDS API as an xarray.DataArray.
        This will work for a single day and a single variable. 
        """
        with TemporaryDirectory() as temp_dir:
            output = f'{temp_dir}/{self.variable}.zip'
            ext = 'grib' if self.request['format'] == 'grib.zip' else 'nc'
            #filename = f'mars_data_0.{ext}'

            self.download(output, time)
            data = self.extract_zipped(output)

            # aggregate the data to a daily value
            daily_data = self.agg_daily(data, time)

            # regrid the data
            regridded_data = grid.apply(daily_data)

            # this will make sure that soil_layers are represented as a coordinate, as expected
            dataset = self.ensure_soil_levels(regridded_data)

            # ensure the variable name is correct
            dataset = xr.Dataset({self.variable: dataset.to_array()})
            for process in self._preprocess:
                modfun = process[0]
                modified = modfun(dataset)
                if process[1]:
                    dataset = xr.merge([dataset, modified])
                else:
                    dataset = modified

            return self.explode_soilLayers(dataset)
    
    def agg_daily(self, data, time):
        """
        Aggregates the data of a single day to a daily value.
        Also aggregates the soil levels to a single value.
        """
        if time is not None:
            if self.variable == 'river_discharge_in_the_last_6_hours':
                data = data.sum(dim='time')  # daily discharge is the sum of the 6-hourly discharge
            else:
                data = data.mean(dim='time', skipna = True) # for other variables is the average
            # remove the step coordinate
            data = data.drop_vars('step')
            # add the time coordinate
            data = data.assign_coords(time = time)
        
        return data
    
    @staticmethod
    def extract_zipped(zipped_file: str) -> xr.Dataset:
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            with TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                data = []
                for filename in os.listdir(temp_dir):
                    if filename.endswith('.grib') or filename.endswith('.nc'):
                        fullname = os.path.join(temp_dir, filename)
                        engine = 'cfgrib' if filename.endswith('.grib') else 'netcdf4'
                        data.append(xr.open_dataarray(fullname, engine=engine).load())
        
        data = xr.merge(data)
        return data
    
    @staticmethod
    def get_times(time_start: datetime.datetime, time_end: datetime.datetime) -> datetime.datetime:
        """
        Get a list of times between two dates.
        EFAS data occurs daily, so we set the delta to 1 day.
        """
        delta = datetime.timedelta(days=1)
        time = time_start
        while time <= time_end:
            yield time
            time += delta

    def ensure_soil_levels(self, data: xr.Dataset) -> xr.Dataset:
        """
        Ensures that the soil levels are represented as a coordinate.
        """
        if self.varopts['model_levels'] != 'soil_levels':
            return data
        
        var_name = self.variable
        if var_name in data.data_vars and 'soilLayer' in data[var_name].dims:
            # The variable is already represented as a coordinate
            return data

        soil_labels = [int(var.replace(f'{var_name}_', ''))-1 for var in data.data_vars if var.startswith(f'{var_name}_')]

        if len(soil_labels) == 0:
            # The variable does not have soil layers
            return data

        new_array = data.to_array(dim='soilLayer')
        new_array['soilLayer'] = soil_labels

        new_set = xr.Dataset({var_name: new_array})

        return new_set

    @staticmethod
    def explode_soilLayers(data: xr.Dataset) -> xr.Dataset:
        """
        Explodes the soil layers into separate variables.
        """
        new_data = xr.Dataset()
        for var_name, variable in data.data_vars.items():
            if 'soilLayer' in variable.dims:
                # This variable uses 'soilLayer' as a coordinate
                for i, layer in enumerate(variable['soilLayer']):
                    # Create a new variable for each layer
                    new_var_name = f'{var_name}_layer{i+1}'
                    new_data[new_var_name] = variable.sel(soilLayer=layer)
            else:
                # This variable does not use 'soilLayer' as a coordinate
                new_data[var_name] = variable

        return new_data

    # Preprocess algorithms specific for data downloaded with EFASDownloader
    def average_soil_layers(self, save_before = False) -> 'EFASDownloader':
        """
        Determines if the soil layers need to be averaged when retrieving the data.
        """
        if self.varopts['model_levels'] == 'soil_levels':
            self._preprocess.append((self._average_soil_layers, save_before))
        
        return self

    @staticmethod
    def _average_soil_layers(data: xr.Dataset) -> xr.Dataset:
        data = data.mean(dim='soilLayer', skipna = True)
        for var in data.data_vars:
            data = data.rename_vars({var: var + '_avg'})
        return data