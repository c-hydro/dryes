
import datetime
import xarray as xr
import zipfile
import os

from typing import Optional
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from . import CDSDownloader

class EFASDownloader(CDSDownloader):

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

    def __init__(self) -> None:
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
        # stub in the request
        self._request = {
            'system_version': 'version_5_0' # this is the operational version as of 2023-11-10
        }
        self.dataset = 'efas-historical'
        super().__init__()

    @property
    def variable(self):
        return self._variable
    
    @property
    def varopts(self):
        return self._varopts
    
    @property
    def request(self):
        return self._request

    @variable.setter
    def variable(self, variable: str):
        available_list = self.get_available_variables.keys()
        # check if the variable is available
        if variable not in available_list:
            msg = f'Variable {variable} is not available. Available variables are: '
            msg += ', '.join(available_list)
            raise ValueError(msg)
        
        # set the variable
        self._variable = variable

        # add the variable-specific parameters
        self._varopts = self.get_available_variables[variable]
        self._request = self.update_request_from_variable(variable)

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

    def download(self, variable: str, output: str, time: Optional[datetime.datetime] = None) -> None:
        """
        Downloads EFAS data from the CDS API. This will work for a single day and a single variable. 
        """
        self.variable = variable
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

    def get_data(self, variable: str, time: Optional[datetime.datetime] = None) -> xr.DataArray:
        """
        Get EFAS data from the CDS API as an xarray.Dataset.
        This will work for a single day and a single variable. 
        """
        with TemporaryDirectory() as temp_dir:

            output = f'{temp_dir}/{variable}.zip'
            ext = str.remove(self.request['format'], '.zip')
            filename = f'mars_data_0.{ext}'

            self.download(variable, output, time)
            data = extract_zipped(output, filename)
            return self.agg_daily_data(data)
    
    def agg_daily_data(self, data):
        """
        Aggregates the data of a single day to a daily value.
        Also aggregates the soil levels to a single value.
        """
        if self.varopts['time']:
            if self.variable == 'river_discharge_in_the_last_6_hours':
                data = data.sum(dim='time')  # daily discharge is the sum of the 6-hourly discharge
            else:
                data = data.mean(dim='time', skipna = True) # for other variables is the average

        if self.varopts['model_levels'] == 'soil_levels':
            data = data.mean(dim='soilLayer', skipna = True) # average over the soil layers
        
        return data

def extract_zipped(zipped_file: str, filename: str = "mars_data_0.nc"):
    with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
        with TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)
            fullname = os.path.join(temp_dir, filename)
            data = xr.open_dataarray(fullname)
    
    return data