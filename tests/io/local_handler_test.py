import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from datetime import datetime

from dryes.io import LocalIOHandler

def make_sample_data():
    # Create a 10x10 array with values from 0 to 99
    data = np.reshape(np.arange(100), (10, 10)).astype(np.float32)

    coords = {
        "x": np.linspace(-50, 40, 10),
        "y": np.linspace(50, -40, 10),
    }

    # Create a DataArray
    xarr = xr.DataArray(data, coords=coords, dims=["y", "x"])

    # Set spatial dimensions
    xarr.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Define the CRS
    xarr.rio.write_crs("EPSG:4326", inplace=True)

    return xarr

class TestLocalIOHandler(unittest.TestCase):
    def setUp(self):
        self.handler1 = LocalIOHandler(path='path/', file='file_%Y%m%d.tif', name = 'test')
        self.handler2 = LocalIOHandler(path='path', file='file.txt')
        self.sample_data = make_sample_data()
        #self.sample_data.rio.to_raster('tests/io/sample_data/output.tif', compress = 'lzw')

    def test_init(self):
        self.assertEqual(self.handler1.dir, 'path/')
        self.assertEqual(self.handler1.file, 'file_%Y%m%d.tif')
        self.assertEqual(self.handler1.name, 'test')
        self.assertEqual(self.handler1.format, 'GeoTIFF')
        self.assertEqual(self.handler1.path_pattern, 'path/file_%Y%m%d.tif')

        self.assertEqual(self.handler2.dir, 'path')
        self.assertEqual(self.handler2.file, 'file.txt')
        self.assertEqual(self.handler2.name, 'file')
        self.assertEqual(self.handler2.format, 'ASCII')
        self.assertEqual(self.handler2.path_pattern, 'path/file.txt')

    @patch('os.path.exists')
    def test_check_data(self, mock_exists):
        mock_exists.return_value = True
        self.assertTrue(self.handler1.check_data())
        mock_exists.assert_called_once_with('path/file_%Y%m%d.tif')

    @patch('os.path.exists')
    @patch('rioxarray.open_rasterio')
    def test_get_data(self, mock_open_rasterio, mock_exists):

        mock_exists.return_value = True
        mock_open_rasterio.return_value = self.sample_data

        test_time = datetime(2021, 1, 1)
        # Now when you call get_data, it should return the mock_data
        self.assertTrue(self.handler1.get_data(test_time).equals(self.sample_data))
        
        # check that the template is updated correctly
        sample_template = self.handler1.template.copy(data = np.full((10, 10), np.nan))
        self.assertTrue(sample_template.equals(sample_template))

        mock_exists.assert_called_once_with('path/file_20210101.tif')
        mock_open_rasterio.assert_called_once_with('path/file_20210101.tif')

    
if __name__ == '__main__':
    unittest.main()