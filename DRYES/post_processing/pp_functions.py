from astropy.convolution import convolve, Gaussian2DKernel
import warnings
from functools import partial
import numpy as np

def gaussian_smoothing(sigma):

    def _gaussian_smoothing(data, _sigma):
        kernel = Gaussian2DKernel(_sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = np.squeeze(data)
            return convolve(data, kernel, nan_treatment = 'interpolate', preserve_nan = True)
        
    return partial(_gaussian_smoothing, _sigma = sigma)