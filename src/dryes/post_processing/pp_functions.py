from astropy.convolution import convolve, Gaussian2DKernel
import warnings
from functools import partial
import numpy as np

def gaussian_smoothing(sigma):

    def _gaussian_smoothing(data, _sigma):
        kernel = Gaussian2DKernel(_sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            original_shape = data.shape
            data = np.squeeze(data)
            smooth_data = convolve(data, kernel, nan_treatment = 'interpolate', preserve_nan = True)

            post_info = {'smoothing_type': 'gaussian',
                         'smoothing_sigma': _sigma}

            return np.reshape(smooth_data, original_shape), post_info
        
    return partial(_gaussian_smoothing, _sigma = sigma)