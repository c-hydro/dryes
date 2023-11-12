from setuptools import setup, find_packages

setup(
    name='DRYES',
    version='0.1',
    packages=find_packages(),
    description='A package for operational calculation of drought indices',
    author='Luca Trotter',
    author_email='luca.trotter@cimafoundation.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='data analysis, meteorological data, satellite data, climatological indices, drought indices',
    install_requires=[
        'numpy',
        'pandas',
        'rasterio',
        'scipy',
        'cdsapi',
        'netCDF4',
        'xarray',
        'cfgrib',
        'rioxarray',
        'scipy'
        # add any additional packages that your project depends on
    ],
)