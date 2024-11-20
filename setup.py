from setuptools import setup, find_packages

setup(
    name='dryes',
    version='3.1.4',
    packages=find_packages(),
    description='A package for operational calculation of environmental indices for drought monitoring',
    author='Luca Trotter',
    author_email='luca.trotter@cimafoundation.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    keywords='data analysis, meteorological data, satellite data, climatological indices, drought indices',
    install_requires=[
        'astropy>=5.3.3',
        'lmoments3>=1.0.6',
        'methodtools>=0.4.7',
        'numpy>=1.24.0',
        'python_dateutil>=2.8.2',
        'rioxarray>=0.15.0',
        'scipy>=1.8.0',
        'xarray>=2023.9.0',
        'deprecated>=1.2.12',
        'matplotlib>=3.8.4',
        'geopandas',
        'boto3',
        'img2pdf',
        'paramiko',
        'scipy',
        'netCDF4'
    ],
    python_requires='>=3.10',
    test_suite='tests'
)