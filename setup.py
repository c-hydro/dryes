from setuptools import setup, find_packages

setup(
    name='DRYES',
    version='0.1',
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
    ],
)