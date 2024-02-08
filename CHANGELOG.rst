=========
Changelog
=========

=======
Version 2.6.13 [2024-02-08]
***************************
- Fixed typo in monthly aggregation tool

Version 2.6.12 [2023-12-11]
***************************
- Modified SPI tool to avoid errors in parameters calculation

Version 2.6.11 [2023-11-13]
***************************
- Switched from xarray.open_rasterio to rioxarray.open_rasterio post xarray v0.20

Version 2.6.10 [2023-10-17]
**************************
- Added support for multiband inputs in SSMI
- Added tool to compute monthly aggregations from daily maps

Version 2.6.9 [2023-10-17]
**************************
- Minor changes to classifier tool for tiff objects by defined thresholds and classes

Version 2.6.8 [2023-10-12]
**************************
- Added new tool to classify tiff objects by defined thresholds and classes

Version 2.6.7 [2023-10-04]
**************************
- Modified SPI tool to avoid errors in parameters calculation

Version 2.6.6 [2023-10-03]
**************************
- Modified tool for monthly air temperature anomaly to add mkdir in output path (if needed)

Version 2.6.5 [2023-10-03]
**************************
- Modified SPI tool to restore original output path

Version 2.6.4 [2023-10-03]
**************************
- Modified SPI tool to substract one month in output path

Version 2.6.3 [2023-10-02]
**************************
- Added a new version of the SPI index

Version 2.6.2 [2023-10-02]
**************************
- Modified SPEI and CDI tool to add mkdir in output path (if needed)

Version 2.6.1 [2023-09-28]
**************************
- Modified tool to aggregate geotiff by regions (e.g., nuts) including min, max e mode

Version 2.6.0 [2023-09-25]
**************************
- Added a new version of the CDI 2D index

Version 2.5.0 [2023-09-25]
**************************
- Added a new version of the SPEI index

Version 2.4.5 [2023-09-22]
**************************
- Minor changes to HSAF SSMI

Version 2.4.4 [2023-09-18]
**************************
- Modified HSAF SSMI index to handle a dynamic mask (e.g., SWE)

Version 2.4.3 [2023-09-11]
**************************
- Added new tool to compute an additive bias correction for rasters based on monthly mean differences with a benchmark dataset

Version 2.4.2 [2023-09-11]
**************************
- Added new tool to aggregate geotiff by regions (e.g., nuts)

Version 2.4.1 [2023-09-07]
**************************
- Modified generic aggregator of hourly rasters to daily summary to include min and max

Version 2.4.0 [2023-08-31]
**************************
- Added generic aggregator of hourly rasters to daily summary 
- Added tool for computation of monthly air temperature anomaly

Version 2.3.0 [2023-08-28]
**************************
- Added a spatialization tool for air temperature based on linear regression of in-situ data over elevation by homogeneous regions

Version 2.2.0 [2023-07-27]
**************************
- Updated aggregator for daily MCM precipitation under tools/tool_processing_transfer_mcm
- Added a new version of the H-SAF fractional snow cover (FSC) index

Version 2.1.0 [2023-07-04]
**************************
- Major directory restructuring
- Added an aggregator for daily MCM precipitation under tools/tool_processing_transfer_mcm
- Added a new version of the SSMI index
- Added a new version of the H-SAF snow-covered-area (SCA) index
- Updated virtual environment with new packages

Version 1.0.0 [2023-04-13]
**************************
- Release for Bolivia operational chain

Version 0.0.1 [2021-09-17]
**************************
- Repo creation, still under construction!
