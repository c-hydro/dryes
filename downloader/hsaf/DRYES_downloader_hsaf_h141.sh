#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES DOWNLOADER - HSAF - H141'
script_version="1.0.0"
script_date='2021/09/29'

virtualenv_folder='/home/fp_virtualenv_python3/'
virtualenv_name='fp_virtualenv_python3_dryes_libraries'
script_folder='/home/dryes-idx/'

# Execution example:
# python3 dryes_downloader_hsaf_h141_h142.py -settings_file dryes_downloader_hsaf_h141.json -time "2020-11-02 12:00"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file='/home/dryes-idx/downloader/hsaf/dryes_downloader_hsaf_h141_h142.py'
settings_file='/home/dryes_downloader_hsaf_h141_history.json'

# Get information (-u to get gmt time)
#time_now=$(date -u +"%Y-%m-%d %H:00")
time_now='2000-05-31 23:20' # DEBUG 
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Activate virtualenv
export PATH=$virtualenv_folder/bin:$PATH
source activate $virtualenv_name

# Add path to pythonpath
export PYTHONPATH="${PYTHONPATH}:$script_folder"
#-----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Info script start
echo " ==================================================================================="
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> START ..."
echo " ==> COMMAND LINE: " python3 $script_file -settings_file $settings_file -time $time_now

# Run python script (using setting and time)
python3 $script_file -settings_file $settings_file -time "$time_now"

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

