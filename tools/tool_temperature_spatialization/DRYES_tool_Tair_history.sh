#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES TOOL T AIR'
script_version="1.0.0"
script_date='2023/08/28'

virtualenv_folder=''
virtualenv_name=''
script_folder=''

# Execution example:
# python3 dryes_tool_interp_Tair_elevation.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file='dryes_tool_interp_Tair_elevation.py'
settings_file='dryes_tool_interp_Tair_elevation.json'

# Get information (-u to get gmt time)
#time_now=$(date -u +"%Y-%m-%d %H:00")
time_now='2021-06-30 23:00' # DEBUG 
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
python3 $script_file -settings_file $settings_file -time_now "$time_now"

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

