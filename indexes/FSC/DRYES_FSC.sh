#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES FSC'
script_version="1.0.1"
script_date='2023/08/30'

virtualenv_folder=''
virtualenv_name=''
script_folder=''

# Execution example:
# python3 dryes_FSC_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM" -year_history_start "yyyy" -year_history_end "yyyy"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file=''
settings_file=''

# Get information (-u to get gmt time)
#time_now=$(date -u +"%Y-%m-%d 00:00" -d "1 day ago")
time_now='2023-03-01 00:00' # DEBUG 
year_history_start='2013' 
year_history_end='2022' # DEBUG 
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
python3 $script_file -settings_file $settings_file -time_now "$time_now" -year_history_start "$year_history_start" -year_history_end "$year_history_end"

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

