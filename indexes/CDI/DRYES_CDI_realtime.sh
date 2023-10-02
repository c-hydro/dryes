#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES CDI'
script_version="1.0.1"
script_date='2023/10/02'

virtualenv_folder=''
virtualenv_name=''
script_folder=''

# Execution example:
# python3 dryes_CDI_main.py -settings_file "configuration.json" -time_now "yyyy-mm-dd HH:MM"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file=''
settings_file=''

# Get information (-u to get gmt time)
time_now=$(date -u +"%Y-%m-%d 00:00")
#time_now='2023-07-03 00:00' # DEBUG 
time_history_start='1992-01-01 00:00' # DEBUG 
time_history_end='2021-12-31 00:00' # DEBUG 
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

