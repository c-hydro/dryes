#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES PET according to HS'
script_version="1.0.0"
script_date='2024/02/09'

virtualenv_folder=''
virtualenv_name=''
script_folder=''

# Execution example:
# python3 dryes_tool_pet_HS.py -settings_file "dryes_tool_pet_HS.json" -time_now "yyyy-mm-dd 00:00"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file=''
settings_file=''

# Get information (-u to get gmt time)
#time_now=$(date -u +"%Y-%m-%d 00:00")
time_now='2022-08-01 00:00' # DEBUG 
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

