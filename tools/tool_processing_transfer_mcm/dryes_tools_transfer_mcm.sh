#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='DRYES TOOLS - TRANSFER DATASETS - HISTORY'
script_version="1.0.0"
script_date='2023/06/15'

# Virtualenv default definition(s)
virtualenv_folder=$HOME/DRYES/envs/
virtualenv_name='dryes_libraries'

# Default script folder(s)
script_folder=$HOME/DRYES/script/
configuration_folder=$script_folder # '/home/dte/utils/'
package_folder=$HOME/DRYES/libraries/dryes/

# Execution example:
# ./dryes_tools_transfer_mcm.sh -t 2020-01-01
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file_transfer=${script_folder}'dryes_tool_transfer_datasets.py'
settings_file_transfer=${configuration_folder}'dryes_tool_transfer_remote2local_mcm.json'

script_file_converter=${script_folder}'dryes_tool_processing_converter.py'
settings_file_converter=${configuration_folder}'dryes_tool_processing_converter_mcm.json'

#-----------------------------------------------------------------------------------------
# Get time information (-u to get gmt time)
# args: -t "%Y-%m-%d"
# you can assign any hour, forced procedure rounds per day then set time 23:00 of the last day
time_now=$(date +"%Y-%m-%d 01:00")

while getopts t: flag
do
    case "${flag}" in
        t) tt=${OPTARG}" 23:00";;
    esac
done

if [ -z "$tt" ]; then time=$time_now; else time=$(date +"$tt"); fi
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Activate virtualenv
export PATH=$virtualenv_folder/bin:$PATH
source activate $virtualenv_name

# Add path to pythonpath
export PYTHONPATH="${PYTHONPATH}:$script_folder"
export PYTHONPATH="${PYTHONPATH}:$package_folder"
#-----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Info script transfer start
echo " ==================================================================================="
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> START ..."
echo " ==> COMMAND LINE: " python $script_file_transfer -settings_file $settings_file_transfer -time $time

# Run python script (using setting and time)
python $script_file_transfer -settings_file $settings_file_transfer -time "$time"

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Info script converter start
echo " ==================================================================================="
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> START ..."
echo " ==> COMMAND LINE: " python $script_file_converter -settings_file $settings_file_converter -time $time

# Run python script (using setting and time)
python $script_file_converter -settings_file $settings_file_converter -time "$time"

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
# ----------------------------------------------------------------------------------------

echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

