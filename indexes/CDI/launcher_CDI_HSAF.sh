#!/bin/bash
#-----------------------------------------------------------------------------------------
# Get file information

working_dir=/home/sequia/drought/code/CDI/

#virtual_env_name="/Users/lauro/miniconda3"
virtual_env_name="sequia"

script_file='CDI_Index_HSAF.py'

#-----------------------------------------------------------------------------------------
# Setting parameters

# Run period in month(s)
months=2

# Get information (-u to get gmt time)
time_now=$(date -u +"%Y%m")

#-----------------------------------------------------------------------------------------

#activate virtual env
echo "source activate $virtual_env_name"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $virtual_env_name

cd $working_dir

#compute index for the last "timerange" months
echo $script_file --dateend=$time_now --timerange=$months -j CDI_config_HSAF.json
python $script_file --dateend=$time_now --timerange=$months -j CDI_config_HSAF.json
echo $script_file --dateend=$time_now --timerange=$months -j CDI_config_HSAF_SPI.json
python $script_file --dateend=$time_now --timerange=$months -j CDI_config_HSAF_SPI.json
