#!/bin/bash
#-----------------------------------------------------------------------------------------
# Get file information

working_dir=/home/sequia/drought/code/combined_index/

#virtual_env_name="/Users/lauro/miniconda3"
virtual_env_name="sequia"

script_file='combined_index_compact.py'

#-----------------------------------------------------------------------------------------
# Setting parameters

# Run period in month(s)
months=2

# Get information (-u to get gmt time)
time_now=$(date -u +"%Y%m")
#-----------------------------------------------------------------------------------------

#activate virtual env
echo "conda activate $virtual_env_name"
export PATH="$HOME/minicondas/bin:$PATH"
#source ~/minicondas/etc/profile.d/conda.sh
source activate $virtual_env_name

cd $working_dir

#compute index for the last "timerange" months
echo $script_file --dateend=$time_now --timerange=$months -j combined_HSAF.json
python $script_file --dateend=$time_now --timerange=$months -j combined_HSAF.json

python $script_file --dateend=$time_now --timerange=$months -j combined_HSAF_SPI.json
