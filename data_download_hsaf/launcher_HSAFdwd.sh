#!/bin/bash
# Execute HSAF Downloader
#-----------------------------------------------------------------------------------------
# Get file information

working_dir="/home/sequia/drought/code/data_download_hsaf/"

virtual_env_name="sequia_plus"

script_file='acquire_H14_v04_op.py'

#-----------------------------------------------------------------------------------------
# Setting parameters

# Run period in day(s)
dayback=30 #25

# Get information (-u to get gmt time)
time_now=$(date -u +"%Y%m%d")

echo "--------------------> SEARCH H14 data in the last $dayback days, up to $time_now"

#-----------------------------------------------------------------------------------------

#activate virtual env
echo "conda activate $virtual_env_name"
export PATH="$HOME/minicondas/bin:$PATH"
#source ~/minicondas/etc/profile.d/conda.sh
source activate $virtual_env_name

export PATH="/usr/local/cdo_1_9/bin:$PATH"

cd $working_dir

#download modis
echo "$script_file -t $time_now -d $dayback"
python $script_file -t $time_now -d $dayback > log/bol-h14_dwd.txt

echo "-------------------> finalised"
