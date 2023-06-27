#!/bin/bash
#-----------------------------------------------------------------------------------------
# Get file information

working_dir=/home/sequia/drought/code/VHI/

#virtual_env_name="/Users/lauro/miniconda3"
virtual_env_name="sequia"

script_file='TempAnom_Index.py'

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

#update statistics
python TCI_stat_base.py

#compute index for the last "timerange" months
echo $script_file --dateend=$time_now --timerange=$months
python $script_file --dateend=$time_now --timerange=$months
