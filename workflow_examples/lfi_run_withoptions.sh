#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <options> <time> [<time_end>]"
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# the DRYES package should be installed in the virtual environment
# if not, install it with:
#pip install -e .

# Check if a single time or a time range was provided
if [ "$#" -eq 2 ]; then
    # Run the Python script with the provided arguments
    python scripts/run_lfi.py -options "$1" -time "$2"
else
    # Run the Python script with the provided arguments
    python scripts/run_lfi.py -options "$1" -time_start "$2" -time_end "$3"
fi