from dryes.indices import SPI
from dryes.utils.args import get_options

# In continuity with DRYES 2.0, this script should take as inputs:
# - a JSON file with the options
# - a date (or range of dates) for which to calculate the index
# (we will not include as inputs the history, as this is likely fixed most of the times)

# run it as follows:
# python path/to/here/run_spi.py -options "path/to/options.json" -time_start "YYYY-MM-DD" -time_end "YYYY-MM-DD"
# or
# python path/to/here/run_spi.py -options "path/to/options.json" -time "YYYY-MM-DD"

def main():
    options = get_options()
    this_spi = SPI(options['index'], options['io'])
    this_spi.compute(**options['run'])

if __name__ == '__main__':
    main()