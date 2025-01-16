import argparse

from d3tools import Options
from d3tools.timestepping import get_date_from_str

from dryes import DRYESIndex

def parse_arguments():
    """
    Parse command line arguments for the workflow.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run workflow with specified parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('workflow_json', type=str, help='JSON file of the workflow')
    parser.add_argument('-s', '--start', type=str, help='Start date to calculate the index [YYYY-MM-DD]')
    parser.add_argument('-e', '--end',   type=str, help='End date to calculate the index [YYYY-MM-DD]')
    parser.add_argument('-S', '--setup', action='store_true', help='Flag to setup the workflow, i.e. calculate parameters')
    
    args = parser.parse_args()

    if not args.setup:
        if not args.start or not args.end:
            parser.error("--start and --end are required if --setup is not specified")

    return args

def main():
    args = parse_arguments()

    # load the options from the json file
    options = Options.load(args.workflow_json)

    # set the start and end date
    start_date = get_date_from_str(args.start) if args.start else None
    end_date   = get_date_from_str(args.end)   if args.end   else None

    # create the index
    index = DRYESIndex.from_options(**options.DRYES_INDEX)

    # run the computation
    index.compute((start_date, end_date), make_parameters=args.setup)
    
if __name__ == '__main__':
    main()