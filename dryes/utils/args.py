import json
import argparse
import os
from datetime import datetime

from ..tools.config.parse import substitute_values

from ..time_aggregation import aggregation_functions as agg
from ..post_processing import pp_functions as pp

def get_options():
    """
    get and the options for the index from the command line
    """
    args = get_args()
    time_start, time_end, options_file = check_args(args)
    index_options, io_options, run_options = parse_json_options(options_file)
    
    run_options['current'] = (time_start, time_end)
    return {'index': index_options, 'io': io_options, 'run': run_options}

def get_json_data(file):
    """
    parse a json file
    """
    with open(file, 'r') as f:
        data = json.load(f)

    data = check_options(data)
    return data

def check_options(data):
    """
    check that the options are valid
    """
    mandatory_keys = ['index_options', 'io_options', 'run_options']
    for key in mandatory_keys:
        if key not in data:
            raise ValueError(f'{key} is a mandatory key in the options file')
    
    run_options = data['run_options']
    try:
        history_start = datetime.strptime(run_options['history_start'], '%Y-%m-%d')
        history_end = datetime.strptime(run_options['history_end'], '%Y-%m-%d')
    except:
        raise ValueError('history_start and history_end must be in the format YYYY-MM-DD')
    if history_start > history_end:
        raise ValueError('history_start must be before history_end')
    if 'timesteps_per_year' not in run_options:
        raise ValueError('timesteps_per_year must be specified in the run options')

    data['run_options']['history_start'] = history_start
    data['run_options']['history_end'] = history_end
    return data

def parse_json_options(file):
    """
    parse options from a json file
    """
    data = get_json_data(file)
    if "tags" in data:
        for i in range(5): # we can't do this with rec = True, because we have the {history_start} and {history_end} tags
            data["tags"] = substitute_values(data["tags"], data["tags"])
    else:
        data["tags"] = {}

    index_options, io_options, run_options = parse_dict_options(data)
    return index_options, io_options, run_options

def parse_dict_options(options:dict):

    # start with the index options
    index_options = options['index_options']
    index_options = substitute_values(index_options, options["tags"])
    if 'agg_fn' in index_options:
        if 'type' in index_options['agg_fn']:
            index_options['agg_fn'] = create_obj_from_dict(index_options['agg_fn'], 'agg')
        else:
            for key, value in index_options['agg_fn'].items():
                index_options['agg_fn'][key] = create_obj_from_dict(value, 'agg')
    if 'post_fn' in index_options:
        if 'type' in index_options['post_fn']:
            index_options['post_fn'] = create_obj_from_dict(index_options['post_fn'], 'pp')
        else:
            for key, value in index_options['post_fn'].items():
                index_options['post_fn'][key] = create_obj_from_dict(value, 'pp')
    
    # then the io options
    io_options = options['io_options']
    io_options = substitute_values(io_options, options["tags"])
    for key, value in io_options.items():
        value["type"] = value["type"].capitalize() + "IOHandler"
        io_options[key] = create_obj_from_dict(value, '')

    # then the run options
    run_options = options['run_options']
    history_start = run_options.pop('history_start')
    history_end   = run_options.pop('history_end')
    run_options.update({'reference': (history_start, history_end)})
    
    return index_options, io_options, run_options

def create_obj_from_dict(obj_dict, space):
    """
    create an object from a dictionary
    """
    # Extract the type of the function
    fn_type = obj_dict.pop('type')

    # Create a string with the remaining key-value pairs formatted as arguments
    args = ', '.join(f'{k}={repr(v)}' for k, v in obj_dict.items())

    # Create the full function call string
    space = f"{space}." if space else ""
    fn_call = f'{space}{fn_type}({args})'

    # Evaluate the function call string
    return eval(fn_call)

def get_args():
    """
    parse arguments from the command line
    two or three arguments are expected:
    - options: a json file with the options
    - time: a date for which to calculate the index
    OR 
    - options
    - time_start: a start date for the time range
    - time_end: an end date for the time range
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-options', help = 'JSON file with the options', required=True)
    parser.add_argument('-time', help = 'Date for which to calculate the index')
    parser.add_argument('-time_start', help = 'Start date for the time range')
    parser.add_argument('-time_end', help = 'End date for the time range')
    args = parser.parse_args()

    return args

def check_args(args):
    """
    make sure one of the times is available and that the json file exists
    """
    if args.time is None and (args.time_start is None or args.time_end is None):
        raise ValueError('Either -time or -time_start and -time_end must be specified')
    elif args.time is not None and (args.time_start is not None or args.time_end is not None):
        raise Warning('Either -time or -time_start and -time_end must be specified, not both. -time will be used.')
    if args.time is not None:
        time_start = args.time
        time_end = args.time
    else:
        time_start = args.time_start
        time_end = args.time_end

    try:
        time_start = datetime.strptime(time_start, '%Y-%m-%d')
        time_end = datetime.strptime(time_end, '%Y-%m-%d')
    except:
        raise ValueError('time must be in the format YYYY-MM-DD')

    if not os.path.exists(args.options):
        raise ValueError(f'{args.options} does not exist')
    
    return time_start, time_end, args.options