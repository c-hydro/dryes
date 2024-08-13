# Define the global variable
global COMBINATION_ALGORITHMS
COMBINATION_ALGORITHMS = {}

def as_DRYES_algorithm(func):
    COMBINATION_ALGORITHMS[func.__name__] = func
    return func