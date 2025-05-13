# Define the global variable
global COMBINATION_ALGORITHMS
COMBINATION_ALGORITHMS = {}

keys = ['input', 'previous', 'static']
def as_DRYES_algorithm(**kwargs):
    def decorator(func):

        # Add the provided arguments as attributes to the function
        for key, value in kwargs.items():
            setattr(func, key, value)

        def wrapper(*args, **kwargs):
            for k in keys:
                if f'{k}_data' in kwargs:
                    if k not in func.__dict__:
                        kwargs.pop(f'{k}_data')
            return func(*args, **kwargs)
        
        # Register the function in the global dictionary
        COMBINATION_ALGORITHMS[func.__name__] = wrapper
        
        # Add the provided arguments as attributes to the wrapper as well
        for key, value in kwargs.items():
            setattr(wrapper, key, value)

        return wrapper
    return decorator