import logging
from time import time

def timer(func):
    """
    A decorator function that measures the execution time of a given function.

    Parameters:
    func (function): The function to be timed.

    Returns:
    function: The wrapped function that measures the execution time.
    """
    
    def wrap(*args, **kwargs):
        starttime = time()
        result = func(*args, **kwargs)
        logging.info(f'Function {func.__name__!r} executed in {(time()-starttime):.4f}s')
        return result
    return wrap