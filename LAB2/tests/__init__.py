import sys
import random
from pathlib import Path

CUR_DIR = Path(__file__).parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR.parent))

import random

def generate_random_test_params(params_list, num_examples=100, min_val=1, max_val=1024, max_batch=32):
    """
    Generates a list of tuples with random integer parameters given a list of parameter names.
    If 'batch' is in the list, its value will be small (<= max_batch).

    Args:
        params_list (list of str): Names of parameters, e.g., ['in_f', 'out_f', 'batch']
        num_examples (int): Number of tuples to generate.
        min_val (int): Minimum value for each parameter.
        max_val (int): Maximum value for each parameter.
        max_batch (int): Maximum batch size if 'batch' is in params_list.

    Returns:
        List[Tuple[int, ...]]: List of tuples with random integers.
    """
    result = []
    for _ in range(num_examples):
        tup = []
        for name in params_list:
            if name.lower() == "batch":
                tup.append(random.randint(min_val, max_batch))
            else:
                tup.append(random.randint(min_val, max_val))
        result.append(tuple(tup))
    return result
