# frutil/tnorms.py
"""
Collection of common T-norm functions.
"""
import numpy as np

def tn_minimum(values: np.ndarray):
    """
    In case of 1D array, returns minimum of the 1D array
    In case of 3D array (nxnx2), returns minimum of the 1D array
    parameters:
    values: 1D array or 3D array of form nxnx2
    """
    if(values.ndim == 1):
        return np.min(values)
    elif(values.ndim == 3):
        return np.min(values, axis=-1)
    raise ValueError("Input must be a 1-dimensional or 3-dimensional numpy array.")
    
# def tn_product(values: np.ndarray):
#     if(values.ndim == 1):
#         return np.prod(values)
#     elif(values.ndim == 3):
#         return np.prod(values, axis=-1)
#     raise ValueError("Input must be a 1-dimensional or 3-dimensional numpy array.")
    
    

# def tn_lukasiewicz(values: np.ndarray):
#     return max(0.0, 1.0 - np.sum(1.0 - values))

