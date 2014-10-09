import numpy as np

# --- Used for percentiles
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin(axis=0)
    return idx