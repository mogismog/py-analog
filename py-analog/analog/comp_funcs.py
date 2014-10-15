#!/usr/bin/env python

import numpy as np
from scipy.stats import rankdata as rd
from numba import autojit

@autojit
def _rank_analog_grid(in_array,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (in_array.shape[0]-1,(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = (1,(((grid_window*2)+1)*((grid_window*2)+1)))
    ranked_array = np.ones((in_array.shape))*-9999
    # --- First, rank data at each array in relevant domain
    for i in range(i_start-grid_window,i_stop+grid_window+1,1): # --- latitudes
        for j in range(j_start-grid_window,j_stop+grid_window+1,1): # --- longitudes
            ranked_array[:,i,j] = rd(in_array[:,i,j],method='average')
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            #temp_array[:,:] = in_array[:,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(temp_array.shape)
            out_array[:,i,j] = np.argsort(np.sum(ranked_array[:-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -ranked_array[-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2),
                                      axis=1,dtype=np.int))
    return out_array


@autojit
def _rmse_analog_grid(in_array,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (in_array.shape[0]-1,(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = (1,(((grid_window*2)+1)*((grid_window*2)+1)))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            out_array[:,i,j] = np.argsort(np.sqrt(np.mean((in_array[:-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -in_array[-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2))**2,
                                      axis=1,dtype=np.float),dtype=np.float))
    return out_array

@autojit
def _mae_analog_grid(in_array,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (in_array.shape[0]-1,(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = (1,(((grid_window*2)+1)*((grid_window*2)+1)))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            out_array[:,i,j] = np.argsort(np.mean(np.abs(in_array[:-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -in_array[-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2)),
                                      axis=1,dtype=np.float))
    return out_array