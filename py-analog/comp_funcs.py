#!/usr/bin/env python

import numpy as np
from scipy.stats import rankdata as rd
from numba import autojit

@autojit
def _rank_analog_grid(train,forecast,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (train.shape[0],(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = (1,(((grid_window*2)+1)*((grid_window*2)+1)))
    ranked_array = np.empty((train.shape[0]+1,forecast.shape[0],forecast.shape[1]))
    # --- Fill in ranked array
    ranked_array[:-1,...] = train
    ranked_array[-1,...] = forecast
    # --- First, rank data at each array in relevant domain
    for i in range(i_start-grid_window,i_stop+grid_window+1,1): # --- latitudes
        for j in range(j_start-grid_window,j_stop+grid_window+1,1): # --- longitudes
            ranked_array[:,i,j] = rd(ranked_array[:,i,j],method='average')
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            #temp_array[:,:] = in_array[:,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(temp_array.shape)
            out_array[:,i,j] = (np.sum(ranked_array[:-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -ranked_array[-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2),
                                      axis=1))
    return out_array


@autojit
def _rmse_analog_grid(train,forecast,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (train.shape[0],(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = ((((grid_window*2)+1)*((grid_window*2)+1)))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            out_array[:,i,j] = (np.sqrt(np.mean((train[:,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -forecast[i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2))**2,
                                      axis=1,dtype=np.float)))
    return out_array

@autojit
def _mae_analog_grid(train,forecast,out_array,i_start,i_stop,j_start,j_stop, grid_window):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    shp1 = (train.shape[0],(((grid_window*2)+1)*((grid_window*2)+1)))
    shp2 = ((((grid_window*2)+1)*((grid_window*2)+1)))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            out_array[:,i,j] = (np.mean(np.abs(train[:,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -forecast[i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2)),
                                      axis=1))
    return out_array

@autojit
def argsort_analogs(analog_array,i_start,i_stop,j_start,j_stop):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            analog_array[:,i,j] = np.argsort(analog_array[:,i,j])
    return analog_array
