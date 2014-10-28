#!/usr/bin/env python

import numpy as np
from scipy.stats import rankdata as rd
from numba import autojit
from scipy.interpolate import Rbf

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
            out_array[:,i,j] = (np.sum(np.abs(ranked_array[:-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp1)
                                      -ranked_array[-1,i-grid_window:i+grid_window+1,j-grid_window:j+grid_window+1].reshape(shp2)),
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
    out_array = np.zeros(analog_array.shape)
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            out_array[:,i,j] = np.argsort(analog_array[:,i,j])
    return analog_array

@autojit
def interp_proba(distances,events,i_start,i_stop,j_start,j_stop,pct_samps):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    probs = np.zeros((distances.shape[-2:]))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            temp_events = events[:,i,j]
            # --- First, find maximum distance based on % of samples used
            # --- scale distances first
            #scaled_dists = np.ones(distances.shape[:2])
            #for v in range(distances.shape[0]):
            #    scaled_dists[v,:] = (distances[v,:,i,j].T - np.min(distances[v,:,i,j]))/(np.max(distances[v,:,i,j]) - np.min(distances[v,:,i,j]))
            #all_distances = np.sqrt(np.sum(scaled_dists.T**2,axis=0)) # --- Euclidean distance from "perfect" match
            #max_rad = np.percentile(all_distances,pct_samps*100.) # --- max distance
            #distance_indices = np.where(all_distances <= max_rad)[0]
            #weights = ((max_rad - all_distances[distance_indices])/(max_rad*all_distances[distance_indices]))**2
            # --- set up RBF function
            #print events[:,i,j].max()
            rbfi = Rbf(distances[0,:,i,j],distances[1,:,i,j],distances[2,:,i,j],events[:,i,j])
            #probs[i,j] = np.sum(weights*temp_events[distance_indices])/np.sum(weights)
            probs[i,j] = rbfi(0,0,0) # --- perfect forecast
    return probs

@autojit
def gen_proba(distances,events,i_start,i_stop,j_start,j_stop,n_analogs):
    """
    Function to find analogous dates based on root mean square error.
    :param array:
        Some 2-d numpy array
    :return self:
    """
    probs = np.zeros((distances.shape[-2:]))
    # --- Now find differences at each forecast grid point and local-domain
    for i in range(i_start,i_stop+1,1): # --- latitudes
        for j in range(j_start,j_stop+1,1): # --- longitudes
            temp_events = events[:,i,j]
            # --- First, find maximum distance based on % of samples used
            # --- scale distances first
            distance_indices = np.where(distances[:,i,j] < n_analogs)[0]
            probs[i,j] = np.sum(temp_events[distance_indices])/len(distance_indices)
    return probs