#!/usr/bin/env python

import warnings

import numpy as np

from utils import find_nearest_idx

from comp_funcs import *

class Analog(object):
    """Various methods to produce a single deterministic/probabilistic forecast via analog method."""

    def __init__(self,grid_window=3, comp_method=['Rank'], field_weights=[],
                 lat_bounds=[], lon_bounds=[], forecast_lats=[], forecast_lons=[], lat_inc=1., lon_inc=1.,
                 forecast_dates=[]):
        """
        Initialize the analog object.

        :param grid_window:
            integer, +/- number of grid points (n/s, e/w) around each forecast grid point
            to include in local domain pattern matching.
            For example, if grid_window = 3, we use 49 grid points
            to calculate differences between forecast/training
            data (3*2 + 1)*(3+2 +1).
        :param comp_method
            list, Method with which to pattern match. Options: ['Rank','MAE','RMSE',].
            if n_vars > 1, can use different methods to compare each variable.
        :param lat_bounds/lon_bounds:
            list or NumPy array
            A list/numpy array of all latitude within the forecast and training data.\n
            len(allLats) = forecastData[0] or forecastData[1] if forecastData.shape == 3.
        :param forecast_last/forecast_lons:
            list or NumPy array
            Either a list/numpy array of min/max forecast lats/lons (defining a domain) or a single point.
        :param lat_inc/lon_inc:
            float
            The change in latitude/longitude each grid point (e.g. dx or dy).
        :param forecast_dates:
            float
            The change in latitude/longitude each grid point (e.g. dx or dy).
        :return: self
        """

        # --- Here, we make sure the method(s) selected is/are available, and if it/they is/are then add
        available_methods = ['rank','rmse','mae','corr',]
        self.comp_method = []
        for mthd in comp_method:
            if any(mthd.lower() in avail for avail in available_methods):
                self.comp_method.append(mthd.lower())
            else:
                warnings.warn('Method {} not an option! Reverting to rank method'.format(mthd.lower()))
                self.comp_method.append('rank')

        # --- Passed the check!
        self.grid_window = grid_window
        self.field_weights = field_weights

        # --- Here, we check to make sure everything is copacetic with the domain specifications
        # ---
        if len(forecast_lats) != len(forecast_lons):
            raise ValueError("len(lat_bounds) != len(lon_bounds)\nWe either need to forecast for a point or domain.")

        # --- Everything cool? Ok, let's do this.
        if len(forecast_lats) == 1:
            self.point_fcst = True
        else:
            self.point_fcst = False

        self.all_lats = np.arange(lat_bounds[0], lat_bounds[-1] + 1, lat_inc)
        self.all_lons = np.arange(lon_bounds[0], lon_bounds[-1] + 1, lon_inc)

        if self.n_vars == 1:
            if self.all_lats.shape[0] != self.train.shape[0]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in training data.")
            if self.all_lons.shape[0] != self.train.shape[1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in training data.")

            if self.all_lats.shape[0] != self.forecast.shape[0]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in forecast data.")
            if self.all_lons.shape[0] != self.forecast.shape[1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in forecast data.")
        else:
            if self.all_lats.shape[0] != self.train.shape[-2]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in training data.")
            if self.all_lons.shape[0] != self.train.shape[-1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in training data.")

            if self.all_lats.shape[0] != self.forecast.shape[-2]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in forecast data.")
            if self.all_lons.shape[0] != self.forecast.shape[-1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in forecast data.")

        # --- If we're dealing with a point forecast, we will find the closest lat/lon grid points to the fcst point.
        if len(forecast_lats) > 1:
            self.closest_lat = self.all_lats[find_nearest_idx(self.all_lats[:], self.forecast_lats[0])[0]]
            self.closest_lon = self.all_lons[find_nearest_idx(self.all_lons[:], self.forecast_lons[0])[0]]
        # --- If not, we want to find a starting/stopping lat/lon indices to generate a domain-based forecast
        else:
            self.forecast_lats = forecast_lats
            self.forecast_lons = forecast_lons
            self.start_lat_idx = np.where(forecast_lats[0] == all_lats)[0]
            self.start_lon_idx = np.where(forecast_lons[0] == all_lons)[0]
            self.stop_lat_idx = np.where(forecast_lats[1] == all_lats)[0]
            self.stop_lon_idx = np.where(forecast_lons[1] == all_lons)[0]

        # --- Great, it passed! Let's add in some more info for our own edification
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.lat_inc = lat_inc
        self.lon_inc = lon_inc


    def __repr__(self):
        return "<Analog(grid_window={}, comp_method={}, field_weights={},lat_bounds={}, lon_bounds={}, forecast_lats={}, forecast_lons={}, lat_inc={}, lon_inc={})>".format(
                             self.grid_window,self.comp_method, self.field_weights, self.lat_bounds,
                             self.lon_bounds, self.forecast_lats, self.forecast_lons, self.lat_inc, self.lon_inc)


    def __str__(self):
        return "Analog(grid_window={}, comp_method={}, field_weights={},lat_bounds={}, lon_bounds={}, forecast_lats={}, forecast_lons={}, lat_inc={}, lon_inc={})".format(
                             self.grid_window,self.comp_method, self.field_weights, self.lat_bounds,
                             self.lon_bounds, self.forecast_lats, self.forecast_lons, self.lat_inc, self.lon_inc)

    def analog_data(self,train,forecast):

        """
        Initialize the training and forecast data with which to find analogs.

        :param forecast:
            NumPy array, Either a 2-d (lat,lon) or 3-d (n_vars,lat,lon) numpy array of "current" forecast data.
        :param train:
            NumPy array, Either a 3-d (time,lat,lon) or 4-d (n_vars,time,lat,lon) numpy array of "past"/training data.
        :return: self
        """
        # --- Here, we check to make sure everything is copacetic between the forecast/train array shapes
        # --- and other pre-defined variables
        if (len(forecast.shape) > 2) and (forecast.shape[0] > 1):
            self.n_vars = forecast.shape[0]
            # --- First, make sure we're dealing with same number of vars between fcst/train
            if forecast.shape[0] != train.shape[0]:
                raise ValueError("Different number of variables between forecast and training data arrays!")
            # --- Make sure there aren't more field_weights than variables
            if len(field_weights) > self.forecast.shape[0]:
                raise ValueError("More field_weights than variables!")
            if len(comp_method) > self.forecast.shape[0]:
                raise ValueError("More methods than variables!")
        else:
            self.n_vars = 1

        # --- Set up the class attributes
        self.forecast = forecast
        self.train = train


    def find_analogs(self):
        """
        Used to find analogs for a single forecast domain.

        :return analog_idxs:
            NumPy array, indices of closest analogs, from best pattern match to worst.
        """

        # --- Pre-generating analog indices array, this should be faster.
        if len(self.fcstLats) <= 1:
            analog_idxs = np.ones(self.train.shape[0]) * -9999.9
        else:
            # --- Make analog indices array first, then fill it in
            analog_idxs = np.ones(self.train.shape) * -9999.9

        # --- Now, let's find the closest analogs
        if self.n_vars == 1: # --- Only doing a single field
            if self.comp_method[0] == 'rank':
                _rank_analog_grid(self.train,self.forecast,analog_idxs,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
            elif self.comp_method[0] == 'rmse':
                _rmse_analog_grid(self.train,self.forecast,analog_idxs,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
            elif self.comp_method[0] == 'mae':
                _mae_analog_grid(self.train,self.forecast,analog_idxs,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
        elif self.n_vars > 1:
            for meth, nvar in enumerate(self.comp_method):
                if meth == 'rank':
                    _rank_analog_grid(self.train[nvar,...],self.forecast[nvar,...],analog_idxs[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)
                elif meth == 'rmse':
                    _rmse_analog_grid(self.train[nvar,...],self.forecast[nvar,...],analog_idxs[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)
                elif meth == 'mae':
                    _mae_analog_grid(self.train[nvar,...],self.forecast[nvar,...],analog_idxs[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)

                # --- Now, we add weights to the distances...
                analog_idxs[nvar,...] *= self.field_weights[nvar]

            # --- sum up distances along variable axis
            analog_idxs = np.sum(analog_idxs,axis=0)

        # --- now find indices of closest ranks
        self.analogs = argsort_analogs(analog_idxs,elf.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx)