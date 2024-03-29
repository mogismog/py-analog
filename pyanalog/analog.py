#!/usr/bin/env python

import warnings
from utils import find_nearest_idx
import numpy as np
from comp_funcs import _rank_analog_grid,_rmse_analog_grid,_mae_analog_grid,argsort_analogs,interp_proba,gen_proba

class Analog(object):
    """Various methods to produce a single deterministic/probabilistic forecast via analog method."""

    def __init__(self,grid_window=3, comp_method=['Rank'], field_weights=[],
                 lat_bounds=[], lon_bounds=[], forecast_lats=[], forecast_lons=[], lat_inc=1., lon_inc=1.,):
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

        # --- If we're dealing with a point forecast, we will find the closest lat/lon grid points to the fcst point.
        if len(forecast_lats) < 1:
            self.closest_lat = self.all_lats[find_nearest_idx(self.all_lats[:], forecast_lats[0])]
            self.closest_lon = self.all_lons[find_nearest_idx(self.all_lons[:], forecast_lons[0])]
        # --- If not, we want to find a starting/stopping lat/lon indices to generate a domain-based forecast
        else:
            self.forecast_lats = forecast_lats
            self.forecast_lons = forecast_lons
            self.start_lat_idx = np.where(forecast_lats[0] == self.all_lats)[0]
            self.start_lon_idx = np.where(forecast_lons[0] == self.all_lons)[0]
            self.stop_lat_idx = np.where(forecast_lats[1] == self.all_lats)[0]
            self.stop_lon_idx = np.where(forecast_lons[1] == self.all_lons)[0]

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


    def find_analogs(self, train, forecast):
        """
        Used to find analogs for a single forecast domain.
        :param forecast:
            NumPy array, Either a 2-d (lat,lon) or 3-d (n_vars,lat,lon) numpy array of "current" forecast data.
        :param train:
            NumPy array, Either a 3-d (time,lat,lon) or 4-d (n_vars,time,lat,lon) numpy array of "past"/training data.
        :return analog_idxs:
            NumPy array, indices of closest analogs, from best pattern match to worst. Same shape as train array.
        """

        # --- Here, we check to make sure everything is copacetic between the forecast/train array shapes
        # --- and other pre-defined variables
        if (len(forecast.shape) > 2) and (forecast.shape[0] > 1):
            n_vars = forecast.shape[0]
            # --- First, make sure we're dealing with same number of vars between fcst/train
            if forecast.shape[0] != train.shape[0]:
                raise ValueError("Different number of variables between forecast and training data arrays!")
            # --- Make sure there aren't more field_weights than variables
            if len(self.field_weights) > forecast.shape[0]:
                raise ValueError("More field_weights than variables!")
            if len(self.comp_method) > forecast.shape[0]:
                raise ValueError("More methods than variables!")
        else:
            n_vars = 1

        if n_vars == 1:
            if self.all_lats.shape[0] != train.shape[0]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in training data.")
            if self.all_lons.shape[0] != train.shape[1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in training data.")

            if self.all_lats.shape[0] != forecast.shape[0]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in forecast data.")
            if self.all_lons.shape[0] != forecast.shape[1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in forecast data.")
        else:
            if self.all_lats.shape[0] != train.shape[-2]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in training data.")
            if self.all_lons.shape[0] != train.shape[-1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in training data.")

            if self.all_lats.shape[0] != forecast.shape[-2]:
                 raise ValueError("Number of latitude grid points from domain boundaries doesn't equal that in forecast data.")
            if self.all_lons.shape[0] != forecast.shape[-1]:
                 raise ValueError("Number of longitude grid points from domain boundaries doesn't equal that in forecast data.")


        # --- Pre-generating analog indices array, this should be faster.
        self.distances = np.zeros(train.shape)
        # --- Now, let's find the closest analogs
        if n_vars == 1: # --- Only doing a single field
            if self.comp_method[0] == 'rank':
                _rank_analog_grid(train,forecast,self.distances,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
            elif self.comp_method[0] == 'rmse':
                _rmse_analog_grid(self.train,self.forecast,self.distances,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
            elif self.comp_method[0] == 'mae':
                _mae_analog_grid(self.train,self.forecast,self.distances,self.start_lat_idx,self.stop_lat_idx,
                                  self.start_lon_idx,self.stop_lon_idx, self.grid_window)
            self.total_distances = self.distances
        elif n_vars > 1:
            for nvar,meth in enumerate(self.comp_method):
                #print "Finding analogs for variable #{}: method {}".format(nvar+1,meth)
                if meth == 'rank':
                    _rank_analog_grid(train[nvar,...],forecast[nvar,...],self.distances[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)
                elif meth == 'rmse':
                    _rmse_analog_grid(train[nvar,...],forecast[nvar,...],self.distances[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)
                elif meth == 'mae':
                    _mae_analog_grid(train[nvar,...],forecast[nvar,...],self.distances[nvar,...],self.start_lat_idx,self.stop_lat_idx,
                                      self.start_lon_idx,self.stop_lon_idx, self.grid_window)

                # --- Now, we add weights to the distances...
                self.distances[nvar,...] *= self.field_weights[nvar]

            # --- sum up distances along variable axis
            #self.total_distances = np.sum(self.distances,axis=0)

        # --- now find indices of closest ranks
        #self.indices = argsort_analogs(self.total_distances,self.start_lat_idx,self.stop_lat_idx,
        #                              self.start_lon_idx,self.stop_lon_idx)

        return self


    def gen_forecast(self,events,pct_samps,interp=False):
        """
        Function to generate probabilities for forecasts...

        :param events:
        :param interp:
        :return proba:
        """

        if interp:
            probs = interp_proba(self.distances,events,self.start_lat_idx,self.stop_lat_idx,self.start_lon_idx,self.stop_lon_idx,pct_samps)
        if not interp:
            probs = gen_proba(self.indices,events,self.start_lat_idx,self.stop_lat_idx,self.start_lon_idx,self.stop_lon_idx,pct_samps)
        return probs
