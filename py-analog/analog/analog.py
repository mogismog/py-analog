#!/usr/bin/env python

import warnings

import numpy as np

from pyanalog.utils import find_nearest_idx

class SFAnalog(object):
    """Various methods to produce a single deterministic/probabilistic forecast via analog method."""

    def __init__(self, forecast, train, grid_window=3, comp_method=['Rank'], field_weights=[],
                 lat_bounds=[], lon_bounds=[], forecast_lats=[], forecast_lons=[], lat_inc=1., lon_inc=1.):
        """
        Initialize the analog object.

        :param forecast:
            NumPy array, Either a 2-d (lat,lon) or 3-d (n_vars,lat,lon) numpy array of "current" forecast data.
        :param train:
            NumPy array, Either a 3-d (time,lat,lon) or 4-d (n_vars,time,lat,lon) numpy array of "past"/training data.
        :param forecast_date:
            datetime object, the forecast date in question.
        :param grid_window:
            integer, +/- number of grid points (n/s, e/w) around each forecast grid point
            to include in local domain pattern matching.
            For example, if grid_window = 3, we use 49 grid points
            to calculate differences between forecast/training
            data (3*2 + 1)*(3+2 +1).
        :param comp_method
            list, Method with which to pattern match. Options: ['Rank','MAE','RMSE','Corr', 'MSE'].
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
        :return: self
        """
        # --- Set up the class attributes
        self.forecast = forecast
        self.train = train
        self.forecast_date = forecast_date

        # --- Here, we check to make sure everything is copacetic.
        if (len(self.forecast.shape) > 2) and (self.forecast.shape[0] > 1):
            self.n_vars = self.forecast.shape[0]
            # --- First, make sure we're dealing with same number of vars between fcst/train
            if self.forecast.shape[0] != self.train.shape[0]:
                raise ValueError("Different number of variables between forecast and training data arrays!")
            # --- Make sure there aren't more field_weights than variables
            if len(field_weights) > self.forecast.shape[0]:
                raise ValueError("More field_weights than variables!")
            if len(comp_method) > self.forecast.shape[0]:
                raise ValueError("More methods than variables!")
        else:
            self.n_vars = 1

        # --- Now, to make sure the method selected is available, and if it is then add
        available_methods = ['rank','rmse','mae','corr','mse']
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
        if len(forecast_lats) > 1:
            self.closest_lat = self.all_lats[find_nearest_idx(self.all_lats[:], self.forecast_lats[0])[0]]
            self.closest_lon = self.all_lons[find_nearest_idx(self.all_lons[:], self.forecast_lons[0])[0]]
        else:
            self.forecast_lats = forecast_lats
            self.forecast_lons = forecast_lons

        # --- Great, it passed! Let's add in some more info for our own edification
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.lat_inc = lat_inc
        self.lon_inc = lon_inc

    def __repr__(self):
        return "<Analog(grid_window={}, comp_method={}, field_weights={},lat_bounds={}, lon_bounds={}, forecast_lats={}, forecast_lons={}, lat_inc={}, lon_inc={})>".format(
                             self.storm_id,self.name, self.tc_status, self.datetime, self.latitude, self.longitude,
                             self.tc_type)


    def find_analogs(self):
        """
        Used to find analogs for a single forecast domain.

        returns:
            analog_idxs - indices of closest analogs, from best pattern match to worst.
        """

        # --- Pre-generating analog indices array, this should be faster.
        if len(self.fcstLats) <= 1:
            analog_idxs = np.ones(self.train.shape[0]) * -9999.9
        else:
            if self.n_vars == 1:
                analog_idxs = np.ones(self.train.shape) * -9999.9
            else:
                analog_idxs = np.ones((self.train.shape[0], self.train.shape[2], self.train.shape[3])) * -9999.9

        # --- Ok, let's do this...
        if self.point_fcst:
            blah = 0
            # --- We then find the closest analogs

class MFAnalog(object):
    """Various methods to produce multiple deterministic/probabilistic forecasts via analog method."""

    def __init__(self, forecast, train, ):
        """
        Initialize the analog object.

        :param forecast:
            NumPy array, Either a 2-d (lat,lon) or 3-d (n_vars,lat,lon) numpy array of "current" forecast data.
        :param train:
            NumPy array, Either a 3-d (time,lat,lon) or 4-d (n_vars,time,lat,lon) numpy array of "past"/training data.
        :return:
        """
        # --- Set up the class attributes
        self.forecast = forecast
        self.train = train


    def analog_params(self, grid_window=3, comp_method=['Rank'], field_weights=[]):
        """
        Set up some parameters for whatever analog method.

        Parameters
        ----------
        grid_window : integer
            +/- number of grid points (n/s, e/w) around each forecast grid point to include in local domain pattern\n
            matching. For example, if grid_window = 3, we use 49 grid points to calculate differences between forecast/training\n
            data (3*2 + 1)*(3+2 +1).
        comp_method : list
            Method with which to pattern match. Options: ['Rank','MAE','RMSE','Corr'].
            if n_vars > 1, can use different methods to compare each variable.
        """

        # --- Here, we check to make sure everything is copacetic.
        if (len(self.forecast.shape) > 2) and (self.forecast.shape[0] > 1):
            self.n_vars = self.forecast.shape[0]
            # --- First, make sure we're dealing with same number of vars between fcst/train
            if self.forecast.shape[0] != self.train.shape[0]:
                raise ValueError("Different number of variables between forecast and training data arrays!")
            # --- Make sure there aren't more field_weights than variables
            if len(field_weights) > self.forecast.shape[0]:
                raise ValueError("More field_weights than variables!")
            if len(comp_method) > self.forecast.shape[0]:
                raise ValueError("More methods than variables!")
        else:
            self.n_vars = 1

        # --- Now, to make sure the method(2) selected is available, and if it is then add
        available_methods = ['rank','rmse','mae','corr']
        self.comp_method = []
        for mthd in comp_method:
            if any(mthd.lower() in avail for avail in available_methods):
                self.comp_method.append(mthd.lower())
            else:
                warnings.warn('Method {} not an option! Reverting to rank method'.format(mthd.lower()))
                self.comp_method.append('rank')



        # --- Everything cool? Ok, let's do this.
        self.grid_window = grid_window
        self.weights = field_weights

        return self

    def domain(self, lat_bounds=[], lon_bounds=[], forecast_lats=[], forecast_lons=[], lat_inc=1, lon_inc=1):
        """
        Main and forecast domain setup.

        Parameters
        ----------
        lat_bounds/lon_bounds : list or NumPy array
            A list/numpy array of all latitude within the forecast and training data.\n
            len(allLats) = forecastData[0] or forecastData[1] if forecastData.shape == 3.
        forecast_last/forecast_lons : list or NumPy array
            Either a list/numpy array of min/max forecast lats/lons (defining a domain) or a single point.
        lat_inc/lon_inc : float
            The change in latitude/longitude each grid point (e.g. dx or dy).
        """
        # --- Here, we check to make sure everything is copacetic.
        if len(forecast_lats) != len(forecast_lons):
            raise ValueError("len(lat_bounds) != len(lon_bounds)\nWe either need to forecast for a point or domain.")

        # --- Everything cool? Ok, let's do this.
        if len(forecast_lats) == 1:
            self.point_fcst = True
        else:
            self.point_fcst = False

        self.allLats = np.arange(lat_bounds[0], lat_bounds[-1] + 1, lat_inc)
        self.allLons = np.arange(lon_bounds[0], lon_bounds[-1] + 1, lon_inc)
        # --- If we're dealing with a point forecast, we will find the closest lat/lon grid points to the fcst point.
        if len(forecast_lats) > 1:
            self.closest_lat = self.allLats[find_nearest_idx(self.allLats[:], self.fcstLats[0])[0]]
            self.closest_lon = self.allLons[find_nearest_idx(self.allLons[:], self.fcstLons[0])[0]]
        else:
            self.fcstLats = forecast_lats
            self.fcstLons = forecast_lons

        return self



