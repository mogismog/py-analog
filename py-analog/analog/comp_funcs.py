#!/usr/bin/env python

import numpy as np
import numba as nb


@jit('void(f8[:,:],f8[:,:],u8[:],u8,u8)')
def _rmse_analog_point(train,forecast,grid_window,):
    """
    Function to find analogous dates based on root mean square error.
    :param train:
    :param forecast:
    :param grid_window:
    :return:
    """

