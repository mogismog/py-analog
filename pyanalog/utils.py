import numpy as np
from datetime import datetime, timedelta
import itertools


def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin(axis=0)
    return idx


def get_1mo_dates(inyr, inmo, indate, byear, eyear):
    """
 Used with netCDF4 files and py-netCDF.

 From 1985-2011, a number of days range to search
 for dates centered around the given date and forecast time,
 returns the applicable dates in a list of daetime objects.
 In other words, returns a list of file name dates for us to
 search for analogs/use in logistic regression/whatever.

 indate - Initial date (1,31)
 inyr - Initial year, YYYY (1985 - )
 inmo - Initial month, (1,12)

 Returns:
 outdates - List of dates meeting the criteria
    """

    fnlist = []

    # print inmo,indate
    try:
        xdate = datetime(byear, inmo, indate)
    except ValueError:
        xdate = datetime(byear, inmo, indate - 1)
    else:
        xdate = datetime(byear, inmo, indate)
    while xdate < datetime(eyear + 1, 1, 1):
        # print xdate
        if xdate.year == inyr:
            try:
                xdate = datetime((xdate.year + 1), inmo, indate)
            except ValueError:
                xdate = datetime((xdate.year + 1), inmo, indate - 1)
            continue
        for datechange in xrange(0, 35):
            tdelta = timedelta(days=datechange)
            analogdate = xdate + tdelta
            # print analogdate,xdate
            if analogdate.year > eyear:
                continue
            if analogdate.year == inyr:
                continue
            if analogdate.month != xdate.month:
                continue
            fnlist.append(analogdate)
        try:
            xdate = datetime((xdate.year + 1), inmo, indate)
        except ValueError:
            xdate = datetime(xdate.year + 1, inmo, indate - 1)

    return fnlist


def get_analog_dates(forecast_date, window, byear, eyear, all_dates=False, month_range=True,):
    """
 Very useful with netCDF4 files and py-netCDF.

 From 1985-2011, a number of days range to search
 for dates centered around the given date and forecast time,
 returns the applicable dates in a list of daetime objects.
 In other words, returns a list of file name dates for us to
 search for analogs/use in logistic regression/whatever.

inputs:
 forecast_date - some datetime object of the forecast date
 window - range of dates in past years to search, e.g. 45 will find dates 45 days before/after indate\n
    if month_range == True, this number represents the number of months before/after forecast_date.month,
    so if window = 1 and forecast_date.month = April, will use data from March-May.
 byear - earliest year for potential dates (byear/1/1)
 eyear - latest year for potential dates (eyear/12/31)

Optional arguments:
 all_dates - If True, data is assumed to be bias-corrected and all *possible* dates (while still taking into account
    cross validation rules) will be used. If true, month_range is ignored.
 month_range - If True, window will be n months before/after fcst month instead of n days before/after fcst date.
    Ignored if all_dates = True


Returns:
 outdates - List of dates meeting the criteria
    """

    fnlist = []

    # --- Here, since we are now using bias-corrected data, we can get additional potential analog dates!
    if all_dates:
        xdate = datetime(byear,1,1)

        while xdate <= datetime(eyear,12,31):
            if xdate.year != forecast_date.year: # --- cross validation
                if np.abs((xdate-forecast_date).days) > 31: # --- more CV, don't want dates too close together

                    fnlist.append(xdate)
            xdate += timedelta(days=1)

    else:
        if month_range:
            try:
                xdate = datetime(byear,forecast_date.month,forecast_date.day)
            except ValueError:
                # --- For leap year issues
                xdate = datetime(byear,forecast_date.month,forecast_date.day-1)

            while xdate < datetime(eyear+1,1,1):
                #print xdate
                if xdate.year == forecast_date.year:
                    try:
                        xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.day)
                    except ValueError:
                        xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.day-1)
                    continue
                for datechange in reversed(xrange(0,100)):
                    if xdate.month > 1:
                        tdelta = timedelta(days=datechange)
                        analogdate = xdate - tdelta
                        if analogdate.year < byear:
                            continue
                        if analogdate.year == forecast_date.year:
                            continue
                        if analogdate < datetime(xdate.year,xdate.month-1,1):
                            continue
                        fnlist.append(analogdate)
                    elif xdate.month == 1:
                        tdelta = timedelta(days=datechange)
                        analogdate = xdate - tdelta
                        if analogdate.year < byear:
                            continue
                        if analogdate.year == forecast_date.year:
                            continue
                        if analogdate < datetime(xdate.year-1,12,1):
                            continue
                        fnlist.append(analogdate)
                for datechange in xrange(1,101):
                    if xdate.month < 12:
                        tdelta = timedelta(days=datechange)
                        analogdate = xdate + tdelta
                        if analogdate.year > eyear:
                            continue
                        if analogdate.year == forecast_date.year:
                            continue
                        try:
                            datetime(xdate.year,xdate.month+2,1)
                        except ValueError: # --- xdate.month == 11
                            if analogdate >= datetime(xdate.year+1,1,1):
                                continue
                        else:
                            if analogdate >= datetime(xdate.year,xdate.month+2,1):
                                continue
                        fnlist.append(analogdate)
                    elif xdate.month == 12:
                        tdelta = timedelta(days=datechange)
                        analogdate = xdate + tdelta
                        if analogdate.year > eyear:
                            continue
                        if analogdate.year == forecast_date.year:
                            continue
                        if analogdate >= datetime(xdate.year+1,2,1):
                            continue
                        fnlist.append(analogdate)
                try:
                    xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.day)
                except ValueError: # --- 2/29 on non-leap year issue
                    xdate = datetime(xdate.year+1,forecast_date.month,forecast_date.day-1)

        if not month_range:
            try:
                xdate = datetime(byear,forecast_date.month,forecast_date.date)
            except ValueError:
                # --- For leap year issues
                xdate = datetime(byear,forecast_date.month,forecast_date.date-1)

            while xdate < datetime(eyear+1,1,1):
                #print xdate
                if xdate.year == forecast_date.year:
                    try:
                        xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.date)
                    except ValueError: # --- 2/29 on non-leap year issue
                        xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.date-1)
                    continue
                for datechange in reversed(xrange(0,window+1)):
                    tdelta = timedelta(days=datechange)
                    analogdate = xdate - tdelta
                    if analogdate.year < byear:
                        continue
                    if analogdate.year == forecast_date.year:
                        continue
                    fnlist.append(analogdate)

                for datechange in xrange(1,window):
                    tdelta = timedelta(days=datechange)
                    analogdate = xdate + tdelta
                    if analogdate.year > eyear:
                        continue
                    if analogdate.year == forecast_date.year:
                        continue
                    fnlist.append(analogdate)

                try:
                    xdate = datetime((xdate.year + 1),forecast_date.month,forecast_date.date)
                except ValueError: # --- 2/29 on non-leap year issue
                    xdate = datetime(xdate.year+1,forecast_date.month,forecast_date.date-1)

    return fnlist