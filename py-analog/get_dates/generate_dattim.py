#!/usr/bin/env python

from datetime import datetime,timedelta
import itertools

def get_1mo_dates(inyr,inmo,indate,byear,eyear):
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
 window - range of dates in past years to search, e.g. 45 will find dates 45 days before/after indate

 Returns:
 outdates - List of dates meeting the criteria
    """

    fnlist = []

    #print inmo,indate
    try:
        xdate = datetime(byear,inmo,indate)
    except ValueError:
        xdate = datetime(byear,inmo,indate-1)
    else:
        xdate = datetime(byear,inmo,indate)
    while xdate < datetime(eyear+1,1,1):
        #print xdate
        if xdate.year == inyr:
            try:
                xdate = datetime((xdate.year + 1),inmo,indate)
            except ValueError:
                xdate = datetime((xdate.year + 1),inmo,indate-1)
            continue
        for datechange in xrange(0,35):
                tdelta = timedelta(days=datechange)
                analogdate = xdate + tdelta
                #print analogdate,xdate
                if analogdate.year > eyear:
                    continue
                if analogdate.year == inyr:
                    continue
                if analogdate.month != xdate.month:
                    continue
                fnlist.append(analogdate)
        try:
            xdate = datetime((xdate.year + 1),inmo,indate)
        except ValueError:
            xdate = datetime(xdate.year+1,inmo,indate-1)

    return fnlist

def get_analog_dates(forecastDate,window,byear,eyear, bias_corr=False, month_range=True, **kwargs):
    """
 Very useful with netCDF4 files and py-netCDF.

 From 1985-2011, a number of days range to search
 for dates centered around the given date and forecast time,
 returns the applicable dates in a list of daetime objects.
 In other words, returns a list of file name dates for us to
 search for analogs/use in logistic regression/whatever.

inputs:
 forecastDate - some datetime object of the forecast date
 window - range of dates in past years to search, e.g. 45 will find dates 45 days before/after indate\n
    if month_range == True, this number represents the number of months before/after forecastDate.month,
    so if window = 1 and forecastDate.month = April, will use data from March-May.
 byear - earliest year for potential dates (byear/1/1)
 eyear - latest year for potential dates (eyear/12/31)

Optional arguments:
 bias_corr - If True, data is assumed to be bias-corrected and supplemental dates (in seasonally-similar time frames)\n
    will be used
 month_range - If True, window will be n months before/after fcst month instead of n days before/after fcst date.

Returns:
 outdates - List of dates meeting the criteria
    """

    bias_corr = bias_corr
    month_range = month_range


    fnlist = []
    date_list = []

    if month_range:
        try:
            xdate = datetime(byear,forecastDate.month,forecastDate.date)
        except ValueError:
            # --- For leap year issues
            xdate = datetime(byear,forecastDate.month,forecastDate.date-1)

        while xdate < datetime(eyear+1,1,1):
            #print xdate
            if xdate.year == forecastDate.year:
                try:
                    xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date)
                except ValueError:
                    xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date-1)
                continue
            for datechange in reversed(xrange(0,100)):
                if xdate.month > 1:
                    tdelta = timedelta(days=datechange)
                    analogdate = xdate - tdelta
                    if analogdate.year < byear:
                        continue
                    if analogdate.year == forecastDate.year:
                        continue
                    if analogdate < datetime(xdate.year,xdate.month-1,1):
                        continue
                    fnlist.append(analogdate)
                elif xdate.month == 1:
                    tdelta = timedelta(days=datechange)
                    analogdate = xdate - tdelta
                    if analogdate.year < byear:
                        continue
                    if analogdate.year == forecastDate.year:
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
                    if analogdate.year == forecastDate.year:
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
                    if analogdate.year == forecastDate.year:
                        continue
                    if analogdate >= datetime(xdate.year+1,2,1):
                        continue
                    fnlist.append(analogdate)
            try:
                xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date)
            except ValueError: # --- 2/29 on non-leap year issue
                xdate = datetime(xdate.year+1,forecastDate.month,forecastDate.date-1)

    if not month_range:
        try:
            xdate = datetime(byear,forecastDate.month,forecastDate.date)
        except ValueError:
            # --- For leap year issues
            xdate = datetime(byear,forecastDate.month,forecastDate.date-1)

        while xdate < datetime(eyear+1,1,1):
            #print xdate
            if xdate.year == forecastDate.year:
                try:
                    xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date)
                except ValueError: # --- 2/29 on non-leap year issue
                    xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date-1)
                continue
            for datechange in reversed(xrange(0,window+1)):
                tdelta = timedelta(days=datechange)
                analogdate = xdate - tdelta
                if analogdate.year < byear:
                    continue
                if analogdate.year == forecastDate.year:
                    continue
                fnlist.append(analogdate)

            for datechange in xrange(1,window):
                tdelta = timedelta(days=datechange)
                analogdate = xdate + tdelta
                if analogdate.year > eyear:
                    continue
                if analogdate.year == forecastDate.year:
                    continue
                fnlist.append(analogdate)

            try:
                xdate = datetime((xdate.year + 1),forecastDate.month,forecastDate.date)
            except ValueError: # --- 2/29 on non-leap year issue
                xdate = datetime(xdate.year+1,forecastDate.month,forecastDate.date-1)

    # --- Here, since we are now using bias-corrected data, we can get additional potential analog dates!
    if bias_corr:

        date_list.append(fnlist)

        if (forecastDate.month < 2) or (forecastDate.month > 9):
           date_list.append(get_1mo_dates(int(forecastDate.year),3,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),4,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),5,1,byear,eyear))
        if (forecastDate.month == 2):
           date_list.append(get_1mo_dates(int(forecastDate.year),4,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),5,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),10,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),11,1,byear,eyear))
        if (forecastDate.month == 3):
           date_list.append(get_1mo_dates(int(forecastDate.year),5,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),10,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),11,1,byear,eyear))
        if (forecastDate.month == 4):
           date_list.append(get_1mo_dates(int(forecastDate.year),9,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),10,1,byear,eyear))
           date_list.append(get_1mo_dates(int(forecastDate.year),11,1,byear,eyear))

        # --- Now flatten and return the list
        date_list = list(itertools.chain.from_iterable(date_list))
        return date_list
    else:
        return fnlist