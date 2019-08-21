import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm as Gaussian


date_format = "%Y-%m-%d"

def date_prev_year(date, years):
  if (years == 0):
    return date

  msday = 1000*60*60*24
  date_wd = pd.Timestamp(date).weekday()

  ly_date = date - pd.Timedelta(years, unit='Y')

  ly_wd = pd.Timestamp(ly_date).weekday()
  ly_date = ly_date - pd.Timedelta((ly_wd - date_wd) * msday, unit='ms')

  ly_wd = ly_date.weekday()
  if (ly_wd != date_wd):
     print("Current weekday != last year weekday: %s != %s" % (date_wd, ly_wd))

  return pd.Timestamp(ly_date.date())


def call_seasons(season_df):
    df = season_df[['StartDate', 'EndDate']]
    dict_of_periods = {}

    for index, row in df.iterrows():
        start_date = row['StartDate']
        end_date = row['EndDate']

        s_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        e_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        day = timedelta(days=1)
        date_list = []

        while s_date <= e_date:
            date_list.append(s_date.strftime('%Y-%m-%d'))
            s_date = s_date + day

        p_name = 'Period ' + str(index + 1)
        dict_of_periods[p_name] = date_list

    return dict_of_periods


def season_wod(dates_df):
    ## This for loop is used to created the wod for each date in a period range
    season_df, season_dict = {}, {}
    for k, v in dates_df.items():
        for d in v:
            wod = pd.Timestamp(d).weekday()
            final_dict = [wod, d]
            if k in season_dict.keys():
                season_dict.setdefault(k, []).append(final_dict)
            else:
                season_dict.setdefault(k, []).append(final_dict)


    ## This for loop is used to reassign every day in a period with respect to it's wod
    for k1, v1 in season_dict.items():
        season_df2 = {}
        for k2, v2 in v1:
            if k2 in season_df2:
                season_df2[k2].append(v2)
            else:
                season_df2[k2] = [v2]
        if k1 in season_df:
            season_df[k1].append([season_df2])
        else:
            season_df[k1] = season_df2

    return season_df


def mad_process_RT(l):
    ## The threshold...
    thld = 3.0
    idx = list()
    x = np.nanmedian(l)
    mad_score = mad(l)
    for ind, val in enumerate(l):
        dev = ((.6745 * (val - x)) / mad_score)
        if abs(dev) >= thld:
            l[ind] = x
            idx.append(1)
        else:
            l[ind] = val
            idx.append(0)
    ret = np.nanmedian(l)

    return ret if ret != np.nan else 0.0, idx


def mad_process_RN(l):
    ## The threshold...
    thld = .25
    idx = list()
    x = np.nanmedian(l)
    mad_score = mad(l)
    for ind, val in enumerate(l):
        dev = ((.6745 * (val - x)) / mad_score)
        if abs(dev) >= thld:
            l[ind] = x
            idx.append(1)
        else:
            l[ind] = val
            idx.append(0)
    ret = np.nanmedian(l)

    return ret if ret != np.nan else 0.0, idx


def assign_outliers_RT(season_dict, dict_sorted, dates_dict):
    outliers_list = []
    outliers_dict = {}
    for k, v in season_dict.items():
        for k1, v1 in v.items():
            date_array, val_array = [], []
            for i in v1:
                if i in dict_sorted:
                    for val in dates_dict[i]:
                        date_array.append(val[0]) 
                        val_array.append(val[1])
                    
            mad_verification = mad_process_RT(val_array)[1]    ## MAD formula applied here..

            for i, val in enumerate(mad_verification):
                if val == 1:
                    outliers_list.append(date_array[i])

    outliers_dict["Outliers"] = outliers_list

    return outliers_dict


def assign_outliers_RN(season_dict, dict_sorted, dates_dict):
    outliers_list = []
    outliers_dict = {}
    for k, v in season_dict.items():
        for k1, v1 in v.items():
            date_array, val_array = [], []
            for i in v1:
                if i in dict_sorted:
                    for val in dates_dict[i]:
                        date_array.append(val[0]) 
                        val_array.append(val[1])
                    
            mad_verification = mad_process_RN(val_array)[1]   ## MAD formula applied here..

            for i, val in enumerate(mad_verification):
                if val == 1:
                    outliers_list.append(date_array[i])

    outliers_dict["Outliers"] = outliers_list

    return outliers_dict


## Assigns revenue values to each date by form of a tuple...
def assign_val_to_dates(date_array, rev_array):
    dates_dict, revenue_dict, d_final_dict = {}, {}, {}
    for (k, v), (k1, v1) in zip(date_array.items(), rev_array.items()):
        for (k2, v2), (k3, v3) in zip(v.items(), v1.items()):
            dates_dict[k] = v2
            revenue_dict[k1] = v3

    for (k3, v3), (k4, v4) in zip(dates_dict.items(), revenue_dict.items()):
        d_final_dict[k3] = list(zip(v3, v4))

    return d_final_dict


## Test the mad definition
def mad(a, c=Gaussian.ppf(3 / 4.), axis=0, center=np.nanmedian):
    ## c \approx .6745
    """
    The Median Absolute Deviation along given axis of an array

    Parameters
    ----------
    a : array-like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately .6745.
    axis : int, optional
        The defaul is 0. Can also be None.
    center : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called center(a). The axis argument will be applied
        via np.apply_over_axes. Otherwise, provide a float.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    a = np.asarray(a)
    if callable(center):
        center = np.apply_over_axes(center, a, axis)

    return np.nanmedian((np.fabs(a - center)) / c, axis=axis)


def returns_range(data, from_date=None, to_date=None):
  def __get(df, index, column, default=np.nan):
    try:
      return df.ix[index][column]
    except:
      return np.nan

  def __dates(date_values, cols_name):

    dates = date_values
    dates = dates.strftime(date_format)
    return dates

  if (from_date != None):
    data = data[from_date:]
  if (to_date != None):
    data = data[:to_date]

  years = sorted(list(set(data.index.year)))
  latest_year = years[-1]
  num_days = 366 # If calendar.isleap(latest_year) else 365

  first_date = datetime.strptime('%s-01-01' % latest_year, '%Y-%m-%d')
  d_index = [pd.Timestamp(first_date + timedelta(days=x)) for x in range(0, num_days)]

  years_back = [int(latest_year - ym) for ym in reversed(years)]

  d_dict, dates_dict, outliers_dict = {}, {}, {}

  for i,date in enumerate(d_index):
    ## Array with rev values...
    d_dict[date.strftime(date_format)] = {(c + '_revenue'): ([__get(data, date_prev_year(date, y), c) 
                                                              for y in years_back]) for c in data.columns.values}

    ## Array with date values
    dates_dict[date.strftime(date_format)] = ({(c + '_dates'): [__dates(date_prev_year(date, y), c)
                                                         for y in years_back] for c in data.columns.values})


  return d_dict, dates_dict

