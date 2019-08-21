# -*- encoding: utf-8 -*-
#
# Copyright (c) 2016 Chronos AS
#
# Authors: Fredrik Stormo, Stefan Remman
# Contact: kjetil.karlsen@chronosit.no

import pandas as pd
from os import path

import matplotlib.pyplot as plt
import matplotlib.dates as mpldates

from datetime import datetime, timedelta
from pandas.tseries.index import DatetimeIndex, Float64Index, Int64Index
from sklearn.preprocessing import Imputer
from collections import OrderedDict

from d2o.utils import logger as log
from d2o.utils.handlers import ExceptionHandler

try:
  import d2o.utils.calc as calc
except Exception as e:
  log.err(ExceptionHandler(e))

#import calc as calc
import random
import copy
import numpy as np
import seaborn as sns

dejfont = {'fontname':'Dejavu Sans Mono'}

def unicode_columns(df, encoding='utf-8'):
  # TODO: Specify the possible encodings.
  """
  Changes encoding of columns to 'encoding'.

  Parameters
  ----------
  df : pd.DataFrame

  encoding : str, default 'utf-8'
    No need to use any other encoding than utf-8, unless you have a specific purpose.

  Returns
  -------
  unicode_columns : list
    The new column keys in unicode encoding.

  Notes
  -----
  Deprecated, used for testing.
  Currently only converts a few scandinavian characters.
  """
  original_columns = df.columns.values
  unicode_columns = []
  for x in original_columns:
    try:
      x = x.encode('utf-8')
      x = x.replace('ø', 'oe').replace('Ø', 'Oe').replace('å', 'aa').replace('Å', 'Aa').replace('æ', 'ae').replace('Æ', 'Ae')
      unicode_columns.append(x.lower())
    except Exception as e:
      print(e)
      unicode_columns.append(x.lower())
  return unicode_columns

def add_mav(df, window_size):
  '''
  Parameters
  ----------
  df : pd.DataFrame

  window_size : int

  Returns
  -------
  combined_df : pd.dataframe
      df with extra columns containing the moving average. Columns with moving average
      is named by appending "_MAV" to the end of the column name/key.
  '''
  column_names = df.columns.values
  mat = df.as_matrix()
  mav_mat = calc.moving_average_mat(mat, window_size)
  mav_column_names = ['%s_MAV(%s)' % (x, window_size) for x in column_names]
  mav_df = pd.DataFrame.from_records(mav_mat, index=df.index, columns=mav_column_names)

  return combine([df, mav_df])

def combine(df_list):
  """
  Combines the data frames in df_list into one data frame.

  Parameters
  ----------
  df_list : list of pd.DataFrame

  Returns
  -------
  concat_df : pd.DataFrame
    The composite data frame of all the data frames in df_list.

  Notes
  -----
  The combination is done by 'adding' the columns of the data frames to one big
  data frame.

  ex: df_1 contains columns 'A and 'B', df_2 contains 'C' and 'D'. Then the composite
  data frame df_composite have columns 'A', 'B', 'C' and 'D'.
  """
  return pd.concat(df_list, axis=1)

def scale(df, scaler):
  """
  Scales the values in each column using the provided scaler.

  Parameters
  ----------
  df : pd.DataFrame

  scaler : d2o.prep.scaling.Scaler instance

  Returns
  -------
  sdf : pd.DataFrame
    Scaled data frame.
  """
  try:
    sdf = pd.DataFrame(scaler.fit(df).transform(df))
  except Exception as e:
    log.err(ExceptionHandler(e))
  #sdf.index = df.index
  #sdf.columns = df.columns
  return sdf

def replace(df, value, replace_with=np.nan):
  """
  Replaces all the instances of 'value' with replace_with.

  Parameters
  ----------
  df : pd.DataFrame

  value : any type
    The values to replace in df.

  replace_with : any type
    If any elements in df is equal to value using the logic element == value
    the that element is changed to 'replace with'.

  Returns
  -------
  df_replaced : pd.DataFrame

  Notes
  -----
  Uses the pd.DataFrame.applymap() method.
  """
  return df.applymap(lambda x: replace_with if x == value else x)

def impute_weekdays(df, include_zeros=False):
  """
  Set NaN's to that weekday's median.

  Parameters
  ----------
  df : pd.DataFrame

  include_zeros : bool, default False

  Returns
  -------
  df : pd.DataFrame
    Dataframe with all zeros set to their weekdays median.
  """
  if (include_zeros):
    df = df.replace(0, np.nan)

  df['Weekday'] = df.index.weekday
  medians = {i:df[df['Weekday'] == i].median() for i in range(0,7)}

  dfs = []
  for i in range(0,7):
    dfs.append(df[df['Weekday']==i].fillna(medians[i]))

  df = pd.concat(dfs).sort_index()
  return df

def expand_date_index(df, precision='day'):
  """
  Adds columns for year, month, week and so on. Highest precision is 'hour'.

  Parameters
  ----------
  df : pd.DataFrame

  precision : str, default 'day'
    Can be set to 'hour' for max precision, anything else will give the standard
    'Weekday', 'Week', 'Month' and 'Year' columns.

  Returns
  -------
  df : pd.DataFrame
    Dataframe with date information expanded in to several columns.
  """
  if (precision == 'hour'):
    df['Seconds'] = df.index.seconds
    df['Hour'] = df.index.hour
  df['Weekday'] = df.index.weekday
  df['Week'] = df.index.week
  df['Month'] = df.index.month
  df['Year'] = df.index.year

  return df

def from_csv(path, *args, **kwargs):
  """
  A wrap of pd.read_csv(). all arguments are passed to that function.
  """
  return pd.read_csv(path, *args, **kwargs)

def replace_rows(df, df_r, indices):
  """
  Replaces rows in df with rows from df_r.
  If a column in df is missing values in any of the rows, then those rows are dropped.

  Parameters
  ----------
  df : pf.DataFrame
    Data frame to replace rows in.

  df_r : pd.DataFrame
    The replacement rows are taken from this data frame.

  Returns
  -------
  dropped : pd.DataFrame
    Data frame with replaced and dropped rows.
  """
  # df = copy.deepcopy(df)
  indices_to_replace = df.loc[indices].dropna().index.values
  df.loc[indices_to_replace] = df_r.loc[indices_to_replace]

  return df.dropna()

def pad_dates(dates, freq='D', length=1):
  """
  Adds padding to the dates provided.

  Parameters
  ----------
  dates : array_like
    List or array with dates in ISO8601 date or datetime format.

  freq : str, default 'D'
    Defines in what scale to pad the dates. valid_frequencies: {'Y','M','W','D','h','m','s', 'ms'}

  length : int
    How much to pad the dates. The dates are padded using 'freq' to pad 'length' in each direction.

  Returns
  -------
  sorted_padded : array_like
    Sorted and padded list or array of dates.

  Notes
  -----
  This function assumes that the date format is ISO 8601 date or datetime format.
  Some other formats may be accepted. See the link provided in the see also tab.

  See also
  --------
  http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.datetime.html
  """
  valid_frequencies = ['Y','M','W','D','h','m','s','ms']
  if (freq not in valid_frequencies):
    raise InvalidOptionError("Invalid freq: '%s'" % freq)

  padded_dates = set()
  for date in dates:
      padded_dates.add(date)
      for i in range(1,length+1):
        padded_dates.add(date - np.timedelta64(length, '%s' % freq))
        padded_dates.add(date + np.timedelta64(length, '%s' % freq))

  return sorted(list(padded_dates))

def smooth_holidays(df, holidays_df, length=1):
  """
  Smoothes out peeks and troughs in df, using indices in holiday_df.

  Parameters
  ----------
  df : pd.DataFrame
    Data Frame to smooth.

  holidays_df : pd.DataFrame
    Defines the indices to use when smoothing. The function does not care about the values
    in holiday_df, only if they are present for a certain row index. This index is then
    used to smooth df.

  length : int, default 1
    This argument is passed to pad_dates(). This defines the 'padding' to the holiday.
    Assume may 1. is a holiday. If length = 1, then april 30. may 1. and may 2. will
    be replaced with a 'smoothed' value.

  Returns
  -------
  smoothed_df : pd.DataFrame
    Data frame with holidays and days leading up to and after, according to length, smoothed.
  """
  # TODO: Add support for hours
  mmd_df = pd.rolling_median(df, 7)
  indices = df.loc[holidays_df.index.values].dropna().index.values
  indices = pad_dates(indices, freq='D', length=length)
  df = replace_rows(df, mmd_df, indices)
  return df

def impute(df, freq='D', method='median'):
  """
  Imputes all missing values in all columns including the index column. Currently only works for
  data frames with indexcolumn with type DatatimeIndex.

  Parameters
  ----------
  df : pd.DataFrame
    Data frame with missing values.

  freq : str, default 'D'
    The resolution used to impute. 'D' means impute days, 'W' means week etc.

  method : str, default 'median'
    What imputatin method to use. valid_methods: {'zero','mean','median', 'nan'}

  Returns
  -------
  imputed : pd.DataFrame

  See also
  --------
  pd.date_range(start, end, freq) is used to impute the index column.
  http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html
  """
  valid_methods = ['zero','mean','median', 'nan']
  method = method.lower()
  if (method not in valid_methods):
    raise InvalidOptionError("Invalid method: '%s'" % method)

  index_type = type(df.index)
  if (index_type == DatetimeIndex):
    range_ = pd.date_range(df.index.min(), df.index.max(), freq=freq)
  elif (index_type == Float64Index):
    raise NotImplementedError("Imputation of floating point indexes is not implemented")
    # TODO: implement pandas.Series.interpolate imputation
  elif (index_type == Int64Index):
    raise NotImplementedError("Imputation of integer indexes is not implemented")

  # Impute indices  (units)
  df = df.reindex(range_)

  # Impute contents (items)
  # TODO: Add MAV imputation
  if (method == 'zero'):
    return df.fillna(0.0)
  elif (method == 'mean'):
    df = df.fillna(df.mean())
  elif (method == 'median'):
    df = df.fillna(df.median())
  elif (method == 'nan'):
    pass
  elif (method == 'mav'):
    raise NotImplementedError("Imputation with moving averages is not implemented")

  return df

def remove_by_percentage(df, percentage, identifier='zero'):
  # TODO: Raise exception if percentage is out of bounds. Currently the function uses values even when they make no sense.
  """
  Removes columns if the amount of 'identifier' in column is greater than or equal to percentage.

  Parameters
  ----------
  df : pd.DataFrame

  percentage : float

  identifier : str, default 'zero'
    What type of row value to check for when counting the number of 'missing' values.
    valid identifiers: {'zero', 'nan'}

  Returns
  -------
  df : pd.DataFrame
    Data frame with columns removed.
  """
  valid = ['zero', 'nan']
  if (identifier not in valid):
    raise Exception("Invalid identifier: '%s', valid identifiers are: %s" % (identifier, valid))

  stats = statistics(df)
  to_remove = [c for c in df.columns if stats[c]['percent_%s' % identifier] >= percentage]

  df = df[[c for c in df.columns if c not in to_remove]]
  return df

def remove_empty_columns(df, threshp=None, include_zeros=False):
  """
  Removes columns in df if the number of nan in the column is greater than 'threshp'.

  Parameters
  ----------
  df : pd.DataFrame

  treshp : float, default None
    Value between 0 and 1.

  include_zeros : bool, default False
    If True, then zeros will be added to the nan count.

  Returns
  -------
  dropped : pd.DataFrame
  """
  if (include_zeros):
    df = df.replace(0.0, np.nan)
  if (threshp != None):
    return df.dropna(thresh=int(np.rint(len(df)*threshp)), axis=1)
  return df.dropna(how='all', axis=1)

def nan_to_none(df):
  """
  Changes all instances of nan to None.
  """
  return df.where((pd.notnull(df)), None)

def create_labels(title="", subtitle="", xlabel="", ylabel=""):
  """
  Creates a dictionary with title, subtitle, xlabel an ylabel.

  Returns
  -------
  labels : dict
  """
  labels = OrderedDict()
  labels['title']    = title
  labels['subtitle'] = subtitle
  labels['xlabel']   = xlabel
  labels['ylabel']   = ylabel

  return labels

def plot_corr(df, sort=False, **kwargs):
  # TODO: sns.clustermap has a metric kwarg that may be interesting to test out and implement.
  # Create the corrmat or clustermap in correlation.py insted?
  """
  Creates a correlation matrix from the datafram and plots it. A plt.show()
  call is needed after this function returns to show the figure on screen.

  Parameters
  ----------
  df : pd.DataFrame

  sort : bool, default False
    If True, creates a clustermap using "method" kwarg to sort.

  **kwargs
  --------
  corr_method : str, default "pearson".
    vallid arguments: 'pearson', 'kendall', 'spearman'.

  sort_method : str, default "average".
    What method to use in sorting/clustering algorithm.

  save_figure : bool, default False

  file_name : str, default "tmp_heatmap.png"
    if "save_figure" = True, figure will be saved in current directory as "file_name".

  abs_values : bool, default False.
    If True, plots the absolute values of the correlation matrix.

  return_corrmat : bool, default False.

  Returns
  -------
  f : plt.Figure, or sns.ClusterGrid

  corrmat : pd.DataFrame
    Only gets returned if kwarg 'return_corrmat' is True.

  See also
  --------
  For available methods see:
  http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
  """
  sns.set(context="paper", font="monospace")
  save_figure = kwargs.get("save_figure", False)
  file_name = kwargs.get("file_name", "tmp_heatmap.png")
  corr_method = kwargs.get("corr_method", "pearson")
  abs_values = kwargs.get("abs_values", False)
  return_corrmat = kwargs.get("return_corrmat", False)

  corrmat = df.corr(method=corr_method)
  if abs_values:
    corrmat = corrmat.abs()
  if (sort):
    sort_method = kwargs.get("sort_method", "average")
    metric = kwargs.get("metric", "euclidian")
    cg = sns.clustermap(corrmat, method=sort_method, metric=metric,
                        cmap=plt.cm.inferno, linewidths=.5)
    #Rotate x and y tick labels so long texts will fitt next to eachother.
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    if save_figure:
      cg.savefig(file_name)
  else:
    f, sns_ax = plt.subplots(figsize=(12, 10))
    hmap = sns.heatmap(corrmat, ax=sns_ax, vmax=.8, linewidths=.5,
                       square=True, cmap=plt.cm.inferno)

    colnames = corrmat.columns.values
    for i, colname in enumerate(colnames):
      if i and colname != colnames[i - 1]:
        ax.axhline(len(colnames) - i, c="black")
        ax.axvline(i, c="black")

    f.tight_layout()

  if return_corrmat:
    if sort: # Sort the columns and rows in the clustered fashion.
      corrmat = corrmat[cg.dendrogram_col.reordered_ind]
      corrmat = corrmat.reindex(corrmat.columns.values)
    return cg, corrmat
  else:
    if sort:
      return cg
    else:
      return f

def plot_hist(df, labels=None, bins=30, rel=False, center_on_bins=False):
  # Return the figure object instead of drawing the figure?
  """
  Creates and draws a figure using matplotlib.pyplot

  Parameters
  ----------
  df : pd.DataFrame

  labels : dict

  bins : int, default 30
    Number of bis to use.

  rel : bool, default False
    IF True, the values in each bin is relative to the number of entries in df.

  center_on_bins : bool, default False
    If True the xticks are centered on the bins instead of left aligned.

  Returns
  -------
  None
  """
  pd.set_option('display.mpl_style', 'default')
  fig, ax = plt.subplots()
  if (rel):
    counts, bins, patches = ax.hist(df, bins=bins, align='left', weights=(np.zeros_like(df) + 100. / df.size))
  else:
    counts, bins, patches = ax.hist(df, bins=bins, align='left')
  if (center_on_bins):
    xticks = bins
  else:
    xticks = ax.get_xticks()
  ax.set_xticks(xticks, minor=False)
  plt.xticks(rotation=45, ha='right')

  if (labels):
    add_labels(ax, labels)

  plt.draw()

def plot_timeseries(df, labels=None, interval=1, saveto=None, *args, **kwargs):
  # TODO: Currently this function does not return a figure object or draw a figure. Only way to
  # get the plot is to save it.
  """
  Plots the columns in df.

  Parameters
  ----------
  df : pd.DataFrame

  labels : dict
    Dictionary, see create_labels().

  interval : int, default 1
    The interval for the ticks on the x-axis.

  saveto : str, default None
    Filepath to save the figure.

  Returns
  -------
  None
  """
  pd.set_option('display.mpl_style', 'default')
  # pd.set_option('display.width', 5000)
  # pd.set_option('display.max_columns', 60)

  fig,ax = plt.subplots()
  ax.set_color_cycle(['r','g','b','c','m','y','k'])

  dates = df.index.to_pydatetime()
  x = range(0, len(dates))
  x_interval = x[::interval]
  dates_interval = dates[::interval]

  ax.plot(x, df, '-v')
  ax.xaxis.set_ticks(x_interval)
  ax.xaxis.set_ticklabels(['%s' % x.strftime('%Y-%m-%d') for x in dates_interval])
  plt.tick_params(axis='x', labelsize=6)
  plt.xticks(rotation=50, ha='right')
  ax.margins(0.01, None)

  if (labels):
    plt.suptitle(labels['title'], fontsize=14)
    plt.figtext(0.5, 0.93, labels['subtitle'], fontsize=10, ha='center', **dejfont)
    plt.xlabel(labels['xlabel'], **dejfont)
    plt.ylabel(labels['ylabel'], **dejfont)

  lines, labels = ax.get_legend_handles_labels()
  plt.legend(df.columns.values.tolist())

  if (saveto):
    figure = plt.gcf()
    figure.set_size_inches(19.2, 5.6)
    plt.savefig(saveto, dpi = 100)

def plot(df, labels=None, saveto=None, *args, **kwargs):
  """
  Plots the columns in df on the active matplotlib figure

  Parameters
  ----------
  df : pd.DataFrame

  labels : dict
    Dictionary, see create_labels().

  saveto : str, default None
    Filepath to save the figure.

  kwargs
  ------
  fontsize : int

  labelsize : int

  Returns
  -------
  None
  """
  pd.set_option('display.mpl_style', 'default')
  fontsize  = kwargs['foexplicitlyntsize'] if ('fontsize' in kwargs.keys()) else 16
  labelsize = kwargs['labelsize'] if ('labelsize' in kwargs.keys()) else 12
  fig,ax = plt.subplots()
  fig.tight_layout()
  ax.set_color_cycle(['r','g','b','c','m','y','k'])

  ax.plot(df.index.values, df)
  plt.tick_params(axis='x', labelsize=labelsize)
  plt.tick_params(axis='y', labelsize=labelsize)
  # plt.xticks(rotation=50, ha='right')
  ax.margins(0.01, None)

  if (kwargs):
    plt.suptitle(kwargs['title'], fontsize=fontsize, **dejfont)
    plt.figtext(0.5, 0.93, kwargs['subtitle'], fontsize=fontsize-4,
     ha='center', **dejfont)
    plt.xlabel(kwargs['xlabel'], fontsize=fontsize, **dejfont)
    plt.ylabel(kwargs['ylabel'], fontsize=fontsize, **dejfont)

  lines, labels = ax.get_legend_handles_labels()
  plt.legend(df.columns.values.tolist())
  if (saveto):
    figure = plt.gcf()
    figure.set_size_inches(19.2, 7)
    plt.savefig(saveto, dpi = 100)

def to_json(df):
  """
  Writes the data frame in json format.

  Parameters
  ----------
  df : pd.DataFrame

  Returns
  -------
  json_string : str
  """
  columns = df.columns.values
  json_string = "{\n"
  for column in columns:
    json_string += '  "%s": %s,\n' % (column, df[[column]].T.values.tolist()[0])
  json_string = json_string[:-2]
  json_string += "\n}\n"
  json_string = json_string.replace("'", '"')
  return json_string

def statistics(df):
  """
  Counts and computes some basic statistics of the dataframe.

  Parameters
  ----------
  df : pd.DataFrame

  Returns
  -------
  stat_df : pd.DataFrame
    Dataframe containing the statistics.

  Notes
  -----
  The rows in stat_df is {'num_nan', 'num_zero', 'percent_nan',
  'percent_zero', 'median','mean', 'standard_deviation', 'variance'}
  """
  statdict = {
    'num_nan':{},
    'num_zero':{},
    'percent_nan':{},
    'percent_zero':{},
    'median':{},
    'mean':{},
    'standard_deviation':{},
    'variance':{}
  }

  length   = len(df)
  num_nan  = df.isnull().sum().to_dict()
  num_zero = (df == 0).astype(float).sum(axis=0).to_dict()
  medians  = df.median().to_dict()
  means     = df.mean().to_dict()
  stds     = df.std().to_dict()
  _vars    = df.var().to_dict()

  statdict['num_nan']            = num_nan
  statdict['num_zero']           = num_zero
  statdict['percent_nan']        = {k:(v/float(length))*100.0 for k,v in num_nan.iteritems()}
  statdict['percent_zero']       = {k:(v/float(length))*100.0 for k,v in num_zero.iteritems()}
  statdict['median']             = medians
  statdict['mean']               = means
  statdict['standard_deviation'] = stds
  statdict['variance']           = _vars

  return pd.DataFrame.from_dict(statdict).T

def distance(df1, df2):
  # TODO: Explicitly convert elements to float?
  """
  Computes the sum of the elementvise absolute difference between the data frames.

  Parameters
  ----------
  df1 : pd.DataFrame

  df2 : pd.DataFrame

  Returns
  -------
  distance : float

  Notes
  -----
  The dataframes must be of same shape and contain floats.
  """
  if (not all([ix[0] == ix[1] for ix in zip(df1.index,df2.index)])):
    raise Exception("Dataframes must have the same index")

  X1 = df1.as_matrix()
  X2 = df2.as_matrix()

  return(np.sum(np.abs(X1-X2)))

def get_path_to_dataset(dataset):
  """
  Used for testing.
  Requires the d2o/data/datasets/testdata/ directory.
  Get path to datset without initializing during import.

  Parameters
  ----------
  dataset : str
    Name.

  Returns
  -------
  file_path : str
    File path to dataset.
  """
  file_path = path.dirname(path.realpath(__file__))
  file_path = file_path[0:file_path.find("/d2o")] + "/d2o/data/datasets/testdata/" + dataset

  return file_path


