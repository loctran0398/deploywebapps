# -*- encoding: utf-8 -*-
import pandas as pd
from os import path

import matplotlib.pyplot as plt
import matplotlib.dates as mpldates

from datetime import datetime, timedelta
from pandas.tseries.index import DatetimeIndex, Float64Index, Int64Index
from sklearn.preprocessing import Imputer
from collections import OrderedDict

import d2o.utils.calc as calc

import random
import copy
import numpy as np
import seaborn as sns

dejfont = {'fontname':'Dejavu Sans Mono'}

def unicode_columns(df, encoding='utf-8'):
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
  return pd.concat(df_list, axis=1)

def scale(df, scaler):
  sdf = pd.DataFrame(scaler.fit_transform(df))
  sdf.index = df.index
  sdf.columns = df.columns
  return sdf

def replace(df, value, replace_with=np.nan):
  return df.applymap(lambda x: replace_with if x == value else x)

def impute_weekdays(df, include_zeros=False):
  """ Set NaN's to that weekday's median """
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
  if (precision == 'hour'):
    df['Seconds'] = df.index.seconds
    df['Hour'] = df.index.hour
  df['Weekday'] = df.index.weekday
  df['Week'] = df.index.week
  df['Month'] = df.index.month
  df['Year'] = df.index.year

  return df

def from_csv(path, *args, **kwargs):
  return pd.read_csv(path, *args, **kwargs)

def replace_rows(df, df_r, indices):
  # df = copy.deepcopy(df)
  indices_to_replace = df.loc[indices].dropna().index.values
  df.loc[indices_to_replace] = df_r.loc[indices_to_replace]

  return df.dropna()

def pad_dates(dates, freq='D', length=1):
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
  # TODO: Add support for hours
  mmd_df = pd.rolling_median(df, 7)
  indices = df.loc[holidays_df.index.values].dropna().index.values
  indices = pad_dates(indices, freq='D', length=1)
  df = replace_rows(df, mmd_df, indices)
  return df

def impute(df, freq='D', method='median'):
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
  valid = ['zero', 'nan']
  if (identifier not in valid):
    raise Exception("Invalid identifier: '%s', valid identifiers are: %s" % (identifier, valid))

  stats = statistics(df)
  to_remove = [c for c in df.columns if stats[c]['percent_%s' % identifier] >= percentage]

  df = df[[c for c in df.columns if c not in to_remove]]
  return df

def remove_empty_columns(df, threshp=None, include_zeros=False):
  if (include_zeros):
    df = df.replace(0.0, np.nan)
  if (threshp != None):
    return df.dropna(thresh=int(np.rint(len(df)*threshp)), axis=1)
  return df.dropna(how='all', axis=1)

def nan_to_none(df):
  return df.where((pd.notnull(df)), None)

def create_labels(title="", subtitle="", xlabel="", ylabel=""):
  labels = OrderedDict()
  labels['title']    = title
  labels['subtitle'] = subtitle
  labels['xlabel']   = xlabel
  labels['ylabel']   = ylabel

  return labels

def plot_corr(df, sort=False, **kwargs):
  # TODO: sns.clustermap has a metric kwarg that may be interesting to test out and implement.
  # TODO: Bug, if plt.show() is used after this function two windows will open, one with
  # the desired heatmap and one empty.
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
  pd.set_option('display.mpl_style', 'default')
  fontsize  = kwargs['fontsize'] if ('fontsize' in kwargs.keys()) else 16
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
  columns = df.columns.values
  json_string = "{\n"
  for column in columns:
    json_string += '  "%s": %s,\n' % (column, df[[column]].T.values.tolist()[0])
  json_string = json_string[:-2]
  json_string += "\n}\n"
  json_string = json_string.replace("'", '"')
  return json_string

def statistics(df):
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
  if (not all([ix[0] == ix[1] for ix in zip(df1.index,df2.index)])):
    raise Exception("Dataframes must have the same index")

  X1 = df1.as_matrix()
  X2 = df2.as_matrix()

  return(np.sum(np.abs(X1-X2)))

def get_path_to_dataset(dataset):
  """
  Simple hack to get path to datsett without initializing during import.
  To be replaced by a datasett class in the future.

  Parameters
  ----------
  dataset : str
    Name of file to get path to.

  Returns
  -------
  file_path : str
    File path to datasett.
  """
  file_path = path.dirname(path.realpath(__file__))
  file_path = file_path[0:file_path.find("/clib")] + "/clib/data/datasets/testdata/" + dataset

  return file_path


