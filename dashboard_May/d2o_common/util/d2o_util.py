import arrow
import re
import numpy as np
import pandas as pd



def detect_d2o_host():
  """
  D2o production server and test sever using different data api host.
  Detect which server is used and return it's data host
  Returns:
    str: ip address of data host.
  """
  import socket
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  print(s.getsockname()[0])
  if s.getsockname()[0] == '172.16.0.23':
    print('Production server detected')
    HOST = "http://10.50.10.4:8485"
  else:
    print('Using testing data host')
    HOST = "http://172.16.0.51:8485"
  s.close()
  return HOST

def is_outlier(points, thresh=3.5):
  """
  Returns a boolean array with True if points are outliers and False
  otherwise.

  Args:
    points(np.array):  An numobservations by numdimensions array of observations
    thresh(float): The modified z-score to use as a threshold. Observations with
          a modified z-score (based on the median absolute deviation) greater
          than this value will be classified as outliers.

  Returns:
    list[bool]: A numobservations-length boolean array.

  References:
      Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
      Handle Outliers", The ASQC Basic References in Quality Control:
      Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

  """
  points = np.array(points)
  if len(points.shape) == 1:
    points = points[:, None]
  median = np.median(points, axis=0)
  diff = np.sum((points - median) ** 2, axis=-1)
  diff = np.sqrt(diff)
  med_abs_deviation = np.median(diff)
  if med_abs_deviation == 0:
    return (diff == 0)

  modified_z_score = 0.6745 * diff / med_abs_deviation

  return modified_z_score > thresh

def get_ratio_non_positive_value_in_series(series_value):
  """
  Get ratio of non-positive value on total values in series
  Args:
    series_value (pd.Series): list of values

  Returns:
    float: ration of non-positive value

  """
  series_value = pd.Series(series_value)
  num_non_positive_vals = len(series_value[series_value <= 0])
  return float(num_non_positive_vals)/len(series_value)

def remove_non_ascii_character(text):
  return re.sub(r'[^\x00-\x7F]+','', text)

def transfer_to_d2o_weekday(pandas_week_day):
  """
  Mapping from pandas day of week: Monday=0, ..., Sunday = 6
  to d2o Day of week:  Sunday = 1, Monday = 2, ..., Saturday = 7
  Args:
    pandas_week_day:

  Returns:

  """
  d2o_week_day = pandas_week_day + 2 if (pandas_week_day + 2) < 8 else 1
  return d2o_week_day

def get_weekday_from_d2o_weekday(d2o_week_day):
  """
  Mapping from pandas day of week: Monday=0, ..., Sunday = 6
  to d2o Day of week:  Sunday = 1, Monday = 2, ..., Saturday = 7
  Args:
    pandas_week_day:

  Returns:

  """
  pandas_week_day = d2o_week_day - 2 if (d2o_week_day - 2) >= 0 else 6
  return pandas_week_day

def get_range_date_bounds(year_back):
  """
  Get the from date and to date of date range
  Args:
    year_back (int): number of year we got back from now to get the date range.
    to_date always current time.
  Returns:
    (datetime.date, datetime.date): from_date and to_date
  """
  date_now = arrow.now()
  from_date = date_now.replace(days=-(365*year_back))
  return from_date.date(), date_now.date()

def add_week_day(df, weekday_colname='weekday'):
  """
  Add column week day to dataframe that have index is datetime.
  Args:
    df (pd.DataFrame): dataframe must have DatetimeIndex
    weekday_colname (str): The name of columns will add to dataframe

  Returns:
    pd.DataFrame
  """
  df[weekday_colname] = list(pd.Series(pd.to_datetime(df.index)).dt.weekday)
  return df


def get_week_day(dates):
  dates = pd.Series(pd.to_datetime(dates))  # make sure that is datetime series
  return list(dates.dt.weekday)

def add_lag_day_cols(df, lag_days=[1],
                     ignore_colums=('Hours', 'season', 'day_of_week'),
                     new_col_name_pattern='%s_lag%s'):
  """
  For each columns in dataframe excpet columns in ingore_columns create a lag
  days column of it, set name as pattern of original column name and lag day
  Args:
    df (pd.DataFrame): data, index must be a datetime type
    lag_days (list): list of lag days need to be create
    ignore_colums (tuple): list of columns will be ignore in add lag day process
    new_col_name_pattern (str): new column name pattern, combine between old name
    and lag day.

  Returns:
    pd.DataFrame: added lag day data
  """
  columns = df.columns
  for lag in lag_days:
    for col in columns:
      if col in ignore_colums:
        continue
      df[new_col_name_pattern % (col, lag)] = df[col].shift(lag)
  return df


def sort_df_by_row(df, ascending=False):
  """
  Sort each row of dataframe by values, return name of columns
  Args:
    df (pd.DataFrame): values dataframe.
         | name1 | name2| name3 |....
      1     4    |   1   |    5
      2     3       5       4
      ...

  Returns:
    pd.DataFrame: ordered name of columns dataframe
         | 1    |  2   |   3 | ....
      1    name3   name1   name2
      2    name2    name5   name6
      ....
  """
  return df.apply(lambda row: pd.Series(list(row.sort_values(ascending=ascending)
                                             .index)),axis=1)

def info_mess(client_id, hotel_id, dep_id):
  return "DEP %s HOTEL %s CLIENT ID %s" % (dep_id, hotel_id, client_id)