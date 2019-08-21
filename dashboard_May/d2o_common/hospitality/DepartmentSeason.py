import calendar
import pandas as pd
from datetime import timedelta

from d2o_common.util import logger as log

VALID_SEASON_TYPE = ['normal', 'labor']
NAN_SEASON_ID = 0 # season id for data points are not in any season.


class DepartmentSeason(object):
  """
  Season for department
  """
  def __init__(self, season_raw_data):
    """
    Initalize department season object
    Args:
      season_raw_data (dict):
    """
    self.season_df = self.create_season_df(season_raw_data)
    if self.season_df is None:
      self.has_season = False
    else:
      self.has_season = True
      self.has_overlap_season = self.is_overlap_season(self.season_df)
      self.mapping_date_to_season = self.create_mapping_date_to_season(self.season_df)

  def get_season_name(self, season_id):
    try:
      name = self.season_df[self.season_df['Id'] == season_id]['Names'].iloc[0]
    except:
      name = ''
    return  name

  def transform_data_to_season_df(self, df):
    """
    Transform to season dataframe, this transform can contain two things add
    the "Season" column to dataframe, add the extend row if this department
    have overlap season so the number row of dataframe may be increase
    Args:
      df (pd.DataFrame): index must be 'Date' and datetime index.

    Returns:
      pd.DataFrame: dataframe with "Season" column
    """
    if df.index.name != "Date":
      raise Exception('Invalida index for dataframe')
    df_merge_season = pd.merge(df, self.mapping_date_to_season, how='outer', on=None, left_index=True, right_index=True)
    df_merge_season = df_merge_season.loc[df.index, :]
    df_merge_season['Season'].fillna(NAN_SEASON_ID, inplace=True)
    df_merge_season = df_merge_season[df_merge_season['Season'] != NAN_SEASON_ID] #drop nan season
    return df_merge_season

  def get_list_season(self):
    list_season = list(self.season_df["Id"].unique())
    return list_season

  @classmethod
  def create_season_df(self, json_data):
    """
    Using d2o api to get season data.
    Returns:
      pd.DataFrame: season data in dataframe format
      |Id|From|To|Names|Color
    """
    try:
      l_df = []
      for period in json_data['Periods']:
        if period['Dates'] == []:
          continue
        else:
          df = pd.DataFrame(period['Dates'])
          df['Color'] = period['Color']
          df['Names'] = period['Name']
          df['From'] = pd.to_datetime(df['From'])
          df['To'] = pd.to_datetime(df['To'])
          df = df[['Id', 'From', 'To', 'Names', 'Color']]
          l_df.append(df)
      from_to_ss_df = pd.concat(l_df, ignore_index=True)
      from_to_ss_df = from_to_ss_df[from_to_ss_df.Color != '#000000']
      return from_to_ss_df
    except Exception, e:
      log.warning('No_season_data')
      return None

  @classmethod
  def create_mapping_date_to_season(self, from_to_season_df):
    """
    Create map from date -> season. Support overlap season base on duplicate index
    1/1/2014 (date) -> 22 (season id)
    1/1/2014 (date) -> 45 (season id)
    Args:
      from_to_season_df (pd.DataFrame): the season dataframe from __get_season_df
      |Id|From|To|Names|Color|
    Returns:
      pd.DataFrame: dates to season dataframe
      index(Date)| Season
      01/01/2013  | 1
      ...
      Note: DataFrame support duplicate index better than Series
    """
    full_year_periods = []
    for row_id in from_to_season_df.index:
      row = from_to_season_df.loc[row_id, :]
      period = pd.Series(row['Id'], index=pd.date_range(row['From'], row['To']))
      full_year_periods.append(period)

    dates_season_series = pd.concat(full_year_periods)
    dates_season_df = pd.DataFrame(dates_season_series, columns=['Season'])
    dates_season_df.index.name = 'Date'
    return dates_season_df

  @classmethod
  def is_overlap_season(self, season_df):
    """
    Checking overlap-season.
    Department have overlap-season if it have one date belongs more than one season.
    Args:
      season_df (pd.DataFrame): season of department
        Id|From|To|Names|Color
    Returns:
      bool:  True if department have overlap-season, False if not.
    """
    sorted_ss_df = season_df.sort_values(['From', 'To']).reset_index()
    sorted_ss_df['From_lag1'] = sorted_ss_df['From'].shift(-1)
    overlap_days =  sorted_ss_df.loc[0:(len(sorted_ss_df) - 1),'To'] - \
                    sorted_ss_df.loc[0:(len(sorted_ss_df) - 1),'From_lag1']
    has_overlap_days = (overlap_days > timedelta(0)).any()
    return has_overlap_days


  @classmethod
  def create_mapping_day_of_year_to_season(self, df_season):
    """
    Create a map that mapping from day of year to season
    Args:
      df_season (pd.DataFrame): the season dataframe from __get_season_df
      |Id|From|To|Names|Color|

    Returns:
      pd.Series: Series with indices are days of year and values is season ids
    """
    df_season = df_season.sort_values('From', ascending=True)
    years = df_season['From'].dt.year.unique()
    leap_years = [year for year in years if calendar.isleap(year)]
    leap_year = leap_years[0]
    df_leap_year = df_season[df_season['From'].dt.year == leap_year]
    season_dayofyear_series = pd.Series(-1, index=pd.
                                        date_range('01-01-%s' % leap_year,
                                                   '31-12-%s' % leap_year))
    for i in df_leap_year.index:
      row = df_leap_year.loc[i,:]
      season_dayofyear_series[pd.date_range(row['From'], row['To'])] = row['Id']
    season_dayofyear_series.index = pd.Series(season_dayofyear_series.index)\
      .dt.dayofyear
    return season_dayofyear_series

  def get_season_for_dates(self, dates):
    """
    Return season id of each date in dates
    Args:
      dates (list): list of like datetime object
    Returns:
      list[str]: list of season id
    """
    dates = pd.Series(pd.to_datetime(dates))
    return list(self.mapping_date_to_season.loc[dates, :].values)
    #return list(self.mapping_dayOfYear_season[dates.dt.dayofyear])
