"""
Copyright: TP7
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import time

#=======================COMPUTATION================================
def compute_interval(cov_val, significance_level):
    """
    Compute confidence interval specific for correlation, based on significance level
    :param cov_val: correlation values
    :param significance_level: significance level values
    :return:
    """
    conf_interval_0 = cov_val * (1 - 2 * significance_level)
    conf_interval_1 = cov_val * (1 + 2 * significance_level)
    return conf_interval_0, conf_interval_1

def compute_error(y, x, capture, arr):
    """
    Compute predicted values (base on capture and arr) and return error of model
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_predicted = x*capture*arr
    s = np.float64((y_predicted - y).abs().mean()) / np.float64(y.abs().mean())
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    result = 100 if np.isinf(s) else s
    return result

def remove_outlier_old(series1, series2):
    """
    Remove outlier values on two series. If one elememt was removed on one series the corresponding element in another
     series would be removed too,. Method to detect outler is use 2 * standard deviation around mean
    :param series1:
    :param series2:
    :return:
    """
    if len(series1) != len(series2):
        print("::>>", len(series1), len(series2))
        raise "series1 and series2 must have a same length"

    series1 = np.array(series1)
    series2 = np.array(series2)

    #remove inf
    cond1 = ~np.isinf(series1) # not nan
    cond2 = ~np.isinf(series1) # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    #remove nan
    cond1 = ~np.isnan(series1) # not nan
    cond2 = ~np.isnan(series2) # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    #remove outlier
    cond1 = (series1 <= (series1.mean() + 2*series1.std())) & (series1 >= (series1.mean() - 2*series1.std()))
    cond2 = (series2 <= (series2.mean() + 2*series2.std())) & (series2 >= (series2.mean() - 2*series2.std()))
    cond = cond1 & cond2

    return pd.Series(series1[cond]), pd.Series(series2[cond])

def remove_outlier(series1, series2, num_day=10):
    """
    Remove outlier values on two series. If one elememt was removed on one series the corresponding element in another
     series would be removed too,. Method to detect outler is use 2 * standard deviation around mean
    :param series1:
    :param series2:
    :return:
    """
    if len(series1) != len(series2):
        print("::>>", len(series1), len(series2))
        raise "series1 and series2 must have a same length"

    series1 = np.array(series1)
    series2 = np.array(series2)

    #remove inf
    cond1 = ~np.isinf(series1) # not nan
    cond2 = ~np.isinf(series1) # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    #remove nan
    cond1 = ~np.isnan(series1) # not nan
    cond2 = ~np.isnan(series2) # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    #remove outlier
    cond1 = (series1 <= (series1.mean() + 2*series1.std())) & (series1 >= (series1.mean() - 2*series1.std()))
    cond2 = (series2 <= (series2.mean() + 2*series2.std())) & (series2 >= (series2.mean() - 2*series2.std()))
    cond = cond1 & cond2

    s1 = pd.Series(series1[cond])
    s2 = pd.Series(series2[cond])

    if len(s1) < num_day:
        num_day = len(s1)

    return s1[:num_day], s2[:num_day]

#=======================WORK WITH DATAFRAME========================
def get_df_by_dates(dates, df, date_col='date'):
    """
    Select rows in dataframe having date in dates
    :param dates: List date
    :param df: dataframe
    :param date_col: date column name in df
    :return: selected dataframe
    """
    # Selection row in df by date
    tmp_df = df
    if 'date' != df.index.name:
        tmp_df = df.set_index('date')
    tmp_df = tmp_df.ix[dates].reset_index('date')
    return tmp_df

#=======================WORK WITH DATES========================
def get_dates_lag(dates, lag):
    # Generate new list dayte with lag from origninal date
    # case 1: date = 14/2/2016
    # case 2: date = 3/9/2016

    if lag == 365:
        list_date = []
        for date in dates:
            if date.month == 2 and date.day == 29:
                list_date.append(date - timedelta(365))
            else:
                list_date.append(date.replace(year=date.year -1))
        return list_date

    return [date - timedelta(days=lag) for date in dates]

def get_dates_by_dayWeek(dates, day_of_week):
    # Get  list date in dates that is day_of_week
    return [date for date in dates if date.weekday() == day_of_week]


#=======================WORK-WITH-SEASON========================
def get_season_dates_from_tf(season_tf, dates):
    """
    Get dates belong a season in dates
    :param season_tf: season timeframe
    :param dates: list date
    :return:
    """
    years = range(min(dates).year, max(dates).year + 1)
    season_dates_in_years = generate_season_date(season_tf, years)
    season_dates = set(season_dates_in_years).intersection(set(dates))
    return sorted(list(season_dates))

def generate_season_date(season_tf, years):
    """
    Generate dates in one season in years
    :param season_tf: season timeframe
    :param years: list year
    :return:
    """
    list_date = []
    # print season_tf
    for year in years:
        for from_date, to_date in reversed(season_tf):
            if (year % 4) != 0:
                if from_date == '02-29':
                    from_date = '03-01'
                if to_date == '02-29':
                    to_date = '02-28'
            begin = pd.datetime.strptime("{0}-{1}".format(year, from_date), '%Y-%m-%d')
            end = pd.datetime.strptime("{0}-{1}".format(year, to_date), '%Y-%m-%d')
            num_day = (end - begin).days + 1
            list_date += [begin + timedelta(days=i) for i in range(num_day)]
    return list_date

def generate_season_date_by_dof(season_tf, num_day, day_of_week, dates):
    """ Get data in season by day of week.

    :param season_tf:
    :param num_day:
    :param day_of_week:
    :param year:
    :return:
    """
    season_dates = pd.Series(get_season_dates_from_tf(season_tf, dates)).sort_values(ascending=False)
    y_dates = list(season_dates[season_dates.dt.weekday == day_of_week])
    if num_day < len(y_dates):
        y_dates = y_dates[:num_day]
    return y_dates

def generate_season_date_by_dof_ver_2(date_ss, num_day, day_of_week):
    """ Get data in season by day of week.

    :param season_tf:
    :param num_day:
    :param day_of_week:
    :param year:
    :return:
    """
    start_time = time.time()
    season_dates = pd.Series(date_ss).sort_values(ascending=False)
    y_dates = list(season_dates[season_dates.dt.weekday == day_of_week])
    if num_day < len(y_dates):
        y_dates = y_dates[:num_day]
    return y_dates


def get_dates_season(dates, l_season):
    """
    Get season label of each date in dates
    :param dates: list date
    :param l_season: season label in a year
    :return:
    """
    if len(l_season) != 366:
        raise "number of l_season must be 366"
    return [l_season[date.replace(year=2016).timetuple().tm_yday - 1] for date in dates]


def read_dep_season(department):
    """
    Read season label of one department
    :param department:
    :return:
    """
    file_name = "season_dep_{0}.csv".format(department)
    df_season_dayinyear = pd.read_csv(file_name)
    l_season = []
    for id in df_season_dayinyear.index:
        num_day = df_season_dayinyear.loc[id, 'to'] - df_season_dayinyear.loc[id, 'from'] + 1
        season = df_season_dayinyear.loc[id, 'season']
        l_season += [season] * num_day
    return l_season

def read_dep_season_timeframe(department, hotel_id, year=2016):
    """
    Read season timeframe of one department
    :param department: department id
    :param hotel_id:
    :param year: year default is 2016 (leap year)
    :return: list season timeframe
    """
    file_name = os.path.join('data', hotel_id, "{0}.csv".format(department))
    df = pd.read_csv(file_name)
    df['from'] = df['from'].apply(lambda x: (timedelta(days=int(x)) + datetime(year=year, day=1, month=1)).strftime("%m-%d"))
    df['to'] = df['to'].apply(lambda x: (timedelta(days=int(x)) + datetime(year=year, day=1, month=1)).strftime("%m-%d"))
    seasons = df['season'].unique()
    df = df.set_index('season')
    return [zip(pd.Series(df.loc[season, 'from']),pd.Series(df.loc[season, 'to'])) for season in seasons]

def read_dep_ss_tf(period_df):
    seasons = period_df['season'].unique()
    period_df = period_df.set_index('season')
    return [zip(pd.Series(period_df.loc[season, 'from']),pd.Series(period_df.loc[season, 'to'])) for season in seasons]

def find_out_season(period, day_cv, year = 2016):
    '''
    output a season for a specific date
    :param period: a period dataframe
    :param day_cv: a specific date, in datetime type
    :param year: (default 2016) to deal with normal year or leap year
    :return: the season of that date
    '''
    day_adj = day_cv + pd.DateOffset(years= (year - day_cv.year))
    for i in xrange(0, len(period)):
        # pd.to_datetime("%s-"%year + period[1][0][0])
        time_list = period[i]
        for j in xrange(0, len(time_list)):
            start_date = pd.to_datetime("%s-"%year + time_list[j][0])
            end_date = pd.to_datetime("%s-"%year + time_list[j][1])
            if (day_adj >= start_date) & (day_adj <= end_date):
                return [time_list, day_cv.weekday()]

def find_out_period_id(period_id_dict, day_cv, year = 2016):
    '''
    output a season for a specific date
    :param period: a period dataframe
    :param day_cv: a specific date, in datetime type
    :param year: (default 2016) to deal with normal year or leap year
    :return: the season of that date
    '''
    day_adj = day_cv + pd.DateOffset(years= (year - day_cv.year))
    for i in xrange(0, len(period_id_dict)):
        # pd.to_datetime("%s-"%year + period[1][0][0])
        time_list = period_id_dict["timeframe"][i][0]
        period_id = period_id_dict["period_id"][i]
        for j in xrange(0, len(time_list)):
            start_date = pd.to_datetime("%s-"%year + time_list[j][0])
            end_date = pd.to_datetime("%s-"%year + time_list[j][1])
            if (day_adj >= start_date) & (day_adj <= end_date):
                return period_id


def perdelta(start, end, delta = timedelta(days=1)):
    '''
    create a list of dates that are equally separated
    :param start: start date
    :param end: end date
    :param delta: (default 1) the separated time you want
    :return: a list of dates that are equally separated
    '''
    curr = start
    while curr <= end:
        yield curr
        curr += delta

#======================WORKING-WITH-DATA================================
def get_type_col(col_name):
    return col_name.split("_")[2]

def get_root_col(col_name):
    return "_".join(col_name.split("_")[0:2])



def get_info_col(col_name, type_info):
    l = col_name.split("_")
    if type_info == 'dep':
        return l[0]
    elif type_info == 'sep':
        return l[1]
    elif type_info == 'col':
        return "_".join(l[:2])
    elif type_info == 'type_data':
        return l[2] # rv,gn,rn

def read_all_data(hotel_id):
    rn_df = pd.read_csv(os.path.join('data','hotel_{}_rn.csv'))
    rv_df = pd.read_csv(os.path.join('data','hotel_{}_rv.csv'))
    gn_df = pd.read_csv(os.path.join('data','hotel_{}_gn.csv'))

    date_s = rn_df['date']
    del rn_df['date']
    del rv_df['date']
    del gn_df['date']

    rn_df.columns = [create_col_name(col, 'rn') for col in rn_df.columns]
    rv_df.columns = [create_col_name(col, 'rv')  for col in rv_df.columns]
    gn_df.columns = [create_col_name(col, 'gn')  for col in gn_df.columns]

    df = pd.concat([date_s, rv_df, rn_df, gn_df], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    return df


def check_total(col_1, col_2):
    '''
    check 2 segments to determine we should keep the correlation data or not
    :param col_1: the 1st segment
    :param col_2: the 2nd segment
    :return: True if 2 segment is in same department, and 1 segment is total of department. Else return False
    '''

    ids_1 = col_1.split("_")
    ids_2 = col_2.split("_")
    if ids_1[0] != ids_2[0]:
        return False
    else:
        if ((int(ids_1[1]) * int(ids_2[1])) == 0) and ((int(ids_1[1]) + int(ids_2[1])) != 0):
            return True
        else:
            return False


#======================== OTB SNAPSHOT ===============================
def compute_otb_adjust_ratio(dep_id, seg_id, snapshot_date, otb_df, num_prev_day=365, range_leadtime=30):
    prev_year_snapshot = snapshot_date - timedelta(days=num_prev_day)
    otb_df['snapshot_date'] = pd.to_datetime(otb_df['snapshot_date'])
    otb_df = otb_df[otb_df['h_id'] == dep_id]
    otb_df = otb_df[otb_df['segment_id'] == seg_id]

    otb_df_cur = otb_df[otb_df['snapshot_date'] == snapshot_date].sort_values(['leadtime'], ascending=True)
    otb_df_cur = otb_df_cur[otb_df_cur['leadtime'] <= range_leadtime]

    otb_df_prev = otb_df[otb_df['snapshot_date'] == prev_year_snapshot].sort_values(['leadtime'], ascending=True)
    otb_df_prev = otb_df_prev[otb_df_prev['leadtime'] <= range_leadtime]

    # print otb_df_cur
    min_length = min(len(otb_df_cur), len(otb_df_prev))
    # print 'Current snapshot: ', snapshot_date, len(otb_df_cur)
    # print 'Previous year snapshot: ', prev_year_snapshot, len(otb_df_prev)
    # print "Min length: ", min_length
    ratio = otb_df_cur['revenue'][:min_length].mean(skipna=False)/otb_df_prev['revenue'][:min_length].mean(skipna=False)
    if np.isnan(ratio):
        ratio = 1
    return ratio


def create_otb(plot_df, all_rv_col_without_hid, otb_dict):
    new_df = plot_df[plot_df.columns]
    for id in new_df.index:
        if id % 100 == 0:
            print(id)
        rv_col = new_df.loc[id, "col"] 
        new_df.loc[id, "filled_values"] = new_df.loc[id, "filled_values"] * otb_dict[rv_col]
    return new_df


#======================SEASON==============================
def add_overlap_season_col(df_season, df_season2, dep_df2):
    dates_in_year = pd.date_range('01-01-2016', '31-12-2016') # choose leap year.
    overlap_seasons = []
    for tf in df_season['timeframe']:
        tf = tf[0] # convert to format [()]
        dates_in_tf_1 = set(get_season_dates_from_tf(tf, dates_in_year))
        tf1_overlap_season = []
        for tf_2 in df_season2['timeframe']:
            tf_2 = tf_2[0]
            dates_in_tf_2 = set(get_season_dates_from_tf(tf_2, dates_in_year))
            overlap_dates =  dates_in_tf_1.intersection(dates_in_tf_2)
            if len(overlap_dates)  > 0:
                tf1_overlap_season.append(tf_2) # Consider to timeframe format in database
        overlap_seasons.append(tf1_overlap_season)
    df_season[dep_df2] = overlap_seasons
    return df_season




