"""
Copyright: TP7
"""

# This module is used to compute the correlation between each pair of departments in one hotel

import pandas as pd
import numpy as np
from d2o.revenue_lib.capture_arr import *
import d2o.revenue_lib.project_lib as lib
import os

def get_series_to_compute_corr(y_col_name, x_col_name, lag, y_dates, df):
    """
    Prepare series to compute correlation
    :param y_col_name: name of column y
    :param x_col_name: name of column x
    :param lag: lag days between y_col and x_col
    :param y_dates: dates series by y_col
    :param df: dataframe contain all data of one hotel
    :return:
    """
    col_dates  = lib.get_dates_lag(y_dates, lag)
    y_series   = lib.get_df_by_dates(y_dates, df)[y_col_name]
    col_series = lib.get_df_by_dates(col_dates, df)[x_col_name]
    # if length of x_col data smaller than of y_col just return data by length x_col
    return y_series[:len(col_series)], col_series

def get_df_corr_data(dep_id, seg_id, season_tf, day_of_week, df):
    """
    Get data to test. It's consistent with the ways getting data in compute_corr
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :return:
    """
    dates = df['date'].sort_values(ascending=False)
    season_dates = pd.Series(lib.get_season_dates_from_tf(season_tf, dates))
    y_dates = list(season_dates[season_dates.dt.weekday == day_of_week])
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    cols = set(df.columns) - {'date', 'day_of_week'}
    y_series = lib.get_df_by_dates(y_dates,df)[y_col]
    df_result = pd.DataFrame({y_col:list(y_series)})
    for lag in [0, 1, 7, 14, 21, 30, 365]:
        for i, col in enumerate(cols):
            df_result["date_{}".format(lag)] = lib.get_dates_lag(y_dates, lag)
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            y_series, col_series = get_series_to_compute_corr(y_col, col, lag, y_dates, df)
            if len(col_series) < len(y_dates):
                df_result["{}_{}".format(col, lag)] = list(col_series) + ['F']*(len(y_dates)-len(col_series))
            else:
                df_result["{}_{}".format(col, lag)] = list(col_series)
    return df_result


# ver 2 is ERROR
def compute_corr_ver_2(dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df):
    """
    Compute explanation (1 -error) for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """

    dates = df['date'].sort_values(ascending=False)
    season_dates = pd.Series(lib.get_season_dates_from_tf(season_tf, dates))
    y_dates = list(season_dates[season_dates.dt.weekday == day_of_week])
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name

    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    for i, col in enumerate(cols):
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            y_series, col_series = get_series_to_compute_corr(y_col, col, lag, y_dates, df)

            cap_arr = compute_capture_arr(y_col, col, lag, season_tf, num_day, day_of_week, df)
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = 1 - lib.compute_error(y_series_rm, col_series, cap_arr['capture'], cap_arr['ARR'])
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


# ver 2 is ERROR
def compute_corr_ver_2_1(dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df, date_ss):
    """
    Compute explanation (1 -error) for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    y_dates_full = [i for i in date_ss if i.weekday() == day_of_week]
    y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    df1_full = lib.get_df_by_dates(y_dates_full, df)

    y_series = df1_full[y_col]

    df1 = lib.get_df_by_dates(y_dates, df1_full)
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    dict_df = {}
    dict_df[0] = df1[df1.columns]
    dict_df["0_full"] = df1_full[df1_full.columns]
    for lag in [1, 7]: #remove lag 365
        dates  = lib.get_dates_lag(y_dates, lag)
        dates_full  = lib.get_dates_lag(y_dates_full, lag)
        df2_full = lib.get_df_by_dates(dates_full, df)
        df2 = lib.get_df_by_dates(dates, df2_full)
        dict_df[lag] = df2
        dict_df[str(lag) + "_full"] = df2_full
    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1, 7]: #remove lag 365
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_series = dict_df[str(lag) + "_full"][col]
            y_series_r = y_series[-len(col_series):]
            df2 = dict_df[lag]
            y_series_rm, col_series_rm = lib.remove_outlier(y_series_r, col_series)
            cap_arr = compute_capture_arr_ver_2(y_col, col, 1, 1, df1, df2, cols)
            cov_val = 1 - lib.compute_error(y_series_rm, col_series_rm, cap_arr['capture'], cap_arr['ARR'])
            if y_col == "69_107_rv" and day_of_week == 6 and season_tf == [('01-01', '02-17'), ('12-23', '12-31')] and lag == 0:
                print('test', col)
                with open('test_caset_69_107_{0}_{1}.txt'.format(season_tf, col), 'w') as f:
                    f.write("\ny_series: {0}\ncol_series:{1}\ny_rm:{2}\ncol_rm: {3}\ncap_arr: {4}\cov_val:{5}".format(\
                        ", ".join(map(str,y_series_r)), ", ".join(map(str, col_series)), ", ".join(map(str, y_series_rm)), ", ".join(map(str, col_series_rm)), str(cap_arr), cov_val))
            #===========================
            if lag == 365:
                y_365_lag = df2[col]
                date_y_365_lag_not_null = df2[~y_365_lag.isnull()]['date']
                if len(date_y_365_lag_not_null.dt.year.unique()) < 3:
                    cov_val = -99  # explaination is big
            #===========================
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)

def compute_corr_ver_2_1_cruise(dep_id, seg_id, season_tf, num_day, day_of_week,
                                significance_level, df, date_ss, cruise):
    """
    Compute explanation (1 -error) for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    y_weekdays = cruise.cruiseday_list(date_ss)
    y_dates_full = [date_ss[i] for i, d in enumerate(y_weekdays) if d == day_of_week]
    y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    df1_full = lib.get_df_by_dates(y_dates_full, df)
    y_series = df1_full[y_col]

    df1 = lib.get_df_by_dates(y_dates, df1_full)
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    dict_df = {}
    dict_df[0] = df1[df1.columns]
    dict_df["0_full"] = df1_full[df1_full.columns]
    for lag in [1, 7]: #remove lag 365
        dates  = lib.get_dates_lag(y_dates, lag)
        dates_full  = lib.get_dates_lag(y_dates_full, lag)
        df2_full = lib.get_df_by_dates(dates_full, df)
        df2 = lib.get_df_by_dates(dates, df2_full)
        dict_df[lag] = df2
        dict_df[str(lag) + "_full"] = df2_full
    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1, 7]: #remove lag 365
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_series = dict_df[str(lag) + "_full"][col]
            y_series_r = y_series[-len(col_series):]
            df2 = dict_df[lag]
            y_series_rm, col_series_rm = lib.remove_outlier(y_series_r, col_series)
            cap_arr = compute_capture_arr_ver_2(y_col, col, 1, 1, df1, df2, cols)
            cov_val = 1 - lib.compute_error(y_series_rm, col_series_rm, cap_arr['capture'], cap_arr['ARR'])
            if y_col == "69_107_rv" and day_of_week == 6 and season_tf == [('01-01', '02-17'), ('12-23', '12-31')] and lag == 0:
                print('test', col)
                with open('test_caset_69_107_{0}_{1}.txt'.format(season_tf, col), 'w') as f:
                    f.write("\ny_series: {0}\ncol_series:{1}\ny_rm:{2}\ncol_rm: {3}\ncap_arr: {4}\cov_val:{5}".format( \
                        ", ".join(map(str,y_series_r)), ", ".join(map(str, col_series)), ", ".join(map(str, y_series_rm)), ", ".join(map(str, col_series_rm)), str(cap_arr), cov_val))
            #===========================
            if lag == 365:
                y_365_lag = df2[col]
                date_y_365_lag_not_null = df2[~y_365_lag.isnull()]['date']
                if len(date_y_365_lag_not_null.dt.year.unique()) < 3:
                    cov_val = -99  # explaination is big
            #===========================
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


# ver 3 is CORRELATION / REGRESSION
def compute_corr_ver_3(dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df):
    """
    Compute correlation for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    dates = df['date'].sort_values(ascending=False)
    season_dates = pd.Series(lib.get_season_dates_from_tf(season_tf, dates))
    y_dates = list(season_dates[season_dates.dt.weekday == day_of_week])
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name

    cols = set(df.columns) - {'date', 'day_of_week'}
    # result_df = pd.DataFrame(columns=['col', 'lag', 'cov_value', "conf_interval_0", "conf_interval_1"])
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])

    # x is some col with some lag
    index = 0
    for i, col in enumerate(cols):
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            y_series, col_series = get_series_to_compute_corr(y_col, col, lag, y_dates, df)
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = np.corrcoef(y_series_rm, col_series)[1, 0]
            conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


# ver 3 is CORRELATION / REGRESSION
def compute_corr_ver_3_1(dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df, date_ss):
    """
    Compute correlation for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    y_dates = [i for i in date_ss if i.weekday() == day_of_week]
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    y_series = lib.get_df_by_dates(y_dates, df)[y_col]
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_dates  = lib.get_dates_lag(y_dates, lag)
            col_series = lib.get_df_by_dates(col_dates, df)[col]
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = np.corrcoef(y_series_rm, col_series)[1, 0]
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


# ver 2 is ERROR
def compute_corr_ver_2_week(dep_id, seg_id, season_tf, significance_level, df, year_input):
    """
    Compute explanation (1 -error) for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    dates = df['date'].sort_values(ascending=False)
    season_dates = pd.Series(lib.get_season_dates_from_tf(season_tf, dates))
    y_dates = list(season_dates)
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    for i, col in enumerate(cols):
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            y_series, col_series = get_series_to_compute_corr(y_col, col, lag, y_dates, df)
            cap_arr = compute_capture_arr_week(y_col, col, lag, season_tf, df, year_input)
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = 1 - lib.compute_error(y_series_rm, col_series, cap_arr['capture'], cap_arr['ARR'])
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


def compute_corr_ver_2_week_1(dep_id, seg_id, season_tf, significance_level, df, year_input, date_ss):
    y_dates = date_ss
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    df1_full = lib.get_df_by_dates(y_dates, df)
    y_series = df1_full[y_col]
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0
    dict_df = {}
    dict_df["0_2"] = df1_full[df1_full.columns]
    dict_df["0_1"] = df1_full[df1_full.columns]
    for lag in [1, 7, 365]:
        col_dates  = lib.get_dates_lag(y_dates, lag)
        df2 = lib.get_df_by_dates(col_dates, df)
        dict_df[str(lag) + "_2"] = df2[df2.columns]
        dict_df[str(lag) + "_1"] = df1_full[(len(df1_full) - len(df2)) : ]
    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1, 7, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            df1 = dict_df[str(lag) + "_1"]
            df2 = dict_df[str(lag) + "_2"]
            col_series = df2[col]
            y_series_rm = y_series[-len(col_series):]
            y_series_rm, col_series = lib.remove_outlier(y_series_rm, col_series)
            cap_arr = compute_capture_arr_ver_2(y_col, col, 1, 1, df1, df2, cols)
            cov_val = 1 - lib.compute_error(y_series_rm, col_series, cap_arr['capture'], cap_arr['ARR'])
            #===========================
            if lag == 365:
                y_365_lag = df2[col]
                date_y_365_lag_not_null = df2[~y_365_lag.isnull()]['date']
                if len(date_y_365_lag_not_null.dt.year.unique()) < 3:
                    cov_val = -99  # explaination is big
            #===========================
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


# ver 3 is CORRELATION / REGRESSION
def compute_corr_ver_3_week(dep_id, seg_id, season_tf, significance_level, df):
    """
    Compute correlation for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    dates = df['date'].sort_values(ascending=False)
    season_dates = pd.Series(lib.get_season_dates_from_tf(season_tf, dates))
    y_dates = list(season_dates)
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])

    # x is some col with some lag
    index = 0
    for i, col in enumerate(cols):
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            y_series, col_series = get_series_to_compute_corr(y_col, col, lag, y_dates, df)
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = np.corrcoef(y_series_rm, col_series)[1, 0]
            conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)



# ver 3 is CORRELATION / REGRESSION
def compute_corr_ver_3_week_1(dep_id, seg_id, season_tf, significance_level, df, date_ss):
    """
    Compute correlation for all pair of deparment_segment in one hotel
    :param dep_id: department id
    :param seg_id: segment id
    :param season_tf: season timeframe
    :param day_of_week:  day of week range 0-6
    :param df: Dataframe contain all data of one hotel
    :param significance_level: significance to compute confidence interval
    :return:
    """
    y_dates = date_ss
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    y_series = lib.get_df_by_dates(y_dates, df)[y_col]
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    # x is some col with some lag
    index = 0

    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1, 7, 14, 21, 30, 365]:
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_dates  = lib.get_dates_lag(y_dates, lag)
            col_series = lib.get_df_by_dates(col_dates, df)[col]
            y_series_rm, col_series = lib.remove_outlier(y_series, col_series)
            cov_val = np.corrcoef(y_series_rm, col_series)[1, 0]
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)
