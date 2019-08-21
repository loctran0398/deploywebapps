"""
Copyright: TP7
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import os
import project_lib as lib


def regression(col_y, col_x, lag, season_tf, num_day, day_of_week, df):
    # Get dataframe for col1
    dates_1 = lib.generate_season_date_by_dof(season_tf, num_day, day_of_week, df['date'])
    df1 = lib.get_df_by_dates(dates_1, df)

    # Get dataframe for col2
    dates_2 = lib.get_dates_lag(dates_1, lag)
    df2 = lib.get_df_by_dates(dates_2, df)
    col1_data = df2[col_x]
    col2_data = df1[col_y]

    col1_data_rm, col2_data_rm = lib.remove_outlier(col1_data, col2_data)
    regr = linear_model.LinearRegression()
    col1_data_rm = col1_data_rm.reshape(len(col1_data_rm), 1)
    try:
        regr.fit(col1_data_rm, col2_data_rm)
        result = {'coef': regr.coef_[0], 'intercept': regr.intercept_, 'score': regr.score(col1_data_rm, col2_data_rm)}
    except:
        print dates_1
        print dates_2
        print col1_data, col2_data
        result = {'coef': 0, 'intercept': 0}
        raise
    return result

def compute_capture_arr(col1, col2, lag, season_tf, num_day, day_of_week, df):
    """
    Compute capture and arr for original column col1 and lag column col2
    :param col1: dep_seg id of original column
    :param col2: dep_seg id of lag column
    :param lag:
    :param season_tf: season time frame
    :param num_day:
    :param day_of_week:
    :param df: dataframe contain data of one hotel
    :return: dictionary contain 'capture', 'ARR', 'capture_unit', 'ARR_unit'
    """
    # Get dataframe for col1

    dates_1 = lib.generate_season_date_by_dof(season_tf, num_day, day_of_week, df['date'])
    df1 = lib.get_df_by_dates(dates_1, df)

    # Get dataframe for col2
    dates_2 = lib.get_dates_lag(dates_1, lag)
    df2 = lib.get_df_by_dates(dates_2, df)
    root_col1 = lib.get_root_col(col1)
    root_col2 = lib.get_root_col(col2)
    type_col2 = lib.get_type_col(col2)
    col1_rv = "{}_rv".format(root_col1)
    col1_rn = "{}_rn".format(root_col1)
    col2_rv = "{}_rv".format(root_col2)
    col2_rn = "{}_rn".format(root_col2)
    col2_gn = "{}_gn".format(root_col2)

    capture = 1
    ARR = 0
    capture_unit = 1
    ARR_unit = 0

    try:
        if root_col1 == root_col2:
            if col1_rn in df.columns: # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = lib.remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in df.columns: # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = lib.remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = lib.remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = lib.remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture':capture, 'ARR':ARR, 'capture_unit':capture_unit, 'ARR_unit':ARR_unit}
    return result


def compute_capture_arr_1year(col1, col2, lag, season_tf, num_day, day_of_week, df):
    """
    Compute capture and arr for original column col1 and lag column col2
    :param col1: dep_seg id of original column
    :param col2: dep_seg id of lag column
    :param lag:
    :param season_tf: season time frame
    :param num_day:
    :param day_of_week:
    :param df: dataframe contain data of one hotel
    :return: dictionary contain 'capture', 'ARR', 'capture_unit', 'ARR_unit'
    """
    # Get dataframe for col1
    df_insample_1year=df[(df['date'] >= "01-01-2015") & (df['date'] <= "12-31-2015")] #"01-01-2012, 12-31-2015"
    dates_1 = lib.generate_season_date_by_dof(season_tf, num_day, day_of_week, df_insample_1year['date'])
    df1 = lib.get_df_by_dates(dates_1, df)

    # Get dataframe for col2
    dates_2 = lib.get_dates_lag(dates_1, lag)
    df2 = lib.get_df_by_dates(dates_2, df)
    root_col1 = lib.get_root_col(col1)
    root_col2 = lib.get_root_col(col2)
    type_col2 = lib.get_type_col(col2)
    col1_rv = "{}_rv".format(root_col1)
    col1_rn = "{}_rn".format(root_col1)
    col2_rv = "{}_rv".format(root_col2)
    col2_rn = "{}_rn".format(root_col2)
    col2_gn = "{}_gn".format(root_col2)

    capture = 1
    ARR = 0
    capture_unit = 1
    ARR_unit = 0

    try:
        if root_col1 == root_col2:
            if col1_rn in df.columns: # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = lib.remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in df.columns: # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = lib.remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = lib.remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = lib.remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture':capture, 'ARR':ARR, 'capture_unit':capture_unit, 'ARR_unit':ARR_unit}
    return result


def compute_capture_arr_ver_2(col1, col2, y_series, col_series, df1, df2, cols):
    """
    Compute capture and arr for original column col1 and lag column col2
    :param col1: dep_seg id of original column
    :param col2: dep_seg id of lag column
    :param lag:
    :param season_tf: season time frame
    :param num_day:
    :param day_of_week:
    :param df: dataframe contain data of one hotel
    :return: dictionary contain 'capture', 'ARR', 'capture_unit', 'ARR_unit'
    """
    root_col1 = "_".join(col1.split("_")[0:2])
    root_col2 = "_".join(col2.split("_")[0:2])
    type_col2 = col2.split("_")[2]
    col1_rv = "{}_rv".format(root_col1)
    col1_rn = "{}_rn".format(root_col1)
    col2_rv = "{}_rv".format(root_col2)
    col2_rn = "{}_rn".format(root_col2)
    col2_gn = "{}_gn".format(root_col2)

    capture = 1
    ARR = 0
    capture_unit = 1
    ARR_unit = 0

    try:
        if root_col1 == root_col2:
            if col1_rn in cols: # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = lib.remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in cols: # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = lib.remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = lib.remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = lib.remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture':capture, 'ARR':ARR, 'capture_unit':capture_unit, 'ARR_unit':ARR_unit}
    return result



def compute_capture_arr_week(col1, col2, lag, season_tf, df, year_input):
    """ neeed to change the comment later
    Compute capture and arr for original column col1 and lag column col2
    :param col1: dep_seg id of original column
    :param col2: dep_seg id of lag column
    :param lag:
    :param season_tf: season time frame
    :param day_of_week:
    :param df: dataframe contain data of one hotel
    :return: dictionary contain 'capture', 'ARR', 'capture_unit', 'ARR_unit'
    """
    # Get dataframe for col1
    dates_1 = lib.generate_season_date(season_tf, year_input)
    df1 = lib.get_df_by_dates(dates_1, df)

    # Get dataframe for col2
    dates_2 = lib.get_dates_lag(dates_1, lag)
    df2 = lib.get_df_by_dates(dates_2, df)
    root_col1 = lib.get_root_col(col1)
    root_col2 = lib.get_root_col(col2)
    type_col2 = lib.get_type_col(col2)
    col1_rv = "{}_rv".format(root_col1)
    col1_rn = "{}_rn".format(root_col1)
    col2_rv = "{}_rv".format(root_col2)
    col2_rn = "{}_rn".format(root_col2)
    col2_gn = "{}_gn".format(root_col2)

    capture = 1
    ARR = 0
    capture_unit = 1
    ARR_unit = 0

    try:
        if root_col1 == root_col2:
            if col1_rn in df.columns: # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = lib.remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in df.columns: # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = lib.remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = lib.remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = lib.remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture':capture, 'ARR':ARR, 'capture_unit':capture_unit, 'ARR_unit':ARR_unit}
    return result


def compute_capture_arr_week_ver_2(col1, col2, y_series, col_series, df1, df2, cols):
    root_col1 = "_".join(col1.split("_")[0:2])
    root_col2 = "_".join(col2.split("_")[0:2])
    type_col2 = col2.split("_")[2]
    col1_rv = "{}_rv".format(root_col1)
    col1_rn = "{}_rn".format(root_col1)
    col2_rv = "{}_rv".format(root_col2)
    col2_rn = "{}_rn".format(root_col2)
    col2_gn = "{}_gn".format(root_col2)

    capture = 1
    ARR = 0
    capture_unit = 1
    ARR_unit = 0

    try:
        if root_col1 == root_col2:
            if col1_rn in cols: # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = lib.remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm/col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in cols: # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = lib.remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm/col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = lib.remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = lib.remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm/col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = lib.remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm/col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = lib.remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm/col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture':capture, 'ARR':ARR, 'capture_unit':capture_unit, 'ARR_unit':ARR_unit}
    return result
