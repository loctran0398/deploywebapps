# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:15:52 2017

@author: Ar
"""
import pandas as pd
import numpy as np
import ast
import time
import statsmodels.api as sm
from datetime import timedelta, datetime
import d2o.revenue_lib.project_lib as lib

from d2o.utils import logger as log
import sys
from d2o_common.legacy_lib import database_api as db_api


def log_json_upload(client_id, log_file_client):
    log_file_client = [log_file_client]
    url = '%s/Forecast/RevenueDriver/Auto/Log/%s' % (db_api.HOST, client_id)
    db_api.post_api(link=url, json_data=log_file_client)
    log.info("Writing revenue_driver_detection_json log to API")
    print("Pass send log to API")

#KL
def set_weight(chosen_rows):
    '''
    :param chosen_rows is list of date
    :output return list weight set by year
    '''
    w = np.ones(len(chosen_rows))
    chosen_year = list(set([row.year for row in chosen_rows]))
    if len(chosen_year) >= 2:
        date1_newyear = datetime(chosen_year[-1:][0], 1, 1)
        index_newyear = [idx for idx, val in enumerate(chosen_rows) if val >= date1_newyear]
        index_oldyear = [idx for idx, val in enumerate(chosen_rows) if val < date1_newyear]
        w[index_newyear] = 4.0
        nsample = len(index_newyear)
        w[len(index_oldyear) + int(nsample * 7.5 // 10):] = 8.0
    else:
        nsample = len(chosen_rows)
        w[int(nsample * 7.5 // 10):] = 8.0
    return w
#KL-End

def sMAPE(y_true, y_pred):
    try:
        y_true = y_true.reset_index(drop=True)
        y_pred = y_pred.reset_index(drop=True)
    except:
        print('sMAPE function: y_true and y_pred should be pd.Series.')
    a = pd.DataFrame()
    a['y_true'] = y_true
    a['y_pred'] = y_pred
    n = len(y_pred)
    a = a[(a['y_true'] != 0) | (a['y_pred'] != 0)]
    #    ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
    return np.sum(np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred']))) / n

def sMAPE_weight(y_true, y_pred, w):
    try:
        y_true = y_true.reset_index(drop=True)
        y_pred = y_pred.reset_index(drop=True)
    except:
        print('sMAPE function: y_true and y_pred should be pd.Series.')
    a = pd.DataFrame()
    a['y_true'] = y_true
    a['y_pred'] = y_pred
#    n = len(y_pred)
#    a = a[(a['y_true'] != 0) | (a['y_pred'] != 0)]
    smape_series = np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred']))
    smape_series = smape_series.replace(np.nan,0)
    #    ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
    return np.sum(w*smape_series)


def v1_compute_error(y, x, capture, arr):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_forecast = x * capture * arr
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


# def v2_compute_error_single_corr(y, x, capture, arr, k_modifier):
#    """
#    Compute predicted values (base on capture and arr) and return error of model ver 1
#    :param y: real value series
#    :param x: predicted value series
#    :param capture: capture values
#    :param arr: ARR values
#    :return: error values
#    """
#    y_min = float(np.percentile(y, 0))
#    y_max = float(np.percentile(y, 100))
#    #======================================================================================
#    a = np.linalg.lstsq(x[:,np.newaxis], y)[0][0]
#    modifier = a/(capture*arr)
#    if np.isinf(modifier) or np.isnan(modifier):
#        modifier = 1.0
#    modifier = min(max(modifier, 1-k_modifier), 1+k_modifier)
#
#    y_forecast = modifier*capture*arr*x
#    #======================================================================================
#    y_forecast = pd.Series([max(min(y_max,temp),y_min) for temp in y_forecast])
#    s = sMAPE(y, y_forecast)
#    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
#    forecast_error = 100 if np.isinf(s) else s
#    return [y_forecast, forecast_error, modifier, y_min, y_max]

def v2_compute_error_single_corr(y, x, capture, arr, k_modifier, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_min = min(y)
    y_max = max(y)
    # ======================================================================================
    w = np.ones(len(y))
    w=set_weight(y_dates)
    mod_wls = sm.WLS(y, x, weights= w)    
    res_wls = mod_wls.fit()
    a = res_wls.params[0]
    modifier = a / (capture * arr)
    if np.isinf(modifier) or np.isnan(modifier):
        modifier = 1.0

    y_forecast = modifier * capture * arr * x
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    s = sMAPE_weight(y, y_forecast, w/sum(w))
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error, modifier, y_min, y_max]

def v2_compute_error_single_corr_for_otb(y, x, k_modifier, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_min = min(y)
    y_max = max(y)
    # ======================================================================================
    w = np.ones(len(y))
    w=set_weight(y_dates)
    mod_wls = sm.WLS(y, x, weights=w)
    res_wls = mod_wls.fit()
    a = res_wls.params[0]
    modifier = 0
    if np.isinf(modifier) or np.isnan(modifier):
        modifier = 1.0
        
    y_forecast =  a * x
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    s = sMAPE_weight(y, y_forecast, w/sum(w))
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    nan_number = y/x
    rate_nan = float(np.isnan(nan_number).sum())/len(nan_number)
    if rate_nan >= 0.6:
        forecast_error = -0.00001
    return [y_forecast, forecast_error, modifier, y_min, y_max]



def v2_compute_error_single(y, x, capture, arr, k_modifier, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_min = min(y)
    y_max = max(y)
    # ======================================================================================
    w = np.ones(len(y))
    w=set_weight(y_dates)
    mod_wls = sm.WLS(y, x, weights=w)
    res_wls = mod_wls.fit()
    a = res_wls.params[0]
    modifier = a / (capture * arr)
    if np.isinf(modifier) or np.isnan(modifier):
        modifier = 1.0

    y_forecast = modifier * capture * arr * x
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    s = sMAPE_weight(y, y_forecast, w/sum(w))
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error, modifier, y_min, y_max, a]


def v2_compute_error(y, x1, capture1, arr1, x2, capture2, arr2, k_modifier, y_dates):
    y_min = min(y)
    y_max = max(y)
    # ======================================================================================
    x = pd.concat([x1, x2], axis=1).max(axis=1)
    #    if rate1 == 0:
    #        x = x2
    #    elif rate2 == 0:
    #        x = x1
    w = np.ones(len(y))
    w=set_weight(y_dates)

    mod_wls = sm.WLS(y, x, weights=w)
    res_wls = mod_wls.fit()
    a = res_wls.params[0]

    modifier1 = a / (capture1 * arr1)
    modifier2 = a / (capture2 * arr2)

    #    mod_wls1 = sm.WLS(y, x1, weights=1./w)
    #    res_wls1 = mod_wls1.fit()
    #    a1 = res_wls1.params[0]
    #    mod_wls2 = sm.WLS(y, x2, weights=1./w)
    #    res_wls2 = mod_wls2.fit()
    #    a2 = res_wls2.params[0]
    #
    #    modifier1 = a1/(capture1*arr1)
    #    modifier2 = a2/(capture2*arr2)

    if np.isinf(modifier1) or np.isnan(modifier1):
        modifier1 = 1.0
    if np.isinf(modifier2) or np.isnan(modifier2):
        modifier2 = 1.0

    y1_forecast = modifier1 * capture1 * arr1 * x1
    y2_forecast = modifier2 * capture2 * arr2 * x2
    y_forecast = pd.concat([y1_forecast, y2_forecast], axis=1).max(axis=1)
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    y1_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y1_forecast])
    y2_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y2_forecast])
    s = sMAPE_weight(y, y_forecast, w/sum(w))
    s1 = sMAPE_weight(y, y1_forecast, w/sum(w))
    s2 = sMAPE_weight(y, y2_forecast, w/sum(w))
    #    print('both: ', 1-s)
    #    print('x1: ', 1-s1)
    #    print('x2: ', 1-s2)
    if min(s, s1, s2) == s1:
        modifier2 = 0.0
        y_forecast = y1_forecast
        s = s1
    elif min(s, s1, s2) == s2:
        modifier1 = 0.0
        y_forecast = y2_forecast
        s = s2
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error, modifier1, modifier2, y_min, y_max]


# def v2_compute_error_single(y, x, capture, arr, k_modifier):
#    """
#    Compute predicted values (base on capture and arr) and return error of model ver 1
#    :param y: real value series
#    :param x: predicted value series
#    :param capture: capture values
#    :param arr: ARR values
#    :return: error values
#    """
#    y_min = float(np.percentile(y, 0))
#    y_max = float(np.percentile(y, 100))
#    #======================================================================================
#    a = np.linalg.lstsq(x[:,np.newaxis], y)[0][0]
#    modifier = a/(capture*arr)
#    if np.isinf(modifier) or np.isnan(modifier):
#        modifier = 1.0
#    modifier = min(max(modifier, 1-k_modifier), 1+k_modifier)
#
#    y_forecast = modifier*capture*arr*x
##======================================================================================
#    y_forecast = pd.Series([max(min(y_max,temp),y_min) for temp in y_forecast])
#    s = sMAPE(y, y_forecast)
#    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
#    forecast_error = 100 if np.isinf(s) else s
#    return [y_forecast, forecast_error, modifier, y_min, y_max]
#
# def v2_compute_error(y, x1, capture1, arr1, x2, capture2, arr2, k_modifier):
#    """
#    Compute predicted values (base on capture and arr) and return error of model ver 2
#    :param y: real value of forecast driver
#    :param x: value of other drivers
#    :param capture: capture values
#    :param arr: ARR values
#    :return: error values
#    """
#    y_min = float(np.percentile(y, 0))
#    y_max = float(np.percentile(y, 100))
#    #======================================================================================
#    x = pd.concat([x1, x2], axis=1).max(axis=1)
#    a = np.linalg.lstsq(x[:,np.newaxis], y)[0][0]
#    modifier1 = a/(capture1*arr1)
#    modifier2 = a/(capture2*arr2)
#    if np.isinf(modifier1) or np.isnan(modifier1):
#        modifier1 = 1.0
#    if np.isinf(modifier2) or np.isnan(modifier2):
#        modifier2 = 1.0
#    modifier1 = min(max(modifier1, 1-k_modifier), 1+k_modifier)
#    modifier2 = min(max(modifier2, 1-k_modifier), 1+k_modifier)
#
#    y1_forecast = modifier1*capture1*arr1*x1
#    y2_forecast = modifier2*capture2*arr2*x2
#    y_forecast = pd.concat([y1_forecast, y2_forecast], axis=1).max(axis=1)
#    #======================================================================================
#    y_forecast = pd.Series([max(min(y_max,temp),y_min) for temp in y_forecast])
#    s = sMAPE(y, y_forecast)
#    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
#    forecast_error = 100 if np.isinf(s) else s
#    return [y_forecast, forecast_error, modifier1, modifier2, y_min, y_max]

def df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, median_lag7_full, median_lag7):
    dict_df = {}
    dict_df[0] = df1[df1.columns]
    dict_df["0_full"] = df1_full[df1_full.columns]
    for lag_temp in [1]:  # remove lag 7, 365
        dates = lib.get_dates_lag(y_dates, lag_temp)
        dates_full = lib.get_dates_lag(y_dates_full, lag_temp)
        df2_full = lib.get_df_by_dates(dates_full, df)
        df2 = lib.get_df_by_dates(dates, df2_full)
        if (lag_temp == 7):
            df2_full.loc[~df2_full[y_col].isnull(), y_col] = median_lag7_full
            df2.loc[~df2[y_col].isnull(), y_col] = median_lag7

        dict_df[lag_temp] = df2
        dict_df[str(lag_temp) + "_full"] = df2_full
    return dict_df


def v2_compute_corr(dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df, date_ss, day_arr,
                    day_1yrs_back,cruise, accuracy_df, k_modifier=1000):
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
    column_otb = [str(i) * 3 + '_' + str(i) * 3 + '_rv' for i in range(5,31)]
    y_weekdays = cruise.cruiseday_list(date_ss)
    y_dates_full = [date_ss[i] for i, d in enumerate(y_weekdays) if d == day_of_week]
    y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
    y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
    df1_full = lib.get_df_by_dates(y_dates_full, df)

    y_series = df1_full[y_col]

    df1 = lib.get_df_by_dates(y_dates, df1_full)
    cols = set(df.columns) - {'date', 'day_of_week'}
    result_df = pd.DataFrame(
        columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    index = 0

    median_lag7_full = np.median(df1_full[df1_full['date'] >= day_1yrs_back][y_col].dropna())
    median_lag7 = np.median(df1[df1['date'] >= day_1yrs_back][y_col].dropna())
    dict_df = df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, median_lag7_full, median_lag7)
    # mm
    #    print('Season: {}, Day of week: {}'.format(season_tf, day_of_week))
    #    import pickle
    #    with open('dict.pickle', 'wb') as handle:
    #        pickle.dump(dict_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for i, col in enumerate(cols):
#        print(col)
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1]:  # remove lag 7, 365
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_series = dict_df[str(lag) + "_full"][col]
            y_series_r = y_series[-len(col_series):]
            # mm
            if (y_series_r.isnull().values.all()) or (col_series.isnull().values.all()) \
                    or (len(y_series_r) == 0) or (len(col_series) == 0):
                result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', \
                                                  "conf_interval_0", "conf_interval_1", "adj_corr_col"])
                len_columns = len(df.columns) - 2
                result_df['col'] = list([str(dep_id) + '_0_rv'] * len_columns)
                result_df['cov_col'] = list(set(df.columns) - {'date', 'day_of_week'})
                result_df['lag'] = np.zeros(len_columns) + 1
                result_df['cov_value'] = np.zeros(len_columns)
                result_df['conf_interval_0'] = np.zeros(len_columns)
                result_df['conf_interval_1'] = np.zeros(len_columns)
                result_df['adj_corr_col'] = [x + '_1' for x in result_df['cov_col']]
                return result_df

            df2 = dict_df[lag]
            y_series_r.index = y_dates
            col_series.index = y_dates
            y_series_rm, col_series_rm = v2_remove_outlier(y_series_r, col_series)
            
            new_y_dates = y_series_rm.index
            y_series_rm = y_series_rm.reset_index(drop = True)
            col_series_rm = col_series_rm.reset_index(drop = True)
#            cap_arr = v2_compute_capture_arr(day_arr, y_col, col, 1, 1, df1, df2, cols)
#            y_forecast, forecast_error, modifier, y_min, y_max = v2_compute_error_single_corr(y_series_rm,
#                                                                                              col_series_rm,
#                                                                                              cap_arr['capture'],
#                                                                                              cap_arr['ARR'],
#                                                                                              k_modifier)
            if  not pd.Series([col]).isin(column_otb).iloc[0]:
    
                cap_arr = v2_compute_capture_arr(day_arr, y_col, col, 1, 1, df1, df2, cols)
                y_forecast, forecast_error, modifier, y_min, y_max = v2_compute_error_single_corr(y_series_rm,
                                                                                                  col_series_rm,
                                                                                                  cap_arr['capture'],
                                                                                                  cap_arr['ARR'],
                                                                                                  k_modifier,new_y_dates)
                # df_value_forecast[col+'_'+str(lag)] = y_forecast.values
                # print (col, y_forecast)
                
                #mm 09282018: apply new accuracy when computing correlation
                cov_val = (1 - forecast_error) 
                
            elif lag == 0:
                y_forecast, forecast_error, modifier, y_min, y_max = v2_compute_error_single_corr_for_otb(y_series_rm,
                                                                                                  col_series_rm,
                                                                                                  k_modifier, new_y_dates)
                cov_val = 1 - forecast_error
            else:
                continue
            #            if (int(dep_id)==258) and (int(seg_id)==0) and (day_of_week==4) and (season_tf == [('07-31', '09-07')]):
            #                kq = pd.DataFrame()
            #                date_temp =df1_full['date'].reset_index(drop=True)
            #                kq['date'] = [i.date().strftime('%m/%d/%Y') for i in date_temp]
            #                kq['y_series_r'] = y_series_r.reset_index(drop=True)
            #                kq['col_series'] = col_series.reset_index(drop=True)
            #                kq[y_col] = y_series_rm.reset_index(drop=True)
            #                kq[col] = col_series_rm.reset_index(drop=True)
            #                kq['cap_arr'] = pd.DataFrame([cap_arr['capture'],cap_arr['ARR']])
            #                kq['modifier'] = pd.DataFrame([modifier])
            #                kq['high_low'] = pd.DataFrame([y_max, y_min])
            #                kq['y_forecast'] = y_forecast.reset_index(drop=True)
            #                kq['acc'] = pd.DataFrame([1-forecast_error])
            #                df1.to_csv('train_data/df1.csv', index = False)
            ##                kq.to_csv('train_data/case_{}_{}_{}_{}.csv'.format(dep_id, seg_id, day_of_week, season_tf), index = False)
            #                dict_df['0_full'].to_csv('train_data/lag0_{}_{}_{}_{}.csv'.format(dep_id, seg_id, day_of_week, season_tf), index = False)
            #                dict_df['1_full'].to_csv('train_data/lag1_{}_{}_{}_{}.csv'.format(dep_id, seg_id, day_of_week, season_tf), index = False)
            #                dict_df['7_full'].to_csv('train_data/lag7_{}_{}_{}_{}.csv'.format(dep_id, seg_id, day_of_week, season_tf), index = False)

            #                writer = pd.ExcelWriter('train_data/case_{}_{}_{}_{}.xlsx'.format(dep_id, seg_id, day_of_week, season_tf))
            #                kq.to_excel(writer,'detail', index = False)
            #                dict_df['0_full'].to_excel(writer,'lag_0', index = False)
            #                dict_df['1_full'].to_excel(writer,'lag_1', index = False)
            #                dict_df['7_full'].to_excel(writer,'lag_7', index = False)
            #                writer.save()
            
            # if y_col == "69_107_rv" and day_of_week == 6 and season_tf == [('01-01', '02-17'),
            #                                                                ('12-23', '12-31')] and lag == 0:
            #     print('test', col)
            #     with open('test_caset_69_107_{0}_{1}.txt'.format(season_tf, col), 'w') as f:
            #         f.write("\ny_series: {0}\ncol_series:{1}\ny_rm:{2}\ncol_rm: {3}\ncap_arr: {4}\cov_val:{5}".format( \
            #             ", ".join(map(str, y_series_r)), ", ".join(map(str, col_series)),
            #             ", ".join(map(str, y_series_rm)), ", ".join(map(str, col_series_rm)), str(cap_arr), cov_val))
            # ===========================
            if lag == 365:
                y_365_lag = df2[col]
                date_y_365_lag_not_null = df2[~y_365_lag.isnull()]['date']
                if len(date_y_365_lag_not_null.dt.year.unique()) < 2:
                    cov_val = -99  # explaination is big
            # ===========================
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


def v2_compute_corr_cruise(client, hotel_id, dep_id, seg_id, season_tf, num_day, day_of_week, significance_level, df,
                           date_ss, day_arr, day_1yrs_back, k_modifier, cruise):
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
    result_df = pd.DataFrame(
        columns=['col', "cov_col", 'lag', 'cov_value', "conf_interval_0", "conf_interval_1", "adj_corr_col"])
    index = 0

    median_lag7_full = np.median(df1_full[df1_full['date'] >= day_1yrs_back][y_col].dropna())
    median_lag7 = np.median(df1[df1['date'] >= day_1yrs_back][y_col].dropna())

    if np.isnan(median_lag7_full):
        print(y_col)
        print('Season: {}, Day of week: {}'.format(season_tf, day_of_week))
        print('No cruise day in 450 days with selected season. Choosing 2 years data instead.')
        day_4yrs_back = day_1yrs_back - timedelta(days=280)
        median_lag7_full = np.median(df1_full[df1_full['date'] >= day_4yrs_back][y_col].dropna())
        median_lag7 = np.median(df1[df1['date'] >= day_4yrs_back][y_col].dropna())
        print('median_lag7_full ', median_lag7_full)

    # mm
    if np.isnan(median_lag7_full):
        log.info("DATABASE %s __ HOTEL %s __ DEP %s __ No data for period %s __ weekday %s \n\n" %
                 (client, hotel_id, dep_id, season_tf, day_of_week))
        #        log_file_client = {'Date': datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d'),
        #                           'H_Id': dep_id, 'Message': 'No Season Data', 'Status': 0}
        #        log_json_upload(client, log_file_client)
        # mm
        result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', \
                                          "conf_interval_0", "conf_interval_1", "adj_corr_col"])
        len_columns = len(df.columns) - 2
        result_df['col'] = list([str(dep_id) + '_0_rv'] * len_columns)
        result_df['cov_col'] = list(set(df.columns) - {'date', 'day_of_week'})
        result_df['lag'] = np.zeros(len_columns) + 1
        result_df['cov_value'] = np.zeros(len_columns)
        result_df['conf_interval_0'] = np.zeros(len_columns)
        result_df['conf_interval_1'] = np.zeros(len_columns)
        result_df['adj_corr_col'] = [x + '_1' for x in result_df['cov_col']]
        return result_df

    dict_df = df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, median_lag7_full, median_lag7)

    for i, col in enumerate(cols):
        if lib.check_total(y_col, col):
            continue
        for lag in [0, 1]:  # remove lag 7, 365
            if (col.split('_')[0] != str(dep_id)) and (lag not in [0, 1]):
                continue
            if (col.split('_')[0] == str(dep_id)) and (lag in [0, 1]):
                continue
            col_series = dict_df[str(lag) + "_full"][col]
            y_series_r = y_series[-len(col_series):]
            # mm
            if (y_series_r.isnull().values.all()) or (col_series.isnull().values.all()) \
                    or (len(y_series_r) == 0) or (len(col_series) == 0):
                result_df = pd.DataFrame(columns=['col', "cov_col", 'lag', 'cov_value', \
                                                  "conf_interval_0", "conf_interval_1", "adj_corr_col"])
                len_columns = len(df.columns) - 2
                result_df['col'] = list([str(dep_id) + '_0_rv'] * len_columns)
                result_df['cov_col'] = list(set(df.columns) - {'date', 'day_of_week'})
                result_df['lag'] = np.zeros(len_columns) + 1
                result_df['cov_value'] = np.zeros(len_columns)
                result_df['conf_interval_0'] = np.zeros(len_columns)
                result_df['conf_interval_1'] = np.zeros(len_columns)
                result_df['adj_corr_col'] = [x + '_1' for x in result_df['cov_col']]
                return result_df

            df2 = dict_df[lag]
            y_series_rm, col_series_rm = v2_remove_outlier(y_series_r, col_series)
            cap_arr = v2_compute_capture_arr(day_arr, y_col, col, 1, 1, df1, df2, cols)
            y_forecast, forecast_error, modifier, y_min, y_max = v2_compute_error_single_corr(y_series_rm,
                                                                                              col_series_rm,
                                                                                              cap_arr['capture'],
                                                                                              cap_arr['ARR'],
                                                                                              k_modifier)
            cov_val = 1 - forecast_error
            if y_col == "69_107_rv" and day_of_week == 6 and season_tf == [('01-01', '02-17'),
                                                                           ('12-23', '12-31')] and lag == 0:
                print('test', col)
                with open('test_caset_69_107_{0}_{1}.txt'.format(season_tf, col), 'w') as f:
                    f.write("\ny_series: {0}\ncol_series:{1}\ny_rm:{2}\ncol_rm: {3}\ncap_arr: {4}\cov_val:{5}".format( \
                        ", ".join(map(str, y_series_r)), ", ".join(map(str, col_series)),
                        ", ".join(map(str, y_series_rm)), ", ".join(map(str, col_series_rm)), str(cap_arr), cov_val))
            # ===========================
            if lag == 365:
                y_365_lag = df2[col]
                date_y_365_lag_not_null = df2[~y_365_lag.isnull()]['date']
                if len(date_y_365_lag_not_null.dt.year.unique()) < 2:
                    cov_val = -99  # explaination is big
            # ===========================
            if "rn" in col:
                conf_interval_0, conf_interval_1 = cov_val, cov_val
            else:
                conf_interval_0, conf_interval_1 = lib.compute_interval(cov_val, significance_level)
            result_df.loc[index] = [y_col, col, lag, cov_val, conf_interval_0, conf_interval_1, col + "_" + str(lag)]
            index += 1
    result_df = result_df.fillna(-99)
    return result_df.sort_values('cov_value', ascending=False)


def v2_remove_outlier(series1, series2):
    """
    Remove outlier values on two series. If one elememt was removed on one series the corresponding element in another
     series would be removed too,. Method to detect outler is use 2 * standard deviation around mean
    :param series1:
    :param series2:
    :return:
    """
    if len(series1) != len(series2):
        #        print("::>>", len(series1), len(series2))
        raise "series1 and series2 must have a same length"

#    series1 = np.array(series1)
#    series2 = np.array(series2)

    # remove inf
    cond1 = ~np.isinf(series1)  # not nan
    cond2 = ~np.isinf(series1)  # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    # remove nan
    cond1 = ~np.isnan(series1)  # not nan
    cond2 = ~np.isnan(series2)  # not nan
    cond = cond1 & cond2
    series1 = series1[cond]
    series2 = series2[cond]

    series1_min = min(series1)
    series1_max = max(series1)
    series2_min = min(series2)
    series2_max = max(series2)

    series2 = series2[(series1 >= series1_min) & (series1 <= series1_max)]
    series1 = series1[(series1 >= series1_min) & (series1 <= series1_max)]

    series1 = series1[(series2 >= series2_min) & (series2 <= series2_max)]
    series2 = series2[(series2 >= series2_min) & (series2 <= series2_max)]

    s1 = pd.Series(series1)
    s2 = pd.Series(series2)
    
    return s1, s2


def v2_compute_capture_arr(day_arr, col1, col2, y_series, col_series, df1, df2, cols):
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
    # get data of 450 days before
    #    day_arr = np.datetime64(datetime.strptime('2017-01-30', '%Y-%m-%d') - timedelta(days=450))
    if not (df1.day_of_week.isnull().any() or df2.day_of_week.isnull().any()):
        df1 = df1.dropna()
        df2 = df2.dropna()
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    temp_df1 = df1
    temp_df2 = df2
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    df1 = df1[df1['date'] >= day_arr]
    df2 = df2[df2['date'] >= day_arr]
    if (len(df1) == 0) or (len(df2) == 0):
        #        print('v2_compute_capture_arr function: There is no data in 450 days before. Use all data.')
        df1 = temp_df1
        df2 = temp_df2
    if len(df1) > len(df2):
        df1 = df1.drop(df1.index[0])
    elif len(df1) < len(df2):
        df2 = df2.drop(df2.index[len(df2) - 1])

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
            if col1_rn in cols:  # exist unit ?
                capture = 1
                col1_rn_rm, col2_rn_rm = v2_remove_outlier(df1[col1_rn], df2[col2_rn])
                ARR = (col1_rn_rm / col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rn_rm / col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                capture = 1
                col1_rv_rm, col2_rm = v2_remove_outlier(df1[col1_rv], df2[col2])
                ARR = (col1_rv_rm / col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                capture_unit = 1
                ARR_unit = (col1_rv_rm / col2_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
        else:
            if col1_rn in cols:  # exist unit ?
                if type_col2 == 'rv':
                    capture = 1
                    col1_rv_rm, col2_rv_rm = v2_remove_outlier(df1[col1_rv], df2[col2_rv])
                    ARR = (col1_rv_rm / col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rv_rm = v2_remove_outlier(df1[col1_rn], df2[col2_rv])
                    ARR_unit = (col1_rn_rm / col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    col1_rn_rm, col2_rn_rm = v2_remove_outlier(df1[col1_rn], df2[col2_rn])
                    capture = (col1_rn_rm / col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = v2_remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm / col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_rn_rm = v2_remove_outlier(df1[col1_rn], df2[col2_rn])
                    ARR_unit = (col1_rn_rm / col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    col1_rn_rm, col2_gn_rm = v2_remove_outlier(df1[col1_rn], df2[col2_gn])
                    capture = (col1_rn_rm / col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    col1_rv_rm, col1_rn_rm = v2_remove_outlier(df1[col1_rv], df1[col1_rn])
                    ARR = (col1_rv_rm / col1_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                    capture_unit = 1
                    col1_rn_rm, col2_gn_rm = v2_remove_outlier(df1[col1_rn], df2[col2_gn])
                    ARR_unit = (col1_rn_rm / col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
            else:
                if type_col2 == 'rv':
                    capture = 1
                    col1_rm, col2_rv_rm = v2_remove_outlier(df1[col1], df2[col2_rv])
                    ARR = (col1_rm / col2_rv_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'rn':
                    capture = 1
                    col1_rm, col2_rn_rm = v2_remove_outlier(df1[col1], df2[col2_rn])
                    ARR = (col1_rm / col2_rn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
                elif type_col2 == 'gn':
                    capture = 1
                    col1_rm, col2_gn_rm = v2_remove_outlier(df1[col1], df2[col2_gn])
                    ARR = (col1_rm / col2_gn_rm).replace([np.inf, -np.inf], np.nan).dropna().median()
    except:
        capture = 1
        ARR = 0

    capture = 0 if pd.isnull(capture) else capture
    ARR = 0 if pd.isnull(ARR) else ARR
    ARR_unit = 0 if pd.isnull(ARR_unit) else ARR_unit
    result = {'capture': capture, 'ARR': ARR, 'capture_unit': capture_unit, 'ARR_unit': ARR_unit}
    return result


def v2_intersect_join(temp_df1, temp_df2):
    y1 = temp_df1.ix[:, 0]
    x1 = temp_df1.ix[:, 1]
    y2 = temp_df2.ix[:, 0]
    x2 = temp_df2.ix[:, 1]
    kq_y = []
    kq_x1 = []
    kq_x2 = []
    dates = temp_df2['date']
    new_dates = []
    flag = y2.astype(int)
    flag[:] = 1
    for i1, val1 in enumerate(y1):
        for i2, val2 in enumerate(y2):
            if (val1 == val2) and (flag[i2] == 1):
                kq_y.append(val2)
                kq_x1.append(x1[i1])
                kq_x2.append(x2[i2])
                flag[i2] = 0
                new_dates.append(dates[i2])
                break
    temp_df = pd.DataFrame()
    temp_df['y_both'] = kq_y
    temp_df['col1_series_rm'] = kq_x1
    temp_df['col2_series_rm'] = kq_x2
    temp_df['date'] = new_dates

    # remove outliers if never do this before, check in v2_remove_outlier function.
    #    y_min = float(np.percentile(kq_y, 5))
    #    y_max = float(np.percentile(kq_y, 95))
    #    temp_df = temp_df[(temp_df['y_both']>=y_min) & (temp_df['y_both']<=y_max)]
    return temp_df


def v2_intersect_join_rn(y_series_rn, y_both):
    y = y_series_rn.ix[:, 0]
    x = y_series_rn.ix[:, 1]

    kq_y = []
    kq_x = []

    flag = y_both.astype(int)
    flag[:] = 1
    for i1, val1 in enumerate(y):
        for i2, val2 in enumerate(y_both):
            if (val1 == val2) and (flag[i2] == 1):
                kq_y.append(val2)
                kq_x.append(x[i1])
                flag[i2] = 0
                break
    temp_df = pd.DataFrame()
    temp_df['y_both'] = kq_y
    temp_df['y_unit'] = kq_x
    return temp_df


def v2_revenue_detection(current_time, day_1yrs_back, hotel_id, dep_id, num_day, significance_level, year_input, \
                         year_input_tree, df, top3_df_v2, season_df_str, day_arr, k_modifier):
    #    #top3_df_v2 is used for model ver 2.
    top3_df_v2 = top3_df_v2[top3_df_v2['Day'] > 0]  # Check error from Day's dictionary.
    top3_df_v2['Type_str'] = top3_df_v2['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
    top3_df_v2['Driver_name'] = top3_df_v2['Source_H_Id'].astype(str) + '_' + top3_df_v2['Source_Segment_Id'].astype(
        str) + '_' + top3_df_v2['Type_str'].astype(str)

    seg_id_list = top3_df_v2['Destination_Segment_Id'].drop_duplicates()
    day_of_week_list = top3_df_v2['Day'].drop_duplicates().sort_values() - 1
    season_df_list = season_df_str[str(dep_id)]

    # Test for an example: one_seg_id, one day_of_week, one season
    #    seg_id_list = seg_id_list[seg_id_list==0]
    #    day_of_week_list = day_of_week_list[day_of_week_list==6]
    #    season_df_list = season_df_list[season_df_list=="[('01-25', '02-14'), ('12-09', '12-29')]"]

    # write to database
    acc_summary = pd.DataFrame(columns=['Driver', 'Acc_max', 'Acc_tree', 'Acc_linear', 'Acc_max_train'])
    id_acc = 0

    top_df_db_int_v2 = []
    db_lag7 = {}
    # column_otb = [str(i) * 3 + '_' + str(i) * 3 + '_rv' for i in range(5,31)]
    column_otb = [str(i) * 3 for i in range(5, 31)]
    for seg_id in seg_id_list:
        acc_summary.loc[id_acc, 'Driver'] = '{}_{}_{}'.format(hotel_id, dep_id, seg_id)
        acc_summary.loc[id_acc, 'Acc_max'] = 0.0
        acc_summary.loc[id_acc, 'Acc_tree'] = 0.0
        acc_summary.loc[id_acc, 'Acc_linear'] = 0.0
        acc_summary.loc[id_acc, 'Acc_max_train'] = 0.0
        df_lag7 = pd.DataFrame()
        has_rn_flag = 0
        for day_of_week in day_of_week_list:
            # tree model
            numdays = (current_time - day_1yrs_back).days + 1
            date_ss_tree = [current_time - timedelta(days=x) for x in range(0, numdays)]
            date_ss_tree.sort()

            y_dates_full_tree = [i for i in date_ss_tree if i.weekday() == day_of_week]
            y_dates_tree = y_dates_full_tree[-(min(num_day, len(y_dates_full_tree))):]
            y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
            y_col_rn = '_'.join([str(dep_id), str(seg_id)]) + '_rn'  # Get roomnight data name
            if y_col_rn in df.columns:
                has_rn_flag = 1
            df1_full_tree = lib.get_df_by_dates(y_dates_full_tree, df)
            y_series_tree = df1_full_tree[y_col]
            df1_tree = lib.get_df_by_dates(y_dates_tree, df1_full_tree)
            cols = set(df.columns) - {'date', 'day_of_week'}

            y_series_lag7 = pd.DataFrame()
            y_series_lag7['date'] = df1_full_tree['date']
            y_series_lag7[y_col] = df1_full_tree[y_col]

            # For upload median revenue, median unit
            df_ss_lag7 = pd.DataFrame(columns=['season', 'median_revenue', 'median_unit'])
            id_ss_lag7 = 0

            for season_tf_str in season_df_list:
                # calculating high low threshold of tree model
                #                day_1yrs_back_lag7 = day_1yrs_back - timedelta(days=7)

                season_tf = list(ast.literal_eval(season_tf_str))
                date_ss_hl = lib.generate_season_date(season_tf, year_input_tree)
                date_ss_hl.sort()
                date_ss_hl = [i for i in date_ss_hl if i >= day_1yrs_back]
                y_dates_hl = [i for i in date_ss_hl if i.weekday() == day_of_week]
                # y_dates_hl_lag7 = lib.get_dates_lag(y_dates_hl, 7)
                df_hl = lib.get_df_by_dates(y_dates_hl, df)[y_col]
                meadian_hl = np.median(df_hl.dropna())
                # Work with median revenue
                df_ss_lag7.loc[id_ss_lag7, 'season'] = season_tf_str
                df_ss_lag7.loc[id_ss_lag7, 'median_revenue'] = meadian_hl
                # Work with median unit
                if has_rn_flag == 1:
                    df_unit = lib.get_df_by_dates(y_dates_hl, df)[y_col_rn]
                    meadian_unit = np.median(df_unit.dropna())
                    df_ss_lag7.loc[id_ss_lag7, 'median_unit'] = meadian_unit
                id_ss_lag7 += 1

                for i in y_dates_hl:
                    y_series_lag7.loc[y_series_lag7['date'] == i, y_col] = meadian_hl

            df_lag7 = pd.concat([df_lag7, y_series_lag7], ignore_index=True)

            dict_df_tree = {}
            dict_df_tree[0] = df1_tree[df1_tree.columns]
            dict_df_tree["0_full"] = df1_full_tree[df1_full_tree.columns]
            for lag_temp in [1]:  # remove lag 7, 365
                dates_tree = lib.get_dates_lag(y_dates_tree, lag_temp)
                dates_full_tree = lib.get_dates_lag(y_dates_full_tree, lag_temp)
                df2_full_tree = lib.get_df_by_dates(dates_full_tree, df)
                df2_tree = lib.get_df_by_dates(dates_tree, df2_full_tree)
                if (lag_temp == 7):
                    df2_full_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]
                    df2_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]

                dict_df_tree[lag_temp] = df2_tree
                dict_df_tree[str(lag_temp) + "_full"] = df2_full_tree

            for season_tf_str in season_df_list:
                # ================================================== WORK WITH INPUTS =========================================================

                # max model
                season_tf = list(ast.literal_eval(season_tf_str))
                date_ss = lib.generate_season_date(season_tf, year_input)
                date_ss.sort()

                y_dates_full = [i for i in date_ss if i.weekday() == day_of_week]
                y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
                #                y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
                df1_full = lib.get_df_by_dates(y_dates_full, df)
                y_series = df1_full[y_col]
                df1 = lib.get_df_by_dates(y_dates, df1_full)
                #                cols = set(df.columns) - {'date', 'day_of_week'}

                # get period_id from season_tf
                # quoclht
                season_period_id = season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id']
                if len(season_period_id) == 1:
                    period_id = int(season_period_id)
                else:
                    period_id = int(season_period_id.iloc[0])

                #                period_id = int(season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id'])
                Day = np.round((day_of_week + 8) % 7 + 1).astype(int)

                # =========================================== COMPUTING ACCURACY OF MODEL VER 2 ===============================================

                toptemp_df = top3_df_v2[
                    (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                    (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]
                top1_name = str(toptemp_df[toptemp_df['Priority'] == 1]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 1]['SourceOffset'].values[0])
                top2_name = str(toptemp_df[toptemp_df['Priority'] == 2]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 2]['SourceOffset'].values[0])
                top3_name = str(toptemp_df[toptemp_df['Priority'] == 3]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 3]['SourceOffset'].values[0])
                # top 3 drivers must be different dept, seg or lag
                # dept is higher priority than different lag
                if (top1_name == top2_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                elif (top1_name == top3_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                else:
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                if ((top1_name == top2_name) and (
                        top1_name == top3_name)):  # if all dept is the same, lag must be different
                    #                    print(toptemp_df)
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    if top2_df.loc[top2_df.index[1], 'SourceOffset'] == 0:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(1)
                    else:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(0)
                    m_top2_df = top2_df

                # max model
                median_lag7_full = np.median(df1_full[df1_full['date'] >= day_1yrs_back][y_col].dropna())
                median_lag7 = np.median(df1[df1['date'] >= day_1yrs_back][y_col].dropna())
                dict_df = df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, \
                                       median_lag7_full, median_lag7)

                # ================================================= Work with top2_df ========================================================

                col1 = top2_df[top2_df['Priority'] == 1]['Driver_name'].values[0]
                col2 = top2_df[top2_df['Priority'] != 1]['Driver_name'].values[0]

                lag1 = top2_df[top2_df['Priority'] == 1]['SourceOffset'].values[0]
                lag2 = top2_df[top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, col1, lag1, col2, lag2)

                col1_series = dict_df[str(lag1) + "_full"][col1]
                y1_series_r = y_series[-len(col1_series):]
                df2 = dict_df[lag1]
                col2_series = dict_df[str(lag2) + "_full"][col2]
                y2_series_r = y_series[-len(col2_series):]
                df2_2 = dict_df[lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(y1_series_r) != len(col1_series):
                    col1_series = col1_series[:-(len(col1_series) - len(y1_series_r))]
                if len(y2_series_r) != len(col2_series):
                    col2_series = col2_series[:-(len(col2_series) - len(y2_series_r))]

                y1_series_rm, col1_series_rm = v2_remove_outlier(y1_series_r, col1_series)
                y2_series_rm, col2_series_rm = v2_remove_outlier(y2_series_r, col2_series)

                # check 2
                temp_df1 = pd.DataFrame(list(zip(y1_series_rm, col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(y2_series_rm, col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']

                temp_df = v2_intersect_join(temp_df1, temp_df2)
                y_both = temp_df['y_both']
                col1_series_rm_both = temp_df['col1_series_rm']
                col2_series_rm_both = temp_df['col2_series_rm']

                cap_arr = v2_compute_capture_arr(day_arr, y_col, col1, 1, 1, df1, df2, cols)
                cap_arr2 = v2_compute_capture_arr(day_arr, y_col, col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                y_forecast_v2, forecast_error_v2, modifier1, modifier2, y_min, y_max = v2_compute_error(y_both, \
                                                                                                        col1_series_rm_both,
                                                                                                        cap_arr[
                                                                                                            'capture'],
                                                                                                        cap_arr['ARR'], \
                                                                                                        col2_series_rm_both,
                                                                                                        cap_arr2[
                                                                                                            'capture'],
                                                                                                        cap_arr2['ARR'],
                                                                                                        k_modifier)

                y_forecast_v2_top1, forecast_error_v2_top1, modifier_top1, y_min_top1, y_max_top1, a_top1 = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier)

                y_forecast_v2_top2, forecast_error_v2_top2, modifier_top2, y_min_top2, y_max_top2, a_top2 = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier)

                cov_val_v2 = 1 - forecast_error_v2
                cov_val_v2_top1 = 1 - forecast_error_v2_top1
                cov_val_v2_top2 = 1 - forecast_error_v2_top2

                if cov_val_v2 < cov_val_v2_top1:
                    cov_val_v2 = cov_val_v2_top1
                    modifier1 = modifier_top1
                    modifier2 = 0.0
                    y_min = y_min_top1
                    y_max = y_max_top1
                #                    y_forecast_v2 = y_forecast_v2_top1
                if cov_val_v2 < cov_val_v2_top2:
                    cov_val_v2 = cov_val_v2_top2
                    modifier1 = 0.0
                    modifier2 = modifier_top2
                    y_min = y_min_top2
                    y_max = y_max_top2
                #                    y_forecast_v2 = y_forecast_v2_top2

                if cov_val_v2_top1 >= cov_val_v2_top2:
                    cov_val_linear = cov_val_v2_top1
                    linear_rate1 = a_top1
                    linear_rate2 = a_top2
                    linear_flag = 1
                else:
                    cov_val_linear = cov_val_v2_top2
                    linear_rate1 = a_top2
                    linear_rate2 = a_top1
                    linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                # ================================================= Work with m_top2_df ========================================================

                m_top2_df['Priority'].values[0] = 1
                m_top2_df['Priority'].values[1] = 2
                m_col1 = m_top2_df[m_top2_df['Priority'] == 1]['Driver_name'].values[0]
                m_col2 = m_top2_df[m_top2_df['Priority'] != 1]['Driver_name'].values[0]

                m_lag1 = m_top2_df[m_top2_df['Priority'] == 1]['SourceOffset'].values[0]
                m_lag2 = m_top2_df[m_top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, m_col1, m_lag1, m_col2, m_lag2)

                m_col1_series = dict_df[str(m_lag1) + "_full"][m_col1]
                m_y1_series_r = y_series[-len(m_col1_series):]
                df2 = dict_df[m_lag1]
                m_col2_series = dict_df[str(m_lag2) + "_full"][m_col2]
                m_y2_series_r = y_series[-len(m_col2_series):]
                df2_2 = dict_df[m_lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(m_y1_series_r) != len(m_col1_series):
                    m_col1_series = m_col1_series[:-(len(m_col1_series) - len(m_y1_series_r))]
                if len(m_y2_series_r) != len(m_col2_series):
                    m_col2_series = m_col2_series[:-(len(m_col2_series) - len(m_y2_series_r))]

                m_y1_series_rm, m_col1_series_rm = v2_remove_outlier(m_y1_series_r, m_col1_series)
                m_y2_series_rm, m_col2_series_rm = v2_remove_outlier(m_y2_series_r, m_col2_series)

                # check 2
                temp_df1 = pd.DataFrame(list(zip(m_y1_series_rm, m_col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(m_y2_series_rm, m_col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']

                temp_df = v2_intersect_join(temp_df1, temp_df2)
                m_y_both = temp_df['y_both']
                m_col1_series_rm_both = temp_df['col1_series_rm']
                m_col2_series_rm_both = temp_df['col2_series_rm']

                m_cap_arr = v2_compute_capture_arr(day_arr, y_col, m_col1, 1, 1, df1, df2, cols)
                m_cap_arr2 = v2_compute_capture_arr(day_arr, y_col, m_col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                m_y_forecast_v2, m_forecast_error_v2, m_modifier1, m_modifier2, m_y_min, m_y_max = v2_compute_error(
                    m_y_both, \
                    m_col1_series_rm_both, m_cap_arr['capture'], m_cap_arr['ARR'], \
                    m_col2_series_rm_both, m_cap_arr2['capture'], m_cap_arr2['ARR'], k_modifier)

                m_y_forecast_v2_top1, m_forecast_error_v2_top1, m_modifier_top1, m_y_min_top1, m_y_max_top1, m_a_top1 = \
                    v2_compute_error_single(m_y1_series_rm.dropna(), m_col1_series_rm.dropna(), m_cap_arr['capture'],
                                            m_cap_arr['ARR'], k_modifier)

                m_y_forecast_v2_top2, m_forecast_error_v2_top2, m_modifier_top2, m_y_min_top2, m_y_max_top2, m_a_top2 = \
                    v2_compute_error_single(m_y2_series_rm.dropna(), m_col2_series_rm.dropna(), m_cap_arr2['capture'],
                                            m_cap_arr2['ARR'], k_modifier)

                m_cov_val_v2 = 1 - m_forecast_error_v2
                m_cov_val_v2_top1 = 1 - m_forecast_error_v2_top1
                m_cov_val_v2_top2 = 1 - m_forecast_error_v2_top2

                if m_cov_val_v2 < m_cov_val_v2_top1:
                    m_cov_val_v2 = m_cov_val_v2_top1
                    m_modifier1 = m_modifier_top1
                    m_modifier2 = 0.0
                    m_y_min = m_y_min_top1
                    m_y_max = m_y_max_top1
                #                    m_y_forecast_v2 = m_y_forecast_v2_top1
                if m_cov_val_v2 < m_cov_val_v2_top2:
                    m_cov_val_v2 = m_cov_val_v2_top2
                    m_modifier1 = 0.0
                    m_modifier2 = m_modifier_top2
                    m_y_min = m_y_min_top2
                    m_y_max = m_y_max_top2
                #                    m_y_forecast_v2 = m_y_forecast_v2_top2

                if m_cov_val_v2_top1 >= m_cov_val_v2_top2:
                    m_cov_val_linear = m_cov_val_v2_top1
                    m_linear_rate1 = m_a_top1
                    m_linear_rate2 = m_a_top2
                    m_linear_flag = 1
                else:
                    m_cov_val_linear = m_cov_val_v2_top2
                    m_linear_rate1 = m_a_top2
                    m_linear_rate2 = m_a_top1
                    m_linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                if m_cov_val_v2 > cov_val_v2:
                    top2_df = m_top2_df
                    y_both = m_y_both
                    col1_series_rm_both = m_col1_series_rm_both
                    col2_series_rm_both = m_col2_series_rm_both
                    y_min = m_y_min
                    y_max = m_y_max
                    cov_val_v2 = m_cov_val_v2
                    cap_arr = m_cap_arr
                    cap_arr2 = m_cap_arr2
                    modifier1 = m_modifier1
                    modifier2 = m_modifier2
                    linear_rate1 = m_linear_rate1
                    linear_rate2 = m_linear_rate2
                    linear_flag = m_linear_flag
                    cov_val_linear = m_cov_val_linear

                    col1_series = m_col1_series
                    y1_series_r = m_y1_series_r
                    col2_series = m_col2_series
                    y2_series_r = m_y2_series_r
                    col1_series_rm = m_col1_series_rm
                    y1_series_rm = m_y1_series_rm
                    col2_series_rm = m_col2_series_rm
                    y2_series_rm = m_y2_series_rm
                    #                    y_forecast_v2 = m_y_forecast_v2
                    lag1 = m_lag1
                    lag2 = m_lag2
                    col1 = m_col1
                    col2 = m_col2

                    # tree model
                rate1 = cap_arr['capture'] * cap_arr['ARR']
                rate2 = cap_arr2['capture'] * cap_arr2['ARR']
                tree_db, highlow_threshold = v2_revenue_detection_tree(dict_df_tree, top2_df, y_series_tree, y_min,
                                                                       y_max, rate1, rate2)
                # new modifiers are re-computed from full data with season and day of week
                _, _, modifier_tree1, _, _, _ = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier)
                _, _, modifier_tree2, _, _, _ = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier)
                for i, v in enumerate(tree_db):
                    if v['Priority'] == 1:
                        v['Modifier'] = modifier_tree1
                    if v['Priority'] == 2:
                        v['Modifier'] = modifier_tree2
                # ================================================= Validation tree model =====================================================
                y = y_both[-4:].reset_index(drop=True)
                x1 = col1_series_rm_both[-4:].reset_index(drop=True)
                x2 = col2_series_rm_both[-4:].reset_index(drop=True)
                case_df = pd.DataFrame(tree_db)

                cov_val_tree_temp = 0.0
                for i, v in enumerate(y):
                    _, forecast_error_tree = tree_validation_error_json(y[i], y_min, y_max, x1[i], x2[i], case_df,
                                                                        highlow_threshold, rate1, rate2)
                    cov_val_tree_temp += 1 - forecast_error_tree
                cov_val_tree = cov_val_tree_temp / len(y)

                _, forecast_error_max = max_validate_error_json(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2,
                                                                rate2)
                cov_val_max = 1 - forecast_error_max

                if linear_flag == 1:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x1, linear_rate1)
                else:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x2, linear_rate2)
                cov_val_linear = 1 - forecast_error_linear
                # ======================================================= WORK WITH RESULTS ===================================================

                y_median_revenue = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_revenue'].values[0]
                # Work with unit data
                if has_rn_flag == 1:
                    y_median_unit = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_unit'].values[0]
                    y_median_unit_rate = y_median_revenue / y_median_unit
                    if np.isnan(y_median_unit_rate) or np.isinf(y_median_unit_rate):
                        y_median_unit_rate = 0.0
                else:
                    y_median_unit = 0.0
                    y_median_unit_rate = 0.0
                #                if  day_of_week == 0:
                #                    a = pd.DataFrame()
                #                    a['x1'] = x1
                #                    a['x2'] = x2
                #                    a['y'] = y
                #                    a['x1_both'] = col1_series_rm_both
                #                    a['x2_both'] = col2_series_rm_both
                #                    a['y_both'] = y_both
                #                    a['rate'] = pd.DataFrame([rate1, rate2])
                #                    a['modifier'] =  pd.DataFrame([modifier1, modifier2])
                #                    a['min_max'] = pd.DataFrame([y_min, y_max])
                #                    a['acc'] = pd.DataFrame([cov_val_linear, cov_val_max, cov_val_tree])
                #                    a.to_csv('{}.csv'.format(season_tf_str), index = False)
                acc_summary.loc[id_acc, 'Acc_max'] += cov_val_max
                acc_summary.loc[id_acc, 'Acc_tree'] += cov_val_tree
                acc_summary.loc[id_acc, 'Acc_linear'] += cov_val_linear
                acc_summary.loc[id_acc, 'Acc_max_train'] += cov_val_v2
                #                rate1 = cap_arr['capture']*cap_arr['ARR']
                #                rate2 = cap_arr2['capture']*cap_arr2['ARR']
                #                print(cov_val_linear, cov_val_max, cov_val_tree)
                if max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_linear:
                    best_model = 0
                elif max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_max:
                    best_model = 1
                else:
                    best_model = 2
                # append results to dataframe
                top2_df = top2_df.reset_index(drop=True)
                
                #TN - replace otb
                for i in range(2):
                    replace_source = str(top2_df['Source_Segment_Id'].iloc[i])
                    if pd.Series(str(replace_source)).isin(column_otb).iloc[0]:
                        leadtime_otb = ''.join(sorted(set(replace_source), key=replace_source.index))
                        if (leadtime_otb == '1' ) or (leadtime_otb == '2'):
                            leadtime_otb = leadtime_otb*2
                        top2_df['SourceOffset'].iloc[i] = leadtime_otb
                        top2_df['Source_Segment_Id'].iloc[i] = 0
                        top2_df['Source_H_Id'].iloc[i] = dep_id
                        top2_df['Type'].iloc[i] = 6
                
                top2_df['Priority'] = [1, 2]
                temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                temp_db = temp_db.rename(
                    columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                temp_db = temp_db.to_dict(orient='records')
                temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                temp_db_float = pd.DataFrame()
                temp_db_float['Rate'] = [rate1, rate2]
                temp_db_float['Linear_Rate'] = [linear_rate1, linear_rate2]
                temp_db_float['Correlation'] = [cov_val_v2, cov_val_v2]
                temp_db_float['Modifier'] = [modifier1, modifier2]
                temp_db_float['HighThreshold'] = [highlow_threshold['top1_high'].values[0],
                                                  highlow_threshold['top2_high'].values[0]]
                temp_db_float['LowThreshold'] = [highlow_threshold['top1_low'].values[0],
                                                 highlow_threshold['top2_low'].values[0]]
                temp_db_float = temp_db_float.to_dict(orient='records')
                for i in range(len(temp_db)):
                    temp_db[i].update(temp_db_float[i])

                temp_db_up = {
                    #                        'TrendModifier': 1.2,
                    'MedianRevenue': y_median_revenue,
                    'MedianUnitRate': y_median_unit_rate,
                    'MedianUnits': y_median_unit,
                    'High': y_max,
                    'Low': y_min,
                    'Sources': temp_db,
                    'Tree': tree_db,
                    'Day': int(Day),
                    'Offset': int(0),
                    'Segment_Id': int(seg_id),
                    'H_Id': int(dep_id),
                    'Period': period_id,
                    'Model': best_model
                }
                top_df_db_int_v2.append(temp_db_up)

        df_lag7 = df_lag7.sort_values(['date'], ascending=True)
        db_lag7[seg_id] = df_lag7

        print('------------------ Dep_Seg {}_{} ------------------'.format(dep_id, seg_id))
        total_iteration = len(day_of_week_list) * len(season_df_list)
        acc_summary.loc[id_acc, 'Acc_max'] = acc_summary.loc[id_acc, 'Acc_max'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_tree'] = acc_summary.loc[id_acc, 'Acc_tree'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_linear'] = acc_summary.loc[id_acc, 'Acc_linear'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_max_train'] = acc_summary.loc[id_acc, 'Acc_max_train'] / total_iteration
        print('Acc_max: {} .'.format(acc_summary.loc[id_acc, 'Acc_max']))
        print('Acc_tree: {} .'.format(acc_summary.loc[id_acc, 'Acc_tree']))
        print('Acc_linear: {} .'.format(acc_summary.loc[id_acc, 'Acc_linear']))
        id_acc += 1
    return [top_df_db_int_v2, db_lag7, acc_summary]


def linear_validate_error_json(y, y_min, y_max, x, linear_rate, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 2
    :param y: real value of forecast driver
    :param x: value of other drivers
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    w=set_weight(y_dates)
    y_forecast = linear_rate * x
    #    y_forecast = pd.Series([max(min(y_max,temp),y_min) for temp in y_forecast])
#    s = sMAPE(y, y_forecast)
    s = sMAPE_weight(y, y_forecast, w/sum(w))
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


def max_validate_error_json(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2, rate2, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 2
    :param y: real value of forecast driver
    :param x: value of other drivers
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    # ======================================================================================
    # method 1
    y1_forecast = modifier1 * rate1 * x1
    y2_forecast = modifier2 * rate2 * x2
    y_forecast = pd.concat([y1_forecast, y2_forecast], axis=1).max(axis=1)
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    w=set_weight(y_dates)
    s = sMAPE_weight(y, y_forecast, w/sum(w))
#    s = sMAPE(y, y_forecast)
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


#def tree_validation_error_json(y, y_min, y_max, x1, x2, case_df, highlow_threshold, rate1, rate2):
#    y = pd.Series(y)
#    x1_temp = x1
#    x2_temp = x2
#    top1_low = highlow_threshold['top1_low'].values[0]
#    top2_low = highlow_threshold['top2_low'].values[0]
#    top1_high = highlow_threshold['top1_high'].values[0]
#    top2_high = highlow_threshold['top2_high'].values[0]
#
#    # get type_of_x
#    if (x1_temp < top1_low) & (x2_temp < top2_low):
#        type_of_x = 1  # low_low
#    elif (x1_temp < top1_low) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
#        type_of_x = 2  # low_med
#    elif (x1_temp < top1_low) & (x2_temp > top2_high):
#        type_of_x = 3  # low_hig
#    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp < top2_low):
#        type_of_x = 4  # med_low
#    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
#        type_of_x = 5  # med_med
#    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp > top2_high):
#        type_of_x = 6  # med_hig
#    elif (x1_temp > top1_high) & (x2_temp < top2_low):
#        type_of_x = 7  # hig_low
#    elif (x1_temp > top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
#        type_of_x = 8  # hig_med
#    elif (x1_temp > top1_high) & (x2_temp > top2_high):
#        type_of_x = 9  # hig_hig
#    # ======================================================================================
#    # get modifier and priority
#    modifier = case_df[case_df['Case'] == type_of_x]['Modifier'].values[0]
#    priority = case_df[case_df['Case'] == type_of_x]['Priority'].values[0]
#    if priority == 1:
#        rate = rate1
#    else:
#        rate = rate2
#
#    if priority == 1:
#        x = x1_temp
#    else:
#        x = x2_temp
#    y_forecast = [modifier * rate * x]
#    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
#    # ======================================================================================
#    s = sMAPE(y, y_forecast)
#    forecast_error = 100 if np.isinf(s) else s
#    return y_forecast, forecast_error


def tree_validation_error_json(y, y_min, y_max, x1, x2, case_df, highlow_threshold, rate1, rate2):
    y = pd.Series(y)
    x1_temp = x1
    x2_temp = x2
    top1_low = highlow_threshold['top1_low'].values[0]
    top2_low = highlow_threshold['top2_low'].values[0]
    top1_high = highlow_threshold['top1_high'].values[0]
    top2_high = highlow_threshold['top2_high'].values[0]

    # get type_of_x
    if (x1_temp < top1_low) & (x2_temp < top2_low):
        type_of_x = 1  # low_low
    elif (x1_temp < top1_low) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 2  # low_med
    elif (x1_temp < top1_low) & (x2_temp > top2_high):
        type_of_x = 3  # low_hig
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp < top2_low):
        type_of_x = 4  # med_low
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 5  # med_med
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp > top2_high):
        type_of_x = 6  # med_hig
    elif (x1_temp > top1_high) & (x2_temp < top2_low):
        type_of_x = 7  # hig_low
    elif (x1_temp > top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 8  # hig_med
    elif (x1_temp > top1_high) & (x2_temp > top2_high):
        type_of_x = 9  # hig_hig
    # ======================================================================================
    # get modifier and priority
    modifier = case_df[case_df['Case'] == type_of_x]['Modifier'].values[0]
    priority = case_df[case_df['Case'] == type_of_x]['Priority'].values[0]
    if priority == 1:
        rate = rate1
    else:
        rate = rate2

    if priority == 1:
        x = x1_temp
    else:
        x = x2_temp
    y_forecast = [modifier * rate * x]
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    # ======================================================================================
    s = sMAPE(y, y_forecast)
    forecast_error = 100 if np.isinf(s) else s
    return y_forecast, forecast_error


def v2_revenue_detection_both(current_time, day_1yrs_back, hotel_id, dep_id, num_day, significance_level, year_input, \
                              year_input_tree, df, top3_df, top3_df_v2, season_df_str, day_arr, k_modifier):
    top3_df = top3_df[top3_df['Day'] > 0]  # Check error from Day's dictionary.
    top3_df['Type_str'] = top3_df['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
    top3_df['Driver_name'] = top3_df['Source_H_Id'].astype(str) + '_' + top3_df['Source_Segment_Id'].astype(str) + '_' + \
                             top3_df['Type_str'].astype(str)

    #    #top3_df_v2 is used for model ver 2.
    top3_df_v2 = top3_df_v2[top3_df_v2['Day'] > 0]  # Check error from Day's dictionary.
    top3_df_v2['Type_str'] = top3_df_v2['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
    top3_df_v2['Driver_name'] = top3_df_v2['Source_H_Id'].astype(str) + '_' + top3_df_v2['Source_Segment_Id'].astype(
        str) + '_' + top3_df_v2['Type_str'].astype(str)

    seg_id_list = top3_df_v2[
        'Destination_Segment_Id'].drop_duplicates()  # top3_df or top3_df_v2 will give the same resutls.
    day_of_week_list = top3_df_v2[
                           'Day'].drop_duplicates().sort_values() - 1  # top3_df or top3_df_v2 will give the same resutls.
    season_df_list = season_df_str[str(dep_id)]

    # Test for an example: one_seg_id, one day_of_week, one season
    #    seg_id_list = seg_id_list[seg_id_list==0]
    #    day_of_week_list = day_of_week_list[day_of_week_list==6]
    #    season_df_list = season_df_list[season_df_list=="[('01-25', '02-14'), ('12-09', '12-29')]"]

    # write to database
    key_list = ['Source_H_Id', 'Day', 'DestinationOffset', 'Destination_Segment_Id', \
                'Type', 'H_Id', 'Period', 'Priority', 'Property', 'SourceOffset', 'Source_Segment_Id']
    top_db_up_v1 = pd.DataFrame()
    top_db_up_float_v1 = pd.DataFrame()

    acc_summary = pd.DataFrame(columns=['Driver', 'Acc_max', 'Acc_tree', 'Acc_linear', 'Acc_max_train', \
                                        'Acc_v1', 'Increase_acc'])
    id_acc = 0
    runtime_acc_v1 = 0.0

    top_df_db_int_v2 = []
    db_lag7 = {}

    for seg_id in seg_id_list:
        acc_summary.loc[id_acc, 'Driver'] = '{}_{}_{}'.format(hotel_id, dep_id, seg_id)
        acc_summary.loc[id_acc, 'Acc_max'] = 0.0
        acc_summary.loc[id_acc, 'Acc_tree'] = 0.0
        acc_summary.loc[id_acc, 'Acc_linear'] = 0.0
        acc_summary.loc[id_acc, 'Acc_max_train'] = 0.0
        acc_summary.loc[id_acc, 'Acc_v1'] = 0.0
        df_lag7 = pd.DataFrame()
        has_rn_flag = 0
        for day_of_week in day_of_week_list:
            # tree model
            numdays = (current_time - day_1yrs_back).days + 1
            date_ss_tree = [current_time - timedelta(days=x) for x in range(0, numdays)]
            date_ss_tree.sort()

            y_dates_full_tree = [i for i in date_ss_tree if i.weekday() == day_of_week]
            y_dates_tree = y_dates_full_tree[-(min(num_day, len(y_dates_full_tree))):]
            y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
            y_col_rn = '_'.join([str(dep_id), str(seg_id)]) + '_rn'  # Get roomnight data name
            if y_col_rn in df.columns:
                has_rn_flag = 1
            df1_full_tree = lib.get_df_by_dates(y_dates_full_tree, df)
            y_series_tree = df1_full_tree[y_col]
            df1_tree = lib.get_df_by_dates(y_dates_tree, df1_full_tree)
            cols = set(df.columns) - {'date', 'day_of_week'}

            y_series_lag7 = pd.DataFrame()
            y_series_lag7['date'] = df1_full_tree['date']
            y_series_lag7[y_col] = df1_full_tree[y_col]

            # For upload median revenue, median unit
            df_ss_lag7 = pd.DataFrame(columns=['season', 'median_revenue', 'median_unit'])
            id_ss_lag7 = 0

            for season_tf_str in season_df_list:
                # calculating high low threshold of tree model
                #                day_1yrs_back_lag7 = day_1yrs_back - timedelta(days=7)

                season_tf = list(ast.literal_eval(season_tf_str))
                date_ss_hl = lib.generate_season_date(season_tf, year_input_tree)
                date_ss_hl.sort()
                date_ss_hl = [i for i in date_ss_hl if i >= day_1yrs_back]
                y_dates_hl = [i for i in date_ss_hl if i.weekday() == day_of_week]
                #                y_dates_hl_lag7 = lib.get_dates_lag(y_dates_hl, 7)
                df_hl = lib.get_df_by_dates(y_dates_hl, df)[y_col]
                meadian_hl = np.median(df_hl.dropna())
                # Work with median revenue
                df_ss_lag7.loc[id_ss_lag7, 'season'] = season_tf_str
                df_ss_lag7.loc[id_ss_lag7, 'median_revenue'] = meadian_hl
                # Work with median unit
                if has_rn_flag == 1:
                    df_unit = lib.get_df_by_dates(y_dates_hl, df)[y_col_rn]
                    meadian_unit = np.median(df_unit.dropna())
                    df_ss_lag7.loc[id_ss_lag7, 'median_unit'] = meadian_unit
                id_ss_lag7 += 1

                for i in y_dates_hl:
                    y_series_lag7.loc[y_series_lag7['date'] == i, y_col] = meadian_hl

            df_lag7 = pd.concat([df_lag7, y_series_lag7], ignore_index=True)

            dict_df_tree = {}
            dict_df_tree[0] = df1_tree[df1_tree.columns]
            dict_df_tree["0_full"] = df1_full_tree[df1_full_tree.columns]
            for lag_temp in [1]:  # remove lag 7, 365
                dates_tree = lib.get_dates_lag(y_dates_tree, lag_temp)
                dates_full_tree = lib.get_dates_lag(y_dates_full_tree, lag_temp)
                df2_full_tree = lib.get_df_by_dates(dates_full_tree, df)
                df2_tree = lib.get_df_by_dates(dates_tree, df2_full_tree)
                if (lag_temp == 7):
                    df2_full_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]
                    df2_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]

                dict_df_tree[lag_temp] = df2_tree
                dict_df_tree[str(lag_temp) + "_full"] = df2_full_tree

            for season_tf_str in season_df_list:
                runtime_start_v1 = time.time()
                # ================================================== WORK WITH INPUTS =========================================================

                # max model
                season_tf = list(ast.literal_eval(season_tf_str))
                date_ss = lib.generate_season_date(season_tf, year_input)
                date_ss.sort()

                y_dates_full = [i for i in date_ss if i.weekday() == day_of_week]
                y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
                #                y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
                df1_full = lib.get_df_by_dates(y_dates_full, df)
                y_series = df1_full[y_col]
                df1 = lib.get_df_by_dates(y_dates, df1_full)
                #                cols = set(df.columns) - {'date', 'day_of_week'}

                # get period_id from season_tf
                period_id = int(season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id'])
                Day = np.round((day_of_week + 8) % 7 + 1).astype(int)

                # =========================================== COMPUTING ACCURACY OF MODEL VER 1 ===============================================

                top1_df = top3_df[
                    (top3_df['H_Id'] == int(dep_id)) & (top3_df['Destination_Segment_Id'] == int(seg_id)) & \
                    (top3_df['Day'] == Day) & (top3_df['Period'] == period_id) & (top3_df['Priority'] == 1)]
                col = top1_df[top1_df['Priority'] == 1]['Driver_name'].values[0]
                lag = top1_df[top1_df['Priority'] == 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, col, lag)
                dict_df_v1 = {}
                dict_df_v1[0] = df1[df1.columns]
                dict_df_v1["0_full"] = df1_full[df1_full.columns]
                for lag_temp in [1]:  # remove lag 7, 365
                    dates = lib.get_dates_lag(y_dates, lag_temp)
                    dates_full = lib.get_dates_lag(y_dates_full, lag_temp)
                    df2_full = lib.get_df_by_dates(dates_full, df)
                    df2 = lib.get_df_by_dates(dates, df2_full)
                    dict_df_v1[lag_temp] = df2
                    dict_df_v1[str(lag_temp) + "_full"] = df2_full

                col_series = dict_df_v1[str(lag) + "_full"][col]
                y_series_r = y_series[-len(col_series):]
                df2 = dict_df_v1[lag]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(y_series_r) != len(col_series):
                    col_series = col_series[:-(len(col_series) - len(y_series_r))]
                y_series_rm, col_series_rm = v2_remove_outlier(y_series_r, col_series)

                # check 2
                if len(y_series_rm) != len(col_series_rm):
                    col_series_rm = col_series_rm[:-(len(col_series_rm) - len(y_series_rm))]

                cap_arr = v2_compute_capture_arr(day_arr, y_col, col, 1, 1, df1, df2, cols)
                y_forecast_v1, forecast_error_v1 = v1_compute_error(y_series_rm, col_series_rm, cap_arr['capture'],
                                                                    cap_arr['ARR'])

                cov_val_v1 = 1 - forecast_error_v1
                conf0_v1, conf1_v1 = lib.compute_interval(cov_val_v1, significance_level)

                # append results to dataframe
                top1_df = top1_df.reset_index(drop=True)
                top1_df['AverageRevenue'] = cap_arr['ARR']
                top1_df['Percent'] = cap_arr['capture']
                top1_df['Property'] = cov_val_v1
                top_db_up_v1 = top_db_up_v1.append(top1_df[key_list], ignore_index=True)
                top_db_up_float_v1 = top_db_up_float_v1.append(top1_df[['Property', \
                                                                        'AverageRevenue', 'Percent']],
                                                               ignore_index=True)

                # combine running time each iteration
                runtime_acc_v1 += time.time() - runtime_start_v1

                # =========================================== COMPUTING ACCURACY OF MODEL VER 2 ===============================================

                toptemp_df = top3_df_v2[
                    (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                    (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]
                top1_name = str(toptemp_df[toptemp_df['Priority'] == 1]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 1]['SourceOffset'].values[0])
                top2_name = str(toptemp_df[toptemp_df['Priority'] == 2]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 2]['SourceOffset'].values[0])
                top3_name = str(toptemp_df[toptemp_df['Priority'] == 3]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 3]['SourceOffset'].values[0])
                # top 3 drivers must be different dept, seg or lag
                # dept is higher priority than different lag
                if (top1_name == top2_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                elif (top1_name == top3_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                else:
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                if ((top1_name == top2_name) and (
                        top1_name == top3_name)):  # if all dept is the same, lag must be different
                    #                    print(toptemp_df)
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    if top2_df.loc[top2_df.index[1], 'SourceOffset'] == 0:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(1)
                    else:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(0)
                    m_top2_df = top2_df

                # max model
                median_lag7_full = np.median(df1_full[df1_full['date'] >= day_1yrs_back][y_col].dropna())
                median_lag7 = np.median(df1[df1['date'] >= day_1yrs_back][y_col].dropna())
                dict_df = df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, \
                                       median_lag7_full, median_lag7)

                # ================================================= Work with top2_df ========================================================

                col1 = top2_df[top2_df['Priority'] == 1]['Driver_name'].values[0]
                col2 = top2_df[top2_df['Priority'] != 1]['Driver_name'].values[0]

                lag1 = top2_df[top2_df['Priority'] == 1]['SourceOffset'].values[0]
                lag2 = top2_df[top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, col1, lag1, col2, lag2)

                col1_series = dict_df[str(lag1) + "_full"][col1]
                y1_series_r = y_series[-len(col1_series):]
                df2 = dict_df[lag1]
                col2_series = dict_df[str(lag2) + "_full"][col2]
                y2_series_r = y_series[-len(col2_series):]
                df2_2 = dict_df[lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(y1_series_r) != len(col1_series):
                    col1_series = col1_series[:-(len(col1_series) - len(y1_series_r))]
                if len(y2_series_r) != len(col2_series):
                    col2_series = col2_series[:-(len(col2_series) - len(y2_series_r))]

                y1_series_rm, col1_series_rm = v2_remove_outlier(y1_series_r, col1_series)
                y2_series_rm, col2_series_rm = v2_remove_outlier(y2_series_r, col2_series)

                # check 2
                temp_df1 = pd.DataFrame(list(zip(y1_series_rm, col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(y2_series_rm, col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']

                temp_df = v2_intersect_join(temp_df1, temp_df2)
                y_both = temp_df['y_both']
                col1_series_rm_both = temp_df['col1_series_rm']
                col2_series_rm_both = temp_df['col2_series_rm']

                cap_arr = v2_compute_capture_arr(day_arr, y_col, col1, 1, 1, df1, df2, cols)
                cap_arr2 = v2_compute_capture_arr(day_arr, y_col, col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                y_forecast_v2, forecast_error_v2, modifier1, modifier2, y_min, y_max = v2_compute_error(y_both, \
                                                                                                        col1_series_rm_both,
                                                                                                        cap_arr[
                                                                                                            'capture'],
                                                                                                        cap_arr['ARR'], \
                                                                                                        col2_series_rm_both,
                                                                                                        cap_arr2[
                                                                                                            'capture'],
                                                                                                        cap_arr2['ARR'],
                                                                                                        k_modifier)

                y_forecast_v2_top1, forecast_error_v2_top1, modifier_top1, y_min_top1, y_max_top1, a_top1 = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier)

                y_forecast_v2_top2, forecast_error_v2_top2, modifier_top2, y_min_top2, y_max_top2, a_top2 = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier)

                cov_val_v2 = 1 - forecast_error_v2
                cov_val_v2_top1 = 1 - forecast_error_v2_top1
                cov_val_v2_top2 = 1 - forecast_error_v2_top2

                if cov_val_v2 < cov_val_v2_top1:
                    cov_val_v2 = cov_val_v2_top1
                    modifier1 = modifier_top1
                    modifier2 = 0.0
                    y_min = y_min_top1
                    y_max = y_max_top1
                #                    y_forecast_v2 = y_forecast_v2_top1
                if cov_val_v2 < cov_val_v2_top2:
                    cov_val_v2 = cov_val_v2_top2
                    modifier1 = 0.0
                    modifier2 = modifier_top2
                    y_min = y_min_top2
                    y_max = y_max_top2
                #                    y_forecast_v2 = y_forecast_v2_top2

                if cov_val_v2_top1 >= cov_val_v2_top2:
                    cov_val_linear = cov_val_v2_top1
                    linear_rate1 = a_top1
                    linear_rate2 = a_top2
                    linear_flag = 1
                else:
                    cov_val_linear = cov_val_v2_top2
                    linear_rate1 = a_top2
                    linear_rate2 = a_top1
                    linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                # ================================================= Work with m_top2_df ========================================================

                m_top2_df['Priority'].values[0] = 1
                m_top2_df['Priority'].values[1] = 2
                m_col1 = m_top2_df[m_top2_df['Priority'] == 1]['Driver_name'].values[0]
                m_col2 = m_top2_df[m_top2_df['Priority'] != 1]['Driver_name'].values[0]

                m_lag1 = m_top2_df[m_top2_df['Priority'] == 1]['SourceOffset'].values[0]
                m_lag2 = m_top2_df[m_top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, m_col1, m_lag1, m_col2, m_lag2)

                m_col1_series = dict_df[str(m_lag1) + "_full"][m_col1]
                m_y1_series_r = y_series[-len(m_col1_series):]
                df2 = dict_df[m_lag1]
                m_col2_series = dict_df[str(m_lag2) + "_full"][m_col2]
                m_y2_series_r = y_series[-len(m_col2_series):]
                df2_2 = dict_df[m_lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(m_y1_series_r) != len(m_col1_series):
                    m_col1_series = m_col1_series[:-(len(m_col1_series) - len(m_y1_series_r))]
                if len(m_y2_series_r) != len(m_col2_series):
                    m_col2_series = m_col2_series[:-(len(m_col2_series) - len(m_y2_series_r))]

                m_y1_series_rm, m_col1_series_rm = v2_remove_outlier(m_y1_series_r, m_col1_series)
                m_y2_series_rm, m_col2_series_rm = v2_remove_outlier(m_y2_series_r, m_col2_series)

                # check 2
                temp_df1 = pd.DataFrame(list(zip(m_y1_series_rm, m_col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(m_y2_series_rm, m_col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']

                temp_df = v2_intersect_join(temp_df1, temp_df2)
                m_y_both = temp_df['y_both']
                m_col1_series_rm_both = temp_df['col1_series_rm']
                m_col2_series_rm_both = temp_df['col2_series_rm']

                m_cap_arr = v2_compute_capture_arr(day_arr, y_col, m_col1, 1, 1, df1, df2, cols)
                m_cap_arr2 = v2_compute_capture_arr(day_arr, y_col, m_col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                m_y_forecast_v2, m_forecast_error_v2, m_modifier1, m_modifier2, m_y_min, m_y_max = v2_compute_error(
                    m_y_both, \
                    m_col1_series_rm_both, m_cap_arr['capture'], m_cap_arr['ARR'], \
                    m_col2_series_rm_both, m_cap_arr2['capture'], m_cap_arr2['ARR'], k_modifier)

                m_y_forecast_v2_top1, m_forecast_error_v2_top1, m_modifier_top1, m_y_min_top1, m_y_max_top1, m_a_top1 = \
                    v2_compute_error_single(m_y1_series_rm.dropna(), m_col1_series_rm.dropna(), m_cap_arr['capture'],
                                            m_cap_arr['ARR'], k_modifier)

                m_y_forecast_v2_top2, m_forecast_error_v2_top2, m_modifier_top2, m_y_min_top2, m_y_max_top2, m_a_top2 = \
                    v2_compute_error_single(m_y2_series_rm.dropna(), m_col2_series_rm.dropna(), m_cap_arr2['capture'],
                                            m_cap_arr2['ARR'], k_modifier)

                m_cov_val_v2 = 1 - m_forecast_error_v2
                m_cov_val_v2_top1 = 1 - m_forecast_error_v2_top1
                m_cov_val_v2_top2 = 1 - m_forecast_error_v2_top2

                if m_cov_val_v2 < m_cov_val_v2_top1:
                    m_cov_val_v2 = m_cov_val_v2_top1
                    m_modifier1 = m_modifier_top1
                    m_modifier2 = 0.0
                    m_y_min = m_y_min_top1
                    m_y_max = m_y_max_top1
                #                    m_y_forecast_v2 = m_y_forecast_v2_top1
                if m_cov_val_v2 < m_cov_val_v2_top2:
                    m_cov_val_v2 = m_cov_val_v2_top2
                    m_modifier1 = 0.0
                    m_modifier2 = m_modifier_top2
                    m_y_min = m_y_min_top2
                    m_y_max = m_y_max_top2
                #                    m_y_forecast_v2 = m_y_forecast_v2_top2

                if m_cov_val_v2_top1 >= m_cov_val_v2_top2:
                    m_cov_val_linear = m_cov_val_v2_top1
                    m_linear_rate1 = m_a_top1
                    m_linear_rate2 = m_a_top2
                    m_linear_flag = 1
                else:
                    m_cov_val_linear = m_cov_val_v2_top2
                    m_linear_rate1 = m_a_top2
                    m_linear_rate2 = m_a_top1
                    m_linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                if m_cov_val_v2 > cov_val_v2:
                    top2_df = m_top2_df
                    y_both = m_y_both
                    col1_series_rm_both = m_col1_series_rm_both
                    col2_series_rm_both = m_col2_series_rm_both
                    y_min = m_y_min
                    y_max = m_y_max
                    cov_val_v2 = m_cov_val_v2
                    cap_arr = m_cap_arr
                    cap_arr2 = m_cap_arr2
                    modifier1 = m_modifier1
                    modifier2 = m_modifier2
                    linear_rate1 = m_linear_rate1
                    linear_rate2 = m_linear_rate2
                    linear_flag = m_linear_flag
                    cov_val_linear = m_cov_val_linear

                    col1_series = m_col1_series
                    y1_series_r = m_y1_series_r
                    col2_series = m_col2_series
                    y2_series_r = m_y2_series_r
                    col1_series_rm = m_col1_series_rm
                    y1_series_rm = m_y1_series_rm
                    col2_series_rm = m_col2_series_rm
                    y2_series_rm = m_y2_series_rm
                    #                    y_forecast_v2 = m_y_forecast_v2
                    lag1 = m_lag1
                    lag2 = m_lag2
                    col1 = m_col1
                    col2 = m_col2

                    # tree model
                rate1 = cap_arr['capture'] * cap_arr['ARR']
                rate2 = cap_arr2['capture'] * cap_arr2['ARR']
                tree_db, highlow_threshold = v2_revenue_detection_tree(dict_df_tree, top2_df, y_series_tree, y_min,
                                                                       y_max, rate1, rate2)
                # new modifiers are re-computed from full data with season and day of week
                _, _, modifier_tree1, _, _, _ = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier)
                _, _, modifier_tree2, _, _, _ = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier)
                for i, v in enumerate(tree_db):
                    if v['Priority'] == 1:
                        v['Modifier'] = modifier_tree1
                    if v['Priority'] == 2:
                        v['Modifier'] = modifier_tree2
                # ================================================= Validation tree model =====================================================
                y = y_both[-4:].reset_index(drop=True)
                x1 = col1_series_rm_both[-4:].reset_index(drop=True)
                x2 = col2_series_rm_both[-4:].reset_index(drop=True)
                case_df = pd.DataFrame(tree_db)

                cov_val_tree_temp = 0.0
                for i, v in enumerate(y):
                    _, forecast_error_tree = tree_validation_error_json(y[i], y_min, y_max, x1[i], x2[i], case_df,
                                                                        highlow_threshold, rate1, rate2)
                    cov_val_tree_temp += 1 - forecast_error_tree
                cov_val_tree = cov_val_tree_temp / len(y)

                _, forecast_error_max = max_validate_error_json(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2,
                                                                rate2)
                cov_val_max = 1 - forecast_error_max

                if linear_flag == 1:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x1, linear_rate1)
                else:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x2, linear_rate2)
                cov_val_linear = 1 - forecast_error_linear
                # ======================================================= WORK WITH RESULTS ===================================================

                y_median_revenue = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_revenue'].values[0]
                # Work with unit data
                if has_rn_flag == 1:
                    y_median_unit = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_unit'].values[0]
                    y_median_unit_rate = y_median_revenue / y_median_unit
                    if np.isnan(y_median_unit_rate) or np.isinf(y_median_unit_rate):
                        y_median_unit_rate = 0.0
                else:
                    y_median_unit = 0.0
                    y_median_unit_rate = 0.0

                acc_summary.loc[id_acc, 'Acc_max'] += cov_val_max
                acc_summary.loc[id_acc, 'Acc_tree'] += cov_val_tree
                acc_summary.loc[id_acc, 'Acc_linear'] += cov_val_linear
                acc_summary.loc[id_acc, 'Acc_max_train'] += cov_val_v2
                acc_summary.loc[id_acc, 'Acc_v1'] += cov_val_v1
                #                rate1 = cap_arr['capture']*cap_arr['ARR']
                #                rate2 = cap_arr2['capture']*cap_arr2['ARR']
                if max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_linear:
                    best_model = 0
                elif max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_max:
                    best_model = 1
                else:
                    best_model = 2
                # append results to dataframe
                top2_df = top2_df.reset_index(drop=True)
                top2_df['Priority'] = [1, 2]
                temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                temp_db = temp_db.rename(
                    columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                temp_db = temp_db.to_dict(orient='records')
                temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                temp_db_float = pd.DataFrame()
                temp_db_float['Rate'] = [rate1, rate2]
                temp_db_float['Linear_Rate'] = [linear_rate1, linear_rate2]
                temp_db_float['Correlation'] = [cov_val_v2, cov_val_v2]
                temp_db_float['Modifier'] = [modifier1, modifier2]
                temp_db_float['HighThreshold'] = [highlow_threshold['top1_high'].values[0],
                                                  highlow_threshold['top2_high'].values[0]]
                temp_db_float['LowThreshold'] = [highlow_threshold['top1_low'].values[0],
                                                 highlow_threshold['top2_low'].values[0]]
                temp_db_float = temp_db_float.to_dict(orient='records')
                for i in range(len(temp_db)):
                    temp_db[i].update(temp_db_float[i])

                temp_db_up = {
                    #                        'TrendModifier': 1.2,
                    'MedianRevenue': y_median_revenue,
                    'MedianUnitRate': y_median_unit_rate,
                    'MedianUnits': y_median_unit,
                    'High': y_max,
                    'Low': y_min,
                    'Sources': temp_db,
                    'Tree': tree_db,
                    'Day': int(Day),
                    'Offset': int(0),
                    'Segment_Id': int(seg_id),
                    'H_Id': int(dep_id),
                    'Period': period_id,
                    'Model': best_model
                }
                top_df_db_int_v2.append(temp_db_up)

        df_lag7 = df_lag7.sort_values(['date'], ascending=True)
        db_lag7[seg_id] = df_lag7

        print('------------------ Dep_Seg {}_{} ------------------'.format(dep_id, seg_id))
        total_iteration = len(day_of_week_list) * len(season_df_list)
        acc_summary.loc[id_acc, 'Acc_max'] = acc_summary.loc[id_acc, 'Acc_max'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_tree'] = acc_summary.loc[id_acc, 'Acc_tree'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_linear'] = acc_summary.loc[id_acc, 'Acc_linear'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_max_train'] = acc_summary.loc[id_acc, 'Acc_max_train'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_v1'] = acc_summary.loc[id_acc, 'Acc_v1'] / total_iteration
        increase_acc_seg_max = acc_summary.loc[id_acc, 'Acc_max_train'] - acc_summary.loc[id_acc, 'Acc_v1']
        acc_summary.loc[id_acc, 'Increase_acc'] = increase_acc_seg_max
        print('Acc_max: {} .'.format(acc_summary.loc[id_acc, 'Acc_max']))
        print('Acc_tree: {} .'.format(acc_summary.loc[id_acc, 'Acc_tree']))
        print('Acc_linear: {} .'.format(acc_summary.loc[id_acc, 'Acc_linear']))
        print('Acc_max_train: {}, Acc_v1: {}, Increase: {} .' \
              .format(acc_summary.loc[id_acc, 'Acc_max_train'], acc_summary.loc[id_acc, 'Acc_v1'],
                      increase_acc_seg_max))
        id_acc += 1

    # change results to dict with exact dtypes
    top_df_db_int_v1 = top_db_up_v1.to_dict(orient='records')
    top_df_db_float_v1 = top_db_up_float_v1.to_dict(orient='records')
    top_df_db_int_v1 = [dict([a, int(x)] for a, x in b.items()) for b in
                        top_df_db_int_v1]  # Python 3 items -> iteritems Python 2
    for i in range(len(top_df_db_int_v1)):
        top_df_db_int_v1[i].update(top_df_db_float_v1[i])

    return [top_df_db_int_v1, top_df_db_int_v2, db_lag7, acc_summary]


def v2_revenue_detection_cruise(current_time, day_1yrs_back, hotel_id, dep_id, num_day, significance_level, year_input, \
                                year_input_tree, df, top3_df_v2, day_arr, k_modifier, cruise, season_df):
    #    #top3_df_v2 is used for model ver 2.
#    top3_df_v2 = top3_df_v2[top3_df_v2['Day'] > 0]  # Check error from Day's dictionary.
    top3_df_v2 = top3_df_v2.copy()  # Check error from Day's dictionary.
    top3_df_v2['Type_str'] = top3_df_v2['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
    top3_df_v2['Driver_name'] = top3_df_v2['Source_H_Id'].astype(str) + '_' + top3_df_v2['Source_Segment_Id'].astype(
        str) + '_' + top3_df_v2['Type_str'].astype(str)

    seg_id_list = top3_df_v2['Destination_Segment_Id'].drop_duplicates()
    day_of_week_list = top3_df_v2['Day'].drop_duplicates().sort_values()  # 1-11

    # Test for an example: one_seg_id, one day_of_week, one season
    #    seg_id_list = seg_id_list[seg_id_list==0]
    #    day_of_week_list = day_of_week_list[day_of_week_list==6]
    #    season_df_list = season_df_list[season_df_list=="[('01-25', '02-14'), ('12-09', '12-29')]"]
    
    # write to database
    acc_summary = pd.DataFrame(columns=['Driver', 'Acc_max', 'Acc_tree', 'Acc_linear', 'Acc_max_train'])
    id_acc = 0

    top_df_db_int_v2 = []
    db_lag7 = {}
    column_otb = [str(i) * 3 for i in range(5, 31)]
    for seg_id in seg_id_list:
        acc_summary.loc[id_acc, 'Driver'] = '{}_{}_{}'.format(hotel_id, dep_id, seg_id)
        acc_summary.loc[id_acc, 'Acc_max'] = 0.0
        acc_summary.loc[id_acc, 'Acc_tree'] = 0.0
        acc_summary.loc[id_acc, 'Acc_linear'] = 0.0
        acc_summary.loc[id_acc, 'Acc_max_train'] = 0.0
        df_lag7 = pd.DataFrame()
        has_rn_flag = 0
        for day_of_week in day_of_week_list:
            print(day_of_week)
            # tree model
            numdays = (current_time - day_1yrs_back).days + 1
            date_ss_tree = [current_time - timedelta(days=x) for x in range(0, numdays)]
            date_ss_tree.sort()

            y_weekdays_tree = cruise.cruiseday_list(date_ss_tree).values
            y_dates_full_tree = [date_ss_tree[i] for i, d in enumerate(y_weekdays_tree) if d == day_of_week]
            if len(y_dates_full_tree)  == 0:
                day_1yrs_back = day_1yrs_back - timedelta(days=280)
                numdays = (current_time - day_1yrs_back).days + 1
                date_ss_tree = [current_time - timedelta(days=x) for x in range(0, numdays)]
                date_ss_tree.sort()

                y_weekdays_tree = cruise.cruiseday_list(date_ss_tree).values
                y_dates_full_tree = [date_ss_tree[i] for i, d in enumerate(y_weekdays_tree) if d == day_of_week]
            y_dates_tree = y_dates_full_tree[-(min(num_day, len(y_dates_full_tree))):]
            y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
            y_col_rn = '_'.join([str(dep_id), str(seg_id)]) + '_rn'  # Get roomnight data name
            if y_col_rn in df.columns:
                has_rn_flag = 1
            df1_full_tree = lib.get_df_by_dates(y_dates_full_tree, df)
            y_series_tree = df1_full_tree[y_col]
            y_series_tree.index = df1_full_tree['date']
            df1_tree = lib.get_df_by_dates(y_dates_tree, df1_full_tree)
            cols = set(df.columns) - {'date', 'day_of_week'}

            y_series_lag7 = pd.DataFrame()
            y_series_lag7['date'] = df1_full_tree['date']
            y_series_lag7[y_col] = df1_full_tree[y_col]

            # For upload median revenue, median unit
            df_ss_lag7 = pd.DataFrame(columns=['season', 'median_revenue', 'median_unit'])
            id_ss_lag7 = 0
            remove_season = []  # TN edited

            if day_of_week == 70:
                season_df_list = season_df['Period_Id'][season_df['Period_Type'] == 1].drop_duplicates().tolist()
            else:
                season_df_list = season_df['Period_Id'][((season_df['Period_Type'] == 0) & (season_df['Day'] != 0))].drop_duplicates()
            for season_tf_str in season_df_list:
                # calculating high low threshold of tree model
                #                day_1yrs_back_lag7 = day_1yrs_back - timedelta(days=7)

                # season_tf = list(ast.literal_eval(season_tf_str))
                # date_ss_hl = lib.generate_season_date(season_tf, year_input_tree)
                date_ss_hl = season_df[season_df['Period_Id'] == season_tf_str]['Date'].reset_index(drop=True)
                # date_ss_hl.sort()
                date_ss_hl = [i for i in date_ss_hl if i >= day_1yrs_back]


                y_weekdays_hl = cruise.cruiseday_list(date_ss_hl).values
                y_dates_hl = [date_ss_hl[i] for i, d in enumerate(y_weekdays_hl) if d == day_of_week]
                #                y_dates_hl_lag7 = lib.get_dates_lag(y_dates_hl, 7)
                df_hl = lib.get_df_by_dates(y_dates_hl, df)[y_col]
                meadian_hl = np.median(df_hl.dropna())
#                print('meadian_hl',meadian_hl)
                if np.isnan(meadian_hl):
                    date_ss_hl = season_df[season_df['Period_Id'] == season_tf_str]['Date'].reset_index(drop=True)
                    date_ss_hl.sort()
                    day_4yrs_back = day_1yrs_back - timedelta(days=280)
                    date_ss_hl = [i for i in date_ss_hl if i >= day_4yrs_back]

                    y_weekdays_hl = cruise.cruiseday_list(date_ss_hl).values
                    y_dates_hl = [date_ss_hl[i] for i, d in enumerate(y_weekdays_hl) if d == day_of_week]
                    #                y_dates_hl_lag7 = lib.get_dates_lag(y_dates_hl, 7)
                    df_hl = lib.get_df_by_dates(y_dates_hl, df)[y_col]
                    meadian_hl = np.median(df_hl.dropna())
                    # print(season_tf, day_of_week, meadian_hl)
                ## TN edited
                if np.isnan(meadian_hl):
                    remove_season.append(season_tf_str)
                    Day = np.round(day_of_week).astype(int)
                    period_id = int(season_tf_str)
                    # set{}
                    # temp_db
                    toptemp_df = top3_df_v2[
                        (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                        (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]

                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    top2_df['Priority'] = [1, 2]
                    temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                    temp_db = temp_db.rename(
                        columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                    temp_db = temp_db.to_dict(orient='records')
                    temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                    temp_db_float = pd.DataFrame()
                    temp_db_float['Linear_Rate'] = [0.0, 0.0]
                    temp_db_float['Rate'] = [0.0, 0.0]
                    # temp_db_float['Correlation'] = [0.0, 0.0]
                    temp_db_float['Modifier'] = [0.0, 0.0]
                    temp_db_float['HighThreshold'] = [0.0,
                                                      0.0]
                    temp_db_float['LowThreshold'] = [0.0,
                                                     0.0]
                    temp_db_float = temp_db_float.to_dict(orient='records')
                    for i in range(len(temp_db)):
                        temp_db[i].update(temp_db_float[i])

                    # tree_db
                    tree_df = pd.DataFrame(columns=['Case', 'Priority', 'Modifier'])
                    tree_df['Case'] = range(1, 10)
                    tree_df['Priority'] = len(tree_df['Case']) * [1]
                    tree_df['Modifier'] = len(tree_df['Case']) * [0.0]
                    tree_db = tree_df[['Case', 'Priority', 'Modifier']]
                    tree_db = tree_db.to_dict(orient='records')
                    tree_db = [dict([a, int(x)] for a, x in b.items()) for b in tree_db]

                    temp_db_up = {
                        #                        'TrendModifier': 1.2,
                        'MedianRevenue': 0.0,
                        'MedianUnitRate': 0.0,
                        'MedianUnits': 0.0,
                        'High': 0.0,
                        'Low': 0.0,
                        'Sources': temp_db,
                        'Tree': tree_db,
                        'Day': int(Day),
                        # 'Offset': int(0),
                        'Segment_Id': int(seg_id),
                        'H_Id': int(dep_id),
                        'Period': period_id,
                        'Model': 1
                    }
                    top_df_db_int_v2.append(temp_db_up)
                    continue
                else:
                    # Work with median revenue
                    df_ss_lag7.loc[id_ss_lag7, 'season'] = season_tf_str
                    df_ss_lag7.loc[id_ss_lag7, 'median_revenue'] = meadian_hl
                    # Work with median unit
                    if has_rn_flag == 1:
                        df_unit = lib.get_df_by_dates(y_dates_hl, df)[y_col_rn]
                        meadian_unit = np.median(df_unit.dropna())
                        df_ss_lag7.loc[id_ss_lag7, 'median_unit'] = meadian_unit
                    id_ss_lag7 += 1

                    for i in y_dates_hl:
                        y_series_lag7.loc[y_series_lag7['date'] == i, y_col] = meadian_hl
            season_df_list_new = list(set(season_df_list) - set(remove_season))  # TN edited
            df_lag7 = pd.concat([df_lag7, y_series_lag7], ignore_index=True)

            dict_df_tree = {}
            dict_df_tree[0] = df1_tree[df1_tree.columns]
            dict_df_tree["0_full"] = df1_full_tree[df1_full_tree.columns]
            for lag_temp in [1]:  # remove lag 7, 365
                dates_tree = lib.get_dates_lag(y_dates_tree, lag_temp)
                dates_full_tree = lib.get_dates_lag(y_dates_full_tree, lag_temp)
                df2_full_tree = lib.get_df_by_dates(dates_full_tree, df)
                df2_tree = lib.get_df_by_dates(dates_tree, df2_full_tree)
                if (lag_temp == 7):
                    df2_full_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]
                    df2_tree.loc[:, y_col] = y_series_lag7.loc[:, y_col]

                dict_df_tree[lag_temp] = df2_tree
                dict_df_tree[str(lag_temp) + "_full"] = df2_full_tree

            for season_tf_str in season_df_list_new:
                print(season_tf_str)

                # ================================================== WORK WITH INPUTS =========================================================

                # max model
                # season_tf = list(ast.literal_eval(season_tf_str))
                # date_ss = lib.generate_season_date(season_tf, year_input)
                # date_ss.sort()
                date_ss = season_df[season_df['Period_Id'] == season_tf_str]['Date'].reset_index(drop=True)
                y_weekdays = cruise.cruiseday_list(date_ss).values
                y_dates_full = [date_ss[i] for i, d in enumerate(y_weekdays) if d == day_of_week]
                y_dates = y_dates_full[-(min(num_day, len(y_dates_full))):]
                #                y_col = '_'.join([str(dep_id), str(seg_id)]) + '_rv'  # Get revenue data name
                df1_full = lib.get_df_by_dates(y_dates_full, df)
                y_series = df1_full[y_col]
                df1 = lib.get_df_by_dates(y_dates, df1_full)
                #                cols = set(df.columns) - {'date', 'day_of_week'}

                # get period_id from season_tf
                # quoclht
                period_id = int(season_tf_str)
                # season_period_id = season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id']
                # if len(season_period_id) == 1:
                #     period_id = int(season_period_id)
                # else:
                #     period_id = int(season_period_id.iloc[0])

                #                period_id = int(season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id'])
                Day = np.round(day_of_week).astype(int)

                # =========================================== COMPUTING ACCURACY OF MODEL VER 2 ===============================================

                toptemp_df = top3_df_v2[
                    (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                    (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]
                top1_name = str(toptemp_df[toptemp_df['Priority'] == 1]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 1]['SourceOffset'].values[0])
                top2_name = str(toptemp_df[toptemp_df['Priority'] == 2]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 2]['SourceOffset'].values[0])
                top3_name = str(toptemp_df[toptemp_df['Priority'] == 3]['Source_H_Id'].values[0]) + '_' \
                            + str(toptemp_df[toptemp_df['Priority'] == 3]['SourceOffset'].values[0])
                # top 3 drivers must be different dept, seg or lag
                # dept is higher priority than different lag
                if (top1_name == top2_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                elif (top1_name == top3_name):
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 1]
                else:
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    m_top2_df = toptemp_df[toptemp_df['Priority'] != 2]
                if ((top1_name == top2_name) and (
                        top1_name == top3_name)):  # if all dept is the same, lag must be different
                    #                    print(toptemp_df)
                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    if top2_df.loc[top2_df.index[1], 'SourceOffset'] == 0:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(1)
                    else:
                        top2_df.loc[top2_df.index[1], 'SourceOffset'] = int(0)
                    m_top2_df = top2_df

                # max model
                median_lag7_full = np.median(df1_full[df1_full['date'] >= day_1yrs_back][y_col].dropna())
                median_lag7 = np.median(df1[df1['date'] >= day_1yrs_back][y_col].dropna())
                dict_df = df_with_lag7(df1, df1_full, y_dates, y_dates_full, df, y_col, \
                                       median_lag7_full, median_lag7)

                # ================================================= Work with top2_df ========================================================

                col1 = top2_df[top2_df['Priority'] == 1]['Driver_name'].values[0]
                col2 = top2_df[top2_df['Priority'] != 1]['Driver_name'].values[0]

                lag1 = top2_df[top2_df['Priority'] == 1]['SourceOffset'].values[0]
                lag2 = top2_df[top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, col1, lag1, col2, lag2)

                col1_series = dict_df[str(lag1) + "_full"][col1]
                y1_series_r = y_series[-len(col1_series):]
                df2 = dict_df[lag1]
                col2_series = dict_df[str(lag2) + "_full"][col2]
                y2_series_r = y_series[-len(col2_series):]
                df2_2 = dict_df[lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(y1_series_r) != len(col1_series):
                    col1_series = col1_series[:-(len(col1_series) - len(y1_series_r))]
                if len(y2_series_r) != len(col2_series):
                    col2_series = col2_series[:-(len(col2_series) - len(y2_series_r))]
                # TN
                if (y1_series_r.isnull().values.all()) or (col1_series.isnull().values.all()) \
                        or (y2_series_r.isnull().values.all()) or (col2_series.isnull().values.all()) \
                        or (len(y1_series_r) == 0) or (len(col1_series) == 0) \
                        or (len(y2_series_r) == 0) or (len(col2_series) == 0):
                    Day = np.round(day_of_week).astype(int)
                    period_id = int(season_tf_str)
                    # season_period_id = season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id']
                    # if len(season_period_id) == 1:
                    #     period_id = int(season_period_id)
                    # else:
                    #     period_id = int(season_period_id.iloc[0])
                    # set{}
                    # temp_db
                    toptemp_df = top3_df_v2[
                        (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                        (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]

                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    top2_df['Priority'] = [1, 2]
                    temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                    temp_db = temp_db.rename(
                        columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                    temp_db = temp_db.to_dict(orient='records')
                    temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                    temp_db_float = pd.DataFrame()
                    temp_db_float['Linear_Rate'] = [0.0, 0.0]
                    temp_db_float['Rate'] = [0.0, 0.0]
                    # temp_db_float['Correlation'] = [0.0, 0.0]
                    temp_db_float['Modifier'] = [0.0, 0.0]
                    temp_db_float['HighThreshold'] = [0.0,
                                                      0.0]
                    temp_db_float['LowThreshold'] = [0.0,
                                                     0.0]
                    temp_db_float = temp_db_float.to_dict(orient='records')
                    for i in range(len(temp_db)):
                        temp_db[i].update(temp_db_float[i])

                    # tree_db
                    tree_df = pd.DataFrame(columns=['Case', 'Priority', 'Modifier'])
                    tree_df['Case'] = range(1, 10)
                    tree_df['Priority'] = len(tree_df['Case']) * [1]
                    tree_df['Modifier'] = len(tree_df['Case']) * [0.0]
                    tree_db = tree_df[['Case', 'Priority', 'Modifier']]
                    tree_db = tree_db.to_dict(orient='records')
                    tree_db = [dict([a, int(x)] for a, x in b.items()) for b in tree_db]

                    temp_db_up = {
                        #                        'TrendModifier': 1.2,
                        'MedianRevenue': 0.0,
                        'MedianUnitRate': 0.0,
                        'MedianUnits': 0.0,
                        'High': 0.0,
                        'Low': 0.0,
                        'Sources': temp_db,
                        'Tree': tree_db,
                        'Day': int(Day),
                        # 'Offset': int(0),
                        'Segment_Id': int(seg_id),
                        'H_Id': int(dep_id),
                        'Period': period_id,
                        'Model': 1
                    }
                    top_df_db_int_v2.append(temp_db_up)
                    continue
#                date_lag1 = df2['date']
#                date_lag2 = df2_2['date']
                y1_series_r.index = df2['date']
                col1_series.index = df2['date']
                y2_series_r.index = df2_2['date']
                col2_series.index = df2_2['date']
                y1_series_rm, col1_series_rm = v2_remove_outlier(y1_series_r, col1_series)
                
                original_series = y1_series_rm.copy()
                date_series_1 = y1_series_rm.index
                y1_series_rm = y1_series_rm.reset_index(drop = True)
                col1_series_rm = col1_series_rm.reset_index(drop = True)
                
                y2_series_rm, col2_series_rm = v2_remove_outlier(y2_series_r, col2_series)
                
                date_series_2 = y2_series_rm.index
                y2_series_rm = y2_series_rm.reset_index(drop = True)
                col2_series_rm = col2_series_rm.reset_index(drop = True)

                # check 2
                temp_df1 = pd.DataFrame(list(zip(y1_series_rm, col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(y2_series_rm, col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']
                temp_df2['date'] = date_series_2
                
                temp_df = v2_intersect_join(temp_df1, temp_df2)
#                date_series = temp_df['col1_series_rm'].values
                y_both = temp_df['y_both']
                col1_series_rm_both = temp_df['col1_series_rm']
                col2_series_rm_both = temp_df['col2_series_rm']
                date_series = temp_df['date'].tolist()
#                date_series = original_series[original_series.isin(y_both)].index.tolist()


                cap_arr = v2_compute_capture_arr(day_arr, y_col, col1, 1, 1, df1, df2, cols)
                cap_arr2 = v2_compute_capture_arr(day_arr, y_col, col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                y_forecast_v2, forecast_error_v2, modifier1, modifier2, y_min, y_max = v2_compute_error(y_both, \
                                                                                                        col1_series_rm_both,
                                                                                                        cap_arr[
                                                                                                            'capture'],
                                                                                                        cap_arr['ARR'], \
                                                                                                        col2_series_rm_both,
                                                                                                        cap_arr2[
                                                                                                            'capture'],
                                                                                                        cap_arr2['ARR'],
                                                                                                        k_modifier,date_series )

                y_forecast_v2_top1, forecast_error_v2_top1, modifier_top1, y_min_top1, y_max_top1, a_top1 = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier, date_series_1)

                y_forecast_v2_top2, forecast_error_v2_top2, modifier_top2, y_min_top2, y_max_top2, a_top2 = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier, date_series_2)

                cov_val_v2 = 1 - forecast_error_v2
                cov_val_v2_top1 = 1 - forecast_error_v2_top1
                cov_val_v2_top2 = 1 - forecast_error_v2_top2

                if cov_val_v2 < cov_val_v2_top1:
                    cov_val_v2 = cov_val_v2_top1
                    modifier1 = modifier_top1
                    modifier2 = 0.0
                    y_min = y_min_top1
                    y_max = y_max_top1
                #                    y_forecast_v2 = y_forecast_v2_top1
                if cov_val_v2 < cov_val_v2_top2:
                    cov_val_v2 = cov_val_v2_top2
                    modifier1 = 0.0
                    modifier2 = modifier_top2
                    y_min = y_min_top2
                    y_max = y_max_top2
                #                    y_forecast_v2 = y_forecast_v2_top2

                if cov_val_v2_top1 >= cov_val_v2_top2:
                    cov_val_linear = cov_val_v2_top1
                    linear_rate1 = a_top1
                    linear_rate2 = a_top2
                    linear_flag = 1
                else:
                    cov_val_linear = cov_val_v2_top2
                    linear_rate1 = a_top2
                    linear_rate2 = a_top1
                    linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                # ================================================= Work with m_top2_df ========================================================

                m_top2_df['Priority'].values[0] = 1
                m_top2_df['Priority'].values[1] = 2
                m_col1 = m_top2_df[m_top2_df['Priority'] == 1]['Driver_name'].values[0]
                m_col2 = m_top2_df[m_top2_df['Priority'] != 1]['Driver_name'].values[0]

                m_lag1 = m_top2_df[m_top2_df['Priority'] == 1]['SourceOffset'].values[0]
                m_lag2 = m_top2_df[m_top2_df['Priority'] != 1]['SourceOffset'].values[0]

                #                print(seg_id, day_of_week, Day, season_tf_str, period_id, m_col1, m_lag1, m_col2, m_lag2)
             
                
                
                m_col1_series = dict_df[str(m_lag1) + "_full"][m_col1]
                m_y1_series_r = y_series[-len(m_col1_series):]
                df2 = dict_df[m_lag1]
                m_col2_series = dict_df[str(m_lag2) + "_full"][m_col2]
                m_y2_series_r = y_series[-len(m_col2_series):]
                df2_2 = dict_df[m_lag2]

                # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
                # check 1
                if len(m_y1_series_r) != len(m_col1_series):
                    m_col1_series = m_col1_series[:-(len(m_col1_series) - len(m_y1_series_r))]
                if len(m_y2_series_r) != len(m_col2_series):
                    m_col2_series = m_col2_series[:-(len(m_col2_series) - len(m_y2_series_r))]

                # TN
                if (m_y1_series_r.isnull().values.all()) or (m_col1_series.isnull().values.all()) \
                        or (m_y2_series_r.isnull().values.all()) or (m_col2_series.isnull().values.all()) \
                        or (len(m_y1_series_r) == 0) or (len(m_col1_series) == 0) \
                        or (len(m_y2_series_r) == 0) or (len(m_col2_series) == 0):
                    Day = np.round(day_of_week).astype(int)
                    period_id = int(season_tf_str)
                    # season_period_id = season_df_str[season_df_str[str(dep_id)] == season_tf_str]['period_id']
                    # if len(season_period_id) == 1:
                    #     period_id = int(season_period_id)
                    # else:
                    #     period_id = int(season_period_id.iloc[0])
                    # set{}
                    # temp_db
                    toptemp_df = top3_df_v2[
                        (top3_df_v2['H_Id'] == int(dep_id)) & (top3_df_v2['Destination_Segment_Id'] == int(seg_id)) & \
                        (top3_df_v2['Day'] == Day) & (top3_df_v2['Period'] == period_id)]

                    top2_df = toptemp_df[toptemp_df['Priority'] != 3]
                    top2_df = top2_df.reset_index(drop=True)
                    top2_df['Priority'] = [1, 2]
                    temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                    temp_db = temp_db.rename(
                        columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                    temp_db = temp_db.to_dict(orient='records')
                    temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                    temp_db_float = pd.DataFrame()
                    temp_db_float['Linear_Rate'] = [0.0, 0.0]
                    temp_db_float['Rate'] = [0.0, 0.0]
                    # temp_db_float['Correlation'] = [0.0, 0.0]
                    temp_db_float['Modifier'] = [0.0, 0.0]
                    temp_db_float['HighThreshold'] = [0.0,
                                                      0.0]
                    temp_db_float['LowThreshold'] = [0.0,
                                                     0.0]
                    temp_db_float = temp_db_float.to_dict(orient='records')
                    for i in range(len(temp_db)):
                        temp_db[i].update(temp_db_float[i])

                    # tree_db
                    tree_df = pd.DataFrame(columns=['Case', 'Priority', 'Modifier'])
                    tree_df['Case'] = range(1, 10)
                    tree_df['Priority'] = len(tree_df['Case']) * [1]
                    tree_df['Modifier'] = len(tree_df['Case']) * [0.0]
                    tree_db = tree_df[['Case', 'Priority', 'Modifier']]
                    tree_db = tree_db.to_dict(orient='records')
                    tree_db = [dict([a, int(x)] for a, x in b.items()) for b in tree_db]

                    temp_db_up = {
                        #                        'TrendModifier': 1.2,
                        'MedianRevenue': 0.0,
                        'MedianUnitRate': 0.0,
                        'MedianUnits': 0.0,
                        'High': 0.0,
                        'Low': 0.0,
                        'Sources': temp_db,
                        'Tree': tree_db,
                        'Day': int(Day),
                        'Offset': int(0),
                        'Segment_Id': int(seg_id),
                        'H_Id': int(dep_id),
                        'Period': period_id,
                        'Model': 1
                    }
                    top_df_db_int_v2.append(temp_db_up)
                    continue
                
                
                m_y1_series_r.index = df2['date']
                m_col1_series.index = df2['date']
                m_y2_series_r.index = df2_2['date']
                m_col2_series.index = df2_2['date']
                
                
                
                m_y1_series_rm, m_col1_series_rm = v2_remove_outlier(m_y1_series_r, m_col1_series)
                
                m_original_series = m_y1_series_rm.copy()
                m_date_series_1 = m_y1_series_rm.index
                m_y1_series_rm = m_y1_series_rm.reset_index(drop = True)
                m_col1_series_rm = m_col1_series_rm.reset_index(drop = True)
                
                m_y2_series_rm, m_col2_series_rm = v2_remove_outlier(m_y2_series_r, m_col2_series)
                
                m_date_series_2 = m_y2_series_rm.index
                m_y2_series_rm = m_y2_series_rm.reset_index(drop = True)
                m_col2_series_rm = m_col2_series_rm.reset_index(drop = True)


                # check 2
                temp_df1 = pd.DataFrame(list(zip(m_y1_series_rm, m_col1_series_rm)))
                temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
                temp_df2 = pd.DataFrame(list(zip(m_y2_series_rm, m_col2_series_rm)))
                temp_df2.columns = ['y2_series_rm', 'col2_series_rm']
                temp_df2['date'] = m_date_series_2

                temp_df = v2_intersect_join(temp_df1, temp_df2)
                m_y_both = temp_df['y_both']
                m_col1_series_rm_both = temp_df['col1_series_rm']
                m_col2_series_rm_both = temp_df['col2_series_rm']
                
                m_date_series = temp_df['date'].tolist()
#                m_date_series = m_original_series[m_original_series.isin(m_y_both)].index.tolist()
                
                m_cap_arr = v2_compute_capture_arr(day_arr, y_col, m_col1, 1, 1, df1, df2, cols)
                m_cap_arr2 = v2_compute_capture_arr(day_arr, y_col, m_col2, 1, 1, df1, df2_2, cols)
                # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
                m_y_forecast_v2, m_forecast_error_v2, m_modifier1, m_modifier2, m_y_min, m_y_max = v2_compute_error(
                    m_y_both, \
                    m_col1_series_rm_both, m_cap_arr['capture'], m_cap_arr['ARR'], \
                    m_col2_series_rm_both, m_cap_arr2['capture'], m_cap_arr2['ARR'], k_modifier, m_date_series)

                m_y_forecast_v2_top1, m_forecast_error_v2_top1, m_modifier_top1, m_y_min_top1, m_y_max_top1, m_a_top1 = \
                    v2_compute_error_single(m_y1_series_rm.dropna(), m_col1_series_rm.dropna(), m_cap_arr['capture'],
                                            m_cap_arr['ARR'], k_modifier, m_date_series_1)

                m_y_forecast_v2_top2, m_forecast_error_v2_top2, m_modifier_top2, m_y_min_top2, m_y_max_top2, m_a_top2 = \
                    v2_compute_error_single(m_y2_series_rm.dropna(), m_col2_series_rm.dropna(), m_cap_arr2['capture'],
                                            m_cap_arr2['ARR'], k_modifier, m_date_series_2)

                m_cov_val_v2 = 1 - m_forecast_error_v2
                m_cov_val_v2_top1 = 1 - m_forecast_error_v2_top1
                m_cov_val_v2_top2 = 1 - m_forecast_error_v2_top2

                if m_cov_val_v2 < m_cov_val_v2_top1:
                    m_cov_val_v2 = m_cov_val_v2_top1
                    m_modifier1 = m_modifier_top1
                    m_modifier2 = 0.0
                    m_y_min = m_y_min_top1
                    m_y_max = m_y_max_top1
                #                    m_y_forecast_v2 = m_y_forecast_v2_top1
                if m_cov_val_v2 < m_cov_val_v2_top2:
                    m_cov_val_v2 = m_cov_val_v2_top2
                    m_modifier1 = 0.0
                    m_modifier2 = m_modifier_top2
                    m_y_min = m_y_min_top2
                    m_y_max = m_y_max_top2
                #                    m_y_forecast_v2 = m_y_forecast_v2_top2

                if m_cov_val_v2_top1 >= m_cov_val_v2_top2:
                    m_cov_val_linear = m_cov_val_v2_top1
                    m_linear_rate1 = m_a_top1
                    m_linear_rate2 = m_a_top2
                    m_linear_flag = 1
                else:
                    m_cov_val_linear = m_cov_val_v2_top2
                    m_linear_rate1 = m_a_top2
                    m_linear_rate2 = m_a_top1
                    m_linear_flag = 2
                #                conf0_v2, conf1_v2 = lib.compute_interval(cov_val_v2, significance_level)

                if m_cov_val_v2 > cov_val_v2:
                    top2_df = m_top2_df
                    y_both = m_y_both
                    col1_series_rm_both = m_col1_series_rm_both
                    col2_series_rm_both = m_col2_series_rm_both
                    y_min = m_y_min
                    y_max = m_y_max
                    cov_val_v2 = m_cov_val_v2
                    cap_arr = m_cap_arr
                    cap_arr2 = m_cap_arr2
                    modifier1 = m_modifier1
                    modifier2 = m_modifier2
                    linear_rate1 = m_linear_rate1
                    linear_rate2 = m_linear_rate2
                    linear_flag = m_linear_flag
                    cov_val_linear = m_cov_val_linear

                    col1_series = m_col1_series
                    y1_series_r = m_y1_series_r
                    col2_series = m_col2_series
                    y2_series_r = m_y2_series_r
                    col1_series_rm = m_col1_series_rm
                    y1_series_rm = m_y1_series_rm
                    col2_series_rm = m_col2_series_rm
                    y2_series_rm = m_y2_series_rm
                    date_series = m_date_series
                    date_series_1 = m_date_series_1
                    date_series_2 = m_date_series_2
#                    date2_final =
                    #                    y_forecast_v2 = m_y_forecast_v2
                    lag1 = m_lag1
                    lag2 = m_lag2
                    col1 = m_col1
                    col2 = m_col2

                    # tree model
                
#                y1_series_rm = y1_series_rm.reset_index(drop = True)
#                col1_series_rm = col1_series_rm.reset_index(drop = True)
#                y2_series_rm = y2_series_rm.reset_index(drop = True)
#                col2_series_rm = col2_series_rm.reset_index(drop = True)
                rate1 = cap_arr['capture'] * cap_arr['ARR']
                rate2 = cap_arr2['capture'] * cap_arr2['ARR']
                tree_db, highlow_threshold = v2_revenue_detection_tree(dict_df_tree, top2_df, y_series_tree, y_min,
                                                                       y_max, rate1, rate2)
                # new modifiers are re-computed from full data with season and day of week
                _, _, modifier_tree1, _, _, _ = \
                    v2_compute_error_single(y1_series_rm.dropna(), col1_series_rm.dropna(), cap_arr['capture'],
                                            cap_arr['ARR'], k_modifier, date_series_1)
                _, _, modifier_tree2, _, _, _ = \
                    v2_compute_error_single(y2_series_rm.dropna(), col2_series_rm.dropna(), cap_arr2['capture'],
                                            cap_arr2['ARR'], k_modifier, date_series_2)
                for i, v in enumerate(tree_db):
                    if v['Priority'] == 1:
                        v['Modifier'] = modifier_tree1
                    if v['Priority'] == 2:
                        v['Modifier'] = modifier_tree2
                # ================================================= Validation tree model =====================================================
#                y = y_both[-4:].reset_index(drop=True)
#                x1 = col1_series_rm_both[-4:].reset_index(drop=True)
#                x2 = col2_series_rm_both[-4:].reset_index(drop=True)
                y = y_both.reset_index(drop=True)
                x1 = col1_series_rm_both.reset_index(drop=True)
                x2 = col2_series_rm_both.reset_index(drop=True)
                case_df = pd.DataFrame(tree_db)
                
                weight = set_weight(date_series)/sum(set_weight(date_series))
                
#                cov_val_tree_temp = 0.0
                error_tree_series = []
                for i, v in enumerate(y):
                    _, forecast_error_tree = tree_validation_error_json(y[i], y_min, y_max, x1[i], x2[i], case_df,
                                                                        highlow_threshold, rate1, rate2)
                    error_tree_series.append(forecast_error_tree)
                cov_val_tree = 1 - sum(weight*np.array(error_tree_series))     
#                    cov_val_tree_temp += 1 - forecast_error_tree
#                cov_val_tree = cov_val_tree_temp / len(y)
                
                
                _, forecast_error_max = max_validate_error_json(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2,
                                                                rate2, date_series)
                cov_val_max = 1 - forecast_error_max

                if linear_flag == 1:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x1, linear_rate1, date_series)
                else:
                    _, forecast_error_linear = linear_validate_error_json(y, y_min, y_max, x2, linear_rate2, date_series)
                cov_val_linear = 1 - forecast_error_linear
                # ======================================================= WORK WITH RESULTS ===================================================

                y_median_revenue = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_revenue'].values[0]
                # Work with unit data
                if has_rn_flag == 1:
                    y_median_unit = df_ss_lag7[df_ss_lag7['season'] == season_tf_str]['median_unit'].values[0]
                    y_median_unit_rate = y_median_revenue / y_median_unit
                    if np.isnan(y_median_unit_rate) or np.isinf(y_median_unit_rate):
                        y_median_unit_rate = 0.0
                else:
                    y_median_unit = 0.0
                    y_median_unit_rate = 0.0

                acc_summary.loc[id_acc, 'Acc_max'] += cov_val_max
                acc_summary.loc[id_acc, 'Acc_tree'] += cov_val_tree
                acc_summary.loc[id_acc, 'Acc_linear'] += cov_val_linear
                acc_summary.loc[id_acc, 'Acc_max_train'] += cov_val_v2
                #                rate1 = cap_arr['capture']*cap_arr['ARR']
                #                rate2 = cap_arr2['capture']*cap_arr2['ARR']
                if max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_linear:
                    best_model = 0
                elif max(cov_val_max, cov_val_tree, cov_val_linear) == cov_val_max:
                    best_model = 1
                else:
                    best_model = 2
                # append results to dataframe
                top2_df = top2_df.reset_index(drop=True)

                for i in range(2):
                    replace_source = str(top2_df['Source_Segment_Id'].iloc[i])
                    if pd.Series(str(replace_source)).isin(column_otb).iloc[0]:
                        leadtime_otb = ''.join(sorted(set(replace_source), key=replace_source.index))
                        if (leadtime_otb == '1' ) or (leadtime_otb == '2'):
                            leadtime_otb = leadtime_otb*2
                        top2_df['SourceOffset'].iloc[i] = leadtime_otb
                        top2_df['Source_Segment_Id'].iloc[i] = 0
                        top2_df['Source_H_Id'].iloc[i] = dep_id
                        top2_df['Type'].iloc[i] = 6

                top2_df['Priority'] = [1, 2]
                temp_db = top2_df[['Source_Segment_Id', 'SourceOffset', 'Priority', 'Source_H_Id', 'Type']]
                temp_db = temp_db.rename(
                    columns={'Source_H_Id': 'H_Id', 'Source_Segment_Id': 'Segment_Id', 'SourceOffset': 'Offset'})
                temp_db = temp_db.to_dict(orient='records')
                temp_db = [dict([a, int(x)] for a, x in b.items()) for b in temp_db]
                temp_db_float = pd.DataFrame()
                temp_db_float['Linear_Rate'] = [linear_rate1, linear_rate2]
                temp_db_float['Rate'] = [rate1, rate2]
                # temp_db_float['Correlation'] = [cov_val_v2, cov_val_v2]
                temp_db_float['Modifier'] = [modifier1, modifier2]
                temp_db_float['HighThreshold'] = [highlow_threshold['top1_high'].values[0],
                                                  highlow_threshold['top2_high'].values[0]]
                temp_db_float['LowThreshold'] = [highlow_threshold['top1_low'].values[0],
                                                 highlow_threshold['top2_low'].values[0]]
                temp_db_float = temp_db_float.to_dict(orient='records')
                for i in range(len(temp_db)):
                    temp_db[i].update(temp_db_float[i])
                if Day == 70:
                    df_orginal = cruise.get_data_original()
                    season_original = df_orginal['Day'][df_orginal['Period_Id'] == season_tf_str]
                    for Day in range(min(season_original),max(season_original)+1):
                        temp_db_up = {
                            #                        'TrendModifier': 1.2,
                            'MedianRevenue': y_median_revenue,
                            'MedianUnitRate': y_median_unit_rate,
                            'MedianUnits': y_median_unit,
                            'High': y_max,
                            'Low': y_min,
                            'Sources': temp_db,
                            'Tree': tree_db,
                            'Day': int(Day),
                            # 'Offset': int(0),
                            'Segment_Id': int(seg_id),
                            'H_Id': int(dep_id),
                            'Period': period_id,
                            'Model': best_model
                        }
                        top_df_db_int_v2.append(temp_db_up)
                else:
                    temp_db_up = {
                        #                        'TrendModifier': 1.2,
                        'MedianRevenue': y_median_revenue,
                        'MedianUnitRate': y_median_unit_rate,
                        'MedianUnits': y_median_unit,
                        'High': y_max,
                        'Low': y_min,
                        'Sources': temp_db,
                        'Tree': tree_db,
                        'Day': int(Day),
                        # 'Offset': int(0), # Destination
                        'Segment_Id': int(seg_id),
                        'H_Id': int(dep_id),
                        'Period': period_id,
                        'Model': best_model
                    }
                    top_df_db_int_v2.append(temp_db_up)

        df_lag7 = df_lag7.sort_values(['date'], ascending=True)
        db_lag7[seg_id] = df_lag7

        print('------------------ Dep_Seg {}_{} ------------------'.format(dep_id, seg_id))
        total_iteration = len(day_of_week_list) * len(season_df_list)
        acc_summary.loc[id_acc, 'Acc_max'] = acc_summary.loc[id_acc, 'Acc_max'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_tree'] = acc_summary.loc[id_acc, 'Acc_tree'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_linear'] = acc_summary.loc[id_acc, 'Acc_linear'] / total_iteration
        acc_summary.loc[id_acc, 'Acc_max_train'] = acc_summary.loc[id_acc, 'Acc_max_train'] / total_iteration
        print('Acc_max: {} .'.format(acc_summary.loc[id_acc, 'Acc_max']))
        print('Acc_tree: {} .'.format(acc_summary.loc[id_acc, 'Acc_tree']))
        print('Acc_linear: {} .'.format(acc_summary.loc[id_acc, 'Acc_linear']))
        id_acc += 1

    return [top_df_db_int_v2, db_lag7, acc_summary]


def v1_validate_error(y, x, capture, arr):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    y_forecast = x * capture * arr
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


def linear_validate_error(y, y_min, y_max, x, linear_rate):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 2
    :param y: real value of forecast driver
    :param x: value of other drivers
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    # ======================================================================================
    # method 1
    y_forecast = linear_rate * x
    # ======================================================================================
    #    y_forecast = pd.Series([max(min(y_max,temp),y_min) for temp in y_forecast])
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


def v2_validate_error(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2, rate2):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 2
    :param y: real value of forecast driver
    :param x: value of other drivers
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """
    # ======================================================================================
    # method 1
    y1_forecast = modifier1 * rate1 * x1.values[0]
    y2_forecast = modifier2 * rate2 * x2.values[0]
    y_forecast = [max(y1_forecast, y2_forecast)]
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]


def v2_validation(hotel_id, dep_id, year_input, test_time, df, top_df_v1, top_df_v2, db_lag7, season_df_str,
                  k_modifier):
    outsample_df = pd.DataFrame(columns=['Date', 'Driver', 'Acc_tree', 'Acc_max', 'Acc_linear', 'Acc_best', 'Acc_v1'])
    result_df = pd.DataFrame(columns=['Date', 'Driver', 'High', 'Low', 'Case', \
                                      'Driver_name_tree', 'Driver_value_tree', 'Modifier_tree', 'Rate_tree', \
                                      'Pred_tree', 'Acc_tree', \
                                      'Driver1_name_max', 'Driver1_value_max', 'Modifier1', 'Rate1_max', \
                                      'Driver2_name_max', 'Driver2_value_max', 'Modifier2', 'Rate2_max', \
                                      'Pred_max', 'Acc_max', \
                                      'Linear_Rate1', 'Pred_linear1', 'Acc_linear1', \
                                      'Driver1_name_v1', 'Driver1_value_v1', 'Rate_v1', \
                                      'Pred_v1', 'Acc_v1', 'Real_value'
                                      ])
    best_df = pd.DataFrame(columns=['Date', 'Driver', 'Acc_best', 'Acc_v1', 'Increase_acc'])
    index = 0

    print('Validation: {}_{}'.format(hotel_id, dep_id))
    top_df_v1 = pd.DataFrame(top_df_v1)
    top_df_v1['Type_str'] = top_df_v1['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
    top_df_v1['Driver_name'] = top_df_v1['Source_H_Id'].astype(str) + '_' + top_df_v1['Source_Segment_Id'].astype(str) \
                               + '_' + top_df_v1['Type_str'].astype(str)

    top_df_v2_df = pd.DataFrame(top_df_v2)
    seg_id_list = top_df_v2_df['Segment_Id'].drop_duplicates()
    season_df_list = season_df_str[str(dep_id)]

    # seg_id_list = seg_id_list[seg_id_list==0]
    # season_df_list = season_df_list[season_df_list=="[('06-11', '07-06')]"]

    for test_day in test_time:
        # detect Day of test day
        day_of_week = test_day.weekday()
        Day = np.round((day_of_week + 8) % 7 + 1).astype(int)
        # detect Period of test day
        for idx, season_tf_str in enumerate(season_df_list):
            season_tf = list(ast.literal_eval(season_tf_str))
            date_ss = lib.generate_season_date(season_tf, year_input)
            date_ss.sort()
            if test_day in date_ss:
                period_id = int(season_df_str.loc[idx, 'period_id'])
                break

        for seg_id in seg_id_list:
            df_lag7 = db_lag7[seg_id]

            y_name = '{}_{}_rv'.format(dep_id, seg_id)
            test_day_y = test_day.strftime('%Y-%m-%d')
            y = df[df['date'] == test_day_y][y_name]

            # ================================================= MODEL OLD =====================================================

            test_df = top_df_v1[
                (top_df_v1['H_Id'] == int(dep_id)) & (top_df_v1['Destination_Segment_Id'] == int(seg_id)) & \
                (top_df_v1['Day'] == Day) & (top_df_v1['Period'] == period_id) & (top_df_v1['Priority'] == 1)]
            x_name = test_df['Driver_name'].values[0]
            lag = int(test_df['SourceOffset'].values[0])
            test_day_x = (test_day - timedelta(days=lag)).strftime('%Y-%m-%d')
            x = df[df['date'] == test_day_x][x_name]

            capture = test_df['Percent'].values[0]
            arr = test_df['AverageRevenue'].values[0]
            rate = capture * arr
            y_forecast_v1, forecast_error_v1 = v1_validate_error(y, x, capture, arr)
            acc_v1 = 1 - forecast_error_v1

            # ================================================= MODEL MAX =====================================================

            test_df_2 = next(
                (item for item in top_df_v2 if item['H_Id'] == int(dep_id) and item['Segment_Id'] == int(seg_id) \
                 and item['Day'] == Day and item['Period'] == period_id), None)
            top2_df = pd.DataFrame(test_df_2['Sources'])
            top2_df['Type_str'] = top2_df['Type'].map({1: 'rn', 2: 'gn', 3: 'rv', 5: 'rv'})
            top2_df['Driver_name'] = top2_df['H_Id'].astype(str) + '_' + top2_df['Segment_Id'].astype(str) \
                                     + '_' + top2_df['Type_str'].astype(str)

            y_min = test_df_2['Low']
            y_max = test_df_2['High']

            x1_name = top2_df['Driver_name'][0]
            lag1 = int(top2_df['Offset'][0])
            test_day_x1 = (test_day - timedelta(days=lag1)).strftime('%Y-%m-%d')
            x1 = df[df['date'] == test_day_x1][x1_name]
            if lag1 == 7:
                x1 = df_lag7[df_lag7['date'] == (test_day)][y_name]
                x1 = x1.rename(x1_name)

            rate1 = top2_df['Rate'][0]
            modifier1 = min(max(top2_df['Modifier'][0], 1 - k_modifier), 1 + k_modifier)
            #                modifier1 = top2_df['Modifier'][0]

            x2_name = top2_df['Driver_name'][1]
            lag2 = int(top2_df['Offset'][1])
            test_day_x2 = (test_day - timedelta(days=lag2)).strftime('%Y-%m-%d')
            x2 = df[df['date'] == test_day_x2][x2_name]
            if lag2 == 7:
                x2 = df_lag7[df_lag7['date'] == (test_day)][y_name]
                x2 = x2.rename(x2_name)
            rate2 = top2_df['Rate'][1]
            modifier2 = min(max(top2_df['Modifier'][1], 1 - k_modifier), 1 + k_modifier)
            #                modifier2 = top2_df['Modifier'][1]

            y_forecast_v2, forecast_error_v2 = v2_validate_error(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2,
                                                                 rate2)
            acc_max = 1 - forecast_error_v2

            # ================================================= MODEL LINEAR =====================================================

            linear_rate = top2_df['Linear_Rate'][0]
            y_forecast_linear, forecast_error_linear = linear_validate_error(y, y_min, y_max, x1, linear_rate)
            acc_linear = 1 - forecast_error_linear

            # ================================================= MODEL TREE =====================================================

            case_df = pd.DataFrame(test_df_2['Tree'])
            y_forecast_tree, forecast_error_tree, x_tree, x_name_tree, modifier_tree, rate_tree, type_of_x = \
                tree_validation_error(y, y_min, y_max, x1, x2, case_df, top2_df)
            acc_tree = 1 - forecast_error_tree

            # =============================================== WITH RESULTS ====================================================

            best_model = test_df_2['Model']
            if best_model == 0:
                acc_best = acc_linear
            elif best_model == 1:
                acc_best = acc_max
            else:
                acc_best = acc_tree
            outsample_df.loc[index] = [test_day_y, '_'.join([str(hotel_id), str(dep_id), str(seg_id)]), acc_tree,
                                       acc_max, acc_linear, acc_best, acc_v1]
            result_df.loc[index] = [test_day_y, '_'.join([str(hotel_id), str(dep_id), str(seg_id)]), y_max, y_min,
                                    type_of_x, \
                                    x_name_tree, x_tree, modifier_tree, rate_tree, \
                                    y_forecast_tree.values[0], acc_tree,
                                    '_'.join([str(x1_name), str(lag1)]), x1.values[0], modifier1, rate1, \
                                    '_'.join([str(x2_name), str(lag2)]), x2.values[0], modifier2, rate2, \
                                    y_forecast_v2.values[0], acc_max, \
                                    linear_rate, y_forecast_linear.values[0], acc_linear, \
                                    '_'.join([str(x_name), str(lag)]), x.values[0], rate, y_forecast_v1.values[0],
                                    acc_v1, y.values[0]]
            best_df.loc[index] = [test_day_y, '_'.join([str(hotel_id), str(dep_id), str(seg_id)]), acc_best, acc_v1,
                                  acc_best - acc_v1]
            index += 1

    outsample_df = outsample_df.sort_values(['Date', 'Driver'], ascending=True)
    result_df = result_df.sort_values(['Date', 'Driver'], ascending=True)
    best_df = best_df.sort_values(['Date', 'Driver'], ascending=True)
    #    df_reshape0 = pd.DataFrame(outsample_df['Driver'].unique(), columns = ['Driver']).sort_values(['Driver'], ascending=True)
    #    df_reshape1 = pd.DataFrame(np.transpose(outsample_df['Pred_tree'].reshape(-1,len(df_reshape0))))
    #    df_reshape2 = pd.DataFrame(np.transpose(outsample_df['Pred_max'].reshape(-1,len(df_reshape0))))
    #    df_reshape3 = pd.DataFrame(np.transpose(outsample_df['Increase_acc'].reshape(-1,len(df_reshape0))))
    #
    #    outsample_df_reshape = pd.concat([df_reshape0, df_reshape1, df_reshape2, df_reshape3], axis=1)
    #    outsample_df_reshape.columns = ['Driver', 'v2_lead5', 'v2_lead10', 'v2_lead30', 'v1_lead5', 'v1_lead10', 'v1_lead30', 'inc_lead5', 'inc_lead10', 'inc_lead30']
    #    outsample_df_reshape.columns = ['Driver'] + [i.strftime('%Y-%m-%d') for i in test_time] \
    #                                    + [i.strftime('%Y-%m-%d') for i in test_time] \
    #                                    + [i.strftime('%Y-%m-%d') for i in test_time]
    return outsample_df, result_df, best_df


def compute_error_tree_nodata(x1, x2, y, top1_low, top1_high, top2_low, top2_high, type_of_x):
    if type_of_x == "low_low":
        x1_temp = x1[(x1 <= top1_low)]
        x2_temp = x2[(x2 <= top2_low)]
        y1_temp = y[(x1 <= top1_low)]
        y2_temp = y[(x2 <= top2_low)]
    elif type_of_x == "low_med":
        x1_temp = x1[(x1 <= top1_low)]
        x2_temp = x2[(x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 <= top1_low)]
        y2_temp = y[(x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "low_hig":
        x1_temp = x1[(x1 <= top1_low)]
        x2_temp = x2[(x2 >= top2_high)]
        y1_temp = y[(x1 <= top1_low)]
        y2_temp = y[(x2 >= top2_high)]
    elif type_of_x == "med_low":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high)]
        x2_temp = x2[(x2 <= top2_low)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high)]
        y2_temp = y[(x2 <= top2_low)]
    elif type_of_x == "med_med":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high)]
        x2_temp = x2[(x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high)]
        y2_temp = y[(x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "med_hig":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high)]
        x2_temp = x2[(x2 >= top2_high)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high)]
        y2_temp = y[(x2 >= top2_high)]
    elif type_of_x == "hig_low":
        x1_temp = x1[(x1 >= top1_high)]
        x2_temp = x2[(x2 <= top2_low)]
        y1_temp = y[(x1 >= top1_high)]
        y2_temp = y[(x2 <= top2_low)]
    elif type_of_x == "hig_med":
        x1_temp = x1[(x1 >= top1_high)]
        x2_temp = x2[(x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 >= top1_high)]
        y2_temp = y[(x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "hig_hig":
        x1_temp = x1[(x1 >= top1_high)]
        x2_temp = x2[(x2 >= top2_high)]
        y1_temp = y[(x1 >= top1_high)]
        y2_temp = y[(x2 >= top2_high)]
    return x1_temp, x2_temp, y1_temp, y2_temp


def v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, type_of_x, y_dates):
    """
    Compute predicted values (base on capture and arr) and return error of model ver 1
    :param y: real value series
    :param x: predicted value series
    :param capture: capture values
    :param arr: ARR values
    :return: error values
    """

    top1_low = float(np.percentile(x1, 33))
    top1_high = float(np.percentile(x1, 66))
    top2_low = float(np.percentile(x2, 33))
    top2_high = float(np.percentile(x2, 66))

    if abs(top1_low - top1_high) < 1e-9:
        #        print(x1.name, x1[0])
        #        print(top1_low, top1_high)
        top1_low = min(top1_high, top1_low)
        top1_high = min(top1_high, top1_low)
    if abs(top2_low - top2_high) < 1e-9:
        #        print(x2.name, x1[0])
        #        print(top2_low, top2_high)
        top2_low = min(top2_high, top2_low)
        top2_high = min(top2_high, top2_low)
    #    print(top1_low, top1_high, top2_low, top2_high)
    #    y_min = min(y)
    #    y_max = max(y)
    if type_of_x == "low_low":
        x1_temp = x1[(x1 <= top1_low) & (x2 <= top2_low)]
        x2_temp = x2[(x1 <= top1_low) & (x2 <= top2_low)]
        y1_temp = y[(x1 <= top1_low) & (x2 <= top2_low)]
    elif type_of_x == "low_med":
        x1_temp = x1[(x1 <= top1_low) & (x2 >= top2_low) & (x2 <= top2_high)]
        x2_temp = x2[(x1 <= top1_low) & (x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 <= top1_low) & (x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "low_hig":
        x1_temp = x1[(x1 <= top1_low) & (x2 >= top2_high)]
        x2_temp = x2[(x1 <= top1_low) & (x2 >= top2_high)]
        y1_temp = y[(x1 <= top1_low) & (x2 >= top2_high)]
    elif type_of_x == "med_low":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high) & (x2 <= top2_low)]
        x2_temp = x2[(x1 >= top1_low) & (x1 <= top1_high) & (x2 <= top2_low)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high) & (x2 <= top2_low)]
    elif type_of_x == "med_med":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
        x2_temp = x2[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "med_hig":
        x1_temp = x1[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_high)]
        x2_temp = x2[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_high)]
        y1_temp = y[(x1 >= top1_low) & (x1 <= top1_high) & (x2 >= top2_high)]
    elif type_of_x == "hig_low":
        x1_temp = x1[(x1 >= top1_high) & (x2 <= top2_low)]
        x2_temp = x2[(x1 >= top1_high) & (x2 <= top2_low)]
        y1_temp = y[(x1 >= top1_high) & (x2 <= top2_low)]
    elif type_of_x == "hig_med":
        x1_temp = x1[(x1 >= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
        x2_temp = x2[(x1 >= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
        y1_temp = y[(x1 >= top1_high) & (x2 >= top2_low) & (x2 <= top2_high)]
    elif type_of_x == "hig_hig":
        x1_temp = x1[(x1 >= top1_high) & (x2 >= top2_high)]
        x2_temp = x2[(x1 >= top1_high) & (x2 >= top2_high)]
        y1_temp = y[(x1 >= top1_high) & (x2 >= top2_high)]
    y2_temp = y1_temp

    if (len(x1_temp) == 0) or (len(x2_temp) == 0) or (len(y1_temp) == 0):
        x1_temp, x2_temp, y1_temp, y2_temp = compute_error_tree_nodata(x1, x2, y, top1_low, top1_high, top2_low,
                                                                       top2_high, type_of_x)
    try:
        a1 = np.linalg.lstsq(x1_temp[:, np.newaxis], y1_temp)[0][0]
        a2 = np.linalg.lstsq(x2_temp[:, np.newaxis], y2_temp)[0][0]
        modifier1 = a1 / rate1
        modifier2 = a2 / rate2

        if np.isinf(modifier1) or np.isnan(modifier1):
            modifier1 = 1.0
        if np.isinf(modifier2) or np.isnan(modifier2):
            modifier2 = 1.0

        y1_forecast = modifier1 * rate1 * x1_temp
        y2_forecast = modifier2 * rate2 * x2_temp

        y1_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y1_forecast])
        y2_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y2_forecast])

        
        w=set_weight(y_dates)
        s1 = sMAPE_weight(y1_temp, y1_forecast, w/sum(w))
        s2 = sMAPE_weight(y2_temp, y2_forecast, w/sum(w))

        if min(s1, s2) == s1:
            y_forecast_tree = y1_forecast
            forecast_error_tree = s1
            modifier = modifier1
            source_name = x1.name
            priority = 1
        else:
            y_forecast_tree = y2_forecast
            forecast_error_tree = s2
            modifier = modifier2
            source_name = x2.name
            priority = 2
    except:
        y_forecast_tree = 0.0
        forecast_error_tree = 1.0
        modifier = 1.0
        source_name = x1.name
        priority = 1

    df_error = pd.DataFrame()
    df_error['Source'] = [source_name]
    df_error['Forecast_error'] = forecast_error_tree
    df_error['Priority'] = priority
    df_error['Modifier'] = modifier

    return y_forecast_tree, df_error


def v2_revenue_detection_tree(dict_df_tree, top2_df, y_series_tree, y_min, y_max, rate1, rate2):
#    y_series_tree = y_series_tree.reset_index(drop = True)
    col1 = top2_df[top2_df['Priority'] == 1]['Driver_name'].values[0]
    col2 = top2_df[top2_df['Priority'] != 1]['Driver_name'].values[0]

    lag1 = top2_df[top2_df['Priority'] == 1]['SourceOffset'].values[0]
    lag2 = top2_df[top2_df['Priority'] != 1]['SourceOffset'].values[0]

    #    print(seg_id, day_of_week, Day, season_tf_str, period_id, col1, lag1, col2, lag2)

    col1_series = dict_df_tree[str(lag1) + "_full"][col1]
    col1_series.index = dict_df_tree[str(lag1) + "_full"]['date']
    y1_series_r = y_series_tree[-len(col1_series):]
    col2_series = dict_df_tree[str(lag2) + "_full"][col2]
    col2_series.index = dict_df_tree[str(lag1) + "_full"]['date']
    y2_series_r = y_series_tree[-len(col2_series):]

    # check error vi df duoc lay tu giua nam nay den giua nam kia nen data co the khac length trong mot so season
    # check 1
    if len(y1_series_r) != len(col1_series):
        col1_series = col1_series[:-(len(col1_series) - len(y1_series_r))]
    if len(y2_series_r) != len(col2_series):
        col2_series = col2_series[:-(len(col2_series) - len(y2_series_r))]
        
    y1_series_date = y2_series_r.copy()
    col2_series_date = col2_series.copy()
    col1_series = col1_series.reset_index(drop = True)
    y1_series_r = y1_series_r.reset_index(drop = True)
    y2_series_r = y2_series_r.reset_index(drop = True)
    col2_series = col2_series.reset_index(drop = True)
    y1_series_rm, col1_series_rm = v2_remove_outlier(y1_series_r, col1_series)
    y2_series_rm, col2_series_rm = v2_remove_outlier(y2_series_r, col2_series)
    y1_series_date = y1_series_date[y1_series_date.isin(y2_series_rm)].index.tolist()
    col2_series_date = col2_series_date[col2_series_date.isin(col2_series_rm)].index.tolist()
# check 2
    temp_df1 = pd.DataFrame(list(zip(y1_series_rm, col1_series_rm)))
    temp_df1.columns = ['y1_series_rm', 'col1_series_rm']
    temp_df2 = pd.DataFrame(list(zip(y2_series_rm, col2_series_rm)))
    temp_df2.columns = ['y2_series_rm', 'col2_series_rm']
    try:
        temp_df2['date'] = y1_series_date
    except:
        temp_df2['date'] = col2_series_date
    temp_df = v2_intersect_join(temp_df1, temp_df2)
    y_both = temp_df['y_both']
    y1_series_date = temp_df['date'].tolist()
    col1_series_rm_both = temp_df['col1_series_rm']
    col2_series_rm_both = temp_df['col2_series_rm']

    # Check: y1_series_rm MUST BE EQUAL TO y2_series_rm
    x1 = col1_series_rm_both.rename(col1)
    x2 = col2_series_rm_both.rename(col2)
    y = y_both.rename(y_series_tree.name)
    y_forecast_tree_ll, df_error_ll = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'low_low', y1_series_date)
    y_forecast_tree_lm, df_error_lm = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'low_med', y1_series_date)
    y_forecast_tree_lh, df_error_lh = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'low_hig', y1_series_date)
    y_forecast_tree_ml, df_error_ml = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'med_low', y1_series_date)
    y_forecast_tree_mm, df_error_mm = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'med_med', y1_series_date)
    y_forecast_tree_mh, df_error_mh = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'med_hig', y1_series_date)
    y_forecast_tree_hl, df_error_hl = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'hig_low', y1_series_date)
    y_forecast_tree_hm, df_error_hm = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'hig_med', y1_series_date)
    y_forecast_tree_hh, df_error_hh = v2_compute_error_tree(y, y_min, y_max, x1, rate1, x2, rate2, 'hig_hig', y1_series_date)

    tree_df = pd.concat([df_error_ll, df_error_lm, df_error_lh, \
                         df_error_ml, df_error_mm, df_error_mh, \
                         df_error_hl, df_error_hm, df_error_hh], ignore_index=True)
    tree_df['Case'] = range(1, 10)

    tree_db = tree_df[['Case', 'Priority', 'Modifier']]
    tree_db = tree_db.to_dict(orient='records')
    tree_db = [dict([a, int(x)] for a, x in b.items()) for b in tree_db]
    tree_db_float = pd.DataFrame()
    tree_db_float['Modifier'] = tree_df['Modifier']
    tree_db_float = tree_db_float.to_dict(orient='records')
    for i in range(len(tree_db)):
        tree_db[i].update(tree_db_float[i])

    # Compute High Low threshold
    top1_low = float(np.percentile(x1, 33))
    top1_high = float(np.percentile(x1, 66))
    top2_low = float(np.percentile(x2, 33))
    top2_high = float(np.percentile(x2, 66))
    if top1_low > top1_high:
        top1_low = top1_high
    if top2_low > top2_high:
        top2_low = top2_high

    highlow_threshold = pd.DataFrame(columns=['top1_high', 'top1_low', 'top2_high', 'top2_low'])
    highlow_threshold.loc[0] = [top1_high, top1_low, top2_high, top2_low]

    return tree_db, highlow_threshold


def tree_validation_error(y, y_min, y_max, x1, x2, case_df, top2_df):
    x1_temp = x1.values[0]
    x2_temp = x2.values[0]
    top1_low = top2_df['LowThreshold'].values[0]
    top2_low = top2_df['LowThreshold'].values[1]
    top1_high = top2_df['HighThreshold'].values[0]
    top2_high = top2_df['HighThreshold'].values[1]

    # get type_of_x
    if (x1_temp < top1_low) & (x2_temp < top2_low):
        type_of_x = 1  # low_low
    elif (x1_temp < top1_low) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 2  # low_med
    elif (x1_temp < top1_low) & (x2_temp > top2_high):
        type_of_x = 3  # low_hig
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp < top2_low):
        type_of_x = 4  # med_low
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 5  # med_med
    elif (x1_temp >= top1_low) & (x1_temp <= top1_high) & (x2_temp > top2_high):
        type_of_x = 6  # med_hig
    elif (x1_temp > top1_high) & (x2_temp < top2_low):
        type_of_x = 7  # hig_low
    elif (x1_temp > top1_high) & (x2_temp >= top2_low) & (x2_temp <= top2_high):
        type_of_x = 8  # hig_med
    elif (x1_temp > top1_high) & (x2_temp > top2_high):
        type_of_x = 9  # hig_hig

    # ======================================================================================
    # get modifier and priority
    modifier = case_df[case_df['Case'] == type_of_x]['Modifier'].values[0]
    priority = case_df[case_df['Case'] == type_of_x]['Priority'].values[0]
    rate = top2_df[top2_df['Priority'] == priority]['Rate'].values[0]
    x_name = top2_df[top2_df['Priority'] == priority]['Driver_name'].values[0] + '_' + \
             str(top2_df[top2_df['Priority'] == priority]['Offset'].values[0])
    if priority == 1:
        x = x1_temp
    else:
        x = x2_temp
    y_forecast = [modifier * rate * x]
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    # ======================================================================================
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return y_forecast, forecast_error, x, x_name, modifier, rate, type_of_x


def v2_get_df_by_dates_offline(dates, df, date_col='date'):
    """
    Select rows in dataframe having date in dates
    :param dates: List date
    :param df: dataframe
    :param date_col: date column name in df
    :return: selected dataframe
    """
    tmp_df = df
    dates = [i.strftime('%Y-%m-%d') for i in dates]
    if 'date' != df.index.name:
        tmp_df = df.set_index('date')
    tmp_df = tmp_df.ix[dates].reset_index('date')
    return tmp_df