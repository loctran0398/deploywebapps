# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:38:01 2019

@author: cong-thanh
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import pickle
import ConfigParser
import argparse
import os, __main__, copy
import itertools
import calendar
import binascii
import time, json, requests
import math

from d2o_common.legacy_lib.database_api import Cruise_day
from datetime import timedelta, datetime
from d2o.revenue_lib import capture_arr
from d2o.utils import putils_revenue as put
from datetime import timedelta, datetime
from d2o_common.legacy_lib import database_api as db_api
from d2o_common.legacy_lib import database_api_v2 as db_api_v2

from d2o.utils import logger as log
from d2o.utils.handlers import ExceptionHandler
from d2o.utils.systools import makedirs
from d2o.utils.timeout_revenue import timeout
from d2o.revenue_lib import project_lib as lib

from d2o.revenue_lib import create_driver_df
from d2o.revenue_lib import department_correlation
import v2_revenue_lib_combine_otb2 as lib2
import validation_rev_lib as val_rev_lib2
import code_season_json as ss



def get_revenue_driver(list_adj_col_d, dep_period_df, dict_driver_df_db, dep_seg, hotel_id, season_tf_str, day_of_week):
    """
    Get top 3 driver from correlation result and old drivers in database
    :param list_adj_col_d: List driver with lag day
    :param dep_period_df: season data contain overlap mapping date between seasons
    :param dict_driver_df_db: driver dataframe from database
    :param dep_seg:
    :param hotel_id:
    :param season_tf_str:
    :param day_of_week:
    :return:
    """
    list_adj_col_d = list_adj_col_d.reset_index(drop=True)
    top3_driver = []
    first_driver_id = 0
    top3_driver = list_adj_col_d[first_driver_id:first_driver_id + 3]
    if len(top3_driver) == 2:
        top3_driver = top3_driver.append(pd.Series(top3_driver[1])).reset_index(drop=True)
    else:
        name1 = "_".join(top3_driver.values[0].split("_")[:1]) + "_" + top3_driver.values[0].split("_")[3]
        name2 = "_".join(top3_driver.values[1].split("_")[:1]) + "_" + top3_driver.values[1].split("_")[3]
        name3 = "_".join(top3_driver.values[2].split("_")[:1]) + "_" + top3_driver.values[2].split("_")[3]
        index = first_driver_id + 3
        while (name1 == name2) and (name1 == name3):
            top3_driver.values[2] = list_adj_col_d[index]
            name3 = "_".join(top3_driver.values[2].split("_")[:1]) + "_" + top3_driver.values[2].split("_")[3]
            index += 1
    return top3_driver

def def_driver_type(col, driver):
    """
    Return driver type (define in database)
    :param col: id of column
    :param driver: driver of this column
    :return: value of type
    """
    dep_seg_1 = col.split("_")[0] + "_" + col.split("_")[1]
    dep_seg_2 = driver.split("_")[0] + "_" + driver.split("_")[1]
    if dep_seg_1 == dep_seg_2:
        result = 5
    else:
        driver_type = driver.split("_")[2]
        if driver_type == "rn":
            result = 1
        elif driver_type == "gn":
            result = 2
        else:
            result = 3
    return result

def final_top(top_df):
    """
    Pre-process before save top 3 driver to database
    :param top_df: Top driver dataframe
    :return: Top driver dataframe after pre-process
    """
    new_df = top_df[top_df.columns]
    new_df = new_df.drop_duplicates()
    new_df = new_df.reset_index(drop=True)
    return new_df    
    
LOG_VERBOSITY = "debug"
PRINT_VERBOSITY = "info"


YEAR_BACK_DEFAULT = 2
model = "error"
significance_level = 0.05
num_day = 100
k_modifier = 1000
date_format = "%Y-%m-%d"


#filename = 'Thon_50_260.csv'
current_path = os.getcwd()
#hotel = 50

def forecast_revenue(filename,current_path,hotel_id, Mydep,segment):

    dep_id = int(Mydep)
    
    name_json = 'json_season_{}.json'.format(dep_id)
#    name_rev = 'Thon_{}_{}.csv'.format(hotel_id,dep_id)
    
    df = pd.read_csv(os.path.join(current_path,'input_data',filename))
    
#    df = put.impute(df, method='zero')
#    df = df.fillna(0)
#    df = df.where(df < 0.0, 0.0)
#    df[df < 0.0] = 0.0
    df= df.rename(columns = {df.iloc[:,0].name:'Date'})
    date_s = df['Date']
    del df['Date']
    
    with open(os.path.join(current_path,'out_put',name_json)) as json_file:
        data_json = json.load(json_file)
    
    rev_all_new_df = ss.convert_json_ss_outlier(data_json, filename, hotel_id, Mydep,segment, current_path)

    rev_all_new_df = rev_all_new_df[rev_all_new_df['Period_Type'] != 1]
    rev_all_new_df = rev_all_new_df[rev_all_new_df['Period_Type'] != 2]
    rev_all_new_df['Day'] = rev_all_new_df['Day'].astype('int')
    list_col = [col.lower() for col in df.columns]
#    for col in df.columns:
#        list_col.append("{0}_{1}_rv".format(col.split('_')[0],col.split('_')[1]))
    df.columns = list_col
    
    all_rv_col_without_hid_only_depid = [i for i in df.columns if (int(str(i.split("_")[0])) == dep_id)]
#    all_rv_col_without_hid_only_depid = ['260_0_rv']
    df = pd.concat([date_s, df], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    
    
    current_time = df['Date'].iloc[-1]
    
    print('START RUNNING FOR HOTEL')
    start_str = (current_time - timedelta(days=365 * YEAR_BACK_DEFAULT + 1)).strftime('%m-%d-%Y')
    
    end_str = (current_time - timedelta(days=0)).strftime('%m-%d-%Y')
    
    tf_input_raw = "%s, %s" % (start_str, end_str)
    tf_input = tf_input_raw.replace(" ", "")
    tf_input = tf_input.split(",")
    
    
    start_time_dep_id = time.time()
    print("dep_id", dep_id, type(dep_id))
    
    #              
    cruise = Cruise_day(current_time, rev_all_new_df, hotel_id,
                        2)  # Download 4 year cruise days into object cruise.
    temp_df = rev_all_new_df.copy()
    temp_day_of_week = temp_df['Day']
    df = df.merge(temp_df[['Date','Day']], on = 'Date', how = 'left')
    df = df.rename(columns = {'Day':'day_of_week'})
    
    start_time_dep_id_preprocess = time.time()
    df_full = df[df.columns]
    
    df_insample = df_full[(df_full['Date'] >= tf_input[0]) & (df_full['Date'] <= tf_input[1])]
    begin_time = df_insample['Date'].iloc[0]
    current_time = df_insample['Date'].iloc[len(df_insample) - 1]
    print(begin_time)
    print(current_time)
    days = (current_time - begin_time).days
    #                    if days
    new_otb_data = pd.DataFrame()
    Mydep = str(dep_id)
    df_insample_otb = df_insample.copy()
    
    print('Export df_insample and df_outsample to file csv.')
    print('Train time: from {} to {}.'.format(begin_time, current_time))
    if len(df_insample_otb) < 450:
        day_arr = np.datetime64(df_insample_otb['date'].iloc[0])
        day_1yrs_back = df_insample_otb['date'].iloc[0]
    else:
        day_arr = np.datetime64(current_time - timedelta(days=450))
        day_1yrs_back = current_time - timedelta(days=450)
    print('450 days back: ', day_arr)
    year_input = list(range(begin_time.year, current_time.year + 1))
    year_input_tree = list(range(day_1yrs_back.year, current_time.year + 1))
    
    print('year_input: {}, year_input_tree: {}.'.format(year_input, year_input_tree))
    print('==================================================\n')
    # ============GET OLD REVENUE DRIVER FROM DATABASE ==========
    dict_df_error_from_database = {}
    start_time_get_old_rev = time.time()
    
    print("--- %s seconds ---" % (time.time() - start_time_dep_id_preprocess))
    time_end_preprocess = time.time()
    start_time_corr = time.time()
    
    filled_list = []
    
    start_time = time.time()
    dict_df_error_v2 = {}
    dict_df_corr = {}
    dict_df_error_week = {}
    dict_df_corr_week = {}
    df_insample_otb['day_of_week'] = [i.weekday() for i in df_insample_otb['Date']]
                        # df_insample_otb['day_of_week'] = df['date'].dt.dayofweek
    
    begin_day = int(min(df_insample_otb['day_of_week'].dropna()))
    end_day = int(max(df_insample_otb['day_of_week'].dropna()))
    temp_df = temp_df.rename(columns = {'Groups': 'Period_Id'})
    temp_df = temp_df.rename(columns = {'Groups': 'Period_Id', 'Date' : 'date'})
    df_insample_otb = df_insample_otb.rename(columns = {'Date': 'date'})
    accuracy_df = 0
    
    for col in all_rv_col_without_hid_only_depid:
        dep_id_corr = col.split("_")[0]
        seg_id = col.split("_")[1]
    
    #                        start_time_col = time.time()
        try:
            l_season_tf = temp_df['Period_Id'][((temp_df['Period_Type'] == 0) & (temp_df['Day'] != 0))|(temp_df['Period_Type'] == 1)].drop_duplicates()  # period
        except Exception as e:
            continue
        for itemp, season_tf in enumerate(l_season_tf):
            key = "{0}_{1}".format(col, season_tf)
            date_ss = temp_df[temp_df['Period_Id'] == season_tf]['date'].reset_index(drop= True)
            end_day = int(max(temp_df['Day'][temp_df['Period_Id'] == season_tf]))
            begin_day = int(min(temp_df['Day'][temp_df['Period_Id'] == season_tf]))
            for day_of_week in xrange(begin_day, end_day + 1):
    #                                    print(season_tf,day_of_week)
                key = "{0}_{1}_{2}".format(col, season_tf, day_of_week)
                if model == "error":
                    # ver 2
                    temp_df_v2 = lib2.v2_compute_corr(dep_id_corr, seg_id,season_tf, num_day,
                                                      day_of_week, \
                                                      significance_level, df_insample_otb, date_ss,
                                                      day_arr, \
                                                      day_1yrs_back, cruise, accuracy_df, k_modifier)
                    dict_df_error_v2[key] = temp_df_v2
    
                if model == "regression" or model == "corr":
                    dict_df_corr[key] = department_correlation.compute_corr_ver_3_1(dep_id_corr,
                                                                                    seg_id,
                                                                                    season_tf,
                                                                                    num_day, \
                                                                                    day_of_week,
                                                                                    significance_level,
                                                                                    df_insample_otb,
                                                                                    date_ss)
                    
    season_df_str = l_season_tf
    feature_list = ["H_Id", "Period", "Day", "Source_H_Id", "Source_Segment_Id",
                                        "Destination_Segment_Id",
                                        "Priority", "Property", "Type", "SourceOffset", "DestinationOffset"]
    
    top_df_db_v2 = pd.DataFrame(0, columns=feature_list, index=range(len(dict_df_error_v2) * 3))
    
                                        
    index = 0
    dates_2016 = pd.date_range('01-01-2016', '31-12-2016')
    
    driver_df = pd.DataFrame(columns=['col', 'col_d', 'corr'], index=dates_2016)
    driver_df.index.name = 'date'
    
    print('\n==================================================')
    # season_df_str.to_csv('train_data/season_df_str_{}_{}.csv'.format(hotel_id, dep_id), index=False)
    # season_df_str.to_csv('validation_data/season_df_str_{}_{}.csv'.format(hotel_id, dep_id), index=False)
    print('Export season_df_str to file csv.')
    # ver 2
    index = 0
    
    for key, df_driver in dict_df_error_v2.iteritems():
        l_element = key.split("_")
        prefix = "_".join(l_element[-2:])
        season_tf_str = l_element[-2]
        day_of_week = int(l_element[-1])
    
        col = "_".join(l_element[:2])
        dep = int(l_element[0])
        seg = int(l_element[1])
        dep_seg_temp = "{}_{}".format(dep, seg)
        dep_period_df = season_tf_str
        top3_driver = get_revenue_driver(df_driver['adj_corr_col'], dep_period_df,
                                         dict_df_error_from_database, dep_seg_temp, hotel_id,
                                         season_tf_str, day_of_week)
        period = int(season_tf_str)
    
        # insert to df
        for i, driver in enumerate(top3_driver):
            col_d_driver = "_".join(driver.split("_")[:3])
            lag = int(driver.split("_")[-1])
    
            top_df_db_v2.loc[index, 'H_Id'] = np.round(dep, 0).astype(int)
            top_df_db_v2.loc[index, 'Period'] = np.round(period, 0).astype(int)
            top_df_db_v2.loc[index, 'Day'] = np.round(day_of_week).astype(int)
            top_df_db_v2.loc[index, 'Source_H_Id'] = np.round(int(driver.split("_")[0])).astype(int)
            top_df_db_v2.loc[index, 'Source_Segment_Id'] = np.round(int(driver.split("_")[1])).astype(
                int)
            top_df_db_v2.loc[index, 'Destination_Segment_Id'] = np.round(seg).astype(int)
            top_df_db_v2.loc[index, 'Priority'] = np.round(i + 1).astype(int)
            df_temp_lag = df_driver[df_driver['cov_col'] == col_d_driver]
            error = df_temp_lag[df_temp_lag['lag'] == lag]['cov_value'].values[0]
            # top_df_db.loc[index, 'property'] = df_driver[df_driver['cov_col'] == col_d_driver]['cov_value'].values[0]
            top_df_db_v2.loc[index, 'Property'] = error
            top_df_db_v2.loc[index, 'Type'] = np.round(def_driver_type(col, driver)).astype(int)
            top_df_db_v2.loc[index, 'SourceOffset'] = np.round(int(driver.split("_")[-1])).astype(int)
            top_df_db_v2.loc[index, 'DestinationOffset'] = 0
            index += 1
    
    # ================================================================
    # ver 2
    
    top_df_db_v2 = final_top(top_df_db_v2)  # remove duplicate rows
    
    key_list = ["H_Id", "Period", "Day", "Source_H_Id", "Source_Segment_Id", "Destination_Segment_Id",
                "Priority", "Type", "SourceOffset", "DestinationOffset"]
    
    # ver 2
    top_df_db_int_v2_single = top_df_db_v2[key_list]
    top_df_db_float = top_df_db_v2[['Property']]
    for i in range(len(top_df_db_float)):
        if top_df_db_float['Property'][i] < 0.1:
            top_df_db_float['Property'][i] = 0.1 * (top_df_db_float['Property'][i] + 99) / 99.1
    
    top_df_db_int_v2_single = top_df_db_int_v2_single.to_dict(orient='records')
    top_df_db_int_v2_single = [dict([a, int(x)] for a, x in b.iteritems()) for b in
                               top_df_db_int_v2_single]
    top_df_db_float = top_df_db_float.to_dict(orient='records')
    for i in range(len(top_df_db_int_v2_single)):
        top_df_db_int_v2_single[i].update(top_df_db_float[i])
    
    print('Export top_df_db_int_v2_single to csv file.')
    print('==================================================\n')
    
    top3_df_v2 = pd.DataFrame(top_df_db_int_v2_single)
    temp_df['Date'] = temp_df['date']
    
    
    top_db_end_v2, db_lag7, acc_summary = lib2.v2_revenue_detection_cruise(current_time, day_1yrs_back,
                                                                                               hotel_id, dep_id, \
                                                                                               num_day, significance_level,
                                                                                               year_input, year_input_tree, \
                                                                                               df_insample_otb, top3_df_v2,
                                                                                               day_arr,
                                                                                               k_modifier, cruise, temp_df)
    
    top3_df_v2.to_csv('top3_df_v2_260.csv')
    with open(os.path.join(current_path,'out_put','top_db_end_v2_{}_{}.json'.format(hotel_id,Mydep)), 'w') as fp:
        json.dump(top_db_end_v2, fp)
    
#    with open(os.path.join(current_path,'out_put','top_db_end_v2_{}_{}.json'.format(hotel_id,Mydep))) as json_file:
#        top_db_end_v2 = json.load(json_file)
        
    client = 'Thon'
    years_back = YEAR_BACK_DEFAULT
    df_insample['date'] = df_insample_otb['date']
    df_insample['Date'] = df_insample_otb['date']
    df_insample_otb['Date'] = df_insample_otb['date']
    
    del df_insample['Date']
    del df_insample_otb['Date']
    del temp_df['date']
    

    


    #df_insample.index = df_insample_otb['date']
    validation = val_rev_lib2.validation_file_export(client,hotel_id, dep_id,df_insample, df_insample_otb,years_back, current_time, cruise,top_db_end_v2,all_rv_col_without_hid_only_depid)
    
#    time_end_model2 = time.time()
#    print('Average accuracy max train: {} .'.format(np.mean(acc_summary['Acc_max_train'])))
#    total_acc += np.mean(acc_summary['Acc_max_train'])
#    num_dep += 1.0
#    # runnning time of models
#    print('================================= PRINT PERFORMANCE =================================')
#    runtime_total_v2 = time_end_model2 - start_time_dep_id
#    print("DATABASE %s __ HOTEL %s __ DEP ID %s __ RUNNING TIME OF MODEL VER 2 __ %s" \
#          % (client, hotel_id, dep_id, runtime_total_v2))
#    
#    # data for summary
#    summary_acc_df = summary_acc_df.append(acc_summary, ignore_index=True)
#    summary_runtime_df.loc[id_sum, 'Driver'] = '{}_{}'.format(hotel_id, dep_id)
#    summary_runtime_df.loc[id_sum, 'Runtime_v2'] = runtime_total_v2
#    id_sum += 1
    return validation