from d2o_common.legacy_lib import database_api as db_api
from d2o_common.legacy_lib.utils import logger as log
from d2o_common.legacy_lib.prep.scaling import Scaler
import d2o_common.legacy_lib.utils.putils as put
import weekly_cycle as wc

#from collections import OrderedDict
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import dateutil.relativedelta as relativedelta
import dateutil.rrule as rrule
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import seaborn as sns
import constrained_kmeans as ckmeans
#import calendar
import time

from scipy.stats import norm as Gaussian
import json
import requests
import warnings
import holidays

#HOST = 'http://10.50.10.4:8485' #production
HOST = 'http://172.16.0.51:8485' #test
#HOST = db_api.HOST
REVENUE_LINK_ONE_DEPT   = "%s/Revenue/Department/Days/{client_id}/?h_id={h_id}&total=true&segment=true&from={from_date}&to={to_date}" % (HOST)
LABOR_LINK_ONE_DEPT     = "%s/Labor/Department/Days/{client_id}/?h_id={h_id}&total=true&segment=true&from={from_date}&to={to_date}" % (HOST)
FOODCOST_LINK_ONE_DEPT  = "%s/FoodCost/Analysis/{client_id}/?h_id={h_id}&from={from_date}&to={to_date}" % (HOST)
SPECIAL_DATES_LINK       = '%s/Season/SpecialDates/{client_id}/&h_id={h_id}' % (HOST)

#===================== write log =====================
def log_json_upload(client, log_file_client):
    log_file_client['Type'] = 2
    log_file_client = [log_file_client]
    url = '%s/PMIAuto/Log/%s' % (db_api.HOST, client)
    db_api.post_api(link=url, json_data=log_file_client)
    log.info("Writing v2 season log to API")
    print("Pass send v2 log to API")
    
#===================== Download & Upload data =====================
def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)

def post_api(link, json_data):
    result = requests.post(url=link, json=json_data)
    return result

def write_to_database(client, Mydep, ss_type, write, json_upload = None, default = True):
    '''
    If default is True, uploading default season.
    '''
    if write:
        success_str = 'Upload data sucessfully\n'
        unsuccess_str = 'Upload data unsucessfully\n'
        if (default is True) or (json_upload is None):
            json_upload = create_default_season(Mydep, ss_type)
            success_str = 'Upload default data sucessfully\n'
            unsuccess_str = 'Upload default data unsucessfully\n'
            
        url = '%s/PMIAuto/Season/%s' % (HOST, client)    
        response = post_api(link = url, json_data = json_upload)
        if ss_type == 0:
            print("Writing Revenue Season to API: %s" % response.status_code)
        elif ss_type == 1:
            print("Writing Labor Season to API: %s" % response.status_code)
        else:
            print("Writing Foodcost Season to API: %s" % response.status_code)
            
        if response.status_code == 200:
            print(success_str)
        else:
            print(unsuccess_str)
            
    return json_upload

def create_default_season(Mydep, ss_type):
    json_default = {"H_Id": int(Mydep), 
                    "ValidHistoryFrom": "1900-01-01T00:00:00", 
                    "Type": int(ss_type), 
                    "Periods": [{
                            "Type": 0, 
                            "SpecialPeriod": 0, 
                            "Value3": 0, "Value2": 3, "Value1": 3, 
                            "Dates": [
                                    {"To": "2021-12-31T00:00:00", "From": "2021-01-01T00:00:00"}, 
                                    {"To": "2020-12-31T00:00:00", "From": "2020-01-01T00:00:00"}, 
                                    {"To": "2019-12-31T00:00:00", "From": "2019-01-01T00:00:00"}, 
                                    {"To": "2018-12-31T00:00:00", "From": "2018-01-01T00:00:00"}, 
                                    {"To": "2017-12-31T00:00:00", "From": "2017-01-01T00:00:00"}, 
                                    {"To": "2016-12-31T00:00:00", "From": "2016-01-01T00:00:00"}, 
                                    {"To": "2015-12-31T00:00:00", "From": "2015-01-01T00:00:00"}
                                     ]
                                }]
                   }
    return json_default

#===================== Get data =====================

def get_data(client, Mydep, years_back, ss_type, write, df):
    '''
    GET DATA
    Revenue: roomrevenue, roomnight
    Labor: productive hours, productivity
    Foodcost: foodrevenue, foodcost%    
    :param client: client ID
    :param Mydep: department ID 
    :param year_back: number of years for getting data
    :returns:
        mav_median_df: pre-processed data
        room_data: dataframe for computing outliers, holidays effect
    '''
    cycle_value = 0
    
#    years_back = len(df)/365
    
    if ss_type == 0:
        col_name = Mydep + '_0_RV'
        room_nights, room_revenue, ending_date = get_rn_rv_data(client, int(Mydep), years_back, df)
        if (room_nights is None) or (room_revenue is None):
            return None, None, col_name, cycle_value, ending_date

        # TODO: Create new a function

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Data (v.2)', 'Status': 20}
#        log_json_upload(client, log_file_client)

        room_nights = put.impute(room_nights, method='zero')
        room_nights[room_nights < 0.0] = 0.0
        room_nights.index.names = ['date']
#        room_nights = conditional_sum(room_nights)
        # room_nights = put.add_mav(room_nights, MAV_SIZE)
#        room_nights.columns = ['%s_RN' % x for x in room_nights.columns.values]
        room_nights_new = data_monday_to_sunday_transform(room_nights, years_back)
        
        room_revenue = put.impute(room_revenue, method='zero')
        room_revenue[room_revenue < 0.0] = 0.0
#        room_revenue = conditional_sum(room_revenue)
        # room_revenue = put.add_mav(room_revenue, MAV_SIZE)
#        room_revenue.columns = ['%s_RV' % x for x in room_revenue.columns.values]
        room_revenue_new = data_monday_to_sunday_transform(room_revenue, years_back)
        
        mav_median_df = put.combine([room_nights_new, room_revenue_new])

        # Have several segments, each segment may have room_night or not
        my_flag_plot = 0  # Department have unit data
        for my_rn_col in mav_median_df.columns:
            if '_0_' in my_rn_col and '_RN' in my_rn_col and sum(mav_median_df[my_rn_col]) <= 1: # total roomnight column
                my_flag_plot = 1  # If sum of total roomnight is zero, there is only revenue

        # If all value in mav_median_df[my_rn_col] is zero, replacing all value by 1
        if my_flag_plot == 1:
            for my_rn_col in mav_median_df.columns:
                if '_RN' in my_rn_col:
                    mav_median_df[my_rn_col] = 1

        mav_rate_median_df = convert_to_rates(mav_median_df)
        mav_median_df = put.combine([mav_median_df, mav_rate_median_df])
        mav_median_df = change_number(mav_median_df)
        mav_median_df = mav_median_df.replace([np.inf, -np.inf], np.nan)
        mav_median_df = mav_median_df.fillna(0.0)

#        mav_median_df.rename(columns={
#            u'%s_0_RN' % (Mydep): 'Room Nights',
#            u'%s_0_RV' % (Mydep): 'Revenue',
#            u'%s_0_RT' % (Mydep): 'Rate'
#            }, inplace=True)
        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting Seasons (v.2)', 'Status': 30}
#        log_json_upload(client, log_file_client)
        return mav_median_df, room_revenue, col_name, cycle_value, ending_date
    
    elif ss_type == 1:
        col_name = 'Productive'
        labor_data, ending_date = get_labor_data(client, Mydep, years_back)
        if labor_data is None:
            return None, None, col_name, cycle_value, ending_date

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Labor Data (v.2)', 'Status': 20}
        log_json_upload(client, log_file_client)

        data_df2 = labor_data[["Productive", "Productivity"]].copy()
#        data_df2.index = labor_data.index

        # TODO: Create new a function
        labor = put.impute(data_df2, method='zero')
        labor[labor < 0.0] = 0.0
        labor.index.names = ['date']
#        labor = conditional_sum(labor)
        labor_new = data_monday_to_sunday_transform(labor, years_back)
        labor_new = labor_new.replace([np.inf, -np.inf], np.nan)
        mav_median_df = labor_new.fillna(0.0)

        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting labor seasons', 'Status': 30}
        log_json_upload(client, log_file_client)

        return mav_median_df, labor, col_name, cycle_value, ending_date
    
    elif ss_type == 2:
        col_name = 'FoodRevenue'
        food_data, ending_date = get_food_data(client, Mydep, years_back)
        if food_data is None:
            return None, None, col_name, cycle_value, ending_date

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Foodcost Data (v.2)', 'Status': 20}
        log_json_upload(client, log_file_client)

        food_data['weekday'] = [xdate.weekday() for xdate in food_data.index]
        weekday_begin = min(map(int, food_data['weekday'].drop_duplicates().values))
        weekday_end = max(map(int, food_data['weekday'].drop_duplicates().values))
        food_data = wc.weekly_cycle_of_hotel(food_data,weekday_begin,weekday_end)
        
        foodpur = food_data['Actual_Purchase']
        # foodpur lag
        foodpur_lag_df = pd.concat([foodpur.shift(3), foodpur.shift(2), foodpur.shift(1)], axis=1)
        foodpur_lag_df.columns = ['Purchase_Lag' + str(x) for x in range(3,0,-1)]
        # foodpur lead
        foodpur_lead_df = pd.concat([foodpur.shift(-1), foodpur.shift(-2), foodpur.shift(-3)], axis=1)
        foodpur_lead_df.columns = ['Purchase_Lead' + str(x) for x in range(1,4)]
        
        df_all_data = pd.concat([foodpur_lag_df, foodpur, foodpur_lead_df], axis=1)
        df_all_data = df_all_data.dropna().mean(axis=1)
        
        foodrev = food_data.loc[df_all_data.index, 'Actual_FoodRevenue']
        foodrev = foodrev.rename('FoodRevenue')
        foodcost = df_all_data / foodrev
        foodcost = foodcost.rename('FoodCost')
        
        food_df = pd.concat([foodrev, foodcost], axis = 1)
        foodrev = pd.DataFrame(foodrev)
        
        food = put.impute(food_df, method='zero')
        food[food < 0.0] = 0.0
        food.index.names = ['date']
#        food = conditional_sum(food)
        food_new = data_monday_to_sunday_transform(food, years_back)
        food_new = food_new.replace([np.inf, -np.inf], np.nan)
        mav_median_df = food_new.fillna(0.0)
        
        cycle_value = (food_data.loc[mav_median_df.index[-1], 'cycle_day'] + 1) % 7

        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting foodcost seasons', 'Status': 30}
        log_json_upload(client, log_file_client)
        
        return mav_median_df, foodrev, col_name, cycle_value, ending_date

def has_data(df, col_name):
    """
    Check whether df has enough data or not
    :param df: dataframe
    :return: True or False
    """
#    REQ_LEN = 364
    REQ_PERCENT_DATA = 80

#    has_year = len(df) >= REQ_LEN
#    if (not has_year):
#      return False

    stats = db_api.statistics(df)
    if (stats[col_name]['percent_nan'] > REQ_PERCENT_DATA or stats[col_name]['percent_zero'] > REQ_PERCENT_DATA):
      return False
    return True

#def has_data(df, col_name):
#    """
#    Check whether df has enough data or not
#    :param df: dataframe
#    :return: True or False
#    """
#    REQ_PERCENT_DATA = 80
#
#    stats = db_api.statistics(df)
#    if (stats.loc[:, col_name]['percent_nan'] > REQ_PERCENT_DATA or stats.loc[:, col_name]['percent_zero'] > REQ_PERCENT_DATA):
#      return False
#    return True

#mm
def fill_missing_date(df):
    '''
    :param df: dataframe contain date column 
    :returns:
        df: dataframe with full date between begin date and end date in df
    '''
    temp_df = pd.Series(df.ix[0, 'date'])
    date_df = df.copy()
    date_df['numdays'] = df['date'].diff()
    for i in range(1, len(date_df)):
        if date_df.ix[i, 'numdays'].days > 1:
            date_list = [date_df.ix[i, 'date'] - timedelta(days = x) for x in range(0, date_df.ix[i, 'numdays'].days)]
        else:
            date_list = [date_df.ix[i, 'date']]
        temp_df = pd.concat([temp_df, pd.Series(date_list)])
    temp_df.name = 'date'
    temp_df = temp_df.drop_duplicates()
    df = df.merge(pd.DataFrame(temp_df), how = 'right')
    df = df.sort_values('date')
    df.index = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in df['date']]
    df = df.fillna(0.0)
    return df

#mm
def create_dates_between_time(from_date, to_date):
    '''
    create date list between begin date and end date
    :param from_date: begin date in list
    :param to_date: end date in list
    :returns:
        date_list: date list between from_date and to_date
    '''
    numdays = (to_date - from_date).days
    date_list = [to_date - timedelta(days = x) for x in range(0, numdays)]
    return date_list

#def get_rn_rv_data(client, hotel_id, years_back):
#    '''
#    GET DATA
#    Revenue: roomrevenue, roomnight   
#    :param client: client ID
#    :param hotel_id: hotel ID 
#    :param year_back: number of years for getting data
#    :returns:
#        room_nights: pre-processed data
#        room_revenue: pre-processed data 
#    '''
#    start_time = time.time()
#    to_date = (datetime.now() - timedelta(days = 2)).date()
#    year_add = - years_back
#    from_date = db_api.add_years(to_date, year_add) - timedelta(days = 14)
#    
#    try:
#        link = REVENUE_LINK_ONE_DEPT.format(client_id=client, h_id=hotel_id, from_date=str(from_date), to_date=str(to_date))
#        df = request_data(link)
#        df = db_api.pivot_raw_data(df)
#        df['date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S').date() for i in df.index]
#    except:
#        log.err("DATABASE %s __ ID %s __ Could not get RN & RV data for this ID" % (client, hotel_id))
#        return None, None, to_date
#    
#    date_list = create_dates_between_time(from_date, to_date)
#    date_df = pd.DataFrame(date_list, columns = ['date'])
#    df = df.merge(pd.DataFrame(date_df), how = 'right')
#    df = df.sort_values('date')
#    df.index = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in df['date']]
#    df = df.fillna(0.0)
#    
#    print('Get revenue data for ID %s __ %s seconds\n' % (hotel_id, time.time() - start_time))
#    del df['date']
#    df.index = pd.to_datetime(df.index)
#
#    df_col = [col for col in list(df) if col.startswith(str(hotel_id))]
#    df = df[df_col]
#    rn_col = [col for col in list(df) if col.endswith('rn')]
#    rv_col = [col for col in list(df) if col.endswith('rv')]
#
#    room_nights = df[rn_col]
#    room_revenue = df[rv_col]
#    room_nights.columns = [col.replace('_rn', '') for col in room_nights.columns]
#    room_revenue.columns = [col.replace('_rv', '') for col in room_revenue.columns]
#    
#    return room_nights, room_revenue, to_date
def get_rn_rv_data(client, hotel_id, years_back, df):
    '''
    GET DATA
    Revenue: roomrevenue, roomnight   
    :param client: client ID
    :param hotel_id: hotel ID 
    :param year_back: number of years for getting data
    :returns:
        room_nights: pre-processed data
        room_revenue: pre-processed data 
    '''
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        df = df.rename(columns = {df.iloc[:,0].name : 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    rv_col = [i for i in df.columns if ('RV' in i)]
    room_revenue = df[rv_col]
#    try:   
#        rn_col = [i for i in df.columns if ('RV' in i)]
#        room_nights = df[rn_col]
#    except:
    rn_col = [i.replace('RV','RN') for i in rv_col]
    room_nights = pd.DataFrame(np.zeros(shape =(df.shape[0],len(rn_col))), columns = rn_col)
    room_nights.index = df['Date']
    to_date = df['Date'].iloc[-1]
    return room_nights, room_revenue, to_date


    
def get_labor_data(client, hotel_id, years_back):
    '''
    GET DATA
    Labor
    :param client: client ID
    :param hotel_id: hotel ID 
    :param year_back: number of years for getting data
    :returns:
        labor_df: pre-processed data
    '''
    start_time = time.time()

    to_date = (datetime.now() - timedelta(days = 2)).date()
    year_add = - years_back
    from_date = db_api.add_years(to_date, year_add) - timedelta(days = 14)
    
    try:
        link = LABOR_LINK_ONE_DEPT.format(client_id=client, h_id=hotel_id, from_date=str(from_date), to_date=str(to_date))
        labor_df = request_data(link)
        labor_df['date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S').date() for i in labor_df['Date']]
    except:

        log.err("DATABASE %s __ ID %s __ Could not get labor data for this ID" % (client, hotel_id))
        return None, to_date
    
    date_list = create_dates_between_time(from_date, to_date)
    date_df = pd.DataFrame(date_list, columns = ['date'])
    labor_df = labor_df.merge(pd.DataFrame(date_df), how = 'right')
    labor_df = labor_df.sort_values('date')
    labor_df.index = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in labor_df['date']]
    labor_df = labor_df.fillna(0.0)
    
    print('Get labor data for ID %s __ %s seconds\n' % (hotel_id, time.time() - start_time))
    del labor_df['date']
    labor_df.index = pd.to_datetime(labor_df['Date'])
    labor_df.drop('Date', axis=1, inplace=True)
    
    return labor_df, to_date

def get_food_data(client, hotel_id, years_back):
    '''
    GET DATA
    FOOD
    :param client: client ID
    :param hotel_id: hotel ID 
    :param year_back: number of years for getting data
    :returns:
        food_df: pre-processed data
    '''
    start_time = time.time()
    to_date = (datetime.now() - timedelta(days = 2)).date()
    year_add = - years_back
    from_date = db_api.add_years(to_date, year_add) - timedelta(days = 14)

    try:
        link = FOODCOST_LINK_ONE_DEPT.format(client_id=client, h_id=hotel_id, from_date=str(from_date), to_date=str(to_date))
        food_df = request_data(link)
        food_df['date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S').date() for i in food_df['Date']]
    except:
        log.err("DATABASE %s __ ID %s __ Could not get foodcost data for this ID" % (client, hotel_id))
        return None, to_date
    
    date_list = create_dates_between_time(from_date, to_date)
    date_df = pd.DataFrame(date_list, columns = ['date'])
    food_df = food_df.merge(pd.DataFrame(date_df), how = 'right')
    food_df = food_df.sort_values('date')
    food_df.index = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in food_df['date']]
    food_df = food_df.fillna(0.0)
    
    print('Get foodcost data for ID %s __ %s seconds\n' % (hotel_id, time.time() - start_time))
    del food_df['date']
    food_df.index = pd.to_datetime(food_df['Date'])
    food_df.drop('Date', axis=1, inplace=True)
    
    return food_df, to_date
            
def data_monday_to_sunday_transform(dataframe, years_back):
    '''
    begin-date of dataframe is Monday and end-date of dataframe is Sunday
    :param dataframe: index of dataframe is date 
    :param year_back: number of years for getting data
    :returns:
        dataframe_new: data always has enough day in week
    '''
    k = 0
    year = dataframe.index[-1].year - years_back
    weekday_end = dataframe.index[-1].weekday()
    if weekday_end != 6:
        k = k - weekday_end - 1
    date_end = dataframe.index[k-1]
    weeknumber_end = int(date_end.strftime("%U")) + 1 #Get more than 7 days with needed data, +1 if using isocalendar
    date_begin, _ = get_start_end_dates(year, weeknumber_end)
    while (date_begin < dataframe.index[0]) and (year < dataframe.index[-1].year):
        date_begin, _ = get_start_end_dates(year, weeknumber_end)
        year +=  1
        
    dataframe_new = dataframe[(dataframe.index >= date_begin) & (dataframe.index <= date_end)]
    if len(dataframe_new) < 364:
        print('Data has less than 52 weeks for this ID')
    
    return dataframe_new

def get_start_end_dates(year, week):
     d = datetime(year,1,1)
     if(d.weekday()<= 3):
         d = d - timedelta(d.weekday())             
     else:
         d = d + timedelta(7-d.weekday())
     dlt = timedelta(days = (week-1)*7)
     return d + dlt,  d + dlt + timedelta(days=6)
 
def conditional_sum(df):
    '''
    Fill hotel data with sum columns of departments
    :param df: Hotel data (dataframe)
    :return: Hotel data with sum columns for all departments (dataframe)
    '''
    new_df = df
    main_id = np.unique(main_id_list(new_df))
    id_list = main_id_list(new_df)
    special_character = '_0'
    for i in range(len(main_id)):
        item = main_id[i]  # start_str = (current_time - timedelta(days = 365 * years_back + 1)).strftime('%m-%d-%Y')

        full_item = item + special_character
        if full_item in list(df.columns.values):
            new_df[item + '_0'] = new_df[item + '_0']
        elif id_list.count(item) > 1:
            new_df[item + '_0'] = new_df.filter(regex=item).sum(axis=1) - new_df.filter(regex=full_item).sum(axis=1)
        else:
            new_df[item + '_0'] = new_df.filter(regex=item).sum(axis=1)
    return new_df
    
def main_id_list(df):
    '''
    Get the list of departments's id
    :param df: Hotel data
    :return: List of departments's id
    '''
    id_list = list()
    df_col_names = list(df.columns.values)  # [1:]
    special_character = '_'

    for i in range(len(df_col_names)):
        special_character_pos = df_col_names[i].find(special_character, 0)
        new_col_names = df_col_names[i][:special_character_pos]
        id_list.append(new_col_names)
    return id_list

def convert_to_rates(data):
    '''
    create dataframe containing Rate Revenue columns
    :param data: dataframe contain room revenue and room night columns of department, segment
    :return: dataframe contains Rate Revenue columns
    '''
    rate_revenue = dict()

    for column in set(['_'.join(x.split('_')[:-1]) for x in data.columns.values]):
        revenue = data[column + '_RV']
        room_nights = data[column + '_RN'].replace([0.0, np.nan, np.inf, -np.inf], 1)
        rate_revenue[column + '_RT'] = revenue / room_nights
    return pd.DataFrame.from_dict(rate_revenue)

def change_number(df):
    df_col_names = list(df.columns.values)[1:]
    prefix = R_prefix_list(df)
#    suffix = R_suffix_list(df)

    for i in range(len(df_col_names)):
        col_RN = prefix[i] + '_RN'
        col_RV = prefix[i] + '_RV'
        if np.sum(df[col_RN]) == float(len(df[col_RN])):
            df[col_RN] = df[col_RV]
        else:
            df[col_RN] = df[col_RN]
    return df

def R_prefix_list(df):
    '''
    Get the prefix_list of data
    :param df: MAV(30) of Revenue/Roomnights of department (ex: '224_0_MAV(30)_RV')
    :return: prefix_list of data (ex:'224_0_MAV(30)')
    '''
    R_prefix_list = list()
    df_col_names = list(df.columns.values)[1:]
    R_character = '_R'
    for i in range(len(df_col_names)):
        R_character_pos = df_col_names[i].find(R_character, 0)
        R_prefix = df_col_names[i][:R_character_pos]
        R_prefix_list.append(R_prefix)
    return R_prefix_list

#===================== Get cruise data =====================

def get_data_cruise(client, Mydep, years_back, ss_type, write):
    '''
    GET CRUISE DATA
    Revenue: roomrevenue, roomnight
    Labor: productive hours, productivity
    Foodcost: foodrevenue, foodcost%
    :param client: client ID
    :param Mydep: department ID
    :param year_back: number of years for getting data
    :returns:
        mav_median_df: pre-processed data
        room_data: dataframe for computing outliers, holidays effect
    '''
    cycle_value = 0

    if ss_type == 0:
        col_name = Mydep + '_0_RV'
        room_nights, room_revenue, ending_date = get_rn_rv_data(client, int(Mydep), years_back)
        if (room_nights is None) or (room_revenue is None):
            return None, None, col_name, cycle_value, ending_date

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Data (v.2)', 'Status': 20}
        log_json_upload(client, log_file_client)


        # TODO: Create new a function
        room_nights = put.impute(room_nights, method='zero')
        room_nights[room_nights < 0.0] = 0.0
        room_nights.index.names = ['date']
        room_nights = conditional_sum(room_nights)
        # room_nights = put.add_mav(room_nights, MAV_SIZE)
        room_nights.columns = ['%s_RN' % x for x in room_nights.columns.values]
        # room_nights_new = data_monday_to_sunday_transform(room_nights, years_back)
        room_nights_new = room_nights

        room_revenue = put.impute(room_revenue, method='zero')
        room_revenue[room_revenue < 0.0] = 0.0
        room_revenue = conditional_sum(room_revenue)
        # room_revenue = put.add_mav(room_revenue, MAV_SIZE)
        room_revenue.columns = ['%s_RV' % x for x in room_revenue.columns.values]
        # room_revenue_new = data_monday_to_sunday_transform(room_revenue, years_back)
        room_revenue_new = room_revenue

        mav_median_df = put.combine([room_nights_new, room_revenue_new])

        # Have several segments, each segment may have room_night or not
        my_flag_plot = 0  # Department have unit data
        for my_rn_col in mav_median_df.columns:
            if '_0_' in my_rn_col and '_RN' in my_rn_col and sum(
                    mav_median_df[my_rn_col]) <= 1:  # total roomnight column
                my_flag_plot = 1  # If sum of total roomnight is zero, there is only revenue

        # If all value in mav_median_df[my_rn_col] is zero, replacing all value by 1
        if my_flag_plot == 1:
            for my_rn_col in mav_median_df.columns:
                if '_RN' in my_rn_col:
                    mav_median_df[my_rn_col] = 1

        mav_rate_median_df = convert_to_rates(mav_median_df)
        mav_median_df = put.combine([mav_median_df, mav_rate_median_df])
        mav_median_df = change_number(mav_median_df)
        mav_median_df = mav_median_df.replace([np.inf, -np.inf], np.nan)
        mav_median_df = mav_median_df.fillna(0.0)

        #        mav_median_df.rename(columns={
        #            u'%s_0_RN' % (Mydep): 'Room Nights',
        #            u'%s_0_RV' % (Mydep): 'Revenue',
        #            u'%s_0_RT' % (Mydep): 'Rate'
        #            }, inplace=True)

        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting Seasons (v.2)', 'Status': 30}
        log_json_upload(client, log_file_client)

        return mav_median_df, room_revenue, col_name, cycle_value

    elif ss_type == 1:
        col_name = 'Productive'
        labor_data = get_labor_data(client, Mydep, years_back)
        if labor_data is None:
            return None, None, col_name, cycle_value, ending_date

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Labor Data (v.2)', 'Status': 20}
        log_json_upload(client, log_file_client)

        data_df2 = labor_data[["Productive", "Productivity"]].copy()
        #        data_df2.index = labor_data.index

        # TODO: Create new a function
        labor = put.impute(data_df2, method='zero')
        labor[labor < 0.0] = 0.0
        labor.index.names = ['date']
        #        labor = conditional_sum(labor)
        labor_new = data_monday_to_sunday_transform(labor, years_back)
        labor_new = labor_new.replace([np.inf, -np.inf], np.nan)
        mav_median_df = labor_new.fillna(0.0)

        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting labor seasons', 'Status': 30}
        log_json_upload(client, log_file_client)

        return mav_median_df, labor, col_name, cycle_value, ending_date

    elif ss_type == 2:
        col_name = 'FoodRevenue'
        food_data = get_food_data(client, Mydep, years_back)
        if food_data is None:
            return None, None, col_name, cycle_value, ending_date

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Processing Foodcost Data (v.2)', 'Status': 20}
        log_json_upload(client, log_file_client)

        df_weekday = cruise_df_dow_weeknumber(client, Mydep)
        df_weekday = df_weekday[['Date', 'weekday']]    

        weekday_begin = min(df_weekday['weekday'])
        weekday_end = max(df_weekday['weekday'])
        
        nb_days = (weekday_end-weekday_begin + 1)
        
#        food_data['weekday'] = [date.weekday() for date in food_data.index]
        food_data = food_data.reset_index()
        food_data = food_data.merge(df_weekday, on = 'Date')    
        
        start_ind = food_data[food_data['weekday'] == weekday_begin].index[0]
        end_ind = food_data[food_data['weekday'] == weekday_end].index[-1]

        food_data = food_data.iloc[start_ind:(end_ind+1), :]
        
        food_data = food_data.set_index('Date')                           
        food_data = wc.weekly_cycle_of_hotel(food_data, weekday_begin, weekday_end)
        
        foodpur = food_data['Actual_Purchase']
        # foodpur lag
        foodpur_lag_df = pd.concat([foodpur.shift(3), foodpur.shift(2), foodpur.shift(1)], axis=1)
        foodpur_lag_df.columns = ['Purchase_Lag' + str(x) for x in range(3, 0, -1)]
        # foodpur lead
        foodpur_lead_df = pd.concat([foodpur.shift(-1), foodpur.shift(-2), foodpur.shift(-3)], axis=1)
        foodpur_lead_df.columns = ['Purchase_Lead' + str(x) for x in range(1, 4)]

        df_all_data = pd.concat([foodpur_lag_df, foodpur, foodpur_lead_df], axis=1)
        df_all_data = df_all_data.dropna().mean(axis=1)

        foodrev = food_data.loc[df_all_data.index, 'Actual_FoodRevenue']
        foodrev = foodrev.rename('FoodRevenue')
        foodcost = df_all_data / foodrev
        foodcost = foodcost.rename('FoodCost')

        food_df = pd.concat([foodrev, foodcost], axis=1)
        foodrev = pd.DataFrame(foodrev)

        food = put.impute(food_df, method='zero')
        food[food < 0.0] = 0.0
        food.index.names = ['date']
        #        food = conditional_sum(food)
#        food_new = data_monday_to_sunday_transform(food, years_back)
        food_new = data_full_week_transform(food, df_weekday, weekday_begin, weekday_end)
        
        food_new = food_new.replace([np.inf, -np.inf], np.nan)
        mav_median_df = food_new.fillna(0.0)

        # cycle_value = (food_data.ix[-1, 'cycle_day'] + 1) % 7
        cycle_value = (food_data.loc[mav_median_df.index[-1], 'cycle_day'] + 1) % nb_days

        start_time_preprocess = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_preprocess)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Detecting foodcost seasons', 'Status': 30}
        log_json_upload(client, log_file_client)

        return mav_median_df, foodrev, col_name, cycle_value, ending_date

def get_weekday_data_cruise(client, Mydep, years_back):
    '''
    GET CRUISE DAY DATA
    Cruise Day usually has off-working day
    :param client: client ID
    :param dep_id: department ID
    :param year_back: number of years for getting data
    :returns:
        data: dataframe contains two Date and Day columns and has enough day in first week and last week
    '''
    cruise_day_data = get_cruise_day(client, Mydep, years_back)
    if len(cruise_day_data) == 0 :
        return None
    cruise_day_data['Date'] = pd.to_datetime(cruise_day_data['Date'], format='%Y-%m-%d')
    cruise_day_data['Day'] = cruise_day_data['Day'].astype('str')
    cruise_day_data = cruise_day_data.rename(columns={'Day': 'weekday'})
    cruise_day_data = begin_week1_end_week11(cruise_day_data)
    # weekday_begin = min(map(int, cruise_day_data['weekday'].drop_duplicates().values))
    # weekday_end = max(map(int, cruise_day_data['weekday'].drop_duplicates().values))
    return cruise_day_data

def get_cruise_day(client, dep_id, years_back):
    '''
    GET CRUISE DAY DATA
    Cruise Day usually has off-working day
    :param client: client ID
    :param dep_id: department ID
    :param year_back: number of years for getting data
    :returns:
        data: dataframe contains two Date and Day columns
    '''
    to_date = (datetime.now() - timedelta(days = 2)).date()
    year_add = - years_back
    from_date = db_api.add_years(to_date, year_add) - timedelta(days = 14)

    link = '%s/Hierarchy/CruiseDays/%s/?h_id_parent=%s&from=%s&to=%s' % (HOST, client, dep_id, \
                                                                         from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
    data = request_data(link)
    return data


def cruise_df_dow_weeknumber(client, Mydep):
    '''
    create CRUISE DAY, WEEK_NUMBER DATAFRAME of each date for 6 years
    Cruise Day usually has off-working day
    :param client: client ID
    :param Mydep: department ID
    :returns:
        data: dataframe contains columns such as: Date, weekday, weeknumber 
        after dropping off-working day in Date columns
    '''
    cruise_day_data = get_cruise_day_6year(client, Mydep)

    # if len(cruise_day_data) == 0:
    #     cruise_day_data = create_date_8year(client,Mydep)
    #     cruise_day_data['Date'] = pd.to_datetime(cruise_day_data['Date'], format='%Y-%m-%d')
    #     cruise_day_data = cruise_day_data.rename(columns={'Day': 'weekday'})
    #     weekday_begin = min(cruise_day_data['weekday'].drop_duplicates().values)
    #     weekday_end = max(cruise_day_data['weekday'].drop_duplicates().values)
    #
    # else:
    # generate 2 year of future
    cruise_day_data['Date'] = pd.to_datetime(cruise_day_data['Date'], format='%Y-%m-%d')

    # cruise_day_data['Day'] = cruise_day_data['Day'].astype('str')
    cruise_day_data = cruise_day_data.rename(columns={'Day': 'weekday'})
    weekday_begin = min(cruise_day_data['weekday'].drop_duplicates().values)
    weekday_end = max(cruise_day_data['weekday'].drop_duplicates().values)
    num_day_of_week = weekday_end - weekday_begin + 1
    today = pd.to_datetime(datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d'))
    pd.to_datetime(datetime.today(), format='%Y-%m-%d')
    if max(cruise_day_data['Date']) == today:
        if cruise_day_data['weekday'].iloc[-1] != weekday_end:
            wk_end_of_df_cruise = cruise_day_data['weekday'].iloc[-1]
            list_date_lack = [cruise_day_data['Date'].iloc[-1] + timedelta(days=x) for x in
                              range(1, num_day_of_week - wk_end_of_df_cruise + 1)]
            lack_date_of_df = pd.DataFrame()
            lack_date_of_df['Date'] = list_date_lack
            lack_date_of_df['weekday'] = [wk_end_of_df_cruise + i for i in
                                          range(1, num_day_of_week - wk_end_of_df_cruise + 1)]
            cruise_day_data = pd.concat([cruise_day_data, lack_date_of_df]).reset_index(drop=True)
            today = cruise_day_data['Date'].iloc[-1]

    num_week = int(np.ceil(365.0 / (weekday_end - weekday_begin + 1)))
    list_2year_future = [today + timedelta(days=x) for x in
                         range(1, num_week * (weekday_end - weekday_begin + 1) * 2 + 1)]
    df_2year_future = pd.DataFrame()
    df_2year_future['Date'] = list_2year_future
    df_2year_future['weekday'] = range(weekday_begin, weekday_end + 1) * num_week * 2
    cruise_day_data = pd.concat([cruise_day_data, df_2year_future]).reset_index(drop=True)

    # begin_date = cruise_day_data['Date'].iloc[-1]+ timedelta(days = 1)
    # end_date = cruise_day_data['Date'].iloc[-1] + timedelta(days = num_week*2*(weekday_end - weekday_begin + 1))
    # dd = [begin_date + timedelta(days=x) for x in range((end_date - begin_date).days + 1)]
    # data_frame_date = pd.DataFrame()
    # data_frame_date['Date'] = dd
    # data_frame_date['weekday'] =  range(weekday_begin,weekday_end + 1)*num_week*2
    # cruise_day_data = pd.concat([cruise_day_data,data_frame_date]).reset_index(drop = True)

    if ((pd.to_datetime(str(datetime.now())[:10], format='%Y-%m-%d') - cruise_day_data['Date'].iloc[0]).days - 2) < (
            365 * 4):
        end_date = cruise_day_data['Date'].iloc[0] - timedelta(days=1)
        begin_date = cruise_day_data['Date'].iloc[0] - timedelta(
            days=int(num_week * 5 * (weekday_end - weekday_begin + 1)))
        dd = [begin_date + timedelta(days=x) for x in range((end_date - begin_date).days + 1)]
        data_frame_date = pd.DataFrame()
        data_frame_date['Date'] = dd
        data_frame_date['weekday'] = range(weekday_begin, weekday_end + 1) * num_week * 5
        cruise_day_data = pd.concat([data_frame_date, cruise_day_data]).reset_index(drop=True)

    cruise_day_data['month_day'] = ['{:02d}'.format(i.day) + '_' + '{:02d}'.format(i.month) for i in
                                    cruise_day_data['Date']]
    cruise_day_data['Year'] = [i.year for i in cruise_day_data['Date']]
    year = min(cruise_day_data['Year'].drop_duplicates().values)
    cruise_day_data = cruise_day_data[cruise_day_data['Date'] >= date(year + 1, 01, 01)]

    cruise_day_data = cruise_day_data.reset_index(drop=True)

    begin_index_wk = cruise_day_data[cruise_day_data['weekday'] == weekday_begin].index[0]
    cruise_day_data = cruise_day_data.iloc[begin_index_wk:, :].reset_index(drop=True)

    # cruise_day_data['Year'] = [i.year for i in cruise_day_data['Date']]
    year_test = cruise_day_data['Year'].drop_duplicates().values
    new_df = pd.DataFrame()

    for i in year_test:
        temp_df = cruise_day_data[cruise_day_data['Year'] == i]
        a = temp_df.index[0]
        b = temp_df.index[-1]
        if temp_df['weekday'].iloc[0] != weekday_begin:
            a = temp_df.index[0] + (weekday_end - temp_df['weekday'].iloc[0] + weekday_begin)
        # temp_df = cruise_day_data.iloc[a:,:]
        if temp_df['weekday'].iloc[-1] != weekday_end:
            b = temp_df.index[-1] + (weekday_end - temp_df['weekday'].iloc[-1] + weekday_begin)
        else:
            b = b + 1
        temp_df = cruise_day_data.iloc[a:b, :]
        temp_df['Year'] = i
        # temp_df.reset_index(drop = True)
        # temp_df['week_number'] = temp_df.index.values + 1

        for i in range(weekday_begin, weekday_end + 1):
            new_temp_df = temp_df[temp_df['weekday'] == i]
            new_temp_df = new_temp_df.reset_index(drop=True)
            new_temp_df['week_number'] = new_temp_df.index.values + 1
            new_df = pd.concat([new_df, new_temp_df])
    new_df = new_df.sort_values(['Date']).reset_index(drop=True)
    # new_df['new_week_number'] = [i.isocalendar()[1] for i in new_df['Date']]

    new_df.index = new_df['Date']
    # df_6year = df_6year.reset_index(drop=True)
    new_df_nan = put.impute(new_df, method='nan')

    lack_date_df = new_df_nan[new_df_nan.isnull().any(axis=1)]
    lack_date_list = lack_date_df.index
    del new_df_nan['month_day']
    bfill_new_df_nan = new_df_nan.fillna(method='bfill')
    bfill_new_df_nan = bfill_new_df_nan.rename(columns={'week_number': 'week_number_ffill'})
    bfill_new_df_nan['Date'] = bfill_new_df_nan.index
    new_df_nan = new_df_nan.fillna(method='ffill')
    new_df_nan['Date'] = new_df_nan.index
    new_df_nan = new_df_nan.merge(bfill_new_df_nan[['Date', 'week_number_ffill']], on='Date')
    new_df_nan.index = new_df_nan['Date']

    new_df_nan = new_df_nan.ix[lack_date_list.values, :].reset_index(drop=True)
    new_df_nan['Year'] = [i.year for i in new_df_nan['Date']]
    new_df_nan['Year'] = new_df_nan['Year'].astype('int')
    new_df_nan['week_number'] = new_df_nan['week_number'].astype('int')
    new_df_nan['week_number_ffill'] = new_df_nan['week_number_ffill'].astype('int')
    new_df_nan = new_df_nan[new_df_nan['Date'] > today - timedelta(days=365)]
    new_df_nan['Date'] = [i if i.year != (today.year - 1) else i + timedelta(days=365) for i in new_df_nan['Date']]
    # new_df_nan = new_df_nan.replace(today.year - 1,today.year)
    if len(new_df_nan) == 0:
        return new_df.reset_index(drop=True)
    else:
        if (new_df_nan['Date'].iloc[0] - new_df_nan['Date'].iloc[-1]).days == 1:
            new_df_nan['week_number'][new_df_nan['Year'] == today.year - 1] = new_df_nan['week_number'].iloc[-1]
            new_df_nan['week_number_ffill'][new_df_nan['Year'] == today.year - 1] = \
            new_df_nan['week_number_ffill'].iloc[-1]
            new_df_nan = new_df_nan.reset_index(drop=True)
            new_df_nan = new_df_nan.replace(new_df_nan['week_number'].iloc[0],
                                            new_df_nan['week_number'].iloc[-1]).reset_index(drop=True)
        new_df_nan = new_df_nan.replace(today.year - 1, today.year)
        new_df_nan = new_df_nan.reset_index(drop=True)
        new_df_nan['new_group'] = ['_'.join([str(new_df_nan['Year'].iloc[i]), str(new_df_nan['week_number'].iloc[i]),
                                             str(new_df_nan['week_number_ffill'].iloc[i])]) for i in new_df_nan.index]
        #    new_df_nan['new_group']
        #    new_df_nan = new_df_nan.sort_values(['Date'])
        #    new_df_nan = new_df_nan.sort_values[]

        new_df_nan['From'] = new_df_nan.groupby('new_group')['Date'].transform('min')
        new_df_nan['To'] = new_df_nan.groupby('new_group')['Date'].transform('max')

        del new_df_nan['Date']
        new_df_nan_from_to = new_df_nan.drop_duplicates().reset_index(drop=True)
        new_df_nan_from_to['nums_day'] = new_df_nan_from_to['To'] - new_df_nan_from_to['From']
        new_df_nan_from_to['nums_day'] = [str(i.days + 1) for i in new_df_nan_from_to['nums_day']]
        new_df_nan_from_to['new_group'] = new_df_nan_from_to[['new_group', 'nums_day']].apply(lambda x: '_'.join(x),
                                                                                              axis=1)
        new_df_nan_from_to = new_df_nan_from_to.sort_values(['From']).reset_index(drop=True)
        #    pd.Series((new_df_nan_from_to['To'].values - new_df_nan_from_to['From'].values)/(86400*10**9))

        # new_df_nan = new_df_nan.reset_index(drop = True)
        new_df_drop = new_df.copy()
        for j in range(0, 3):
            for index, i in new_df_nan_from_to['new_group'].iteritems():
                print ('off-working-date', i)
                if int(i.split('_')[0]) == today.year:
                    date_check = str(new_df_nan_from_to['From'][index].day) + '_' + str(
                        new_df_nan_from_to['From'][index].month)
                    try:
                        index_1 = new_df_drop[(new_df_drop['Year'] == int(i.split('_')[0]) + j) & (
                                    new_df_drop['week_number'] == int(i.split('_')[1])) & (
                                                          new_df_drop['weekday'] == int(
                                                      weekday_end - weekday_begin + 1))].index[0]
                    except:
                        continue
                    if today <= (index_1 + timedelta(days=int(i.split('_')[3]))):
                        if (date_check == '1_1') and (index_1.year != today.year):
                            index_1 = new_df_drop[(new_df_drop['Year'] == int(i.split('_')[0]) + j) & (
                                    new_df_drop['week_number'] == int(i.split('_')[2])) &
                                                  (new_df_drop['weekday'] == float(weekday_begin))].index[0]
                            print(index_1)
                            index_1 = index_1 - timedelta(days=1)
                        new_df_drop = new_df_drop.drop(new_df_drop[(new_df_drop['Date'] > index_1) & (
                                    new_df_drop['Date'] <= index_1 + timedelta(days=int(i.split('_')[3])))].index)
                        new_df_drop = count_week_after_drop_lack_date(new_df_drop,
                                                                      index_1 + timedelta(days=int(i.split('_')[3])),
                                                                      year_test, weekday_begin, weekday_end)
                        new_df_drop.index = new_df_drop['Date']
                    else:
                        continue

        new_df_drop = new_df_drop.reset_index(drop=True)
        return new_df_drop

def count_week_after_drop_lack_date(new_df_drop,index_1,year_test, weekday_begin, weekday_end):
    len_2year_future = len(new_df_drop[new_df_drop['Date'] > index_1])
    list_wk_new = int(len_2year_future // 11.0) * range(1, 12) + range(1, int(len_2year_future % 11.0) + 1)

    new_df_drop['weekday'][new_df_drop['Date'] > index_1] = list_wk_new
    new_df_drop = new_df_drop.reset_index(drop=True)
    new_df_drop['Year'] = [i.year for i in new_df_drop['Date']]
    full_df_drop_lack_date = pd.DataFrame()
    for i in year_test:
        temp_df = new_df_drop[new_df_drop['Year'] == i]
        a = temp_df.index[0]
        b = temp_df.index[-1]
        if temp_df['weekday'].iloc[0] != weekday_begin:
            a = temp_df.index[0] + (weekday_end - temp_df['weekday'].iloc[0] + weekday_begin)
        # temp_df = cruise_day_data.iloc[a:,:]
        if temp_df['weekday'].iloc[-1] != weekday_end:
            b = temp_df.index[-1] + (weekday_end - temp_df['weekday'].iloc[-1] + weekday_begin)
        else:
            b = b + 1
        temp_df = new_df_drop.iloc[a:b, :]
        temp_df['Year'] = i

        for i in range(weekday_begin, weekday_end + 1):
            new_df_drop_1year = temp_df[temp_df['weekday'] == i]
            new_df_drop_1year = new_df_drop_1year.reset_index(drop=True)
            new_df_drop_1year['week_number'] = new_df_drop_1year.index.values + 1
            full_df_drop_lack_date = pd.concat([full_df_drop_lack_date, new_df_drop_1year])

    full_df_drop_lack_date = full_df_drop_lack_date.sort_values(['Date']).reset_index(drop=True)
    return full_df_drop_lack_date

def get_cruise_day_6year(client, dep_id):
    '''
    create CRUISE DAY DATAFRAME for recent 5 year FROM API LINK 
    Cruise Day usually has off-working day
    :param client: client ID
    :param Mydep: department ID
    :returns:
        data: dataframe include columns : Date, DAY
    '''
    HOST = db_api.HOST
    to_date = datetime.now()

    from_date = to_date - timedelta(days= 5 * 365)

    link = HOST  + '/Hierarchy/CruiseDays/{}/?h_id_parent={}&from={}-{:02}-{:02}&to={}-{:02}-{:02}'.format(client, dep_id,
                                                                                                           from_date.year,
                                                                                                           from_date.month,
                                                                                                           from_date.day,
                                                                                                           to_date.year,
                                                                                                           to_date.month,
                                                                                                           to_date.day)
    data = request_data(link)
    return data

# def get_cruise_day_6year(client, dep_id):
#     HOST = db_api.HOST
#     to_date = datetime.now()  + timedelta(days= 365 * 2)
#
#     from_date = to_date - timedelta(days= 7 * 365)
#
#     link = HOST  + '/Hierarchy/CruiseDays/{}/?h_id_parent={}&from={}-{:02}-{:02}&to={}-{:02}-{:02}'.format(client, dep_id,
#                                                                                                            from_date.year,
#                                                                                                            from_date.month,
#                                                                                                            from_date.day,
#                                                                                                            to_date.year,
#                                                                                                            to_date.month,
#                                                                                                            to_date.day)
#     data = request_data(link)
#     return data

def data_full_week_transform(data_df, df_weekday, weekday_begin, weekday_end):
    data_df = data_df.reset_index()
    data_df = data_df.rename(columns = {'date':'Date'})
    
    data_df = data_df.merge(df_weekday, on = 'Date')
    
    start_ind = data_df[data_df['weekday'] == weekday_begin].index[0]
    end_ind = data_df[data_df['weekday'] == weekday_end].index[-1]

    data_df = data_df.iloc[start_ind:(end_ind+1), :]
    data_df = data_df.rename(columns = {'Date':'date'})
    data_df = data_df.set_index('date')   
    del data_df['weekday']
    return data_df                  
    
def begin_week1_end_week11(cruise_day_data):
    '''
    begin-date of dataframe is begin weekday  (1) and end-date of dataframe end weekday (11)
    :param cruise_day_data: dataframe has more than 364 observations
    :returns:
        dataframe_new: data always has enough day in week
        ( start by begin-weekday (1) and end by end-weekday (11))
    '''
    date_begin = cruise_day_data['Date'].iloc[-1] - timedelta(days = 364)
    index_begin = cruise_day_data[cruise_day_data['Date'] >= date_begin].index[0]

    weekday_begin = min(map(int, cruise_day_data['weekday'].drop_duplicates().values))
    weekday_end = max(map(int, cruise_day_data['weekday'].drop_duplicates().values))

    if int(cruise_day_data['weekday'].iloc[index_begin]) != weekday_begin:
        index_begin = index_begin - int(cruise_day_data['weekday'].iloc[index_begin]) + 1
    cruise_day_data = cruise_day_data.iloc[index_begin:].reset_index(drop = True)

    if int(cruise_day_data['weekday'].iloc[-1]) != weekday_end:
        end_index =   cruise_day_data.tail(1).index[0] - int(cruise_day_data['weekday'].iloc[-1]) + weekday_begin
        cruise_day_data = cruise_day_data[:end_index]
    return cruise_day_data

#===================== Get OTB data =====================

def get_otb_data(client, Mydep, years_back, df_year):
    time_get_otb = time.time()
    try:
        # get data otb
        #        current_time = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d') - timedelta(days=2)
        #        from_time = current_time - timedelta(days=years_back*365+30)

        current_time = df_year['Date'].iloc[-1]
        from_time = df_year['Date'].iloc[0]
        link = '%s/Otb/Archive/Total/%s/?h_id=%s&from=%s&to=%s' % (HOST, client, \
                                                                   Mydep, from_time.strftime('%Y-%m-%d'),
                                                                   current_time.strftime('%Y-%m-%d'))
        response = requests.get(link)
    except:
        print('No otb data')
        return None
    json_response = json.loads(response.text)
    json_response_df = pd.DataFrame(json_response['Items'])

    if len(json_response_df) > 0:
        json_response_df['Date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in json_response_df['Date']]
        json_response_df['ImportDate'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in
                                          json_response_df['ImportDate']]
        json_response_df['LeadTime'] = (json_response_df['Date'] - json_response_df['ImportDate']).dt.days
    a = json_response_df['LeadTime']

    new_json_response_df = json_response_df[(a == 0) | (a == 1) | (a == 2) | (a == 3) | (a == 4)]
    otb_data = new_json_response_df.groupby(['Date'])['Revenue'].mean()
    otb_data = put.impute(otb_data, method='zero')
    delta = (json_response_df['Date'][-1:] - otb_data.index[-1]).astype('timedelta64[D]').values[0]
    if delta != 0:
        otb_data_df = pd.DataFrame()
        zero_otb_period = json_response_df['Date'][-1:] - otb_data.index[-1]
        datelist = pd.date_range(otb_data.index[-1], periods=delta).tolist()
        zero_series = pd.Series([0] * len(datelist), index=datelist)
        otb_data = pd.concat([otb_data, zero_series], axis=0)
        otb_data_df['otb_data'] = otb_data.values
        otb_data_df['Date'] = otb_data.index
        otb_data_df.index = otb_data.index
        new_otb_data_df = otb_data_df.drop_duplicates(subset='Date', keep='last')
        new_otb_data = new_otb_data_df
    else:
        #        new_otb_data = otb_data
        new_otb_data = pd.DataFrame()
        new_otb_data['Date'] = otb_data.index
        new_otb_data['otb_data'] = otb_data.values
        # begin Monday, end Sunday
        # last_day = otb_data.shape[0]
        # otb_first = otb_data.index[last_day - n].weekday()
        # first_day = last_day - n - otb_first
        #
        # weekday_last = otb_data.index[-1].weekday()
        # if weekday_last != 6:
        #     last_day = last_day - weekday_last - 1
        #
        # new_otb_data = otb_data[:last_day]
        # new_otb_data.to_csv('otb_{}.csv'.format(Mydep))
        # otb_flag = 0

    print ('time_get_otb', time.time() - time_get_otb)

    if len(df_year['Date']) != len(new_otb_data['otb_data']):
        lack_date = list(set(df_year['Date'].values) - set(new_otb_data['Date'].values))
        new_lack_data = pd.DataFrame()
        new_lack_data['Date'] = lack_date
        new_lack_data['otb_data'] = 0
        new_otb_data = pd.concat([new_otb_data, new_lack_data]).reset_index(drop=True)
    return new_otb_data

#===================== Transform data to dict year =====================

def transform_data_to_1year_cruise(mav_median_df, cruise_day_data):
    '''
    Transform data into 1 year dataframe
    :param mav_median_df: data after removing nan, inf, negative value
    :param cruise_day_data:data has weekday of each date
    :returns:
        df_index: dataframe of 1 year with columns ['Date', 'weekday', 'week_number', 'period', 'Year']
        df_year: dataframe of 1 year with Date and revenue, roomnight, rate of all segments
        nb_years: 1 years
    '''
    nb_years = 1
    mav_median_df['Date'] = mav_median_df.index
    cols = mav_median_df.columns.difference(['Date']).tolist()

    # cruise_day_data, weekday_begin, weekday_end = sslib.get_weekday_data_cruise(client, Mydep, years_back)
    weekday_begin = min(map(int, cruise_day_data['weekday'].drop_duplicates().values))
    weekday_end = max(map(int, cruise_day_data['weekday'].drop_duplicates().values))
    mav_median_df = cruise_day_data.merge(mav_median_df, on='Date')
    mav_median_df =  mav_median_df.reset_index(drop = True)
    if int(mav_median_df['weekday'].iloc[0]) != weekday_begin:
        mav_median_df = mav_median_df[(weekday_end - int(mav_median_df['weekday'].iloc[0]) +1):]

    if int(mav_median_df['weekday'].iloc[-1]) != weekday_end:
        mav_median_df = begin_week1_end_week11(mav_median_df)
    # create week_number for mav_median_df


    new_df = pd.DataFrame()
    for i in range(weekday_begin, weekday_end + 1):
        temp_df = mav_median_df[mav_median_df['weekday'] == str(i )].reset_index(drop=True)
        temp_df['week_number'] = temp_df.index + 1
        new_df = pd.concat([new_df, temp_df])
    total_data = new_df.sort_values(['Date']).reset_index(drop = True)
    if total_data.shape[0] > 365:
        total_data = total_data[(weekday_end-weekday_begin+1):]


    df_year = total_data[['Date'] + cols]
    df_index = total_data[['Date','weekday', 'week_number']]
    return df_index, df_year, nb_years

def transform_data_to_1year(mav_median_df, col_name):
    '''
    Transform data to dict
    :param mav_median_df: data after removing nan, inf, negative value
    :returns:
        dict_index: dictionary contains dataframe of each year with columns ['Date', 'weekday', 'week_number', 'period', 'Year']
        dict_year: dictionary contains dataframe of each year with Date and revenue, roomnight, rate of all segments
        dict_tf: dictionary contains dataframe of each year with revenue, roomnight, rate of all segments grouped by week
        df_date: dataframe contains removed dates of previous years
    '''
    
    total_data = mav_median_df.copy()
    warnings.filterwarnings("ignore")    

    cols = total_data.columns.difference(['Date']).tolist()
    total_data['Date'] = [i.to_datetime() for i in total_data.index]
    total_data['Year'] = [i.year for i in total_data['Date']] 
    total_data['weekday'] = [str(i.weekday()) for i in total_data['Date']]
    
    year_test = total_data['Year'].drop_duplicates().values.tolist()
    
    total_data['week_number'] = [str(i.isocalendar()[1]) for i in total_data['Date']]
    total_data['period'] = total_data[['week_number', 'weekday']].apply(lambda x: '_'.join(x), axis=1)
    total_data['week_number'] = [int(i) for i in total_data['week_number']]
    total_data = total_data.sort_values(by = ['Date']).reset_index(drop = True)
    
    dict_index, dict_year, dict_tf, df_date = create_dict_years(total_data, col_name, cols, year_test)    
    
    warnings.filterwarnings("default")
    
    return dict_index, dict_year, dict_tf, df_date  

def create_dict_years(total_data, col_name, cols, year_test):
    '''
    Transform data to dict
    :param mav_median_df: data after removing nan, inf, negative value
    :returns:
        dict_index: dictionary contains dataframe of each year with columns ['Date', 'weekday', 'week_number', 'period', 'Year']
        dict_year: dictionary contains dataframe of each year with Date and revenue, roomnight, rate of all segments
        dict_tf: dictionary contains dataframe of each year with revenue, roomnight, rate of all segments grouped by week
        df_date: dataframe contains removed dates of previous years
    '''
    df = total_data.copy()
    df_date = df[['Date', col_name]][df['week_number'] == 53].reset_index(drop=True)
    df = df[df['week_number'] != 53].reset_index(drop=True)
    index_cols = list(set(df.columns) - set(cols))
    
    dict_index = {}
    dict_year = {}
    dict_pivot = {}
    temp_df = pd.DataFrame()
    begin_week = df['week_number'].iloc[-1] + 1
    
    for i in year_test[:-1][::-1]:
        begin_index = df.index[(df['Year'] == i) & (df['week_number'] == begin_week)][0]
        if len(temp_df) == 0:
            temp_df = df.iloc[begin_index:,:]
        else:
            last_index = temp_df.index[0]
            temp_df = df.iloc[begin_index:last_index,:]
        dict_index[i+1] = temp_df[index_cols]
        dict_year[i+1] = temp_df[['Date'] + cols]
        temp_df_pv= temp_df.pivot_table(index = ['week_number'], columns = 'weekday', values = cols).reset_index(drop=True)
#        if temp_df_pv['week_number'].iloc[-1] == 53:
#            temp_df_pv = temp_df_pv.iloc[:-1]
        dict_pivot[i+1] = temp_df_pv
        
    return dict_index, dict_year, dict_pivot, df_date 

#===================== Evaluate data quality =====================

def compute_data_quality(dict_index, dict_year, dict_tf, frac):
    '''
    Calculating the volatile then choosing number of using data (year)
    :param dict_index: dictionary contains dataframe of each year with columns ['Date', 'weekday', 'week_number', 'period', 'Year']
    :param dict_year: dictionary contains dataframe of each year with revenue, roomnight, rate of all segments
    :param dict_tf: dictionary contains dataframe of each year with revenue, roomnight, rate of all segments grouped by week
    :param frac: threshold of volatility
    :returns:
        df_index: dataframe of chosen year with columns ['Date', 'Year', 'weekday', 'period', 'week_number']
        df_year: dataframe of chosen year with Date and revenue, roomnight, rate of all segments
        df_tf: dataframe of chosen year with revenue, roomnight, rate of all segments grouped by week
        nb_years: number of chosen years
    '''
    
    years = sorted(dict_tf.keys(), reverse = True)
    cols = list(dict_year[years[0]].columns.values)
#    cols = [x for x in dict_year[years[0]].columns.values if '_' not in x]
    
    if len(dict_year) == 3:        
        #index
        idx_s3 = dict_index[years[0]]
        idx_s2 = dict_index[years[1]]
        idx_s1 = dict_index[years[2]]
        idx_s23 = pd.concat([idx_s2, idx_s3])
#        idx_s13 = pd.concat([idx_s1, idx_s3])
        idx_s123 = pd.concat([idx_s1, idx_s2, idx_s3])
        
        #data raw
        df_s3 = dict_year[years[0]][cols]
        df_s2 = dict_year[years[1]][cols]
        df_s1 = dict_year[years[2]][cols]
        df_s23 = pd.concat([df_s2, df_s3])
#        df_s13 = pd.concat([df_s1, df_s3])
        df_s123 = pd.concat([df_s1, df_s2, df_s3])
        
        #data pivoted
        cols.remove('Date')
        data_s3 = dict_tf[years[0]][cols]
        data_s2 = dict_tf[years[1]][cols]
        data_s1 = dict_tf[years[2]][cols]
        data_s23 = pd.concat([data_s2, data_s3])                    
        data_s13 = pd.concat([data_s1, data_s3])                    
        data_s123 = pd.concat([data_s1, data_s2, data_s3])
        
        s3 = data_s3.std().sum()
        s13 = data_s13.std().sum()
        s23 = data_s23.std().sum()                
        s123 = data_s123.std().sum()
        
        signal = '3'
        nb_years = 1
        if (s23 <= (1+frac) * s3) and (s23 >= (1-frac) * s3):
            signal = '23'
            nb_years = 2
#        if (s13 <= (1+frac) * s3) and (s13 >= (1-frac) * s3) and (abs(s13-s3) < abs(s23 - s3)):
#            signal = '13'
        if (s23 <= (1+frac) * s3) and (s23 >= (1-frac) * s3) and \
           (s13 <= (1+frac) * s3) and (s13 >= (1-frac) * s3) and \
           (abs(s123 - s3) < abs(s13 - s3)) and (abs(s123 - s3) < abs(s23 - s3)):
            signal = '123'
            nb_years = 3
            
        print('s3: {}, s13: {}, s23: {}, s123: {}'.format(s3,s13, s23, s123))
        print('Chosen data: ', signal)
        
        if signal == '3':
            return idx_s3, df_s3, data_s3, nb_years
#        elif signal == '13':
#            return idx_s13, df_s13, data_s13
        elif signal == '23':
            return idx_s23, df_s23, data_s23, nb_years
        else:
            return idx_s123, df_s123, data_s123, nb_years
    elif len(dict_year) == 2:
        #index
        idx_s2 = dict_index[years[0]]
        idx_s1 = dict_index[years[1]]
        idx_s12 = pd.concat([idx_s1, idx_s2])
        
        #data raw
        df_s2 = dict_year[years[0]][cols]
        df_s1 = dict_year[years[1]][cols]
        df_s12 = pd.concat([df_s1, df_s2])
        
        #data pivoted
        cols.remove('Date')
        data_s2 = dict_tf[years[0]][cols]
        data_s1 = dict_tf[years[1]][cols]
        
        data_s12 = pd.concat([data_s1, data_s2])                    
        
        s2 = data_s2.std().sum()
        s12 = data_s12.std().sum()
        
        signal = '2'
        nb_years = 1
        if (s12 <= (1 + frac) * s2) and (s12 >= (1-frac) * s2):
            signal = '12'
            nb_years = 2
            
        print('s2: {}, s12: {}'.format(s2,s12))
        print('Chosen data: ', signal)
        
        if signal == '2':
            return idx_s2, df_s2, data_s2, nb_years
        else:
            return idx_s12, df_s12, data_s12, nb_years
    else:
        #index
        idx_s1 = dict_index[years[0]]
        
        #data raw
        df_s1 = dict_year[years[0]][cols]
        
        #data pivoted
        cols.remove('Date')
        data_s1 = dict_tf[years[0]][cols]
        
        print('Chosen data: ', str(1))
        nb_years = 1
        return idx_s1, df_s1, data_s1, nb_years

#===================== Do PCA =====================

def get_PC1(df_index, df_year, col_name):
    '''
    Do PCA and normalize PC1 data
    :param df_year: dataframe of chosen year with Date and revenue, roomnight, rate of all segments
    :param Mydep: department ID
    :return: data: dataframe with columns ['Date', 'pc1_Mydep_0', 'data_Mydep_0', 'Year', 'weekday', 'period', 'week_number']
    '''

    pca_data = Scaler().fit(df_year).transform(df_year)
#    colnames = pca_data.columns.values
    matrix = pca_data.as_matrix()

    num_cols = matrix.shape[1]
    pca = PCA(copy=True, n_components=num_cols, whiten=True)
    vals = pca.fit_transform(matrix)

#    pca_dict = OrderedDict()

    pc1 = vals[:, 0]
    pc1_nor = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1))
#            pc2 = vals[:, 1]
#            pc2_nor = (pc2 - np.min(pc2)) / (np.max(pc2) - np.min(pc2))
#            pc3 = vals[:, 2]
#            pc3_nor = (pc3 - np.min(pc3)) / (np.max(pc3) - np.min(pc3))

    # export excel files
    data = pd.DataFrame()
    data['Date'] =  df_year.index
    data['pc1_' + col_name] = pc1_nor
#            data['pc2_' + col_name] = pc2_nor
#            data['pc3_' + col_name] = pc3_nor
    data['data_' + col_name] = df_year[col_name].values
    data = data.merge(df_index, on = 'Date')
    
    return data

#===================== Clustering =====================

#def group_season_pc1(data, col_name, lower_bound, upper_bound):
#    '''
#    Clustering
#    :param data: dataframe with columns ['Date', 'pc1_Mydep_0', 'data_Mydep_0', 'Year', 'weekday', 'period', 'week_number']
#    :param col_name
#    :param lower_bound
#    :param upper_bound
#    :return: data_grouped
#    '''
#    
#    data_temp = data.copy()
#    # Get column name of pc & revenue
#    pca_col = 'pc1_' + col_name
#    data_col = 'data_' + col_name
#
#    # Prepare data
#    data_temp['week_year'] =  ['_'.join([str(i.isocalendar()[1]), str(i.isocalendar()[0])]) for i in data_temp['Date']]
#    data_temp['From']= data_temp.groupby('week_year')['Date'].transform('min')
#    data_temp['To']= data_temp.groupby('week_year')['Date'].transform('max')
#     
#    data_pv = data_temp.pivot_table(index = ['From', 'To'], columns = 'weekday', values = pca_col).reset_index()
#    
#    data_arr = data_pv.ix[:, 2:].as_matrix()
#    
#    # Transform to 1 year data
#    num_years = len(data_arr)/52
#    shape = [52*x for x in range(1, num_years)]
#    data_arr_rs = np.split(data_arr, shape)
#    data_arr_rs = np.concatenate(data_arr_rs, axis = 1)
##    data_arr_rs[np.isnan(data_arr_rs)] = 0.0
#
#    # Constrained kMeans
#    if num_years == 1:
#        min_cluster = 6
#        max_cluster = 9
#    else:
#        min_cluster = 15
#        max_cluster = 18
#        
#    num_best = pd.DataFrame()
#    idx = 0
#    best = None
#    cluster = None
#    a = time.time()
#    for nb_cluster in range(min_cluster, max_cluster):
#        min_size = 52/nb_cluster
#        groups_size = [min_size] * nb_cluster
#        (C, temp_clusters, f) = ckmeans.constrained_kmeans(data_arr_rs, groups_size, maxiter=None)
#        try:
#            BIC = ckmeans.compute_bic(C, temp_clusters, nb_cluster, data_arr_rs)
#        except:
#            BIC = 9999999
#        if not best or (BIC < best):
#            best = BIC
#            min_size_best = min_size
#            nb_cluster_best = nb_cluster
#            cluster = temp_clusters
#        num_best.loc[idx, 'min_size'] = min_size
#        num_best.loc[idx, 'nb_cluster'] = nb_cluster
#        num_best.loc[idx, 'BIC'] = BIC
#        idx += 1
#    print('num_years: {}'.format(num_years))
#    print(num_best)
#    print('PC1 __ min_size: {}, nb_cluster: {}, BIC: {}, runtime: {}'.format(min_size_best, nb_cluster_best, best, time.time() - a))
#    
#    data_pv = data_pv[-52:].reset_index(drop=True)     
#    data_pv['Group'] = cluster
##    data_pv_export = data_pv.copy(deep = True)
#
#    data_pv.set_index(['From', 'To', 'Group'], inplace = True)
#
#    data_melt = data_pv.stack().reset_index(name = 'PCA')    
#    data_melt.drop_duplicates(['From', 'To'], inplace = True)
#    
#    data_cluster = data_temp.merge(data_melt[['From', 'To', 'Group']], on = ['From', 'To'])
#    data_cluster['date'] = [str(x)[0:10] for x in data_cluster['Date']]
#                
#    # Plot  
##    plot_season_revenue(data_cluster, data_col)
##    plot_season_pca(data_cluster, pca_col)
#
#    # Remove outlier 
#    data_cluster = remove_outlier(data_cluster, data_col, lower_bound, upper_bound)
#    
#    out_data_rev = data_cluster.pivot_table(index = ['From', 'To', 'Group'], columns = 'weekday', values =  data_col).reset_index()
#    out_data_rev = out_data_rev[['From', 'To', '0', '1', '2', '3', '4', '5', '6', 'Group']]
#    out_data_rev.columns = ['From', 'To', 0, 1, 2, 3, 4, 5, 6, 'Group']
#    out_data_rev['Mean'] = out_data_rev.iloc[:, 2:-1].mean(axis = 1)
#    
#    # Export data Revenue
##    writer = pd.ExcelWriter('data_raw/results/pc1/pc1_{}.xlsx'.format(dep_name))
##    out_data_rev.to_excel(writer, 'Revenue', index = False)   
##    writer.save()
#    
#    return out_data_rev

def group_season_pc1(data, col_name, lower_bound, upper_bound, cruise_flag):
    '''
    Clustering
    :param data: dataframe with columns ['Date', 'pc1_Mydep_0', 'data_Mydep_0', 'Year', 'weekday', 'period', 'week_number']
    :param col_name
    :param lower_bound
    :param upper_bound
    :return: data_grouped
    '''

    weekday_begin = int(data['weekday'].drop_duplicates().values[0])
    weekday_end = int(data['weekday'].drop_duplicates().values[-1])

    data_temp = data.copy()
    # Get column name of pc & revenue
    pca_col = 'pc1_' + col_name
    data_col = 'data_' + col_name

    # Prepare data
    if cruise_flag == False:
        data_temp['week_year'] = ['_'.join([str(i.isocalendar()[1]), str(i.isocalendar()[0])]) for i in data_temp['Date']]
        data_temp['From'] = data_temp.groupby('week_year')['Date'].transform('min')
        data_temp['To'] = data_temp.groupby('week_year')['Date'].transform('max')
    else:
        data_temp['From'] = data_temp.groupby('week_number')['Date'].transform('min')
        data_temp['To'] = data_temp.groupby('week_number')['Date'].transform('max')
        
    data_pv = data_temp.pivot_table(index=['From', 'To'], columns='weekday', values=pca_col).reset_index()

    data_arr = data_pv.ix[:, 2:].as_matrix()
    
    number_of_weeks = len(data_temp['week_number'].drop_duplicates().values)
    
    # Transform to 1 year data
    num_years = len(data_arr) / number_of_weeks
    shape = [number_of_weeks * x for x in range(1, num_years)]
    data_arr_rs = np.split(data_arr, shape)
    data_arr_rs = np.concatenate(data_arr_rs, axis=1)
#    data_arr_rs[np.isnan(data_arr_rs)] = 0.0

    # Constrained kMeans
    if num_years == 1:
        min_cluster = 9
        max_cluster = 14
    else:
        min_cluster = 15
        max_cluster = 18
    if number_of_weeks < max_cluster:
        max_cluster = number_of_weeks - 1
        min_cluster = max_cluster - 5
    num_best = pd.DataFrame()
    idx = 0
    best = None
    cluster = None
    a = time.time()
    for nb_cluster in range(min_cluster, max_cluster):
        min_size = number_of_weeks / nb_cluster
        groups_size = [min_size] * nb_cluster
        (C, temp_clusters, f) = ckmeans.constrained_kmeans(data_arr_rs, groups_size, maxiter=None)
        try:
            BIC = ckmeans.compute_bic(C, temp_clusters, nb_cluster, data_arr_rs)
        except:
            BIC = 9999999
        if not best or (BIC < best):
            best = BIC
            min_size_best = min_size
            nb_cluster_best = nb_cluster
            cluster = temp_clusters
        num_best.loc[idx, 'min_size'] = min_size
        num_best.loc[idx, 'nb_cluster'] = nb_cluster
        num_best.loc[idx, 'BIC'] = BIC
        idx += 1
    print('num_years: {}'.format(num_years))
    print(num_best)
    print('PC1 __ min_size: {}, nb_cluster: {}, BIC: {}, runtime: {}'.format(min_size_best, nb_cluster_best, best, time.time() - a))

    data_pv = data_pv[-number_of_weeks:].reset_index(drop=True)
    data_pv['Group'] = cluster
#    data_pv_export = data_pv.copy(deep = True)

    data_pv.set_index(['From', 'To', 'Group'], inplace=True)

    data_melt = data_pv.stack().reset_index(name='PCA')
    data_melt.drop_duplicates(['From', 'To'], inplace=True)

    data_cluster = data_temp.merge(data_melt[['From', 'To', 'Group']], on=['From', 'To'])
    data_cluster['date'] = [str(x)[0:10] for x in data_cluster['Date']]

    # Plot
 #    plot_season_revenue(data_cluster, data_col)
 #    plot_season_pca(data_cluster, pca_col)

    # Remove outlier
    data_cluster = remove_outlier(data_cluster, data_col, lower_bound, upper_bound)
    
    out_data_rev = data_cluster.pivot_table(index=['From', 'To', 'Group'], columns='weekday', values=data_col).reset_index()
    out_data_rev = out_data_rev[['From', 'To'] + [str(i) for i in range(weekday_begin,weekday_end + 1)] + ['Group']]
    out_data_rev.columns = ['From', 'To'] + range(weekday_begin,weekday_end + 1) + ['Group']
    out_data_rev['Mean'] = out_data_rev.iloc[:, 2:-1].mean(axis=1)

    # Export data Revenue
#    writer = pd.ExcelWriter('data_raw/results/pc1/pc1_{}.xlsx'.format(dep_name))
#    out_data_rev.to_excel(writer, 'Revenue', index = False)   
#    writer.save()

    return out_data_rev

def remove_outlier(data, rev_col, lower_bound, upper_bound):
    data.loc[data[rev_col] < np.percentile(data[rev_col], lower_bound), rev_col] = np.percentile(data[rev_col], lower_bound)  
    data.loc[data[rev_col] > np.percentile(data[rev_col], upper_bound), rev_col] = np.percentile(data[rev_col], upper_bound)
    
    return data

def plot_season_revenue(data_df, rev_col):
    date_df = pd.DataFrame({'index': range(len(data_df)), 'date': data_df['date']})
    step_size_tick = range(0, len(data_df) + 1, 30)
    date_df = date_df[date_df['index'].isin(step_size_tick)]
        
    plt.clf()
    fig, ax = plt.subplots(figsize=(100,40))
    sns.barplot(x='date', y=  rev_col, hue='Group', data= data_df)
    plt.xticks(date_df['index'], date_df['date'], rotation = 90)
    plt.title(rev_col)
    plt.ylabel('Values')
    plt.xlabel('Date')
    change_width(ax, 1.0)
    plt.savefig('{}.png'.format(rev_col))
    plt.show()
    plt.close()    
    
def plot_season_pca(data_df, pca_col):
    date_df = pd.DataFrame({'index': range(len(data_df)), 'date': data_df['date']})
    step_size_tick = range(0, len(data_df) + 1, 30)
    date_df = date_df[date_df['index'].isin(step_size_tick)]
        
    plt.clf()
    fig, ax = plt.subplots(figsize=(50,20))
    sns.barplot(x='date', y=  pca_col, hue='Group', data= data_df)
    plt.xticks(date_df['index'], date_df['date'], rotation = 90)
    plt.title(pca_col, fontsize=18)
    plt.ylabel('Values', fontsize=18)
    plt.xlabel('Date', fontsize=18)
#    change_width(ax, .2)
    change_width(ax, 1.0)
    plt.savefig('{}.png'.format(pca_col))
    plt.show()
    plt.close()

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

#===================== Re-grouping & naming season =====================

#def regroup_and_set_name(data_grouped, Mydep):
#    '''
#    Re-grouping and naming
#    :param data_grouped: dataframe
#    :param Mydep: department ID
#    :return: data_conv_pv
#    '''
#
#    col_name = 'data_' + Mydep + '_0'
#    group_name = 'Group'
#    data_conv = convert_data(data_grouped, col_name, group_name)
#
##    df_new = data_conv[['Date', col_name, 'Group', 'weekday']]
##    df_new['weekday'] = [i.weekday() for i in df_new['Date']]
#
#    data_conv[col_name] = (data_conv[col_name] - data_conv[col_name].min()) / (data_conv[col_name].max() - data_conv[col_name].min())
#    data_conv_pv = data_conv.pivot_table(index=['From', 'To', 'Group'], columns='weekday', values=col_name)
#    data_conv_pv = data_conv_pv.reset_index()
#    out_level = cal_level(data_conv_pv, data_conv, col_name)
#
#    # Create new group
#    data_conv_pv, out_level = create_new_group(data_conv_pv, out_level)
#    data_conv_pv = data_conv_pv.loc[~data_conv_pv[['From', 'To']].duplicated(), :]
#    
#    return data_conv_pv, out_level
#
#def convert_data(data_grouped, col_name, group_name):
#    data_conv = data_grouped.copy()
#    data_conv = data_conv[['From', 'To', group_name] + range(7)]
#    data_conv.set_index(['From', 'To', group_name], inplace = True)  
#    
#    data_conv = data_conv.stack().reset_index()    
#    data_conv.columns = ['From', 'To', 'Group', 'weekday', col_name]    
#    
#    date_range_var = [pd.date_range(data_conv['From'][i], data_conv['To'][i]) for i in range(0, len(data_conv), 7)]          
#    data_conv['Date']  = [j for i in date_range_var for j in i]
#    data_conv['date'] = [str(i)[:10] for i in data_conv['Date']]    
#    data_conv['weekday'] = [i.weekday() for i in data_conv['Date']]
#    
#    return data_conv
#
#def cal_level(data, data_conv, col_name):
#    data_temp = data.copy()
#    # STD by Wekkday
#    data_temp['STD_WEEKDAY'] = data_temp[range(7)].std(axis = 1)
#
#    # Create volatile range
#    from_std = 0
#    level_std = 1
#    range_vol_ls = []
#    
#    for i in [20, 40, 60, 80, 100]:
#        to_std = np.percentile(data_temp['STD_WEEKDAY'], i)
#        if i == 100:
#            to_std = 999999999
#        
#        range_vol_ls.append([from_std, to_std, level_std])
#        from_std = to_std
#        level_std += 1
#    
#    range_vol_df = pd.DataFrame(range_vol_ls, columns = ['From', 'To', 'vol_level'])
#    
#    # Caculate volatile by Season
#    
#    vol_season = data_conv.groupby('Group')[col_name].std().reset_index()
#    vol_level_df = merge_by_range(vol_season, range_vol_df, col_name)
#    vol_level_df.rename(columns = {col_name: 'vol_rev'}, inplace = True)
#    
#    # Mean by Week
#    data_temp['Mean'] = data_temp[range(7)].mean(axis = 1)
#    
#    
#    # Create Mean range
#    from_mean = 0
#    level_mean = 1
#    range_mean_ls = []
#    
#    for i in [20, 40, 60, 80, 100]:
#        to_mean = np.percentile(data_temp['Mean'], i)
#        if i == 100:
#            to_mean = 999999999
#        
#        range_mean_ls.append([from_mean, to_mean, level_mean])
#        from_mean = to_mean
#        level_mean += 1
#    
#    range_mean_df = pd.DataFrame(range_mean_ls, columns = ['From', 'To', 'rev_level'])
#    
#    # Caculate mean by Season    
#    mean_season = data_conv.groupby('Group')[col_name].mean().reset_index()
#    mean_level_df = merge_by_range(mean_season, range_mean_df, col_name)
#    mean_level_df.rename(columns = {col_name: 'rev_mean'}, inplace = True)
#    
#    # Output
#    output_df = vol_level_df[['Group', 'vol_rev', 'vol_level']].merge(mean_level_df[['Group', 'rev_mean', 'rev_level']], on = 'Group')
#    output_df = output_df[['Group', 'rev_mean', 'vol_rev', 'rev_level', 'vol_level']]
#    return output_df

def regroup_and_set_name(data_grouped, Mydep):
    '''
    Re-grouping and naming
    :param data_grouped: dataframe
    :param Mydep: department ID
    :return: data_conv_pv
    '''
    list_wk = [i for i in data_grouped.columns.values if isinstance(i, (int, float))]
    weekday_begin = list_wk[0]
    weekday_end = list_wk[-1]

    col_name = 'data_' + Mydep + '_0'
    group_name = 'Group'
    data_conv = convert_data(data_grouped, col_name, group_name, weekday_begin, weekday_end)

#    df_new = data_conv[['Date', col_name, 'Group', 'weekday']]
#    df_new['weekday'] = [i.weekday() for i in df_new['Date']]

    data_conv[col_name] = (data_conv[col_name] - data_conv[col_name].min()) / (data_conv[col_name].max() - data_conv[col_name].min())
    data_conv_pv = data_conv.pivot_table(index=['From', 'To', 'Group'], columns='weekday', values=col_name)
    data_conv_pv = data_conv_pv.reset_index()
    out_level = cal_level(data_conv_pv, data_conv, col_name, weekday_begin, weekday_end)

    # Create new group
    data_conv_pv, out_level = create_new_group(data_conv_pv, out_level)

    return data_conv_pv, out_level

def convert_data(data_grouped, col_name, group_name, weekday_begin, weekday_end):
    data_conv = data_grouped.copy()
    data_conv = data_conv[['From', 'To', group_name] + range(weekday_begin, weekday_end+1)]
    data_conv.set_index(['From', 'To', group_name], inplace=True)

    data_conv = data_conv.stack().reset_index()
    data_conv.columns = ['From', 'To', 'Group', 'weekday', col_name]
    
    date_range_var = [pd.date_range(data_conv['From'][i], data_conv['To'][i]) for i in range(0, len(data_conv), weekday_end - weekday_begin + 1)]
    data_conv['Date'] = [j for i in date_range_var for j in i]
    data_conv['date'] = [str(i)[:10] for i in data_conv['Date']]
    data_conv['weekday'] = range(weekday_begin, weekday_end+1)*(data_conv.shape[0]/(weekday_end - weekday_begin + 1))

    return data_conv

def cal_level(data_conv_pv, data_conv, col_name, weekday_begin, weekday_end):
    # data, data_conv, col_name = data_conv_pv, data_conv, col_name
    
    data_temp = data_conv_pv.copy()
    
    # Mean by Week
    data_temp['Mean'] = data_temp[range(weekday_begin, weekday_end+1)].mean(axis=1)   
    
    # Create Mean range
    from_mean = 0
    level_mean = 1
    range_mean_ls = []
    
    for i in [20, 40, 60, 80, 100]:
        to_mean = np.percentile(data_temp['Mean'], i, interpolation = 'nearest')
        if i == 100:
            to_mean = 999999999
            
        range_mean_ls.append([from_mean, to_mean, level_mean])
        from_mean = to_mean
        level_mean += 1
        
    range_mean_df = pd.DataFrame(range_mean_ls, columns=['From', 'To', 'rev_level'])
    range_mean_df = range_mean_df.drop_duplicates(subset = ['From','To']).reset_index(drop = True)
    range_mean_df = range_mean_df[range_mean_df['From'] != range_mean_df['To']].reset_index(drop = True)
    # Caculate mean by Season
    mean_season = data_conv.groupby('Group')[col_name].mean().reset_index()
    mean_level_df = merge_by_range(mean_season, range_mean_df, col_name)
    mean_level_df.rename(columns={col_name: 'rev_mean'}, inplace=True)
    
    # STD by Week
    data_temp['STD_WEEKDAY'] = data_temp[range(weekday_begin, weekday_end+1)].std(axis=1) / \
                                data_temp[range(weekday_begin, weekday_end+1)].mean(axis=1)
                                
    data_temp['STD_WEEKDAY'] = data_temp['STD_WEEKDAY'].replace(np.nan, 0)
    # Create volatile range
    from_std = 0
    level_std = 1
    range_vol_ls = []
    
    for i in [20, 40, 60, 80, 100]:
        to_std = np.percentile(data_temp['STD_WEEKDAY'], i, interpolation = 'nearest')
        if i == 100:
            to_std = 999999999
            
        range_vol_ls.append([from_std, to_std, level_std])
        from_std = to_std
        level_std += 1
        
    range_vol_df = pd.DataFrame(range_vol_ls, columns=['From', 'To', 'vol_level'])
    range_vol_df = range_vol_df.drop_duplicates(subset=['From', 'To']).reset_index(drop=True)
    range_vol_df = range_vol_df[range_vol_df['From'] != range_vol_df['To']].reset_index(drop=True)
    # Caculate volatile by Season
    vol_season = (data_conv.groupby('Group')[col_name].std()/data_conv.groupby('Group')[col_name].mean()).reset_index()
    vol_season = vol_season.replace(np.nan,0)
    
    vol_level_df = merge_by_range(vol_season, range_vol_df, col_name)
    vol_level_df.rename(columns={col_name: 'vol_rev'}, inplace=True)

    # Output
    output_df = vol_level_df[['Group', 'vol_rev', 'vol_level']].merge(mean_level_df[['Group', 'rev_mean', 'rev_level']], on='Group')
    output_df = output_df[['Group', 'rev_mean', 'vol_rev', 'rev_level', 'vol_level']]
    return output_df

def merge_by_range(data, range_df, nor_col):
    data = data.assign(key=1)
    range_df = range_df.assign(key=1)
    df_merge = pd.merge(data, range_df, on='key').drop('key',axis=1)
    
    df_out = df_merge.query('{} >= From and {} < To'.format(nor_col, nor_col))
    return df_out

def create_new_group(data_raw, data_level):
    out_level_group = data_level.groupby(['rev_level', 'vol_level'])['Group'].min().reset_index()
    out_level_group['Group'] = range(out_level_group.shape[0])
    data_level.rename(columns = {'Group':'old_group'}, inplace = True)
    data_level = data_level.merge(out_level_group, on = ['rev_level', 'vol_level'])
    
    data_raw.rename(columns = {'Group':'old_group'}, inplace = True)
    data_raw = data_raw.merge(data_level[['old_group', 'Group']], on = 'old_group')
    # data_level_sort =  data_level.sort_values(['rev_mean','vol_rev'])

    #TN - drop regroup
    # data_raw['Group'] = data_raw['old_group']
    # data_level['Group'] = data_level['old_group']
    #
    # list_group = []
    # for i in range(1, 6):
    #     for j in range(1, 6):
    #         list_group.append(str(i) + '_' + str(j))
    # length_group = len(data_level['Group'])
    # list_group = list_group[:length_group]
    # data_level['rev_level'] = [int(i.split('_')[0]) for i in list_group]
    # data_level['vol_level'] = [int(i.split('_')[1]) for i in list_group]


    return data_raw, data_level

#===================== Generate data 3 previous years and 2 future years =====================

def generate_6years_df(data_conv_pv, cycle_value):
    '''
    Generating data 3 previous years, 1 current year and 2 future years
    :param data_conv_pv: dataframe contains columns 'From','To','Group' and '0,1,...,6' columns
    :param 
    :return: full_df : dataframe contains 'From', 'To', 'Group','Year' of 6 years
    '''
    current_df = data_conv_pv[['From', 'To', 'Group']].sort_values('From').reset_index(drop=True)
    current_year = sorted(list(set([i.year for i in current_df['From']])))[-1]
    full_year = range(current_year-3, current_year+3)
    from_weeknumber = current_df['From'].iloc[0].isocalendar()[1]
    to_weeknumber = current_df['To'].iloc[-1].isocalendar()[1]
    
    full_df = current_df.copy()
    prev_year = sorted(range(current_year-3, current_year), reverse = True)
    for idx, yr in enumerate(prev_year):
        tmp_df = current_df.copy()
        
        from_start = iso_to_gregorian(tmp_df['From'].iloc[0].year - idx - 1, from_weeknumber, 1)
        from_end = iso_to_gregorian(tmp_df['From'].iloc[-1].year - idx - 1, to_weeknumber, 1)       
        rr = rrule.rrule(rrule.WEEKLY,byweekday=relativedelta.MO,dtstart=from_start)
        from_range = rr.between(from_start, from_end, inc=True)
        from_range = [i for i in from_range if i.isocalendar()[1] != 53]
        tmp_df['From'] = from_range
        
        to_start = iso_to_gregorian(tmp_df['To'].iloc[0].year - idx - 1, from_weeknumber, 7)
        to_end = iso_to_gregorian(tmp_df['To'].iloc[-1].year - idx - 1, to_weeknumber, 7)
        rr = rrule.rrule(rrule.WEEKLY,byweekday=relativedelta.SU,dtstart=to_start)
        to_range = rr.between(to_start, to_end, inc=True)
        to_range = [i for i in to_range if i.isocalendar()[1] != 53]
        tmp_df['To'] = to_range

        full_df = pd.concat([tmp_df, full_df])

    next_year = range(current_year+1, current_year+3)
    for idx, yr in enumerate(next_year):
        tmp_df = current_df.copy()
        
        from_start = iso_to_gregorian(tmp_df['From'].iloc[0].year + idx + 1, from_weeknumber, 1)
        from_end = iso_to_gregorian(tmp_df['From'].iloc[-1].year + idx + 1, to_weeknumber, 1)       
        rr = rrule.rrule(rrule.WEEKLY,byweekday=relativedelta.MO,dtstart=from_start)
        from_range = rr.between(from_start, from_end, inc=True)
        from_range = [i for i in from_range if i.isocalendar()[1] != 53]
        tmp_df['From'] = from_range
        
        to_start = iso_to_gregorian(tmp_df['To'].iloc[0].year + idx + 1, from_weeknumber, 7)
        to_end = iso_to_gregorian(tmp_df['To'].iloc[-1].year + idx + 1, to_weeknumber, 7)
        rr = rrule.rrule(rrule.WEEKLY,byweekday=relativedelta.SU,dtstart=to_start)
        to_range = rr.between(to_start, to_end, inc=True)
        to_range = [i for i in to_range if i.isocalendar()[1] != 53]
        tmp_df['To'] = to_range
        
        full_df = pd.concat([full_df, tmp_df])

    #Create lack dates
    leap_year = [i+1 for i in full_year if datetime(i, 12, 31).isocalendar()[1] == 53]
    leap_list = list(set(leap_year) & set(full_year))
    lackdate_temp = pd.DataFrame()
    
    for leap in leap_list:
        temp_df = pd.DataFrame(columns = ['From', 'To', 'Group']) 
        temp_df['From'] = [iso_to_gregorian(leap - 1, 53, 1)]
        temp_df['To'] = [iso_to_gregorian(leap - 1, 53, 7)]
        temp_df['Group'] = [np.nan]
        
        full_df = pd.concat([full_df, temp_df])
        lackdate_temp = pd.concat([lackdate_temp, temp_df])

    full_df = full_df.reset_index(drop=True).sort_values(['From'])
    full_df = full_df.fillna(method='ffill')
    full_df['Group'] = [int(x) for x in full_df['Group']]
    
    #If foodcost season, cycle_value is defferent to zero => start at the beginning of weekly cycle.
    full_df['From'] = [i - timedelta(days = cycle_value) for i in full_df['From']]
    full_df['To'] = [i - timedelta(days = cycle_value) for i in full_df['To']]
    full_df['year'] = [i.year for i in full_df['From']]
    
    del lackdate_temp['Group']
    lackdate_temp = lackdate_temp.merge(full_df, on = ['From','To'])

    lack_date = []
    group = []
    for i in range(len(lackdate_temp)):
        lack_date = lack_date + [lackdate_temp['From'][i] + timedelta(days=x) for x in range((lackdate_temp['To'][i] - lackdate_temp['From'][i]).days + 1)]
        group = group + [lackdate_temp['Group'][i]]*((lackdate_temp['To'][i] - lackdate_temp['From'][i]).days + 1)
    lackdate_df = pd.DataFrame()
    lackdate_df['From'] = lack_date
    lackdate_df['To'] = lack_date
    lackdate_df['Group'] = group
    lackdate_df['year'] = [i.year for i in lackdate_df['From']]
    
    return full_df, lackdate_df

def iso_year_start(iso_year):
    "The gregorian calendar date of the first day of the given ISO year"
    fourth_jan = datetime(iso_year, 1, 4)
    delta = timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta 

def iso_to_gregorian(iso_year, iso_week, iso_day):
    "Gregorian calendar date for the given ISO year, week and day"
    year_start = iso_year_start(iso_year)
    return year_start + timedelta(days=iso_day-1, weeks=iso_week-1)

def generate_6year_df_cruise(data_conv_pv, client, Mydep, cycle_value):
    '''
    Generating data 3 previous years, 1 current year and 2 future years
    :param data_conv_pv: dataframe contains columns 'From','To','Group' and '1,2,...,11' columns
    :param
    :return:
        full_df : dataframe contains 'From', 'To', 'Group','Year' of 6 years
        df_weekday: dataframe contain Date and Weekday columns
        lack_date_from_to: dataframe contains off-working days
    '''
    data = data_conv_pv.copy()
    data = data.sort_values(['From'])
    list_wk = [i for i in data.columns.values if isinstance(i, (int, float,long))]
    weekday_begin = list_wk[0]
    weekday_end = list_wk[-1]
    data = data[['From', 'To', 'Group'] + range(weekday_begin,weekday_end + 1)]
    data.set_index(['From', 'To', 'Group'], inplace = True)

    data = data.stack().reset_index()
    data.columns = ['From', 'To', 'Group','weekday', 'values']
    
    data = data[['From', 'To', 'Group','weekday']]

    date_range_var = [pd.date_range(data['From'][i], data['To'][i]) for i in range(0, len(data), weekday_end - weekday_begin + 1)]
    data['Date']  = [j for i in date_range_var for j in i]
    data['date'] = [str(i)[:10] for i in data['Date']]
    new_data = data[['Date','Group']]

    df_dow_weeknumber = cruise_df_dow_weeknumber(client, Mydep)

    data_frame = pd.merge(new_data,df_dow_weeknumber, on = 'Date')
    data_frame = data_frame.sort_values('Date').reset_index(drop = True)
    shift_week = data_frame['week_number'].iloc[-1] - data_frame['week_number'].iloc[0]
    if shift_week >= 0:
        df_prev_year = data_frame[data_frame['Year'] == min(data_frame['Year'].drop_duplicates().values)]
        df_next_year = data_frame[data_frame['Year'] == max(data_frame['Year'].drop_duplicates().values)]
        df_prev_year['week_number'] = df_prev_year['week_number'] + shift_week + 1
        data_frame = pd.concat([df_prev_year,df_next_year]).reset_index(drop = True)
    data_frame['week_number'] = [str(i) for i in data_frame['week_number']]
    data_frame['weekday'] = [str(i) for i in data_frame['weekday']]

    data_frame['period'] = data_frame[['week_number', 'weekday']].apply(lambda x: '_'.join(x), axis=1)

    data_frame = data_frame[['period','Group']]


    df_dow_weeknumber['week_number'] = [str(i) for i in df_dow_weeknumber['week_number']]
    df_dow_weeknumber['weekday'] = [str(i) for i in df_dow_weeknumber['weekday']]
    df_dow_weeknumber['period'] = df_dow_weeknumber[['week_number', 'weekday']].apply(lambda x: '_'.join(x), axis=1)

    data_frame_merge = df_dow_weeknumber[['Date','period','week_number','weekday','Year']].merge(data_frame, how = 'left')
    data_frame_merge = data_frame_merge.sort_values(['Date']).reset_index(drop = True)
    data_frame_merge = data_frame_merge.fillna(method='ffill')

    b_index = df_dow_weeknumber[(df_dow_weeknumber['Year'] == new_data['Date'].iloc[0].year - 3) & (
                df_dow_weeknumber['period'] == data_frame['period'].iloc[0])].index.values[0]
    e_index = df_dow_weeknumber[(df_dow_weeknumber['Year'] == new_data['Date'].iloc[-1].year +2) & (
                df_dow_weeknumber['period'] == data_frame['period'].iloc[-1])].index.values[0]
    date_begin = df_dow_weeknumber['Date'][b_index]
    date_end = df_dow_weeknumber['Date'][e_index]

    # df_6year =  data_frame_merge[(data_frame_merge['Date'] >= date_begin) & (data_frame_merge['Date'] <= date_end)].reset_index(drop = True)
    df_6year = data_frame_merge[(data_frame_merge['Date'] >= date_begin)].reset_index(drop=True)
    df_6year.index = df_6year['Date']
    # df_6year = df_6year.reset_index(drop=True)
    df_6year_nan = put.impute(df_6year, method='nan')

    # df_6year = df_6year.fillna(method='ffill')

    # create df_lack_date
    lack_date_df = df_6year_nan[df_6year_nan.isnull().any(axis = 1)]
    lack_date_list = lack_date_df.index
    df_6_year_fillna = df_6year_nan.fillna(method='ffill')
    df_6_year_fillna['Date'] = df_6_year_fillna.index
    lack_date_df_fillna = df_6_year_fillna.ix[lack_date_list.values,:].reset_index(drop = True)
    lack_date_df_fillna['new_group'] = ['_'.join([str(lack_date_df_fillna['Year'].iloc[i]), str(lack_date_df_fillna['week_number'].iloc[i])]) for i in lack_date_df_fillna.index]
    lack_date_df_fillna['From'] = lack_date_df_fillna.groupby('new_group')['Date'].transform('min')
    lack_date_df_fillna['To'] = lack_date_df_fillna.groupby('new_group')['Date'].transform('max')
    lack_date_from_to = lack_date_df_fillna.drop_duplicates().reset_index(drop=True)
    lack_date_from_to = lack_date_from_to.rename(columns={'Year': 'year'})
    lack_date_from_to['From'] = [i - timedelta(days=cycle_value) for i in lack_date_from_to['From']]
    lack_date_from_to['To'] = [i - timedelta(days=cycle_value) for i in lack_date_from_to['To']]
    lack_date_from_to = lack_date_from_to[['Group', 'From', 'To']].drop_duplicates().reset_index(drop = True)
    lack_date_from_to['Group'] = max(data_frame['Group'].drop_duplicates()) + 1




    df_6year = df_6year.reset_index(drop = True)

    new_df_6year = df_6year[['Date','week_number','Group','Year','weekday']]
    # new_df_6year['Year'] = [i.year for i in new_df_6year['Date']]

    new_df_6year['new_group'] = ['_'.join([str(new_df_6year['Year'].iloc[i]), str(new_df_6year['week_number'].iloc[i])]) for i in new_df_6year.index]
    new_df_6year['From'] = new_df_6year.groupby('new_group')['Date'].transform('min')
    new_df_6year['To'] = new_df_6year.groupby('new_group')['Date'].transform('max')

    full_df = new_df_6year[['From', 'To', 'Group', 'Year']].drop_duplicates().reset_index(drop=True)
    full_df = full_df.rename(columns={'Year': 'year'})
    df_weekday = new_df_6year[['Date','weekday']]

    # If foodcos season, cycle_value is defferent to zero => start at the beginning of weekly cycle.
    full_df['From'] = [i - timedelta(days=cycle_value) for i in full_df['From']]
    full_df['To'] = [i - timedelta(days=cycle_value) for i in full_df['To']]

    return full_df, df_weekday, lack_date_from_to

#===================== Outlier detection =====================

#def outlier_detection(room_data, full_df, col_name):
#    '''
#    Detecting outliers
#    :param room_data: dataframe with columns, e.g. ['50_0_RV', '50_224_RV', ...]
#    :param full_df: dataframe with columns ['From', 'To', 'Group', 'year']
#    :return: outlier_date: list of datestring
#    '''
#    temp_data = room_data[[x for x in room_data.columns.values if col_name in x]]
#    temp_data.index.name = 'Date'
#    temp_data = temp_data.reset_index()
#    temp_data = merge_by_range_rev(temp_data, full_df[['From', 'To', 'Group']], 'Date')
#    temp_data['weekday'] = [x.weekday() for x in temp_data['Date']]
#
#    # outlier_date = temp_data.groupby(['Group', 'weekday']).apply(lambda x: (mad_process_RV(x[['Date', col_name]])))
#    outlier_date = temp_data.groupby(['weekday']).apply(lambda x: (mad_process_RV(x[['Date', col_name,'Group']])))
#    outlier_date = [j for i in outlier_date for j in i]
#    
#    return outlier_date

def outlier_detection(room_data, full_df, col_name, df_weekday, cruise_flag):
    '''
    Detecting outliers
    :param room_data: dataframe with columns, e.g. ['50_0_RV', '50_224_RV', ...]
    :param full_df: dataframe with columns ['From', 'To', 'Group', 'year']
    :return: outlier_date: list of datestring
    '''
    temp_data = room_data[[x for x in room_data.columns.values if col_name in x]]
    temp_data.index.name = 'Date'
    temp_data = temp_data.reset_index()
    temp_data = merge_by_range_rev(temp_data, full_df[['From', 'To', 'Group']], 'Date')
    if cruise_flag == False:
        temp_data['weekday'] = [x.weekday() for x in temp_data['Date']]
    else:
        temp_data = temp_data.merge(df_weekday, on='Date')

    weekday_begin = min(map(int, temp_data['weekday'].drop_duplicates().values))
    weekday_end = max(map(int, temp_data['weekday'].drop_duplicates().values))

    # outlier_date = temp_data.groupby(['Group', 'weekday']).apply(lambda x: (mad_process_RV(x[['Date', col_name]])))
    outlier_date = temp_data.groupby(['weekday']).apply(lambda x: (mad_process_RV(x[['Date', col_name, 'Group']])))
    outlier_date = [j for i in outlier_date for j in i]

    return outlier_date , weekday_begin, weekday_end

def merge_by_range_rev(revenue_data, full_df, col):
    revenue_data = revenue_data.assign(key=1)
    full_df = full_df.assign(key=1)
    df_merge = pd.merge(revenue_data, full_df, on='key').drop('key',axis=1)
    
    df_out = df_merge.query('{} >= From and {} <= To'.format(col, col))
#    del df_out['From'], df_out['To']
    return df_out

def mad_process_RV(df):
    '''
    groupby weekday in order to calculate std of all points of each weekday.
    divide data into each year --> calculate std (5 point around the point)
    groupby season --> each season, weekday: compare std(all point) and std(5 point) ; compare dev and threshold
    if number of 0 of(season, weekday) > 75% --> threshold = 3.0
    '''
    df = df.reset_index(drop=True)
    df['Year'] = [i.year for i in df['Date']]
    ## The threshold...
    thld = 6.5
    merge_list = []
    year_test = df['Year'].drop_duplicates().values.tolist()
    for i in year_test:
        temp_df = df[df['Year'] == i]
        temp_df['std'] = temp_df.iloc[:, 1].rolling(5, center=True, min_periods=2).std().values
        around_cond = temp_df.iloc[:, 1].std()
        temp_df = temp_df.reset_index(drop=True)
        for k, data in temp_df.groupby(['Group']):
            df_ss_wk = data.copy()
            df_ss_wk = df_ss_wk.reset_index(drop=True)
            idx = list()
            l = df_ss_wk.iloc[:, 1].tolist()
            per_zero = l.count(0) / float(len(l))
            if per_zero > 0.75 and per_zero < 1:
                thld = 3.0
            x = np.nanmedian(l)
            mad_score = mad(l)
            for ind, val in enumerate(l):
                dev = ((.6745 * (val - x)) / mad_score)
                if abs(dev) >= thld and df_ss_wk['std'].iloc[ind] >= around_cond:
                    idx.append(1)
                else:
                    idx.append(0)
            index = [i for i, val in enumerate(idx) if val == 1]
            date_list = [ii.strftime('%Y-%m-%d') for ii in df_ss_wk['Date'][index]]
            merge_list = merge_list + date_list
    return merge_list

def mad(a, c = Gaussian.ppf(3 / 4.), axis=0, center=np.nanmedian):
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
    result = np.nanmedian((np.fabs(a - center)) / c, axis=axis)
    if result == 0:
        result = np.nanmean((np.fabs(a - center)) / c, axis=axis)
    return result

#===================== Holiday effect =====================

#def get_event_data(client, Mydep):
#    '''
#    Get event data
#    :param client: client_id
#    :param Mydep: department ID
#    :return: event_data (type: dataframe)
#    '''
#    link = SPECIAL_DATES_LINK.format(client_id = client, h_id = Mydep)
#    event_date = request_data(link)
#    event_date = explode(event_date, 'Dates')
#
#    event_date['Dates'] = pd.to_datetime(event_date['Dates'], format='%Y-%m-%d')
#    event_date['Year'] = [str(event_date['Dates'][i])[:4] for i in event_date.index]
#
#    return event_date

def get_event_data(client, Mydep):
    '''
    Get event data
    :param client: client_id
    :param Mydep: department ID
    :return: event_data (type: dataframe)
    return: holiday_data: dataframe with columns ['Id', 'EventName', 'From', 'To']
    param event_date: dataframe, (e.g. columns: ['Dates', 'Id', 'Name', 'Year'])
    '''
    Dates , Id, Name, Year = [], [], [], []
    for j in range(date.today().year  -4 , date.today().year):
        k = 0
        for ptr in holidays.US(years = j).items():
            Dates.append(pd.to_datetime(ptr[0]))
            Id.append(k)
            Name.append(ptr[1])
            Year.append(j)
            k +=1
    event_date  = pd.DataFrame()
    event_date['Dates'] = Dates
    event_date['Id'] = Id
    event_date['Name'] = Name
    event_date['Year'] = Year

    return event_date    
    
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

#def get_holiday_data(room_data, event_date, full_df, col_name, cycle_value):
#    '''
#    Computing holidays effect and finding how long they lasted
#    :param room_data: dataframe, (e.g. columns: ['Date', u'50_0_RV', u'50_224_RV', u'50_225_RV', ..])
#    :param event_date: dataframe, (e.g. columns: ['Dates', 'Id', 'Name', 'Year'])
#    :param full_df: dataframe, (e.g. columns: ['From', 'To', 'Group', 'year')
#    :return: holiday_data: dataframe with columns ['Id', 'EventName', 'From', 'To']
#    '''
#    room_data.reset_index(inplace=True)
#    room_data.rename(columns={'index': 'Date'}, inplace=True)
#
#    event_date['From'] = [back_days_transform([x, x], 14)[0] for x in event_date['Dates']]
#    event_date['To'] = [back_days_transform([x, x], 14)[1] for x in event_date['Dates']]
#
#    min_date_data = room_data["Date"].iloc[0]
#    max_date_data = room_data["Date"].iloc[-1]
#
#    event_date = event_date[event_date['From'] >= min_date_data]
#    event_date = event_date[event_date['To'] <= max_date_data]
#
#    year_event = range(max(room_data["Date"].iloc[0].year, room_data["Date"].iloc[-1].year - 3),
#                       room_data["Date"].iloc[-1].year + 1)
#    year_event = [str(x) for x in year_event]
#
#    # Calculate effect event
#    result_ls = []
#    for i in event_date['Name'].unique():
#        # i = 'Easter'
#        event_id = int(event_date.loc[(event_date['Name'] == i), 'Id'].unique()[0])
#        # Filter data
#        event_filter = event_date[(event_date['Name'] == i) & (event_date['Year'].isin(year_event))]
#        rev_7days, rev_30days, start_date, end_date = combine_4year_revenue(event_filter, room_data, col_name, after=0, before=0)
#
#        flag, mean_frac = data_event2(rev_7days, rev_30days)
#
#        day_after, day_before = expand_effect_range(flag, event_filter, room_data, col_name)
#
#        rev_7days, rev_30days, start_date, end_date = combine_4year_revenue(event_filter, room_data, col_name, day_after, day_before)
#        flag, mean_frac = data_event2(rev_7days, rev_30days)
#        result_ls.append([i, event_id, flag, mean_frac, str(end_date.year), start_date, end_date, day_after, day_before])
#
#    result_df = pd.DataFrame(result_ls, columns=['EventName', 'Id', 'Signal', 'Frac', 'year', 'Start_day', 'End_day', 'day_after', 'day_before'])
#    result_df = event_date[['Dates', 'Name', 'Year']].merge(result_df, left_on=['Name'], right_on=['EventName'], how='left')
#
#    start_day_new = []
#    end_day_new = []
#    for i in result_df.index:
#        date_range_temp = [result_df['Dates'].iloc[i], result_df['Dates'].iloc[i]]
#        start_day_temp, end_day_temp = transform_full_week(date_range_temp, result_df['day_after'].iloc[i], result_df['day_before'].iloc[i])
#        start_day_new.append(start_day_temp)
#        end_day_new.append(end_day_temp)
#
#
#    result_df['Start_day'] = start_day_new
#    result_df['End_day'] = end_day_new
#
#    holiday_data = pd.DataFrame(columns=['Id', 'EventName', 'From', 'To'])
#
#
#    for i in result_df.index:
#        if result_df.loc[i, 'Signal'] != False:
#            holiday_data.loc[i, 'Id'] = int(result_df.loc[i, 'Id'])
#            holiday_data.loc[i, 'EventName'] = result_df.loc[i, 'EventName']
#            holiday_data.loc[i, 'From'] = result_df['Start_day'].iloc[i]
#            holiday_data.loc[i, 'To'] = result_df['End_day'].iloc[i]
#    
#    holiday_data['From'] = [i - timedelta(days = cycle_value) for i in holiday_data['From']]
#    holiday_data['To'] = [i - timedelta(days = cycle_value) for i in holiday_data['To']]
#    
#    return holiday_data

def get_holiday_data(room_data, event_date, col_name, client, Mydep, weekday_begin, weekday_end, cycle_value, df_weekday):
    '''
    Computing holidays effect and finding how long they lasted
    :param room_data: dataframe, (e.g. columns: ['Date', u'50_0_RV', u'50_224_RV', u'50_225_RV', ..])
    :param event_date: dataframe, (e.g. columns: ['Dates', 'Id', 'Name', 'Year'])
    :param full_df: dataframe, (e.g. columns: ['From', 'To', 'Group', 'year')
    :return: holiday_data: dataframe with columns ['Id', 'EventName', 'From', 'To']
    '''

    if (weekday_end - weekday_begin + 1) <= 7:
        event_date['weekday'] = [i.weekday() for i in event_date['Dates']]
        df_weekday = 0
    else:
        # df_weekday = cruise_df_dow_weeknumber(client, Mydep)

        df_weekday = df_weekday.rename(columns={'Date': 'Dates'})
        df_weekday['weekday'] = df_weekday['weekday'].astype('int')
        # df_weekday = df_weekday[['Dates', 'weekday']]
        # df_weekday
        event_date = pd.merge(event_date, df_weekday, on='Dates', how='inner')
    event_date['Name'] = [i.replace(' (Observed)','') for i in event_date['Name']]
    copy_event_date = event_date.copy()
    
#    event_date = event_date[event_date['Name'] == 'Thanksgiving']

    room_data = room_data.reset_index()
    room_data = room_data.rename(columns={'index': 'Date'})

    nb_days = (weekday_end - weekday_begin) + 1

    event_date['From'] = [back_days_transform([x, x], nb_days * 2)[0] for x in event_date['Dates']]
    event_date['To'] = [back_days_transform([x, x], nb_days * 2)[1] for x in event_date['Dates']]

    min_date_data = room_data["Date"].iloc[0]
    max_date_data = room_data["Date"].iloc[-1]

    event_date = event_date[event_date['From'] >= min_date_data]
    event_date = event_date[event_date['To'] <= max_date_data]

    year_event = range(max(room_data["Date"].iloc[0].year, room_data["Date"].iloc[-1].year - 3),
                       room_data["Date"].iloc[-1].year + 1)
    year_event = [x for x in year_event]
    #    col_name = [x for x in room_data.columns if '_0_RV' in x][0]

    # Calculate effect event
    
    
    result_ls = []
    for i in event_date['Name'].unique():
        # i = 'Easter'
        event_id = int(event_date.loc[(event_date['Name'] == i), 'Id'].unique()[0])
        # Filter data
        event_filter = event_date[(event_date['Name'] == i) & (event_date['Year'].isin(year_event))]

        # if not isinstance(df_weekday, (int, long)):
        #     ls_ind = check_begin_day(event_filter, 0, 0,
        #                     weekday_begin, weekday_end, df_weekday)
        #     if len(ls_ind) != 0:
        #         event_filter = event_filter.loc[ls_ind, : ]
        #     else:
        #         continue

        rev_7days, rev_30days, start_date, end_date = combine_4year_revenue_cruise(event_filter, room_data, col_name,
                                                                                   0, 0, weekday_begin, weekday_end)

        flag, mean_frac = data_event2(rev_7days, rev_30days)

        day_after, day_before = expand_effect_range_cruise(flag, event_filter, room_data, col_name,
                                                           weekday_begin, weekday_end, df_weekday)

        rev_7days, rev_30days, start_date, end_date = combine_4year_revenue_cruise(event_filter, room_data, col_name,
                                                                                   day_after, day_before,
                                                                                   weekday_begin, weekday_end)
        flag, mean_frac = data_event2(rev_7days, rev_30days)
        result_ls.append(
            [i, event_id, flag, mean_frac, str(end_date.year), start_date, end_date, day_after, day_before])

    result_df = pd.DataFrame(result_ls, columns=['EventName', 'Id', 'Signal', 'Frac', 'year', 'Start_day', 'End_day',
                                                 'day_after', 'day_before'])
    result_df = copy_event_date[['Dates', 'Name', 'Year', 'weekday']].merge(result_df, left_on=['Name'],
                                                                       right_on=['EventName'],
                                                                       how='left')

    if len(result_ls) != 0:
        start_day_new = []
        end_day_new = []
        for i in result_df.index:
            date_range_temp = [result_df['Dates'].iloc[i], result_df['Dates'].iloc[i]]
            result_df_filter = result_df.loc[i]
            start_day_temp, end_day_temp = transform_full_week_cruise(date_range_temp, result_df_filter,
                                                                      weekday_begin, weekday_end,
                                                                      result_df['day_after'].iloc[i],
                                                                      result_df['day_before'].iloc[i])
            start_day_new.append(start_day_temp)
            end_day_new.append(end_day_temp)

        result_df['Start_day'] = start_day_new
        result_df['End_day'] = end_day_new

    result_df.loc[result_df['Signal'].isnull(), 'Signal'] = False

    # result_df = result_df[['Dates', 'Name', 'Year', 'EventName', 'Id', 'Signal']]

    holiday_data = pd.DataFrame(columns=['Id', 'EventName', 'From', 'To'])
    for i in result_df.index:
        if result_df.loc[i, 'Signal'] != False:
            holiday_data.loc[i, 'Id'] = int(result_df.loc[i, 'Id'])
            holiday_data.loc[i, 'EventName'] = result_df.loc[i, 'EventName']
            holiday_data.loc[i, 'From'] = result_df['Start_day'].iloc[i]
            holiday_data.loc[i, 'To'] = result_df['End_day'].iloc[i]

    holiday_data['From'] = [i - timedelta(days=cycle_value) for i in holiday_data['From']]
    holiday_data['To'] = [i - timedelta(days=cycle_value) for i in holiday_data['To']]
    holiday_data = holiday_data.reset_index(drop = True)
    holiday_data['Range_Day'] =  [create_dates_between_time(holiday_data['From'][i] - timedelta(days =1), holiday_data['To'][i]) for i in range(len(holiday_data))]
    list_holiday = []
    for i in range(len(holiday_data)):
        list_holiday = list_holiday + holiday_data['Range_Day'][i]
    del holiday_data['Range_Day']
    print('########################################end - holiday######################')
    return holiday_data, list_holiday

def back_days_transform(given_range_date, nb_days = 30):
    from_date = given_range_date[0]
    to_date = given_range_date[1]

    new_from_date = from_date - timedelta(days= nb_days)
    new_to_date = to_date + timedelta(days= nb_days)

    return [new_from_date, new_to_date]

#def combine_4year_revenue(event_filter, room_revenue, col_name, after=0, before=0):
#    rev_7days_4year = []
#    rev_30days_4year = []
#
#    for j in event_filter.index:
#        event_day = datetime(event_filter['Dates'][j].year, event_filter['Dates'][j].month,
#                             event_filter['Dates'][j].day)
#        date_range = [event_day, event_day]
#
#        # 7 days
#        event_7days = transform_full_week(date_range, after, before)
#        range_event_7days = pd.date_range(start=event_7days[0], end=event_7days[1])
#        df_7days = room_revenue[room_revenue['Date'].isin(range_event_7days)]
#        df_7days.reset_index(inplace=True)
#        rev_7days_4year.append(df_7days[col_name])
#        start_date = df_7days['Date'].iloc[0]
#        end_date = df_7days['Date'].iloc[-1]
#
#        # 30 days
#        event_30days = back_days_transform(date_range, 15)
#        range_event_30days = pd.date_range(start=event_30days[0], end=event_30days[1])
#        df_30days = room_revenue[room_revenue['Date'].isin(range_event_30days)]
#        df_30days.reset_index(inplace=True)
#        rev_30days_4year.append(df_30days[col_name])
#
#    rev_7days_4year = pd.concat(rev_7days_4year, axis=1)
#    rev_30days_4year = pd.concat(rev_30days_4year, axis=1)
#
#    rev_7days_4year[col_name + '_4year'] = rev_7days_4year.apply(lambda x: pd.ewma(x, alpha=0.6, adjust=False)[-1],
#                                                                 axis=1)
#    rev_30days_4year[col_name + '_4year'] = rev_30days_4year.apply(lambda x: pd.ewma(x, alpha=0.6, adjust=False)[-1],
#                                                                   axis=1)
#
#    return rev_7days_4year[col_name + '_4year'], rev_30days_4year[col_name + '_4year'], start_date, end_date

def combine_4year_revenue_cruise(event_filter, room_revenue, col_name, after, before, weekday_begin, weekday_end):
    '''
    transform 4 year data into 1 year data
    :param room_revenue: dataframe, (e.g. columns: ['Date', u'50_0_RV', u'50_224_RV', u'50_225_RV', ..])
    :param event_filter: dataframe, (e.g. columns: ['Dates', 'Id', 'Name', 'Year'])
    :param col_name: target department
    :after :
    :before :
    :weekday_begin :
    :weekday_end :
    :return: 
        rev_7days_4year: pandas Series contains combined revenue of 7 days
        rev_30days_4year: pandas Series contains combined revenue of 30 days
        start_date: begin date of 7 days
        end_date: end date of 7 days
    '''
    if len(event_filter) == 0:
        return 0, 0, 0, 0

    rev_7days_4year = []
    rev_30days_4year = []

    nb_days = (weekday_end - weekday_begin) + 1

    for j in event_filter.index:
        # j = 2
        event_day = datetime(event_filter['Dates'][j].year, event_filter['Dates'][j].month, event_filter['Dates'][j].day)
        date_range = [event_day, event_day]

        event_filter_lev2 = event_filter.loc[j, :]

        # 7 days
        event_7days = transform_full_week_cruise(date_range, event_filter_lev2, weekday_begin, weekday_end, after, before)
        range_event_7days = pd.date_range(start=event_7days[0], end=event_7days[1])
        df_7days = room_revenue[room_revenue['Date'].isin(range_event_7days)]
        df_7days.reset_index(inplace=True)
        rev_7days_4year.append(df_7days[col_name])
        start_date = df_7days['Date'].iloc[0]
        end_date = df_7days['Date'].iloc[-1]

        # 30 days
        event_30days = back_days_transform(date_range, nb_days * 2)
        range_event_30days = pd.date_range(start=event_30days[0], end=event_30days[1])
        df_30days = room_revenue[room_revenue['Date'].isin(range_event_30days)]
        df_30days.reset_index(inplace=True)
        rev_30days_4year.append(df_30days[col_name])

    rev_7days_4year = pd.concat(rev_7days_4year, axis=1)
    rev_30days_4year = pd.concat(rev_30days_4year, axis=1)

    rev_7days_4year[col_name + '_4year'] = rev_7days_4year.apply(lambda x: pd.ewma(x, alpha=0.6, adjust=False)[-1], axis=1)
    rev_30days_4year[col_name + '_4year'] = rev_30days_4year.apply(lambda x: pd.ewma(x, alpha=0.6, adjust=False)[-1], axis=1)

    return rev_7days_4year[col_name + '_4year'], rev_30days_4year[col_name + '_4year'], start_date, end_date

# after is day to_date after, can minus before can minus to grow up
#def transform_full_week(given_range_date,after=0,before=0):
#    from_date = given_range_date[0]
#    to_date = given_range_date[1]
#
#    new_from_date = from_date - timedelta(days=from_date.weekday()+before)
#    new_to_date = to_date + timedelta(days=6 - to_date.weekday()+after)
#
#    return [new_from_date, new_to_date]

def transform_full_week_cruise(given_range_date, event_filter_lev2, weekday_begin, weekday_end, after, before):
    '''
    expand date range between from_date and to_date based on after, before
    '''
    from_date = given_range_date[0]
    to_date = given_range_date[1]


    new_from_date = from_date - timedelta(days= int(event_filter_lev2['weekday'] - weekday_begin + before))
    new_to_date = to_date + timedelta(days= int(weekday_end - event_filter_lev2['weekday'] + after))

    return [new_from_date, new_to_date]

#def expand_effect_range(flag, event_filter, room_revenue, col_name):
#    '''
#    Save outliers in json
#    :param flag: string or boolean variable (e.g.: 'positive', 'negative', False)
#    :param event_filter: dataframe (columns: ['Dates', u'Id', 'Name', 'Year'])
#    :param room_revenue: dataframe (e.g. columns: ['Date', u'50_0_RV', u'50_224_RV', u'50_225_RV', ..])
#    :param col_name: string (e.g.: '50_0_RV')
#    :return: day_after, day_before: int, (e.g.: 7, 7)
#    '''
#    day_after = 0
#    day_before = 0
#    if flag == 'positive':
#        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue(event_filter, room_revenue, col_name, 7,
#                                                                              -7)
#        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
#        if flag1 == 'positive':
#            day_after = 7
#        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue(event_filter, room_revenue, col_name, -7,
#                                                                              7)
#        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
#        if flag1 == 'positive':
#            day_before = 7
#
#    if flag == 'negative':
#        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue(event_filter, room_revenue, col_name, 7,
#                                                                              -7)
#        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
#        if flag1 == 'negative':
#            day_after = 7
#        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue(event_filter, room_revenue, col_name, -7,
#                                                                              7)
#        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
#        if flag1 == 'negative':
#            day_before = 7
#    return day_after, day_before


def expand_effect_range_cruise(flag, event_filter, room_revenue, col_name, weekday_begin, weekday_end, df_weekday = None):
    '''
    Save outliers in json
    :param flag: string or boolean variable (e.g.: 'positive', 'negative', False)
    :param event_filter: dataframe (columns: ['Dates', u'Id', 'Name', 'Year'])
    :param room_revenue: dataframe (e.g. columns: ['Date', u'50_0_RV', u'50_224_RV', u'50_225_RV', ..])
    :param col_name: string (e.g.: '50_0_RV')
    :return: day_after, day_before: int, (e.g.: 7, 7)
    '''
    day_after = 0
    day_before = 0

    nb_days = (weekday_end - weekday_begin + 1)

    event_filter_org = event_filter.copy(deep = True)


    if (flag == 'positive') or (flag == 'negative'):

        if not isinstance(df_weekday, (int, long)):
            ls_ind = check_begin_day(event_filter_org, nb_days, -nb_days,
                                     weekday_begin, weekday_end, df_weekday)

            event_filter = event_filter_org.loc[ls_ind, :]
        else:
            event_filter = event_filter_org


        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue_cruise(event_filter, room_revenue, col_name,
                                                                                     nb_days, -nb_days,
                                                                                     weekday_begin, weekday_end)
        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
        if flag1 == flag:
            day_after = nb_days

        if not isinstance(df_weekday, (int, long)):
            ls_ind = check_begin_day(event_filter_org, -nb_days, nb_days,
                                     weekday_begin, weekday_end, df_weekday)

            event_filter = event_filter_org.loc[ls_ind, :]
        else:
            event_filter = event_filter_org


        rev_7days, rev_30days, start_date1, end_date1 = combine_4year_revenue_cruise(event_filter, room_revenue, col_name,
                                                                                     -nb_days, nb_days,
                                                                                     weekday_begin, weekday_end)
        flag1, mean_frac1 = data_event2(rev_7days, rev_30days)
        if flag1 == flag:
            day_before = nb_days

    return day_after, day_before

def check_begin_day(event_filter, after, before,
                    weekday_begin, weekday_end, df_weekday):
    ls_ind = []

    for j in event_filter.index:
        # j = 2
        event_day = datetime(event_filter['Dates'][j].year, event_filter['Dates'][j].month,
                             event_filter['Dates'][j].day)
        date_range = [event_day, event_day]

        event_filter_lev2 = event_filter.loc[j, :]

        # 7 days
        event_7days = transform_full_week_cruise(date_range, event_filter_lev2, weekday_begin, weekday_end,
                                                 after, before)

        # Check whether start-date is a begin-of-weekday
        back_date = '{}-{:02}-{:02}'.format(event_7days[0].year, event_7days[0].month, event_7days[0].day)
        if len(df_weekday.loc[df_weekday['Dates'] == back_date, 'weekday']) != 0:
            if df_weekday.loc[df_weekday['Dates'] == back_date, 'weekday'].values[0] == weekday_begin:
                ls_ind.append(j)

    return ls_ind

def data_event2(df_7days, df_30days):
    df_7days = pd.Series(df_7days)
    df_30days = pd.Series(df_30days)

    mean_7days = df_7days.mean()
    mean_30days = df_30days.mean()

    mean_frac = mean_7days / mean_30days

    if (mean_frac >= 1.3):
        flag = 'positive'
    elif (mean_frac <= 0.7):
        flag = 'negative'
    else:
        flag = False

    return flag, mean_frac

#===================== Creating json upload =====================

def period_sample_func():
    return {'Type': 0, 'SpecialPeriod': 0, 
            'Value1': 0, 'Value2': 0,
            'Value3': 0, 'Dates':[]}

def save_outlier_json(outlier_date, json_upload):
    '''
    Save outliers in json
    :param outlier_date: list of datestring
    :param json_upload: json variable
    :return: json_upload: json variable
    '''
    if len(outlier_date) > 0:
        period_sample = period_sample_func()
        period_sample['Type'] = 2
        period_sample['Value1'] = 0
        period_sample['Value2'] = 0
        for i in range(len(outlier_date)):
            period_sample['Dates'].append({'From': outlier_date[i],
                                           'To': outlier_date[i]})

        json_upload['Periods'].append(period_sample)
        return json_upload
    else:
        return json_upload

def save_cruise_json(cruise_lack_date, json_upload):
    '''
    Save outliers in json
    :param cruise_lack_date: lack_date_from_to: dataframe contains off-working days
    :param json_upload: json variable
    :return: json_upload: json variable
    '''
    if len(cruise_lack_date) > 0:
        period_sample = period_sample_func()
        period_sample['Type'] = 0
        period_sample['Value1'] = 0
        period_sample['Value2'] = 0
        for i in range(len(cruise_lack_date)):
            period_sample['Dates'].append({'From': cruise_lack_date['From'][i].strftime('%Y-%m-%d'),
                                           'To': cruise_lack_date['To'][i].strftime('%Y-%m-%d')})

        json_upload['Periods'].append(period_sample)
        return json_upload
    else:
        return json_upload



def save_holiday_json(holiday_data, json_upload, dep_id):
    '''
    Save holiday in json
    :param holiday_data: dataframe, (e.g. columns: ['Id', 'EventName', 'From', 'To'])
    :param json_upload: json variable
    :param dep_id:  int
    :return: json_upload: dataframe with columns ['Id', 'EventName', 'From', 'To']
    '''
    if len(holiday_data) > 0:
        for i in holiday_data['EventName'].unique():
            holiday_filter = holiday_data[holiday_data['EventName'] == i]
            period_sample = period_sample_func()
            period_sample['Type'] = 1
            period_sample['Value1'] = 0
            period_sample['Value2'] = 0
            period_sample['SpecialPeriod'] = int(holiday_filter['Id'].values[0])
            for j in holiday_filter.index:
                period_sample['Dates'].append({'From': holiday_filter['From'][j].strftime('%Y-%m-%d'),
                                               'To': holiday_filter['To'][j].strftime('%Y-%m-%d')})
            json_upload['Periods'].append(period_sample)

        json_upload['H_Id'] = int(dep_id)
        return json_upload
    else:
        return json_upload