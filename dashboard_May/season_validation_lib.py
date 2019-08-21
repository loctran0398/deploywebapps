from d2o_common.legacy_lib import database_api as db_api
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import json
import numpy as np
import os
import time
import sys


HOST = db_api.HOST

def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)

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
    #  ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
    error = np.sum(np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred']))) / n
    return round(error, 2)

def add_years(d, years):
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).
    """
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))
    
def compute_sMAPE(data_df, hotel, dep_id):
    
    smape_1month_base = sMAPE(data_df['actual'][-30:], data_df['base_forecast'][-30:])
    smape_3month_base = sMAPE(data_df['actual'][-90:], data_df['base_forecast'][-90:])
    smape_6month_base = sMAPE(data_df['actual'][-180:], data_df['base_forecast'][-180:])
    smape_12month_base = sMAPE(data_df['actual'], data_df['base_forecast'])
    
    smape_1month_rev = sMAPE(data_df['actual'][-30:], data_df['rev_forecast'][-30:])
    smape_3month_rev = sMAPE(data_df['actual'][-90:], data_df['rev_forecast'][-90:])
    smape_6month_rev = sMAPE(data_df['actual'][-180:], data_df['rev_forecast'][-180:])
    smape_12month_rev = sMAPE(data_df['actual'], data_df['rev_forecast'])

    result = {}
    result['Driver'] = '_'.join([str(hotel), str(dep_id)])
    result['sMAPE_1month_base'] = smape_1month_base
    result['sMAPE_3month_base'] = smape_3month_base
    result['sMAPE_6month_base'] = smape_6month_base
    result['sMAPE_12month_base'] = smape_12month_base
    result['sMAPE_1month_rev'] = smape_1month_rev
    result['sMAPE_3month_rev'] = smape_3month_rev
    result['sMAPE_6month_rev'] = smape_6month_rev
    result['sMAPE_12month_rev'] = smape_12month_rev
    
    return result

def get_data_validation(client, hotel, Mydep, from_date, to_date):         
    try:
        link_base = '%s/PMIAuto/Season/Days/Revenue/%s/?h_id=%s&from=%s&to=%s' % \
                                (HOST, client, Mydep, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
        json_base = request_data(link_base)
    except:
        print("WARNING: Cannot get predicted data of base model from server request")

    base_df = json_base[['Date', 'Day', 'Period_Id', 'Period_Type', 'Revenue']]
    base_df.columns = ['Date', 'Day', 'Period_Id', 'Period_Type', 'base_forecast']
    response_flag = 0
    num = 0
    while (response_flag != 200) and (num < 4):
        try:
            if num >= 2:
                json_rev = pd.DataFrame()
                temp_from_date = from_date
                temp_to_date = from_date + timedelta(days = 365)
                while (temp_from_date <= to_date) and (temp_to_date <= to_date): 
                    print('Try getting data from %s to %s' % (temp_from_date.strftime('%Y-%m-%d'), temp_to_date.strftime('%Y-%m-%d')))
                    link_rev = '%s/PMIAuto/RevenueDriver/Analysis/%s/?h_id=%s&from=%s&to=%s' % \
                               (HOST, client, Mydep, temp_from_date.strftime('%Y-%m-%d'), temp_to_date.strftime('%Y-%m-%d'))
                    d2o_response = requests.get(link_rev)
                    json_rev = pd.concat([json_rev, pd.DataFrame(json.loads(d2o_response.text))], ignore_index = True)
                    
                    temp_from_date = temp_to_date + timedelta(days = 1)
                    temp_to_date = temp_to_date + timedelta(days = 365)
                    if temp_to_date > to_date:
                        temp_to_date = to_date
                        
                    response_flag = d2o_response.status_code
                    print('Status: %s' % (response_flag))
            else:    
                link_rev = '%s/PMIAuto/RevenueDriver/Analysis/%s/?h_id=%s&from=%s&to=%s' % \
                                    (HOST, client, Mydep, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
                response = requests.get(link_rev)
                json_rev = pd.DataFrame(json.loads(response.text))
                response_flag = response.status_code
        except:
            response_flag = 400
            json_rev = pd.DataFrame()
            
        if response_flag != 200:
            num += 1
            print("Status code: %s. Try: %s times." % (response_flag, num))
            time.sleep(20)
            
    if (response_flag != 200) or (len(json_rev) == 0):
        print(link_rev)
        print("WARNING: Cannot get predicted data of revenue model from server request.")
        sys.exit(0) #continue

    rev_list = []
    for i in range(len(json_rev['Dates'])):
        rev_list.append([json_rev['Dates'][i]['Date'],
                        json_rev['Dates'][i]['Department']['ActualRevenue'],
                        json_rev['Dates'][i]['AutoDriverBest']['ActualRevenue']])
    
    rev_df = pd.DataFrame(rev_list, columns=['Date', 'actual', 'rev_forecast'])
    data_df = pd.merge(rev_df, base_df, on = 'Date')
    
    data_df.loc[data_df['actual'] < 0, 'actual'] = 0.0
    data_df.loc[data_df['rev_forecast'] < 0, 'rev_forecast'] = 0.0
    data_df.loc[data_df['base_forecast'] < 0, 'base_forecast'] = 0.0
    
    return data_df

def create_dates_between_time(from_date, to_date):
    numdays = (to_date - from_date).days + 1
    date_list = [(to_date - timedelta(days = x)).strftime('%Y-%m-%d') for x in range(0, numdays)]
    return date_list

def make_validation_file_revenue(client, hotel, Mydep, ss_type, data_conv_pv, nb_years, data_df, ending_date):
    try: 
        data_df['Date'] = [(datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')).strftime('%Y-%m-%d') for i in data_df['Date']]
        data_df = data_df.set_index('Date')
        
        obj_df = data_conv_pv.copy()
        obj_df['From'] = [i.strftime('%Y-%m-%d') for i in obj_df['From']]
        obj_df['Period_Id'] = [int(data_df.loc[i, 'Period_Id']) for i in obj_df['From']]
        weekday_list = (obj_df.columns).difference(['From', 'To', 'old_group', 'Group','Period_Id'])
        period_list = obj_df['Period_Id'].unique()
        
        obj_dict = {}
        for i in period_list:
            for j in weekday_list:
                obj_dict[str(i) + '_' + str(j+1)] = (obj_df[obj_df['Period_Id'] == i][j]).std()
                
        obj_dict_df = pd.DataFrame.from_dict(obj_dict, orient = 'index')
        obj_dict_df = obj_dict_df.reset_index()
        obj_dict_df.columns = ['Period_Day', 'Std_Value']
        
        validation_df = pd.DataFrame()
        validation_df['Date'] = data_df.index
        validation_df['Period_Id'] = [str(int(data_df.loc[idate, 'Period_Id'])) for idate in validation_df['Date']]
    #    validation_df['Day'] = [int(data_df.loc[idate, 'Day']) for idate in validation_df['Date']]
        validation_df['Day'] = [str(int((datetime.strptime(idate, '%Y-%m-%d').weekday()+ 8)%7 + 1)) for idate in validation_df['Date']]
        validation_df['Outlier'] = [1 if data_df.loc[idate, 'Period_Type'] == 2 else 0 for idate in validation_df['Date']]
        validation_df['Holiday'] = [1 if data_df.loc[idate, 'Period_Type'] == 1.0 else 0 for idate in validation_df['Date']]
        validation_df['Actual'] = [data_df.loc[idate, 'actual'] for idate in validation_df['Date']]
        validation_df['Base_Forecast'] = [data_df.loc[idate, 'base_forecast'] for idate in validation_df['Date']]
        validation_df['Rev_Forecast'] = [data_df.loc[idate, 'rev_forecast'] for idate in validation_df['Date']]
        validation_df['sMAPE_base'] = [sMAPE(pd.Series(data_df.loc[idate, 'actual']), pd.Series(data_df.loc[idate, 'base_forecast'])) \
                     for idate in validation_df['Date']]
        validation_df['sMAPE_rev'] = [sMAPE(pd.Series(data_df.loc[idate, 'actual']), pd.Series(data_df.loc[idate, 'rev_forecast'])) \
                     for idate in validation_df['Date']]
        
        validation_df['Period_Day'] = validation_df[['Period_Id', 'Day']].apply(lambda x: '_'.join(x), axis=1)
        validation_df = validation_df.merge(obj_dict_df, on = 'Period_Day')
        del validation_df['Period_Day']
        
        validation_df['Season_Type'] = ss_type
        validation_df['Data_Quality'] = nb_years
        validation_df['Nb_of_Periods'] = len(period_list)    
        
        validation_df = validation_df.sort_values('Date').reset_index(drop=True)
        
        dir_result = '/home/tp7/d2o_service/department_seasons/season_validation/%s' % (client)
        if not os.path.isdir(dir_result):
            os.mkdir(dir_result)
               
        file_name = os.path.join(dir_result, 'season_revenue_{}_{}_{}.csv'.format(hotel, Mydep, ending_date.strftime('%Y-%m-%d')))
        validation_df.to_csv(file_name, index = False)
        print('Export validation file successfully')
        
    except Exception as e:
        print('Export validation file unsuccessfully')
        print(e)
    
def make_validation_file_labor(client, hotel, Mydep, ss_type, data_conv_pv, nb_years, data_df, ending_date):
    try: 
        data_df['Date'] = [(datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')).strftime('%Y-%m-%d') for i in data_df['Date']]
        data_df = data_df.set_index('Date')
        
        obj_df = data_conv_pv.copy()
        obj_df['From'] = [i.strftime('%Y-%m-%d') for i in obj_df['From']]
        obj_df['Period_Id'] = [int(data_df.loc[i, 'Period_Id']) for i in obj_df['From']]
        weekday_list = (obj_df.columns).difference(['From', 'To', 'old_group', 'Group','Period_Id'])
        period_list = obj_df['Period_Id'].unique()
        
        obj_dict = {}
        for i in period_list:
            for j in weekday_list:
                obj_dict[str(i) + '_' + str(j+1)] = (obj_df[obj_df['Period_Id'] == i][j]).std()
                
        obj_dict_df = pd.DataFrame.from_dict(obj_dict, orient = 'index')
        obj_dict_df = obj_dict_df.reset_index()
        obj_dict_df.columns = ['Period_Day', 'Std_Value']
        
        validation_df = pd.DataFrame()
        validation_df['Date'] = data_df.index
        validation_df['Period_Id'] = [str(int(data_df.loc[idate, 'Period_Id'])) for idate in validation_df['Date']]
    #    validation_df['Day'] = [int(data_df.loc[idate, 'Day']) for idate in validation_df['Date']]
        validation_df['Day'] = [str(int((datetime.strptime(idate, '%Y-%m-%d').weekday()+ 8)%7 + 1)) for idate in validation_df['Date']]
        validation_df['Outlier'] = [1 if data_df.loc[idate, 'Period_Type'] == 2 else 0 for idate in validation_df['Date']]
        validation_df['Holiday'] = [1 if data_df.loc[idate, 'Period_Type'] == 1.0 else 0 for idate in validation_df['Date']]

        validation_df['Period_Day'] = validation_df[['Period_Id', 'Day']].apply(lambda x: '_'.join(x), axis=1)
        validation_df = validation_df.merge(obj_dict_df, on = 'Period_Day')
        del validation_df['Period_Day']
        
        validation_df['Season_Type'] = ss_type
        validation_df['Data_Quality'] = nb_years
        validation_df['Nb_of_Periods'] = len(period_list)    
        
        validation_df = validation_df.sort_values('Date').reset_index(drop=True)
        
        dir_result = '/home/tp7/d2o_service/department_seasons/season_validation/%s' % (client)
        if not os.path.isdir(dir_result):
            os.mkdir(dir_result)
               
        file_name = os.path.join(dir_result, 'season_labor_{}_{}_{}.csv'.format(hotel, Mydep, ending_date.strftime('%Y-%m-%d')))
        

        validation_df.to_csv(file_name, index = False)
        print('Export validation file successfully')
        
    except Exception as e:
        print('Export validation file unsuccessfully')
        print(e)



