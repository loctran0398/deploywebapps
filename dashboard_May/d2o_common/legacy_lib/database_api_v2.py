from sklearn import metrics
import pandas as pd
import numpy as np
import requests
import json
import sys
import os
import time
import socket
from datetime import datetime, date, timedelta
from d2o_common.util import logger as log

HOST = "http://172.16.0.51:8485"

#===================== API links =====================
def get_ip_address(print_flag = True):
    '''
    Get IP address of working server
    :param print_flag: print server 's information if True
    :return HOST: IP address 
    '''
    original_stdout = sys.stdout
    if print_flag is False:
        sys.stdout = open(os.devnull, 'w')
        
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
    if s.getsockname()[0].split('.')[3] == '23':
        print('Production server')
        HOST = "http://10.50.10.4:8485"
        print(HOST)
    else:
        print('Test server')
        HOST = "http://172.16.0.51:8485"
        print(HOST)
    s.close()
    sys.stdout = original_stdout
    return HOST

#HOST = get_ip_address(print_flag = False)
CLIENT_LINK                 = "%s/Client/Clients" % (HOST)
HOTEL_IN_DATABASE_LINK      = "%s/Hierarchy/Properties/{client_id}" % (HOST)
DEPTS_IN_HOTEL_LINK         = "%s/Hierarchy/Departments/{client_id}/?h_id_parent={h_id_parent}" % (HOST)
ONE_DEPT_IN_HOTEL_LINK      = "%s/Hierarchy/Department/{client_id}/?h_id={h_id}" % (HOST)

SEASON_LINK                 = "%s/PMIAuto/Season/{client_id}/?h_id={h_id}&type={ss_type}" % (HOST)
SEASONDATES_LINK            = "%s/PMIAuto/Season/Days/{client_id}/?h_id={h_id}&type={ss_type}&from={from_time}&to={to_time}" % (HOST)

OTB_LINK                    = '%s/Otb/Archive/Total/{client_id}/?h_id={h_id}&from={from_time}&to={to_time}' % (HOST)

LOG_LINK                    = '%s/PMIAuto/Log/{client_id}' % (HOST)

ANALYZER_POST_LINK          = '%s/PMIAuto/Analyzer/Selected/Model/{client_id}' % (HOST)
FOODCOST_POST_LINK          = '%s/PMIAuto/FoodCost/{client_id}' % (HOST)

#===================== write log =====================        
def write_log(client, dep_id, message, status, service_type, write):
    '''
    Wite log to server
    :param client: client ID
    :param dep_id: department ID
    :param message: string of log messages
    :param status: status code
    :param write: write log to server if True
    '''
    if write:
        log_file_client = {'Date': datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d'),
                                       'H_Id': dep_id, 'Message': message, 'Status': status, 'Type': service_type}
        log_file_client = [log_file_client]
        url = LOG_LINK.format(client_id = client)
        post_api(link=url, json_data=log_file_client)
        log.info("Writing JSON log to API")

#===================== Download & Upload data =====================
def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)

def post_api(link, json_data):
    response = requests.post(url = link, json = json_data)
    return response

def upload_analyzer_database(client, upload_db):
    url = ANALYZER_POST_LINK.format(client_id = client)
    response = post_api(link=url, json_data = upload_db)
    log.info("Writing Analyzer to API: %s" % response.status_code)
    return response

def upload_foodcost_database(client, upload_db):
    url = FOODCOST_POST_LINK.format(client_id = client)
    response = post_api(link=url, json_data = upload_db)
    log.info("Writing Auto Purchase to API: %s" % response.status_code)
    return response

#===================== Get information from API =====================
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

def get_all_client():
    return request_data(CLIENT_LINK)

def get_hotels_in_database(client):
    link = HOTEL_IN_DATABASE_LINK.format(client_id=client)
    return request_data(link)

def get_depts_info_in_hotel(client, hotel_id):
    link = DEPTS_IN_HOTEL_LINK.format(client_id=client, h_id_parent=hotel_id)
    return request_data(link)

def get_one_dept_info_in_hotel(client, hotel_id):
    link = ONE_DEPT_IN_HOTEL_LINK.format(client_id=client, h_id=hotel_id)
    return json.loads(requests.get(link).text)

def get_segs_info_in_dept(client, dept_id):
    dept_df = get_one_dept_info_in_hotel(client, dept_id)
    seg_df = pd.DataFrame.from_dict(dept_df['Segments'])
    room_df = pd.DataFrame()
    room_df['Id'] = [0]
    room_df['Name'] = dept_df['Name']
    seg_df = pd.concat([room_df, seg_df], ignore_index = True)
    return seg_df

def get_valid_starting_date(client, dep_id, ss_type):
    link = SEASON_LINK.format(client_id=client, h_id=dep_id, ss_type=ss_type)
    df = request_data(link)
    return datetime.strptime(df['ValidHistoryFrom'][0], '%Y-%m-%dT%H:%M:%S').date()

def get_depts_season(client, hotel_id, remove_hotel_id = True):
    '''
    Get all deparments which have season data in a hotel
    :param client: client ID
    :param hotel_id: hotel ID 
    :param remove_hotel_id: include room department if True
    :return deps: dictionary contains deparments which can generate season
    '''
    try:
        dept_df = get_depts_info_in_hotel(client, hotel_id)
        dept_df = dept_df[(dept_df['Revenue'] == True) | (dept_df['Labor'] == True) | (dept_df['FoodCost'] == True)]
        if remove_hotel_id:
            dept_df = dept_df[dept_df['H_Id'] != int(hotel_id)]
        dept_season = dept_df['H_Id'].values
        return dept_season.tolist()
    except Exception as e:
        print('No satisfied departments for this hotel')
        print(e)
        return []

def get_depts_analyzer(client, hotel_id):
    dept_df = get_depts_info_in_hotel(client, hotel_id)
    dept_df = dept_df[dept_df['Revenue'] == True]
    l_df = []
    for i in dept_df.index:
        segs = dept_df.loc[i, 'Segments']
        segs.append({u'Id': 0, u'Name': u'Total'})
        df_t = pd.DataFrame(segs)
        df_t['hotel_id'] = str(hotel_id)
        df_t['name'] = dept_df.loc[i, 'Name'] + "_" + df_t['Name']
        df_t['id']   = str(dept_df.loc[i, 'H_Id']) + "_" + df_t['Id'].astype(str)
        df_t['otb'] = dept_df.loc[i, 'Otb']
        l_df.append(df_t[['hotel_id', 'name', 'id', 'otb']])
    return pd.concat(l_df, ignore_index=True)

def get_depts_foodcost(client, hotel_id):
    dept_df = get_depts_info_in_hotel(client, hotel_id)
    l_df = []
    for i in dept_df.index:
        segs = dept_df.loc[i, 'Segments']
        segs.append({u'Id': 0, u'Name': u'Total'})
        df_t = pd.DataFrame(segs)
        df_t['hotel_id'] = str(hotel_id)
        df_t['name'] = dept_df.loc[i, 'Name'] + "_" + df_t['Name']
        df_t['id']   = str(dept_df.loc[i, 'H_Id']) + "_" + df_t['Id'].astype(str)
        df_t['foodcost'] = dept_df.loc[i, 'FoodCost']
        l_df.append(df_t[['hotel_id', 'name', 'id', 'foodcost']])
    return pd.concat(l_df, ignore_index=True)

#===================== Get data from API =====================
def get_season_data(client, dep_id, from_time, current_time, type_season):
    '''
    Get season data from a time range, remove off-working dates and outliers
    :param client: client ID
    :param dep_id: department ID
    :param from_time: begin time with format %Y-%m-%d
    :param current_time: end time with format %Y-%m-%d
    :param type_season: type of season (revenue, labor, foodcost))
    :return season_df: dataframe which contains dates of month, days of week, period id 
    '''
    if type_season == 'revenue':
        link = SEASONDATES_LINK.format(client_id=client, h_id=dep_id, ss_type=0, from_time=from_time, to_time=current_time)
    if type_season == 'labor':
        link = SEASONDATES_LINK.format(client_id=client, h_id=dep_id, fss_type=1, rom_time=from_time, to_time=current_time)
    if type_season == 'foodcost':
        link = SEASONDATES_LINK.format(client_id=client, h_id=dep_id, ss_type=2, from_time=from_time, to_time=current_time)
        
    season_df = request_data(link)
    season_df['Date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in season_df['Date']]
    
    #Remove off-working dates
    season_df = season_df.loc[~((season_df['Day'] == 0) & (season_df['Period_Type'] != 1))]
    #Remove outliers
    season_df = season_df.loc[season_df['Period_Type'] != 2, ['Date', 'Day', 'Period_Id']].reset_index(drop=True)
    
    return season_df

def get_otb_data(client, dep_id, from_time, current_time):
    '''
    Get otb data from a time range
    :param client: client ID
    :param dep_id: department ID
    :param from_time: begin time with format %Y-%m-%d
    :param current_time: end time with format %Y-%m-%d
    :return otb_df: dataframe which contains revenue value and leadtime could be computed from Date - ImportDate
    '''
    otb_df = pd.DataFrame()
    response_flag = 0
    num = 0
    
    while (response_flag != 200) and (num < 4):
        try:
            if num >= 2:
                temp_from_date = from_time
                temp_to_date = from_time + timedelta(days = 365)
                while (temp_from_date <= current_time) and (temp_to_date <= current_time): 
                    print('Try getting otb data from %s to %s' % (temp_from_date.strftime('%Y-%m-%d'), temp_to_date.strftime('%Y-%m-%d')))
                    link = OTB_LINK.format(client_id = client, h_id = dep_id, \
                           from_time = temp_from_date.strftime('%Y-%m-%d'), to_time = temp_to_date.strftime('%Y-%m-%d'))
                    response = requests.get(link)
                    json_response = json.loads(response.text)
                    otb_df = pd.concat([otb_df, pd.DataFrame(json_response['Items'])], ignore_index = True)
                    
                    temp_from_date = temp_to_date + timedelta(days = 1)
                    temp_to_date = temp_to_date + timedelta(days = 365)
                    if temp_to_date > current_time:
                        temp_to_date = current_time
                        
                    response_flag = response.status_code
                    print('Status: %s' % (response_flag))
            else:    
                link = OTB_LINK.format(client_id = client, h_id = dep_id, \
                           from_time = from_time.strftime('%Y-%m-%d'), to_time = current_time.strftime('%Y-%m-%d'))
                response = requests.get(link)
                json_response = json.loads(response.text)
                otb_df = pd.DataFrame(json_response['Items'])
                response_flag = response.status_code
        except:
            response_flag = 400
            otb_df = pd.DataFrame()
            
        if response_flag != 200:
            num += 1
            print("Status code: %s. Try: %s times." % (response_flag, num))
            time.sleep(20)
            
    if (response_flag != 200) or (len(otb_df) == 0):
        print(link)
        print("WARNING: Cannot get otb data from server request.")
    
    return otb_df 

def get_data_foodcost(client, dep_id, from_time, current_time):
    '''
    client, hotel_id, dep_id: 
    from_time, to_time: datetime '%Y-%m-%d'
    '''
    HOST = get_ip_address()
    foodrev_link = '%s/FoodCost/Analysis/%s/?h_id=%s&from=%s&to=%s' % (HOST, client, \
               dep_id, from_time.strftime('%Y-%m-%d'), current_time.strftime('%Y-%m-%d'))
    foodcost_df = request_data(foodrev_link)
    
    if len(foodcost_df) > 0:
        foodcost_df['Date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in foodcost_df['Date']]
        foodcost_df = foodcost_df[['Date','Actual_Purchase','Actual_FoodRevenue']]
    return foodcost_df

#===================== Measurement error =====================
def sMAPE(y_true, y_pred, weight = None):
    '''
    Return accuracy measure based on percentage errors
    :param y_true: Series of actual values
    :param y_pred: Series of forecasted values
    return: symmetric mean absolute percentage error (sMAPE)
    '''
    try:
        y_true = y_true.reset_index(drop=True)
        y_pred = y_pred.reset_index(drop=True)
    except:
        print('sMAPE function: y_true and y_pred should be pd.Series.')
    a = pd.DataFrame()
    a['y_true'] = y_true
    a['y_pred'] = y_pred
    n = len(y_pred)
    a = a[(a['y_true']!=0) | (a['y_pred']!=0)]
#    ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
    if weight is None:
        smape = np.sum(np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred'])))/n
    else:
        smape = np.sum(np.abs(a['y_pred'] - a['y_true']) * weight[a.index] / (np.abs(a['y_true']) + np.abs(a['y_pred'])))/n
    return smape

def regularized_sMAPE(y_true, y_pred, weight = None):
    '''
    Return accuracy measure based on percentage errors
    :param y_true: Series of actual values
    :param y_pred: Series of forecasted values
    return: symmetric mean absolute percentage error (sMAPE)
    '''
    try:
        y_true = y_true.reset_index(drop=True)
        y_pred = y_pred.reset_index(drop=True)
    except:
        print('sMAPE function: y_true and y_pred should be pd.Series.')
    a = pd.DataFrame()
    a['y_true'] = y_true
    a['y_pred'] = y_pred
    n = len(y_pred)
    a = a[(a['y_true']!=0) | (a['y_pred']!=0)]
#    ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
    if weight is None:
        regularized_smape = np.sum(np.abs(a['y_pred'] - a['y_true']) / np.maximum((np.abs(a['y_true']) + \
                                          np.abs(a['y_pred'])), min(np.percentile(y_true, 40), np.percentile(y_pred, 40)))) / n
    else:
        regularized_smape = np.sum(np.abs(a['y_pred'] - a['y_true']) * weight[a.index] / np.maximum((np.abs(a['y_true']) + \
                                          np.abs(a['y_pred'])), min(np.percentile(y_true, 40), np.percentile(y_pred, 40)))) / n
    return regularized_smape

def MAPE(real_values, forecast_values):
  y_true, y_pred = np.array(real_values), np.array(forecast_values)
  ape = np.abs((y_true - y_pred) / y_true)
  valid_mask = ~np.ma.masked_invalid(ape).mask
  return ape[valid_mask].mean()

def MAE(real_values, forecast_values):
  return metrics.mean_absolute_error(real_values, forecast_values)