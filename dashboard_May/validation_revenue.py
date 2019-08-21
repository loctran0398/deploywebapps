from d2o_common.legacy_lib import database_api as db_api
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import json
from d2o_common.legacy_lib.utils import logger as log
import numpy as np
import time


def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)
    
def get_rn_rv_data(client, hotel_id, year_back):
    try:
        ##Issue 2
        df, has_data_flag = db_api.get_data_for_dep_season(client, hotel_id, year_back)
#        print('has_rv_data_flag', has_data_flag)

        df.index = pd.to_datetime(df.index)

        df_col = [col for col in list(df) if col.startswith(str(hotel_id))]
        df = df[df_col]
        rn_col = [col for col in list(df) if col.endswith('rn')]
        rv_col = [col for col in list(df) if col.endswith('rv')]

        room_nights = df[rn_col]
        room_revenue = df[rv_col]
        room_nights.columns = [col.replace('_rn', '') for col in room_nights.columns]
        room_revenue.columns = [col.replace('_rv', '') for col in room_revenue.columns]
        return room_nights, room_revenue

    except:
        log.err("DATABASE %s __ ID %s __ Could not get RN & RV data for this ID" % (client, hotel_id))

        
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
    return np.sum(np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred']))) / n



def improvement_perc(smape_v1, smape_v2):
    return (smape_v1 - smape_v2) / smape_v1

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
    
def revenue_validation_func(client_id, hotel, dep_id, years_back):
    to_date = (datetime.now() - timedelta(days = 2)).date()
    year_add = - years_back
    from_date = add_years(to_date, year_add)

    link_v2 = HOST + '/Season/Auto/V2/Revenue/{}/?h_id={}&from={}-{}-{}&to={}-{}-{}&segment_id=0'.format(client_id,
                                                                                                         dep_id,
                                                                                                         from_date.year,
                                                                                                         from_date.month,
                                                                                                         from_date.day,
                                                                                                         to_date.year,
                                                                                                         to_date.month,
                                                                                                         to_date.day)

    data_predict = request_data(link_v2)
    data_predict['Date'] = pd.to_datetime(data_predict['Date'], format='%Y-%m-%d')

    room_nights, room_revenue = get_rn_rv_data(client=client_id, hotel_id = int(dep_id), year_back=years_back)
    room_revenue = room_revenue.reset_index()

    data_predict = data_predict[data_predict['Date'].isin(room_revenue['date'])]

    col_name = [x for x in room_revenue.columns if '_0' in x]

    smape_1month_v2 = sMAPE(room_revenue[col_name][-30:], data_predict['Revenue'][-30:])
    smape_3month_v2 = sMAPE(room_revenue[col_name][-90:], data_predict['Revenue'][-90:])
    smape_6month_v2 = sMAPE(room_revenue[col_name][-180:], data_predict['Revenue'][-180:])
    smape_12month_v2 = sMAPE(room_revenue[col_name][-365:], data_predict['Revenue'][-365:])
    link_v1 = HOST + '/Season/Auto/Revenue/{}/?h_id={}&from={}-{}-{}&to={}-{}-{}&segment_id=0'.format(client_id,
                                                                                                      dep_id,
                                                                                                      from_date.year,
                                                                                                      from_date.month,
                                                                                                      from_date.day,
                                                                                                      to_date.year,
                                                                                                      to_date.month,
                                                                                                      to_date.day)
    data_predict = request_data(link_v1)
    data_predict['Date'] = pd.to_datetime(data_predict['Date'], format='%Y-%m-%d')
    data_predict = data_predict[data_predict['Date'].isin(room_revenue['date'])]

    smape_1month_v1 = sMAPE(room_revenue[col_name][-30:], data_predict['Revenue'][-30:])
    smape_3month_v1 = sMAPE(room_revenue[col_name][-90:], data_predict['Revenue'][-90:])
    smape_6month_v1 = sMAPE(room_revenue[col_name][-180:], data_predict['Revenue'][-180:])
    smape_12month_v1 = sMAPE(room_revenue[col_name][-365:], data_predict['Revenue'][-365:])

    result = pd.DataFrame()
    result['Driver'] = ['_'.join([str(hotel), str(dep_id)])]
    result['sMAPE_1month_v1'] = ['{:,.2%}'.format(smape_1month_v1)]
    result['sMAPE_1month_v2'] = ['{:,.2%}'.format(smape_1month_v2)]
    result['Improvement_1month'] = ['{:,.2%}'.format(improvement_perc(smape_1month_v1, smape_1month_v2))]
    result['Absolute_Improvement_1month'] = ['{:,.2%}'.format(smape_1month_v1 - smape_1month_v2)]
    
    result['sMAPE_3month_v1'] = ['{:,.2%}'.format(smape_3month_v1)]
    result['sMAPE_3month_v2'] = ['{:,.2%}'.format(smape_3month_v2)]
    result['Improvement_3month'] = ['{:,.2%}'.format(improvement_perc(smape_3month_v1, smape_3month_v2))]
    result['Absolute_Improvement_3month'] = ['{:,.2%}'.format(smape_3month_v1 - smape_3month_v2)]
    
    result['sMAPE_6month_v1'] = ['{:,.2%}'.format(smape_6month_v1)]
    result['sMAPE_6month_v2'] = ['{:,.2%}'.format(smape_6month_v2)]
    result['Improvement_6month'] = ['{:,.2%}'.format(improvement_perc(smape_6month_v1, smape_6month_v2))]
    result['Absolute_Improvement_6month'] = ['{:,.2%}'.format(smape_6month_v1 - smape_6month_v2)]
    
    result['sMAPE_12month_v1'] = ['{:,.2%}'.format(smape_12month_v1)]
    result['sMAPE_12month_v2'] = ['{:,.2%}'.format(smape_12month_v2)]
    result['Improvement_12month'] = ['{:,.2%}'.format(improvement_perc(smape_12month_v1, smape_12month_v2))]
    result['Absolute_Improvement_12month'] = ['{:,.2%}'.format(smape_12month_v1 - smape_12month_v2)]
    

    return result

def get_info_to_run(list_client):
    info = []
    for client in list_client:
        print(client)
        list_hotel = get_list_hotel(client)
        print('list hotel: ', len(list_hotel))
        for hotel in list_hotel:
            info.append((client, hotel))

    return info

def get_list_hotel(client):
    try:
        d = db_api.get_hotels_in_database(client)['Id'].values
        return d
    except:
        return []
    
def sorted_info(l_info):
    d = {}
    for client, hotel_id in l_info:
        deps = get_list_dep(client, hotel_id)
        d[(client, hotel_id)] = len(deps)
        print(client, hotel_id, len(deps))

    l = sorted(d, key=d.get)  # increasing
    for client, hotel_id in l:
        print(client, hotel_id, d[(client, hotel_id)])
    return l

def get_list_dep(client, hotel, remove_hotel_id = True):
    try:
        df = db_api.get_depts_in_hotel(client, hotel)
        if len(df) == 0:
            return []

        df = df[df['Revenue'] == True]
        if remove_hotel_id:
            df = df[df['H_Id'] != int(hotel)]
        deps = df['H_Id'].values
        return deps
    except:
        return []
    
HOST = db_api.HOST
dep_error = []

#Thon
start_time = time.time()

client = 'EB19AB58-3D2E-4325-95B8-92403AB23191'
years_back = 1
val_list = []

all_clients = db_api.get_all_client()
all_clients = all_clients[all_clients['Id'] == client]
run_clients = all_clients['Id'].values
info = get_info_to_run(run_clients)
info = sorted_info(info)
list_error = []

to_date = (datetime.now() - timedelta(days = 2)).date()
year_add = - years_back
from_date = add_years(to_date, year_add)
print('From: %s To: %s' % (from_date, to_date))
num_dep = 0
num_hotel = 0

for client, hotel in info:
    dep_list = get_list_dep(client, hotel, remove_hotel_id=False).tolist()
    print('\nHotel {} __ Dep list: {}'.format(hotel, dep_list))
    num_hotel += 1
    num_dep += len(dep_list)
    
    for dep_id in dep_list:
        hotel_dep = '_'.join([str(hotel), str(dep_id)])
        if hotel_dep not in list_error:
            try:
                val_res = revenue_validation_func(client, hotel, dep_id, years_back)
                val_list.append(val_res)
            except:
                if hotel_dep not in dep_error:
                    dep_error.append(hotel_dep)
                continue    
        val_df = pd.concat(val_list, axis = 0)
        
val_df.to_csv('validation_Thon.csv', index = False)

print('\nTOTAL HOTELS: %s __ TOTAL DEPTS: %s' % (num_hotel, num_dep))

print('num_dep_error: %s\n %s' % (len(dep_error), dep_error))
print("\n--- TOTAL TIME %s seconds ---" % (time.time() - start_time))

#Test Hotels
#start_time = time.time()
#
#info = [('B689DF88-1321-471D-B290-9025AB2C9C4B','1'),('F7F010AF-BCFE-4E28-A3D5-BFD0AE513341','146'),
#        ('604F54CA-6E1C-4CB5-B58F-A79E0835A2AC','10'),('604F54CA-6E1C-4CB5-B58F-A79E0835A2AC','392'),
#        ('604F54CA-6E1C-4CB5-B58F-A79E0835A2AC','991'),('ED3992A1-441D-415F-92DF-9F4BF1C35957','5'),
#        ('7C71CD95-07D0-40F5-9339-E27FDC2A614D','16'),('FD2E31CF-A5E6-40C9-8226-15A77962A020','2'),
#        ('E0764B13-B814-4BC1-B292-3BBD0B0A550B','179'),('ED3992A1-441D-415F-92DF-9F4BF1C35957','94'),
#        ('7B373366-9C79-48BE-985E-7A7B35A9A5C1','1'),('60DFCF71-A82D-4B70-8534-2FF2D1535092','1'),
#        ('D3F7E51E-0E59-4BAD-88E7-331708108B86','617'),('8DFDC6FE-3152-4065-B9FD-E0D735DCB311','2'),
#        ('EB19AB58-3D2E-4325-95B8-92403AB23191','34'),('604F54CA-6E1C-4CB5-B58F-A79E0835A2AC','702'),
#        ('EB19AB58-3D2E-4325-95B8-92403AB23191','1161'),('EB19AB58-3D2E-4325-95B8-92403AB23191','51'),
#        ('87A8B68F-7C6B-43E4-8109-BCFE44393DAA','57'),('DA64EF5B-AC36-4038-A30E-1999E3C85CDA','37'),
#        ('3502A4B8-23FF-42E2-BF70-7038A901EAFC','1')]
#
#years_back = 1
#val_list = []
#list_error = []
#
#to_date = (datetime.now() - timedelta(days = 2)).date()
#year_add = - years_back
#from_date = add_years(to_date, year_add)
#print('From: %s To: %s' % (from_date, to_date))
#num_dep = 0
#num_hotel = 0
#
#for client, hotel in info:
#    dep_list = get_list_dep(client, hotel, remove_hotel_id=False).tolist()
#    print('\nHotel {} __ Dep list: {}'.format(hotel, dep_list))
#    num_hotel += 1
#    num_dep += len(dep_list)
#
#    for dep_id in dep_list:
#        if '_'.join([str(hotel), str(dep_id)]) not in list_error:
#            try:
#                val_res = revenue_validation_func(client, hotel, dep_id, years_back)
#                val_list.append(val_res)
#            except:
#                if hotel_dep not in dep_error:
#                    dep_error.append(hotel_dep)
#                continue    
#        val_df = pd.concat(val_list, axis = 0)
#
#val_df.to_csv('validation_TestHotels.csv', index = False)
#
#print('\nTOTAL HOTELS: %s __ TOTAL DEPTS: %s' % (num_hotel, num_dep))
#
#print('num_dep_error: %s\n %s' % (len(dep_error), dep_error))
#print("\n--- TOTAL TIME %s seconds ---" % (time.time() - start_time))

