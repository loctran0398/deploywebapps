from datetime import datetime, timedelta
from d2o_common.legacy_lib import database_api as db_api
#HOST = db_api.HOST
import statsmodels.api as sm
import pandas as pd
import numpy as np
import requests
import time
import json
#import argparse
import sys
import os
current_path = os.getcwd()
# import changepy
# from changepy import pelt
# from changepy.costs import normal_mean, poisson, normal_meanvar, exponential
# leadtime_default = [0, 15, 41, 61, 91, 151, 701]


#def get_ip_address():
#    import socket
#    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#    s.connect(("8.8.8.8", 80))
#    print(s.getsockname()[0])
#    if s.getsockname()[0].split('.')[3] == '23':
#        print('Production server')
#        HOST = "http://10.50.10.4:8485"
#    else:
#        print('Test server')
#        HOST = "http://172.16.0.51:8485"
#    s.close()
#    return HOST


#HOST = get_ip_address()
## DEPT_SEASON_LINK = "%s/Season/Auto/V2/{client_id}/?h_id={h_id}&type=0" % (HOST)
## DEPT_SEASON_LINK = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=0" % (HOST)
#DEPT_SEASON_LINK = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=0" % (HOST)
## DEPT_SEASON_LINK = "%s/Season/Auto/V2/{client_id}/?h_id={h_id}&type=0" % (HOST)
#
#DEPT_SEASON_LABOR_LINK = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=1" % (HOST)
#DEPT_SEASON_FOOD_LINK = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=2" % (HOST)
# http://172.16.0.51:8485/Season/Auto/V2/Revenue/{client_id}/?h_id={dept_id}&from=2014-6-1&to=2018-6-1&segment_id=0




#===============================================================
def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)

def has_data(df):
    """
    Check whether df has enough data or not
    :param df: dataframe
    :return: True or False
    """
    REQ_PERCENT_DATA = 80


    stats = db_api.statistics(df)
    if (stats.loc[:, 1]['percent_nan'] > REQ_PERCENT_DATA or stats.loc[:, 1]['percent_zero'] > REQ_PERCENT_DATA):
      return False
    return True


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
    a = a[(a['y_true']!=0) | (a['y_pred']!=0)]
#    ## Note: The reson we divided by n is to solve the case y_true == y_pred == 0
#     np.sum(np.abs(a['y_pred'] - a['y_true']) / np.maximum((np.abs(a['y_true']) + \
#                                                            np.abs(a['y_pred'])), min(np.percentile(y_true, 40),
#                                                                                      np.percentile(y_pred, 40)))) / n
#     np.sum(np.abs(a['y_pred'] - a['y_true']) / (np.abs(a['y_true']) + np.abs(a['y_pred']))) / n
    return np.sum(np.abs(a['y_pred'] - a['y_true']) /(np.abs(a['y_true']) + np.abs(a['y_pred'])))/n



def get_otb_data_many_years(client, Mydep, from_time, current_time):
    response_flag = 0
    num = 0
    time_get_otb = time.time()
    while (response_flag != 200) and (num < 4):
        try:
            # get data otb
            #        current_time = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d') - timedelta(days=2)
            #        from_time = current_time - timedelta(days=years_back*365+30)

            # current_time = datetime.strptime('2018-04-30', '%Y-%m-%d')
            # current_time = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')
            # from_time = current_time - timedelta(days = 150)
            link = '%s/Otb/Archive/Total/%s/?h_id=%s&from=%s&to=%s' % (HOST, client, \
                                                                       Mydep, from_time.strftime('%Y-%m-%d'),
                                                                       current_time.strftime('%Y-%m-%d'))
            response = requests.get(link)

        except:
            print('No otb data')
        response_flag = response.status_code
        if response_flag != 200:
            num += 1
            print("WARNING: Cannot get data from server request. Status code: %s" % (response_flag))
            print("Try: {} times.".format(num))
        # return None

    json_response = json.loads(response.text)
    json_response_df = pd.DataFrame(json_response['Items'])

    if len(json_response_df) > 0:
        json_response_df['Date'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in json_response_df['Date']]
        json_response_df['ImportDate'] = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S') for i in
                                          json_response_df['ImportDate']]
        json_response_df['LeadTime'] = (json_response_df['Date'] - json_response_df['ImportDate']).dt.days
    # a = json_response_df['LeadTime']

    # new_json_response_df = json_response_df[
    #     (json_response_df['LeadTime'] == 0) | (json_response_df['LeadTime'] == 5) | (
    #                 json_response_df['LeadTime'] == 10) | (json_response_df['LeadTime'] == 15)]
    new_json_response_df = json_response_df[(json_response_df['LeadTime'] >= 5) & (json_response_df['LeadTime'] <= 30)]

    # otb_data = put.impute(new_json_response_df, method='zero')

    return new_json_response_df


def add_season_to_df(data,client, dep_id):
    '''
    input: dataframe contains date
    output: add period to dataframe
    :param data:
    :return:
    '''
    try:
        from_time = data['Date'].iloc[0]
        current_time = data['Date'].iloc[-1]
        temp_df = get_season_data(client, dep_id, 0)
        print('get season sucessfully')
        temp_df = cv_ssdef_df(temp_df)
        period_id_dict_temp = cv_period_id_dict(temp_df, current_time.year)
        period_id_dict_temp['period_id'] = [int(i) for i in period_id_dict_temp['period_id']]
        period_id_list = period_id_dict_temp['period_id'].drop_duplicates()
        # df_season = convert_df_season(current_time, from_time, period_id_dict_temp)
        weekday_list = range(7)
        year_test = list(set(range(from_time.year, current_time.year + 1)))
        for period_id in period_id_list:
            season_tf = period_id_dict_temp[period_id_dict_temp['period_id'] == period_id]['timeframe'].values[0][0]
            for weekday in weekday_list:
                date_ss_weekday = get_similar_days(weekday, season_tf, year_test)
                date_ss_weekday = [i for i in date_ss_weekday if from_time <= i <= current_time]
                data.loc[data['Date'].isin(date_ss_weekday), 'Period'] = str(
                    period_id)
    #    data_revenue = db_api.get_data_for_dep_season(client, hotel_id, year_back)
    #    data_revenue.to_csv('test_data/rv_data_{}_{}.csv'.format(hotel_id, dep_id), index = False)
    except:
        print('No revenue season')

    return data


def get_otb(years_back, current_time):
    begin_time = current_time - timedelta(days=years_back * 365)
    # days = (current_time - begin_time).days
    new_otb_data = pd.DataFrame()
    # Mydep = '774'
    try:
        for i in range(years_back):
            begin_time_otb = begin_time + timedelta(days=i * 365)
            end_time = begin_time + timedelta(days=365) + timedelta(days=i * 365)
            if i == range(years_back)[-1]:
                print(i)
                end_time = current_time
            otb_data = get_otb_data_many_years(client, Mydep, begin_time_otb, end_time)
            new_otb_data = pd.concat([new_otb_data, otb_data])
    except:
        print('No otb data')
        return None
    return new_otb_data



    
def take_driver_rev_otb(driver_data, segment_id):
    new_data = driver_data[driver_data['Segment_Id'] == segment_id][['Day', 'Period', 'Sources', 'High', 'Low', 'Tree', 'Model']].reset_index(drop=True)
#    new_data['High'] = new_data['High'].round(3)
#    new_data['Low'] = new_data['Low'].round(3)
    new_data['Driver_1'] = 0
    new_data['Driver_2'] = 0
    for i in range(len(new_data)):
        new_data['Driver_1'][i] = str(int(pd.DataFrame(new_data['Sources'][i][0].items()).iloc[5, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][0].items()).iloc[2, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][0].items()).iloc[6, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][0].items()).iloc[9, 1]))
        new_data['Driver_2'][i] = str(int(pd.DataFrame(new_data['Sources'][i][1].items()).iloc[5, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][1].items()).iloc[2, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][1].items()).iloc[6, 1])) + '_' + str(
            int(pd.DataFrame(new_data['Sources'][i][1].items()).iloc[9, 1]))
    new_data['Period'] = new_data['Period'].astype('str')
    new_data['Day'] = new_data['Day'].astype('str')
    new_data['Period_Day'] = new_data[['Period', 'Day']].apply(lambda x: '_'.join(x), axis=1)
    del new_data['Period']
    del new_data['Day']

    return new_data    


def join_day_period(regression_data):
    regression_data['Day'] = [i.weekday() for i in regression_data['Date']]
    regression_data['Day'] = [i + 2 if i != 6 else 1 for i in regression_data['Day']]
    regression_data['Day'] = regression_data['Day'].astype('str')
    regression_data['Period'] = regression_data['Period'].astype('str')
    regression_data['Period_Day'] = regression_data[['Period', 'Day']].apply(lambda x: '_'.join(x), axis=1)
    del regression_data['Period']
    del regression_data['Day']
    return regression_data

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
    # y1_forecast = modifier1 * rate1 * x1.values[0]
    # y2_forecast = modifier2 * rate2 * x2.values[0]
    y1_forecast = modifier1 * rate1 * x1.values
    y2_forecast = modifier2 * rate2 * x2.values
    # y_forecast = [max(y1_forecast, y2_forecast)]
    y_forecast = np.maximum(y1_forecast, y2_forecast)
    y_forecast = pd.Series([temp for temp in y_forecast])
    # ======================================================================================
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    s = sMAPE(y, y_forecast)
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return [y_forecast, forecast_error]

def max_model_pred_value(data_final, new_driver_data, Mydep, segment_id):
    k_modifier = 1000
    forecast_revenue = pd.DataFrame()
    for i in range(len(new_driver_data)):
        df_cal_max = data_final[data_final['Period_Day'] == new_driver_data['Period_Day'][i]]
        max_df_pred = df_cal_max[['Date', 'Period_Day']]
        


        df_json_paras_max = pd.DataFrame(new_driver_data['Sources'][i])
        df_json_paras_max['Linear_Rate'] = df_json_paras_max['Linear_Rate'].round(3)
        df_json_paras_max['Rate'] = df_json_paras_max['Rate'].round(3)
        df_json_paras_max['HighThreshold'] = df_json_paras_max['HighThreshold'].round(3)   
        df_json_paras_max['LowThreshold'] = df_json_paras_max['LowThreshold'].round(3)

        y_min = new_driver_data['Low'][i]
        y_max = new_driver_data['High'][i]

        modifier1 = min(max(df_json_paras_max['Modifier'][0], 1 - k_modifier), 1 + k_modifier)
        modifier2 = min(max(df_json_paras_max['Modifier'][1], 1 - k_modifier), 1 + k_modifier)

        rate1 = df_json_paras_max['Rate'][0]
        rate2 = df_json_paras_max['Rate'][1]

        x1 = df_cal_max[new_driver_data['Driver_1'][i]]
        x2 = df_cal_max[new_driver_data['Driver_2'][i]]
        y = df_cal_max[Mydep + '_' + str(segment_id) + '_0_3']
        y_forecast_v2, forecast_error_v2 = v2_validate_error(y, y_min, y_max, x1, modifier1, rate1, x2, modifier2,
                                                             rate2)
        driver_1 = new_driver_data['Driver_1'][i]
        driver_2 = new_driver_data['Driver_2'][i]
        max_df_pred['Driver_1'] = driver_1
        max_df_pred['Name_Driver_1'] = str({'H_Id': driver_1.split('_')[0], 'Segment_Id': driver_1.split('_')[1], 'Offset': driver_1.split('_')[2], 'Type': driver_1.split('_')[3]})
        max_df_pred['Driver_2'] = driver_2
        max_df_pred['Name_Driver_2'] = str({'H_Id': driver_2.split('_')[0], 'Segment_Id': driver_2.split('_')[1], 'Offset': driver_2.split('_')[2], 'Type': driver_2.split('_')[3]})
        max_df_pred['Linear_Rate_1'] =  df_json_paras_max['Linear_Rate'][0]
        max_df_pred['Rate_1'] = rate1
        max_df_pred['Value_1'] = x1
        max_df_pred['Modifier_1'] = modifier1
        max_df_pred['Rate_2'] = rate2
        max_df_pred['Value_2'] = x2
        max_df_pred['Modifier_2'] = modifier2
        max_df_pred['Actual_Rev'] = y.values
        max_df_pred['ForecastRevenue'] = y_forecast_v2.values
        forecast_revenue = pd.concat([forecast_revenue, max_df_pred])
        # acc_max = 1 - forecast_error_v2
    return forecast_revenue

def tree_model(data_final, new_driver_data, Mydep, segment_id):
    # k_modifier = 1000
    forecast_revenue = pd.DataFrame()
    for i in range(len(new_driver_data)):
        df_cal_max = data_final[data_final['Period_Day'] == new_driver_data['Period_Day'][i]]
        max_df_pred = df_cal_max[['Date','Period_Day']]
        max_df_pred = max_df_pred.reset_index(drop = True)
        max_df_pred['Rate_Tree'] = pd.Series([])
        max_df_pred['Modifier_Tree'] = pd.Series([])
        max_df_pred['Tree_Value'] = pd.Series([])
        df_json_paras_max = pd.DataFrame(new_driver_data['Sources'][i])
        case_df = pd.DataFrame(new_driver_data['Tree'][i])

        y_min = new_driver_data['Low'][i]
        y_max = new_driver_data['High'][i]

        # modifier1 = min(max(df_json_paras_max['Modifier'][0], 1 - k_modifier), 1 + k_modifier)
        # modifier2 = min(max(df_json_paras_max['Modifier'][1], 1 - k_modifier), 1 + k_modifier)
        #
        # rate1 = df_json_paras_max['Rate'][0]
        # rate2 = df_json_paras_max['Rate'][1]

        x1 = df_cal_max[new_driver_data['Driver_1'][i]].reset_index(drop = True)
        x2 = df_cal_max[new_driver_data['Driver_2'][i]].reset_index(drop = True)



        y = df_cal_max[Mydep + '_' + str(segment_id) + '_0_3'].reset_index(drop = True)


        cov_val_tree_temp = 0.0
        y_forecast = []
        for i, v in enumerate(y):
            y_forecast_tree, forecast_error_tree, modifier, rate, x = tree_validation_error(y[i], y_min, y_max, x1[i], x2[i], case_df,
                                                                         df_json_paras_max)
            cov_val_tree_temp += 1 - forecast_error_tree
            y_forecast.append(y_forecast_tree[0])
            max_df_pred['Rate_Tree'][i] = rate
            max_df_pred['Modifier_Tree'][i] = modifier
            max_df_pred['Tree_Value'][i] = x
        cov_val_tree = cov_val_tree_temp / len(y)


        # acc_tree = 1 - forecast_error_tree
        max_df_pred['High'] = y_max
        max_df_pred['Low'] = y_min
        max_df_pred['ForecastRevenue'] = y_forecast
        forecast_revenue = pd.concat([forecast_revenue, max_df_pred])
        # acc_max = 1 - forecast_error_v2
    return forecast_revenue

def linear_model(data_final, new_driver_data, Mydep, segment_id):

    forecast_revenue = pd.DataFrame()
    for i in range(len(new_driver_data)):
        df_cal_max = data_final[data_final['Period_Day'] == new_driver_data['Period_Day'][i]]
        max_df_pred = df_cal_max[['Date','Period_Day']]

        df_json_paras_max = pd.DataFrame(new_driver_data['Sources'][i])

        y_min = new_driver_data['Low'][i]
        y_max = new_driver_data['High'][i]


        x1 = df_cal_max[new_driver_data['Driver_1'][i]].reset_index(drop=True)
        # x2 = df_cal_max[new_driver_data['Driver_2'][i]].reset_index(drop=True)

        y = df_cal_max[Mydep + '_' + str(segment_id) + '_0_3'].reset_index(drop=True)

        linear_rate = df_json_paras_max['Linear_Rate'][0]
        y_forecast_linear, forecast_error_linear = linear_validate_error(y, y_min, y_max, x1, linear_rate)

        # acc_tree = 1 - forecast_error_tree
        max_df_pred['ForecastRevenue'] = y_forecast_linear.values
        forecast_revenue = pd.concat([forecast_revenue, max_df_pred])
    return forecast_revenue 
    

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



def tree_validation_error(y, y_min, y_max, x1, x2, case_df, top2_df):
    # x1_temp = x1.values[0]
    # x2_temp = x2.values[0]
    x1_temp = x1
    x2_temp = x2
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
    # x_name = top2_df[top2_df['Priority'] == priority]['Driver_name'].values[0] + '_' + \
    #          str(top2_df[top2_df['Priority'] == priority]['Offset'].values[0])
    if priority == 1:
        x = x1_temp
    else:
        x = x2_temp
    y_forecast = [modifier * rate * x]
    y_forecast = pd.Series([temp for temp in y_forecast])
    y_forecast = pd.Series([max(min(y_max, temp), y_min) for temp in y_forecast])
    # ======================================================================================
    s = sMAPE(pd.Series([y]), pd.Series(y_forecast))
    # s = s.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_error = 100 if np.isinf(s) else s
    return y_forecast, forecast_error, modifier, rate, x



def add_df_otb(years_back, current_time, regression_data):
    try:
        new_otb_data = get_otb(years_back, current_time)
        new_otb_data_leadtime = new_otb_data


        data_pv_otb = new_otb_data_leadtime.pivot_table(index=['Date'], columns='LeadTime',
                                                        values='Revenue').reset_index()
        dict = {}
        for i in data_pv_otb.columns:
            if i != 'Date' and i != 'Period_Day' and i != Mydep + '_0_0_3':
                dict.update({i: Mydep + '_0_' + str(i) + '_6'})
            else:
                dict.update({i: i})
        data_pv_otb = data_pv_otb.rename(columns=dict)
        regression_data = regression_data.merge(data_pv_otb, on='Date', how='left')
        
    except:
        print('no otb data')
    regression_data = regression_data.replace(np.nan, 0)
    regression_data = join_day_period(regression_data)
    return regression_data    

def validation_revenue(prepare_data_forecast_rev,new_driver_data, Mydep, segment_id):
    
    prepare_data_forecast_rev['Day'] = prepare_data_forecast_rev['Day'].astype('str')
    prepare_data_forecast_rev['Period'] = prepare_data_forecast_rev['Period'].astype('str')
    prepare_data_forecast_rev['Period_Day'] = prepare_data_forecast_rev[['Period', 'Day']].apply(lambda x: '_'.join(x), axis=1)
    
    max_forecast_revenue =  max_model_pred_value(prepare_data_forecast_rev,new_driver_data, Mydep, segment_id)
    max_forecast_revenue = max_forecast_revenue.sort_values(['Date']).reset_index(drop=True)


    forecast_revenue_tree =  tree_model(prepare_data_forecast_rev, new_driver_data, Mydep, segment_id)
    forecast_revenue_tree = forecast_revenue_tree.sort_values(['Date']).reset_index(drop=True)
    
    # forecast_revenue_tree.to_csv('tree_model_{}.csv'.format(Mydep), index = False)
    
    forecast_revenue_linear = linear_model(prepare_data_forecast_rev, new_driver_data, Mydep, segment_id)
    forecast_revenue_linear = forecast_revenue_linear.sort_values(['Date']).reset_index(drop=True)


    best_forecast_revenue = pd.DataFrame()
    for i in range(len(new_driver_data)):
        best_model = new_driver_data['Model'][i]
        period_day = new_driver_data['Period_Day'][i]
        df_cal_max = prepare_data_forecast_rev[prepare_data_forecast_rev['Period_Day'] == new_driver_data['Period_Day'][i]]
        y = df_cal_max[Mydep + '_' + str(segment_id) + '_0_3'].reset_index(drop=True)
        forecsating_data_0 = forecast_revenue_linear[forecast_revenue_linear['Period_Day'] == period_day][
            ['Date', 'ForecastRevenue']]
        forecsating_data_1 = max_forecast_revenue[max_forecast_revenue['Period_Day'] == period_day][
            ['Date', 'ForecastRevenue']]
        forecsating_data_2 = forecast_revenue_tree[forecast_revenue_tree['Period_Day'] == period_day][
            ['Date', 'ForecastRevenue']]

#        smape_0 = sMAPE(y, forecsating_data_0['ForecastRevenue'])
#        smape_1 = sMAPE(y, forecsating_data_1['ForecastRevenue'])
#        smape_2 = sMAPE(y, forecsating_data_2['ForecastRevenue'])
#        print(best_model,smape_0,smape_1,smape_2)
#        min_smape = min(smape_0,smape_1,smape_2)
        if  best_model == 0:
            forecsating_data =forecsating_data_0
        elif best_model == 1:
            forecsating_data =forecsating_data_1
        else:
            forecsating_data =forecsating_data_2
#        if best_model == 0:
#            forecsating_data = forecast_revenue_linear[forecast_revenue_linear['Period_Day'] == period_day][['Date','ForecastRevenue']]
#            print(sMAPE(y,forecsating_data['ForecastRevenue']))
#        elif best_model == 1:
#            forecsating_data = max_forecast_revenue[max_forecast_revenue['Period_Day'] == period_day][['Date','ForecastRevenue']]
#            print(sMAPE(y,forecsating_data['ForecastRevenue']))
#        
#        else:
#            forecsating_data = forecast_revenue_tree[forecast_revenue_tree['Period_Day'] == period_day][['Date','ForecastRevenue']]
#            print(sMAPE(y,forecsating_data['ForecastRevenue']))

        best_forecast_revenue = pd.concat([best_forecast_revenue,forecsating_data])
    del forecast_revenue_linear['Period_Day']
    del forecast_revenue_tree['Period_Day']
    best_forecast_revenue = best_forecast_revenue.sort_values(['Date']).reset_index(drop=True)
    forecast_revenue_linear = forecast_revenue_linear.rename(columns={'ForecastRevenue': 'Linear_Forecast'})
    max_forecast_revenue = max_forecast_revenue.rename(columns={'ForecastRevenue': 'Max_Forecast'})
    forecast_revenue_tree = forecast_revenue_tree.rename(columns={'ForecastRevenue': 'Tree_Forecast'})
    best_forecast_revenue = best_forecast_revenue.rename(columns={'ForecastRevenue': 'Best_Forecast'})
    return best_forecast_revenue, forecast_revenue_linear, max_forecast_revenue, forecast_revenue_tree    

def read_file_json(client,hotel_id,Mydep,path_json):
    try:
        with open(path_json + 'top_db_end_v2_{}_{}_{}.json'.format(client,hotel_id,Mydep), 'r') as f:
            datastore = json.load(f)
        driver_data = pd.DataFrame.from_dict(datastore)
    except:
        print('error')
        sys.exit(0)
    new_driver_data = take_driver_rev_otb(driver_data)
    new_driver_data = new_driver_data['High'].round(3)
    new_driver_data = new_driver_data['Low'].round(3)
    return new_driver_data
    
def read_file_json_from_api(client,hotel_id,Mydep):
    try:
        link = 'http://172.16.0.51:8485/Forecast/RevenueDriver/Auto/{}/?h_id={}'.format(client,Mydep)
        response = requests.get(link)
        json_response = json.loads(response.text)
        driver_data = pd.DataFrame.from_dict(json_response)
#        driver_data = request_data(link)
    except:
        print('error')
        sys.exit(0)
    new_driver_data = take_driver_rev_otb(driver_data)
    return new_driver_data    

def export_validation(forecast_revenue_linear, forecast_revenue_tree, max_forecast_revenue, best_forecast_revenue):

    final_validation =  forecast_revenue_linear.merge(forecast_revenue_tree.merge(max_forecast_revenue.merge(best_forecast_revenue, on = 'Date', how = 'inner'), on = 'Date', how = 'inner'), on = 'Date', how = 'inner')
#    final_validation['Linear_Smape'] = pd.Series([])
#    final_validation['Max_Smape'] = pd.Series([])
#    final_validation['Tree_Smape'] = pd.Series([])
#    final_validation['Best_Smape'] = pd.Series([])
#    
    final_validation['Linear_Smape'] = np.abs(final_validation['Linear_Forecast'] - final_validation['Actual_Rev'])/np.abs(final_validation['Linear_Forecast'] + final_validation['Actual_Rev'])
    final_validation['Max_Smape']= np.abs(final_validation['Max_Forecast'] - final_validation['Actual_Rev'])/np.abs(final_validation['Max_Forecast'] + final_validation['Actual_Rev'])
    final_validation['Tree_Smape'] = np.abs(final_validation['Tree_Forecast'] - final_validation['Actual_Rev'])/np.abs(final_validation['Tree_Forecast'] + final_validation['Actual_Rev'])
    final_validation['Best_Smape'] = np.abs(final_validation['Best_Forecast'] - final_validation['Actual_Rev'])/np.abs(final_validation['Best_Forecast'] + final_validation['Actual_Rev'])
    final_validation = final_validation.replace(np.nan,0)

    file_validation = final_validation[['Date','Period_Day', 'Driver_1', 'Name_Driver_1','Linear_Rate_1','Rate_1','Modifier_1',
                                         'Value_1','Driver_2', 'Name_Driver_2','Rate_2','Modifier_2', 'Value_2' ,'Rate_Tree', 
                                         'Modifier_Tree', 'Tree_Value', 'High', 'Low', 'Linear_Forecast', 'Max_Forecast','Tree_Forecast',
                                         'Best_Forecast', 'Actual_Rev','Linear_Smape','Max_Smape','Tree_Smape','Best_Smape']]
                                         
    
    return file_validation            

def add_lag1_df_python(df_insample,df_insample_otb,season_original_df, dep_id):        
        
    # del data_lag['Date']
    date = df_insample[['date']]
    date.columns = ['Date']
    all_col = list(set(df_insample.columns) - {'date', 'day_of_week'})
    df_no_date = df_insample[all_col]
    df_no_date_lag = df_no_date.shift(1)
    dict_name = {}
    for i in df_no_date.columns:
        if i != 'Date' and i != 'Period':
            strings = i.split('_')
            dict_name.update({i: strings[0] + '_' + strings[1] + '_0' + '_' + str(pd.Series(strings[2]).map({'rn' : 1,'gn' : 2, 'rv' : 3}).values[0])})
        else:
            dict_name.update({i: i})
    df_no_date = df_no_date.rename(columns=dict_name)


    try:
        df_otb = df_insample_otb[[str(i) * 3 + '_' + str(i) * 3 + '_rv' for i in range(5,31)]]
        df_otb.columns = [str(dep_id) + '_0_' + str(i) + '_6' for i in range(5,31)]
    except:
        df_otb = pd.DataFrame()
    dict_lag = {}                             
    for i in df_no_date_lag.columns:
        if i != 'Date' and i != 'Period':
            dict_lag.update({i: i.split('_')[0] + '_' + i.split('_')[1] + '_1' + '_' + str(pd.Series(i.split('_')[2]).map({'rn' : 1,'gn' : 2, 'rv' : 3}).values[0])})
        else:
            dict_lag.update({i: i})
    df_no_date_lag = df_no_date_lag.rename(columns=dict_lag)

        
    
    
    
    data_all_lag = pd.concat([date,df_no_date, df_no_date_lag,df_otb], axis=1)
    data_all_lag = data_all_lag.dropna()
    data_all_lag = data_all_lag.merge(season_original_df[['Date','Day','Period_Id']], on = 'Date', how = 'left')
    data_all_lag = data_all_lag.rename(columns = {'Period_Id' : 'Period'})
#    data_all_lag['Date'] = [datetime.strptime(datetime(year=i.year, month=i.month, day=i.day).strftime('%Y-%m-%d'), '%Y-%m-%d') for i in data_all_lag['Date']]
#        data_all_lag['Date'] = data_all_lag.index
#        data_all_lag = data_all_lag.reset_index(drop=True)  
#            
#        [str(i) * 3 + '_' + str(i) * 3 + '_rv' for i in data_pv_otb.columns]
    
    return data_all_lag
    
def read_file_json(client,hotel_id,Mydep,top_db_end_v2, segment_id):
    try:
        driver_data = pd.DataFrame.from_dict(top_db_end_v2)
    except:
        print('error in json of validation')
        return None
    new_driver_data = take_driver_rev_otb(driver_data, segment_id)
    return new_driver_data
    
    
def validation_file_export(client,hotel_id, dep_id,df_insample, df_insample_otb,years_back, current_time, cruise,top_db_end_v2,all_rv_col_without_hid_only_depid):
    current_time = df_insample_otb['date'].iloc[-1].strftime('%Y-%m-%d')
    season_original_df = cruise.get_data_original()
    season_original_df['Date'] = pd.to_datetime(season_original_df['Date'])
    season_original_df = season_original_df.rename(columns = {'Groups':'Period_Id'})
    prepare_data_forecast_rev = add_lag1_df_python(df_insample,df_insample_otb,season_original_df, dep_id)
    for col in all_rv_col_without_hid_only_depid:
        segment_id = int(col.split('_')[1])
        try:
        
            new_driver_data = read_file_json(client,hotel_id,dep_id,top_db_end_v2,segment_id)
            Mydep = str(dep_id)
            print('get_best_model')                 
            best_forecast_revenue, forecast_revenue_linear, max_forecast_revenue, forecast_revenue_tree = validation_revenue(prepare_data_forecast_rev,new_driver_data, Mydep, segment_id)
                
            export_validation_file = export_validation(forecast_revenue_linear, forecast_revenue_tree, max_forecast_revenue, best_forecast_revenue)
            
            
            dir_result = current_path      
            if not os.path.isdir(dir_result):
                os.mkdir(dir_result)
#            file_name = os.path.join(dir_result, 'out_put', 'revenue_{}_{}_{}_{}.csv'.format(hotel_id, Mydep,segment_id,current_time))
            file_name = os.path.join(dir_result, 'out_put', 'revenue_{}_{}_{}.csv'.format(hotel_id, Mydep,segment_id))

            
            export_validation_file.to_csv(file_name, index = False)
#            print('export_validation_successfully_{}_{}_{}'.format(Mydep,segment_id,current_time))
            print('export_validation_successfully_{}_{}'.format(Mydep,segment_id))
    #            prepare_data_forecast_rev.to_csv(path + 'drivers_{}_{}.csv'.format(hotel_ids, Mydep), index = False)
        except Exception as e:
            print('Could not export validation file\n')
            print(e)
            export_validation_file = pd.DataFrame()
    return export_validation_file
