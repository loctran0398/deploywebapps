from d2o_common.legacy_lib import database_api_v2 as db_api
from d2o_common.util import logger as log
import pandas as pd
import argparse
import time
from datetime import datetime

import season_lib as sslib
import season_validation_lib as vlib

import os
import json

def generate_season_by_type(client, dep_list, years_back, revenue_type, labor_type, food_type, otb_type, write, df_error, validation):
    '''
    Detect season and create json file based on type of season (revenue, labor, foodcost)
    :param client: client ID
    :param hotel: hotel ID (value '0' if running for dept)
    :param dep_list: list of departments
    :param years_back: number of years for getting data
    :param r_type: revenue information of department
    :param l_type: labor information of department
    :param f_type: foodcost information of department
    :param otb_type: otb information of department
    :param write: write log to API if True
    :param df_error: report status of season detection of each department
    :param validation: create or update validation file if True
    :returns:
        df_error: report status of season detection of each department after running
        json_upload: json file for uploading
    '''
    start_time_all_dept = time.time()
    for dep_id in dep_list:
        try:
            Mydep = str(dep_id)
            try:
                #Get information of deparment
                season_dict = db_api.get_one_dept_info_in_hotel(client, dep_id)
                if season_dict['IsProperty'] is True:
                    hotel = str(season_dict['H_Id'])
                else:
                    hotel = str(season_dict['ParentId'])
                if len(season_dict) == 0:
                    print("DATABASE %s __ DEP ID %s __ No information of Hierarchy database" % (client, dep_id))
                    log.info("DATABASE %s __ DEP ID %s __ No information of Hierarchy database" % (client, dep_id)) 
                    continue #sys.exit(0)
            except:
                print("DATABASE %s __ DEP ID %s __ No information of Hierarchy database" % (client, dep_id))
                log.info("DATABASE %s __ DEP ID %s __ No information of Hierarchy database" % (client, dep_id))    
                continue #sys.exit(0)
            
            if (otb_type == 't') or (otb_type == 'true'):
                o_type = season_dict['Otb']
            else:
                o_type = False
                
            if (revenue_type == 't') or (revenue_type == 'true'):
                r_type = season_dict["Revenue"]
            else:
                r_type = False
                
            if (labor_type == 't') or (labor_type == 'true'):
                l_type = season_dict["Labor"]
            else:
                l_type = False
                
            if (food_type == 't') or (food_type == 'true'):
                f_type = season_dict["FoodCost"]
            else:
                f_type = False
                
            print('\nDATABASE %s __ DEP ID %s\nREVENUE TYPE %s __ LABOR TYPE %s __ FOODCOST TYPE %s' % (client, Mydep, r_type, l_type, f_type))
            
            if r_type == True:
#                a = time.time()
                ss_type = 0
                df_error, json_upload = generate_season(client, hotel, Mydep, years_back, ss_type, o_type, write, df_error)
#                print('RUNNING TIME __ ID {}: {} seconds'.format(hotel, time.time() - a))  
                    
            if l_type == True:
                ss_type = 1
                df_error, json_upload = generate_season(client, hotel, Mydep, years_back, ss_type, o_type, write, df_error)
                    
            if f_type == True:
                ss_type = 2
                df_error, json_upload = generate_season(client, hotel, Mydep, years_back, ss_type, o_type, write, df_error)
                                        
        except Exception as e:
            temp_df_error = pd.DataFrame()
            temp_df_error['Driver'] = ['{}_{}'.format(hotel, Mydep)]
            temp_df_error['Type'] = [ss_type]
            temp_df_error['Error'] = [e]
            df_error = df_error.append(temp_df_error, ignore_index=True)
            
    log.info("DATABASE %s __ ID %s __ RUNNING TIME FOR THIS HOTEL __ %s" % (client, hotel, time.time() - start_time_all_dept))
    print('RUNNING TIME FOR HOTEL __ ID {}: {} seconds'.format(hotel, time.time() - start_time_all_dept))

    return df_error

def generate_season(client, hotel, Mydep, years_back, ss_type, otb_type, write, df_error, df, current_path):
    '''
    Generate season and create json file for each type of season in a department
    :param client: client ID
    :param hotel: hotel ID (value '0' if running for dept)
    :param Mydep: department ID
    :param years_back: number of years for getting data
    :param ss_type: type of season
    :param otb_type: otb information of department
    :param write: write log to API if True
    :param df_error: report status of season detection of each department
    
    :returns:
        df_error: report status of season detection of each department after running
        json_upload: json file for uploading
    '''
    frac = 0.05
    lower_bound = 0  # Boundary of percentile to remove outliers
    upper_bound = 100  # Boundary of percentile to remove outliers

#    cruise_day_data = sslib.get_weekday_data_cruise(client, Mydep, years_back)
    cruise_day_data = None
    if cruise_day_data is None:
        mav_median_df, room_data, col_name, cycle_value, ending_date = sslib.get_data(client, Mydep, years_back, ss_type, write,df)
        cruise_flag = False
    else:
        mav_median_df, room_data, col_name, cycle_value, ending_date = sslib.get_data_cruise(client, Mydep, years_back, ss_type, write)
        cruise_flag = True

    if (mav_median_df is None) or ((cruise_flag == False) and (len(mav_median_df) < 364)):
        log.info("DATABASE %s __ ID %s __ Insufficient data for this ID" % (client, Mydep))

        start_time_mydep = time.time()
        log_file_client = {'Date': datetime.fromtimestamp(int(start_time_mydep)).strftime('%Y-%m-%d'),
                           'H_Id': Mydep, 'Message': 'Insufficient data', 'Status': 0}
        sslib.log_json_upload(client, log_file_client)

        json_default = sslib.write_to_database(client, Mydep, ss_type, write, json_upload=None, default=True)

        temp_df_error = pd.DataFrame()
        temp_df_error['Driver'] = ['{}_{}'.format(hotel, Mydep)]
        temp_df_error['Type'] = [ss_type]
        temp_df_error['Error'] = ['Insufficient data for this ID']
        df_error = df_error.append(temp_df_error, ignore_index=True)

        return df_error, json_default

    #Transform data to dict year
    if cruise_flag == False:
        dict_index, dict_year, dict_tf, df_date = sslib.transform_data_to_1year(mav_median_df, col_name)
        #Evaluate data quality
        df_index, df_year, df_tf, nb_years = sslib.compute_data_quality(dict_index, dict_year, dict_tf, frac)
    else:
        #Evaluate data quality
        df_index, df_year, nb_years = sslib.transform_data_to_1year_cruise(mav_median_df, cruise_day_data)

    from_date = min(df_year['Date'])
    to_date = max(df_year['Date'])

    if sslib.has_data(df_year, col_name) is False:
        log.info("DATABASE %s __ ID %s __ Insufficient data for this ID" % (client, Mydep))
        print('Insufficient data after computing data quality for this ID')
        json_default = sslib.write_to_database(client, Mydep, ss_type, write, json_upload=None, default=True)

        temp_df_error = pd.DataFrame()
        temp_df_error['Driver'] = ['{}_{}'.format(hotel, Mydep)]
        temp_df_error['Type'] = [ss_type]
        temp_df_error['Error'] = ['Insufficient data after computing data quality']
        df_error = df_error.append(temp_df_error, ignore_index=True)

        return df_error, json_default

    #Adding otb data
    years_temp = int(round(len(df_year) / 365)) + 1
    if years_temp == 1 and otb_type == True:
        try:
            otb_data = sslib.get_otb_data(client, Mydep, years_temp, df_year)
            df_year = df_year.merge(otb_data, on='Date')
        except:
            print('no_otb')
    
    df_year = df_year.set_index('Date')
    
    #Do PCA
    data = sslib.get_PC1(df_index, df_year, col_name)

    #Clustering
    data_grouped = sslib.group_season_pc1(data, col_name, lower_bound, upper_bound, cruise_flag)
    
    #Re-grouping & naming
    data_conv_pv, out_level = sslib.regroup_and_set_name(data_grouped, Mydep)
    
    #Generating data 3 previous years and 2 future years
    if cruise_flag == False:
        full_df, lackdate_df  = sslib.generate_6years_df(data_conv_pv, cycle_value)
        df_weekday = pd.DataFrame()
    else:
        full_df, df_weekday, cruise_lack_date  = sslib.generate_6year_df_cruise(data_conv_pv, client, Mydep, cycle_value)

    #Adding outlier date
    outlier_date , weekday_begin, weekday_end = sslib.outlier_detection(room_data, full_df, col_name, df_weekday, cruise_flag)
    
    #Adding  holidays
    print('########################################begin - holiday######################')
    event_date = sslib.get_event_data(client, Mydep)
    holiday_data, list_holiday = sslib.get_holiday_data(room_data, event_date, col_name, client, Mydep, weekday_begin, weekday_end, cycle_value, df_weekday)
#    list_holiday = []
    #Adding back lack dates into full_df
    if cruise_flag == False:
        full_df = full_df[~full_df['From'].isin(lackdate_df['From'])]
        full_df = pd.concat([full_df, lackdate_df]).sort_values('From').reset_index(drop=True)

    #TN 09182018: remove special dates from normal periods
#    full_df = full_df[~(full_df['From'].isin(pd.Series(list_holiday)))].reset_index(drop=True)
    #Creating json upload
    # Define json
    json_upload = {'H_Id': int(Mydep), 
                   'ValidHistoryFrom': from_date.strftime('%Y-%m-%d'), 
                    'Type': int(ss_type), 
                    'Periods':[]
                  } 
    
    for i in range(full_df['Group'].nunique()):
        data_filter = full_df[full_df['Group'] == i]
        period_sample = sslib.period_sample_func()
        period_sample['Value1'] = int(out_level.loc[out_level['Group']==i, 'rev_level'].values[0])
        period_sample['Value2'] = int(out_level.loc[out_level['Group']==i, 'vol_level'].values[0])
        for j in data_filter.index:
            period_sample['Dates'].append({'From': data_filter['From'][j].strftime('%Y-%m-%d'),
                                            'To': data_filter['To'][j].strftime('%Y-%m-%d')})
    
        json_upload['Periods'].append(period_sample)
    
    json_upload = sslib.save_outlier_json(outlier_date, json_upload)
    json_upload = sslib.save_holiday_json(holiday_data, json_upload, int(Mydep))
    
    if cruise_flag == True:
        json_upload = sslib.save_cruise_json(cruise_lack_date, json_upload)
#    link = os.path.join(current_path,'out_put','json_season_{}.json'.format(Mydep))
    link = current_path + '/out_put/json_season_{}.json'.format(Mydep)

    with open(link, 'w') as outfile:
        json.dump(json_upload, outfile)

    time_end_write_todb = time.time()
    log_file_client = {'Date': datetime.fromtimestamp(int(time_end_write_todb)).strftime('%Y-%m-%d'),
                       'H_Id': Mydep, 'Message': 'Compiling Results (v.2)', 'Status': 40}
#    sslib.log_json_upload(client, log_file_client)

#    json_upload = sslib.write_to_database(client, Mydep, ss_type, write, json_upload, default=False)

#    if validation == 1:
#        data_df = vlib.get_data_validation(client, hotel, Mydep, from_date, to_date)
#        try:
#            if ss_type == 0:
#                vlib.make_validation_file_revenue(client, hotel, Mydep, ss_type, data_conv_pv, nb_years, data_df, ending_date)
#            elif ss_type == 1:
#                vlib.make_validation_file_labor(client, hotel, Mydep, ss_type, data_conv_pv, nb_years, data_df, ending_date)
#        except Exception as e:
#            print(e)
    
    

        
    return df_error, json_upload


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser("Generates seasons V2 and writes result to database",
#                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument("-c", "--client-id", help="client ID of database", default=None)
#    parser.add_argument("-i", "--hotel-ids", help="Comma separated list of hotel ids", default=None)
#    parser.add_argument("-X", "--hotel-or-dept", help="Run for hotel or dept", default='dep')
#    parser.add_argument("-Y", "--years", help="Number of years of data to include", default=-1)
#    parser.add_argument("-G", "--type-labor", help="Labor type of department or hotel", default='t')
#    parser.add_argument("-R", "--type-revenue", help="Revenue type of department or hotel", default='t')
#    parser.add_argument("-F", "--type-foodcost", help="Foodcost type of department or hotel", default='t')
#    parser.add_argument("-N", "--no-write", help="Run, but do not write results to database", default=False)
#    parser.add_argument("-A", "--anims", help="Create animations and plots", default=False)
#    parser.add_argument("-v", "--validation", help="Create validation file", default=0)
##    parser.add_argument("-fr", "--from-time", help="Validation begin time", default=None)
##    parser.add_argument("-to", "--to-time", help="Validation end time", default=None)
#    
#    #=============================
#    args = parser.parse_args()
#    client = args.client_id
#    hotel_ids = args.hotel_ids
#    hotel_or_dept = args.hotel_or_dept
#    
#    YEAR_BACK_DEFAULT = 3
#    if int(args.years) != -1:
#        years_back = int(args.years)
#    else:
#        years_back = YEAR_BACK_DEFAULT
#        
#    revenue_type = args.type_revenue
#    labor_type = args.type_labor
#    food_type = args.type_foodcost
#    
#    no_write = args.no_write
#    make_plots = args.anims
#    validation = int(args.validation)
#    
#    write = not no_write
#    
#    #=============================
##    hotel_or_dept = 'dep'
##    years_back = 3
##    hotel_ids = '258'
##    client = 'EB19AB58-3D2E-4325-95B8-92403AB23191'
##    write = True
##    revenue_type = 't'
##    labor_type = 't'
##    food_type = 't'
##    validation = 0
#
## ================================================================
#    log.info('START RUNNING SEASON DETECTION')
#    start_time = time.time()
#    
#    otb_type = False
#    df_error = pd.DataFrame(columns = ['Driver', 'Type', 'Error'])    
#    if hotel_or_dept == 'hotel':
#        hotel_list = hotel_ids.split(',')
#        for hotel in hotel_list:
#            dep_list = db_api.get_depts_season(client, hotel, remove_hotel_id=False)
#            print('Hotel {} __ Dep list: {}\n'.format(hotel, dep_list))
#            df_error = generate_season_by_type(client, dep_list, years_back, revenue_type, labor_type, food_type, otb_type, \
#                                                                           write, df_error, validation)
#     
#    elif hotel_or_dept == 'dep':
#        dep_list = hotel_ids.split(',')
#        print('Dep list: {}\n'.format(dep_list))
#        df_error = generate_season_by_type(client, dep_list, years_back, revenue_type, labor_type, food_type, otb_type, \
#                                                                       write, df_error, validation)
#            
#    print("--- TOTAL TIME RUN ALL %s seconds ---" % (time.time() - start_time))
#    
#    if len(df_error) != 0:
#        print(df_error)


        
            
            