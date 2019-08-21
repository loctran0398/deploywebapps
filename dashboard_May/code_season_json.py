# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 13:41:56 2019

@author: cong-thanh
"""
#import json
import pandas as pd
import os
from datetime import timedelta
#import numpy as np

def unpack(df, column, fillna=None):
    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
        del ret[column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
        del ret[column]
    return ret
 


# convert json data  

#def convert_json_ss(data_json, name_rev, hotel, Mydep, segment, current_path):
##    with open(os.path.join(current_path,'out_put',name_json)) as json_file:
##        data = json.load(json_file)
#    
#    my_cmap = [
#            '#97040c', '#9d4b00', '#d4b400', '#0515a9', '#026f02',
#            '#a0300e', '#d85013', '#ffdd1c', '#454ea0', '#20a308',
#            '#d8017a', '#fc6621', '#f1ef00', '#3d8ddc', '#3cc923',
#            '#fc4eb0', '#fb9262', '#fffe69', '#3fcdfd', '#88d63c',
#            '#ff96d1', '#ffc0a4', '#fffeab', '#9de6fe', '#98edb5'
#        ]    
#    
#    df = pd.DataFrame(data_json)
#    df= df.rename(columns = {df.iloc[:,0].name:'Date'})
#    
#    full_df =pd.DataFrame((d for idx, d in df['Periods'].iteritems()))
#    full_df = full_df[full_df['Type'] != 2].reset_index(drop = True)
#    list_date = []
#    all_new_df = pd.DataFrame() 
#    for k,dates in enumerate(full_df['Dates']):
#        new_df = pd.DataFrame()
#        one_list_dates =[]
#        for j in range(len(dates)):
#           one_list_dates.extend([pd.to_datetime(dates[j][u'From']) + timedelta(days = x) for x in range((pd.to_datetime(dates[j][u'To']) - pd.to_datetime(dates[j][u'From'])).days + 1)])
#        new_df['Date'] =  one_list_dates
#        new_df['Periods'] = full_df['Type'].iloc[k]
#        new_df['Value1'] = full_df['Value1'].iloc[k]
#        new_df['Value2'] = full_df['Value2'].iloc[k]
#        new_df['Groups'] = k
#        new_df['colors'] = my_cmap[k]
#        all_new_df = pd.concat([all_new_df,new_df])
#    
#    # take revenue from data.csv    
#        
#    rev_df = pd.read_csv(os.path.join(current_path, 'input_data' ,name_rev))
##    rev_df = rev_df.replace(np.nan, 0.0)
##    rev_df[rev_df < 0.0] = 0.0
#
#    rev_col = Mydep + '_' + str(segment) + '_RV'
#    
#    rev_df= rev_df.rename(columns = {rev_df.iloc[:,0].name:'Date'})
#    
#    rev_df = rev_df[[rev_col,'Date']]
#    rev_df['Date'] = pd.to_datetime(rev_df['Date'])
#    
#    rev_all_new_df = rev_df.merge(all_new_df, on = 'Date', how = 'left')
#    rev_all_new_df = rev_all_new_df.rename(columns = {'Periods' : 'Period_Type'})
#    rev_all_new_df['Day'] = [i.weekday() for i in rev_all_new_df['Date']]
#
#    return  rev_all_new_df
    
def convert_json_ss_outlier(data_json, name_rev, hotel, Mydep, segment, current_path):
#    with open(os.path.join(current_path,'out_put',name_json)) as json_file:
#        data = json.load(json_file)
    
    my_cmap = [
            '#97040c', '#9d4b00', '#d4b400', '#0515a9', '#026f02',
            '#a0300e', '#d85013', '#ffdd1c', '#454ea0', '#20a308',
            '#d8017a', '#fc6621', '#f1ef00', '#3d8ddc', '#3cc923',
            '#fc4eb0', '#fb9262', '#fffe69', '#3fcdfd', '#88d63c',
            '#ff96d1', '#ffc0a4', '#fffeab', '#9de6fe', '#98edb5'
        ]    
    
    df = pd.DataFrame(data_json)
    df= df.rename(columns = {df.iloc[:,0].name:'Date'})
    
    full_df =pd.DataFrame((d for idx, d in df['Periods'].iteritems()))
#    full_df = full_df[full_df['Type'] != 2].reset_index(drop = True)
#    list_date = []
    all_new_df = pd.DataFrame() 
    for k,dates in enumerate(full_df['Dates']):
        new_df = pd.DataFrame()
#        df_holidays = pd.DataFrame()
        one_list_dates =[]
        for j in range(len(dates)):
           one_list_dates.extend([pd.to_datetime(dates[j][u'From']) + timedelta(days = x) for x in range((pd.to_datetime(dates[j][u'To']) - pd.to_datetime(dates[j][u'From'])).days + 1)])
        new_df['Date'] =  one_list_dates
        new_df['Periods'] = full_df['Type'].iloc[k]
        new_df['Value1'] = full_df['Value1'].iloc[k]
        new_df['Value2'] = full_df['Value2'].iloc[k]
        new_df['Groups'] = str(k)
        new_df['colors'] = my_cmap[k]        

        if int(full_df['Type'].iloc[k]) == 2:
            new_df['colors'] = '#1d2021'
            new_df['Groups'] = 'outlier'
#            new_df['Groups'] = np.nan
        
        if int(full_df['Type'].iloc[k]) == 1:
#            df_holidays = new_df.copy()
#            df_holidays['colors'] = '#813F0B'
#            df_holidays['Groups'] = 'holiday'
#            new_df['Periods'] = 0
#            new_df['Groups'] = np.nan
            new_df['colors'] = '#813F0B'
            new_df['Groups'] = 'holiday'
            
        all_new_df = pd.concat([all_new_df,new_df])
    
    all_new_df = all_new_df.reset_index(drop = True)    
    outlier_date = all_new_df['Date'][all_new_df['Periods'] == 2].values.tolist()
#    holiday_date = all_new_df['Date'][all_new_df['Periods'] == 1].values.tolist()

    all_new_df['colors'][(all_new_df['Date'].isin(outlier_date)) & (all_new_df['Periods'] == 0)] = '#1d2021'
    
    all_new_df_drop = all_new_df[all_new_df.columns - ['Groups']].drop_duplicates()
    
    all_new_df = all_new_df.ix[all_new_df_drop.index]
    all_new_df.index = all_new_df['Date']
    
    

#    all_new_df['colors'][(all_new_df['Date'].isin(holiday_date)) & (all_new_df['Periods'] == 0)] = '#813F0B'

    
#    all_new_df = all_new_df.sort_values('Date')
#    all_new_df_k = all_new_df.fillna(method='ffill')
    
    # take revenue from data.csv    
        
    rev_df = pd.read_csv(os.path.join(current_path, 'input_data' ,name_rev))
#    rev_df = rev_df.replace(np.nan, 0.0)
#    rev_df[rev_df < 0.0] = 0.0

    rev_col = Mydep + '_' + str(segment) + '_RV'
    
    rev_df= rev_df.rename(columns = {rev_df.iloc[:,0].name:'Date'})
    
    rev_df = rev_df[[rev_col,'Date']]
    rev_df['Date'] = pd.to_datetime(rev_df['Date'])
    
    
    rev_all_new_df = rev_df.merge(all_new_df, on = 'Date', how = 'right')
    rev_all_new_df = rev_all_new_df[(rev_all_new_df['Date'] >= rev_df['Date'].iloc[0]) & (rev_all_new_df['Date'] <= rev_df['Date'].iloc[-1])]
    rev_all_new_df = rev_all_new_df.rename(columns = {'Periods' : 'Period_Type'})
    rev_all_new_df['Day'] = [str(i.weekday()) for i in rev_all_new_df['Date']]
    
    return  rev_all_new_df