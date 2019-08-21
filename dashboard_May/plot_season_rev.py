#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:20:01 2019

@author: tp7
"""


import department_seasons_json_v2 as season_jsons
import rev_json_val as forecasted_rev
import code_season_json as ss
import os
import pandas as pd
import json
import numpy as np
import plotly.graph_objs as go

layout_1 = {
      "legend": {
        "x": 0.3, 
        "y": 1.0, 
        "orientation": "h", 
        "yanchor": "bottom"
      }, 
      "margin": {
        "r": 40, 
        "t": 40, 
        "b": 40, 
        "l": 40
      }, 
      "plot_bgcolor": "rgb(250, 250, 250)", 
      "xaxis": {"rangeselector": {
          "x": 0, 
          "y": 1.0, 
          "bgcolor": "rgba(150, 200, 250, 0.4)", 
          "buttons": [
            {
              "count": 1, 
              "label": "1mo", 
              "step": "month", 
              "stepmode": "backward"
            }, 
            {
              "count": 3, 
              "label": "3mo", 
              "step": "month", 
              "stepmode": "backward"
            },
            {
              "count": 1, 
              "label": "1yr", 
              "step": "year", 
              "stepmode": "backward"
            },
            {"step": "all"}
          ], 
          "font": {"size": 13}
        }}, 
      
      }

layout_category = {
      "legend": {
        "x": 0.3, 
        "y": 1.0, 
        "orientation": "h", 
        "yanchor": "bottom"
      }, 
      "margin": {
        "r": 40, 
        "t": 40, 
        "b": 40, 
        "l": 40
      }, 
      "plot_bgcolor": "rgb(250, 250, 250)", 
      "xaxis": {"rangeselector": {
          "x": 0, 
          "y": 1.0, 
          "bgcolor": "rgba(150, 200, 250, 0.4)", 
          "buttons": [
            {
              "count": 1, 
              "label": "1mo", 
              "step": "month", 
              "stepmode": "backward"
            }, 
            {
              "count": 3, 
              "label": "3mo", 
              "step": "month", 
              "stepmode": "backward"
            },
            {
              "count": 1, 
              "label": "1yr", 
              "step": "year", 
              "stepmode": "backward"
            },
            {"step": "all"}
          ], 
          "font": {"size": 13}
        }, 'type': 'category'}, 
      
      }      
      
layout_2 = {
      "legend": {
        "x": 0.3, 
        "y": 0.9, 
        "orientation": "h", 
        "yanchor": "bottom"
      }, 
      "margin": {
        "r": 40, 
        "t": 40, 
        "b": 40, 
        "l": 40
      }, 
      "plot_bgcolor": "rgb(250, 250, 250)", 
      "xaxis": {"rangeselector": {
          "x": 0, 
          "y": 0.9, 
          "bgcolor": "rgba(150, 200, 250, 0.4)", 
          "buttons": [
            {
              "count": 1, 
              "label": "reset", 
              "step": "all"
            }, 
            {
              "count": 1, 
              "label": "1yr", 
              "step": "year", 
              "stepmode": "backward"
            }, 
            {
              "count": 3, 
              "label": "3 mo", 
              "step": "month", 
              "stepmode": "backward"
            }, 
            {
              "count": 1, 
              "label": "1 mo", 
              "step": "month", 
              "stepmode": "backward"
            }, 
            {"step": "all"}
          ], 
          "font": {"size": 13}
        }}, 
      
      }  
      
      
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


def list_weekday(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values):
    
    
    hotel = filename.split('_')[1]
    name_json = 'json_season_{}.json'.format(Mydep)
    with open(os.path.join(current_path,'out_put',name_json)) as json_file:
        json_upload = json.load(json_file)
    
    
    
    rev_all_new_df = ss.convert_json_ss_outlier(json_upload, filename, hotel, Mydep, segment, current_path)
    if period_dropdown_values == 'all':
        weekday = rev_all_new_df['Day'].drop_duplicates()
    else:
        weekday = rev_all_new_df['Day'][rev_all_new_df['Groups'] == str(period_dropdown_values)].drop_duplicates()
    weekday = weekday.sort_values()    
    weekday = pd.Series(['all']).append(weekday)
    
    return weekday



def list_period(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks):
    hotel = filename.split('_')[1]
    df = pd.read_csv(os.path.join(UPLOAD_DIRECTORY , filename))
    client = str(filename.split('_')[0])
    years_back = len(df)/365
    ss_type = 0    
    o_type = False
    write = False
    df_error = pd.DataFrame(columns = ['Driver', 'Type', 'Error'])
    if submit_n_clicks == 0:
        name_json = 'json_season_{}.json'.format(Mydep)
        with open(os.path.join(current_path,'out_put',name_json)) as json_file:
            json_upload = json.load(json_file)
    else:
        df_error, json_upload = season_jsons.generate_season(client, hotel, Mydep, years_back, ss_type, o_type, write, df_error, df, current_path)
    
    rev_all_new_df = ss.convert_json_ss_outlier(json_upload, filename, hotel, Mydep, segment, current_path)
    rev_all_new_df = rev_all_new_df.sort_values('Groups')
    
    list_period = rev_all_new_df['Groups'].drop_duplicates()
    
    list_period = pd.Series(['all']).append(list_period)
    return list_period
     
     
def plot_season(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values, weekday_dropdown_values):
    
    name_json = 'json_season_{}.json'.format(Mydep)
    with open(os.path.join(current_path,'out_put',name_json)) as json_file:
        json_upload = json.load(json_file)
    
    hotel = filename.split('_')[1]
    rev_col = Mydep + '_0_RV'
    rev_all_new_df = ss.convert_json_ss_outlier(json_upload, filename, hotel, Mydep, segment, current_path)
    rev_all_new_df = rev_all_new_df.sort_values('Groups')
    
    holiday_all_new_df = rev_all_new_df.copy().reset_index(drop = True)
    
    holiday_all_new_df[rev_col][holiday_all_new_df['Groups'] != 'holiday'] = 0
    holiday_all_new_df[rev_col][holiday_all_new_df['Groups'] == 'holiday'] = max(rev_all_new_df[rev_col])
    
    holiday_all_new_df = holiday_all_new_df[holiday_all_new_df['Period_Type'] != 2]

    holiday_all_new_df = holiday_all_new_df.sort_values('Period_Type', ascending = False)


    df_duplicates = holiday_all_new_df.ix[holiday_all_new_df[['Date']].drop_duplicates().index]
    df_duplicates = df_duplicates.sort_values('Date')
    
    
    true_holiday = rev_all_new_df['Groups'].drop_duplicates().isin(['holiday']).sum()
    rev_all_new_df = rev_all_new_df[rev_all_new_df['Period_Type'] == 0]
    rev_all_new_df['Date'] = [i.strftime('%Y-%m-%d') for i in rev_all_new_df['Date']]
    list_data = []

    if period_dropdown_values == 'all' and weekday_dropdown_values == 'all':
        
    
        for i in sorted(rev_all_new_df['Groups'].drop_duplicates()):
            dicts = {
          "x": rev_all_new_df['Date'][rev_all_new_df['Groups'] == i].tolist(), 
          "y": rev_all_new_df[rev_col][rev_all_new_df['Groups'] == i].tolist(), 
          "marker": {"color": rev_all_new_df['colors'][rev_all_new_df['Groups'] == i].tolist()}, 
          "name": str(i), 
          "type": "bar", 
          "yaxis": "y"
        }
            list_data.append(dicts)
            
        layout_full = layout_1.copy()
     
    elif period_dropdown_values != 'all' and period_dropdown_values != 'outlier' and weekday_dropdown_values == 'all':
        rev_all_new_df = rev_all_new_df[rev_all_new_df['Groups'] == str(period_dropdown_values)]
#        rev_all_new_df = rev_all_new_df.sort_values('Date')                                
        for i in rev_all_new_df['Day'].drop_duplicates():
            dicts = {
          "x": rev_all_new_df['Date'][rev_all_new_df['Day'] == i].tolist(), 
          "y": rev_all_new_df[rev_col][rev_all_new_df['Day'] == i].tolist(), 
          "marker": {"color": rev_all_new_df['colors'][rev_all_new_df['Day'] == i].tolist()}, 
          "name": str(i), 
          "type": "bar", 
          "yaxis": "y"
        }
            list_data.append(dicts)
            
        layout_full = layout_1.copy()

    elif period_dropdown_values == 'all' and weekday_dropdown_values != 'all':
        rev_all_new_df = rev_all_new_df[rev_all_new_df['Day'] == str(weekday_dropdown_values)]
#        rev_all_new_df = rev_all_new_df.sort_values('Date')                                
        for i in sorted(rev_all_new_df['Groups'].drop_duplicates()):
            dicts = {
          "x": rev_all_new_df['Date'][rev_all_new_df['Groups'] == i].tolist(), 
          "y": rev_all_new_df[rev_col][rev_all_new_df['Groups'] == i].tolist(), 
          "marker": {"color": rev_all_new_df['colors'][rev_all_new_df['Groups'] == i].tolist()}, 
          "name": str(i), 
          "type": "bar", 
          "yaxis": "y"
        }
            list_data.append(dicts)
            
        layout_full = layout_1.copy()
        
    elif period_dropdown_values == 'outlier' and weekday_dropdown_values == 'all':
        
        rev_all_new_df = rev_all_new_df[rev_all_new_df['colors'] == '#1d2021']
        for i in sorted(rev_all_new_df['Day'].drop_duplicates()):
            dicts = {
          "x": rev_all_new_df['Date'][rev_all_new_df['Day'] == i].tolist(), 
          "y": rev_all_new_df[rev_col][rev_all_new_df['Day'] == i].tolist(), 
          "marker": {"color": rev_all_new_df['colors'][rev_all_new_df['Day'] == i].tolist()}, 
          "name": str(i), 
          "type": "bar", 
          "yaxis": "y"
        }
            list_data.append(dicts)
        
        layout_full = layout_category.copy()
    
    elif period_dropdown_values == 'outlier' and weekday_dropdown_values != 'all':
        
        rev_all_new_df = rev_all_new_df[rev_all_new_df['colors'] == '#1d2021']
        rev_all_new_df = rev_all_new_df[rev_all_new_df['Day'] == str(weekday_dropdown_values)]
        for i in sorted(rev_all_new_df['Day'].drop_duplicates()):
            dicts = {
          "x": rev_all_new_df['Date'][rev_all_new_df['Day'] == i].tolist(), 
          "y": rev_all_new_df[rev_col][rev_all_new_df['Day'] == i].tolist(), 
          "marker": {"color": rev_all_new_df['colors'][rev_all_new_df['Day'] == i].tolist()}, 
          "name": str(i), 
          "type": "bar", 
          "yaxis": "y"
        }
            list_data.append(dicts)    
            
        layout_full = layout_category.copy()
      
    else:
        rev_all_new_df = rev_all_new_df[rev_all_new_df['Groups'] == str(period_dropdown_values)]
        rev_all_new_df = rev_all_new_df[rev_all_new_df['Day'] == str(weekday_dropdown_values)]
        rev_all_new_df = rev_all_new_df.sort_values('Date')                                
        dicts = {
          "x": rev_all_new_df['Date'].tolist(), 
          "y": rev_all_new_df[rev_col].tolist(), 
          "marker": {"color": rev_all_new_df['colors'].tolist()}, 
          "name": str(period_dropdown_values) + '_' + str(weekday_dropdown_values), 
          "type": "bar", 
          "yaxis": "y"
        }
           
       
        list_data.append(dicts)
        layout_full = layout_category.copy()
        
    if true_holiday != 0:
#       
       dicts_holiday = {
       "x" : df_duplicates['Date'].tolist(),
       "y" : df_duplicates[rev_col].tolist(),
#       "connectgaps": True,
#       "hoverinfo" : holiday_all_new_df['Date'].tolist()+holiday_all_new_df[rev_col].tolist(),
       "fill": "tozeroy", 
       "fillcolor": "rgba(224, 102, 102, 0.2)", 
       "mode": "none", 
       "name": "holidays", 
       "type": "scatter"
#       "uid": "bd8f2b", 
#       "xsrc": "Dreamshot:3959:496082", 
#       "ysrc": "Dreamshot:3959:044fc9"
       }
       
#       dicts_holiday = {
#       "x" : holiday_all_new_df['Date'].tolist(),
#       "y" : holiday_all_new_df[rev_col].tolist(),
#       "connectgaps": True,
##       "hoverinfo" : holiday_all_new_df['Date'].tolist()+holiday_all_new_df[rev_col].tolist(),
#       "fill": "tozeroy", 
#       "fillcolor": "rgba(224, 102, 102, 0.2)", 
#       "mode": "none", 
#       "name": "Halftime", 
#       "type": "scatter", 
#       "uid": "bd8f2b", 
#       "xsrc": "Dreamshot:3959:496082", 
#       "ysrc": "Dreamshot:3959:044fc9"
#        'stackgroup' : 'one'
#       }
       
       
       list_data.append(dicts_holiday)
        
    return {'data' : list_data, 'layout': layout_full}
#    return {'data' : list_data}

def plot_revenue(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values, weekday_dropdown_values):
    
    
    hotel = int(filename.split('_')[1])
#    df = pd.read_csv(os.path.join(UPLOAD_DIRECTORY , filename))
    rev_col = Mydep + '_0'
    
    if submit_n_clicks == 0:
        forecast_output = 'revenue_{}_{}.csv'.format(hotel,rev_col)
        revenue_forecast = pd.read_csv(os.path.join(current_path,'out_put',forecast_output))
        
    else:
        revenue_forecast = forecasted_rev.forecast_revenue(filename,current_path,hotel, Mydep, segment)
    
    revenue_forecast['Groups'] = [i.split('_')[0] for i in revenue_forecast['Period_Day']]
    revenue_forecast['Day'] = [i.split('_')[1] for i in revenue_forecast['Period_Day']]

    

    list_data = []

    if period_dropdown_values == 'all' and weekday_dropdown_values == 'all':
        
        layout_full = layout_1.copy()
        
        trace_line = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Actual_Rev'], 
          "line": {"width": 1}, 
          "marker": {"color": "#000000"}, 
          "mode": "lines", 
          "name": "Actual Rev", 
          "type": "scatter", 
    #      "yaxis": "y2"
        }
        
        trace_best = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Best_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#FF0000"}, 
          "mode": "lines", 
          "name": "Best FC Rev", 
          "type": "scatter", 
        }
        
        trace_linear = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Linear_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#f18b00"}, 
          "mode": "lines", 
          "name": "Linear FC Rev", 
          "type": "scatter", 
        }
        
        trace_tree = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Tree_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#0000FF"}, 
          "mode": "lines", 
          "name": "Tree FC Rev", 
          "type": "scatter", 
        }
        
        trace_max = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Max_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": '#008000'}, 
          "mode": "lines", 
          "name": "Max FC Rev", 
          "type": "scatter", 
        }  
    
        list_data = [trace_line,trace_best,trace_linear,trace_max, trace_tree]

    elif period_dropdown_values != 'all' and period_dropdown_values != 'outlier' and weekday_dropdown_values == 'all':
        layout_full = layout_category.copy()
        revenue_forecast = revenue_forecast[revenue_forecast['Groups'] == str(period_dropdown_values)]
        trace_line = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Actual_Rev'], 
          "line": {"width": 1}, 
          "marker": {"color": "#000000"}, 
          "mode": "lines+markers", 
          "name": "Actual Rev", 
          "type": "scatter", 
    #      "yaxis": "y2"
        }
        
        trace_best = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Best_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#FF0000"}, 
          "mode": "lines+markers", 
          "name": "Best FC Rev", 
          "type": "scatter", 
        }
        
        trace_linear = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Linear_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#f18b00"}, 
          "mode": "lines+markers", 
          "name": "Linear FC Rev", 
          "type": "scatter", 
        }
        
        trace_tree = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Tree_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#0000FF"}, 
          "mode": "lines+markers", 
          "name": "Tree FC Rev", 
          "type": "scatter", 
        }
        
        trace_max = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Max_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": '#008000'}, 
          "mode": "lines+markers", 
          "name": "Max FC Rev", 
          "type": "scatter", 
        }  
    
        list_data = [trace_line,trace_best,trace_linear,trace_max, trace_tree]

    elif period_dropdown_values == 'all' and weekday_dropdown_values != 'all':
        
        layout_full = layout_2.copy()
        
        revenue_forecast = revenue_forecast[revenue_forecast['Day'] == str(weekday_dropdown_values)]
        trace_line = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Actual_Rev'], 
          "line": {"width": 1}, 
          "marker": {"color": "#000000"}, 
          "mode": "lines+markers", 
          "name": "Actual Rev", 
          "type": "scatter", 
    #      "yaxis": "y2"
        }
        
        trace_best = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Best_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#FF0000"}, 
          "mode": "lines+markers", 
          "name": "Best FC Rev", 
          "type": "scatter", 
        }
        
        trace_linear = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Linear_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#f18b00"}, 
          "mode": "lines+markers", 
          "name": "Linear FC Rev", 
          "type": "scatter", 
        }
        
        trace_tree = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Tree_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#0000FF"}, 
          "mode": "lines", 
          "name": "Tree FC Rev", 
          "type": "scatter", 
        }
        
        trace_max = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Max_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": '#008000'}, 
          "mode": "lines+markers", 
          "name": "Max FC Rev", 
          "type": "scatter", 
        }  
    
        list_data = [trace_line,trace_best,trace_linear,trace_max, trace_tree]            
    else:
        layout_full = layout_category.copy()
        revenue_forecast = revenue_forecast[revenue_forecast['Groups'] == str(period_dropdown_values)]
        revenue_forecast = revenue_forecast[revenue_forecast['Day'] == str(weekday_dropdown_values)]
        trace_line = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Actual_Rev'], 
          "line": {"width": 1}, 
          "marker": {"color": "#000000"}, 
          "mode": "lines+markers", 
          "name": "Actual Rev", 
          "type": "scatter", 
    #      "yaxis": "y2"
        }
        
        trace_best = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Best_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#FF0000"}, 
          "mode": "lines+markers", 
          "name": "Best FC Rev", 
          "type": "scatter", 
        }
        
        trace_linear = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Linear_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#f18b00"}, 
          "mode": "lines+markers", 
          "name": "Linear FC Rev", 
          "type": "scatter", 
        }
        
        trace_tree = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Tree_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": "#0000FF"}, 
          "mode": "lines+markers", 
          "name": "Tree FC Rev", 
          "type": "scatter", 
        }
        
        trace_max = {
          "x": revenue_forecast['Date'], 
          "y": revenue_forecast['Max_Forecast'], 
          "line": {"width": 1}, 
          "marker": {"color": '#008000'}, 
          "mode": "lines+markers", 
          "name": "Max FC Rev", 
          "type": "scatter", 
        }  
    
        list_data = [trace_line,trace_best,trace_linear,trace_max, trace_tree]
        

    return {'data' : list_data, 'layout': layout_full}



def smape_revenue(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values, weekday_dropdown_values):
    hotel = int(filename.split('_')[1])
#    df = pd.read_csv(os.path.join(UPLOAD_DIRECTORY , filename))
    rev_col = Mydep + '_0'
    
    try:
        forecast_output = 'revenue_{}_{}.csv'.format(hotel,rev_col)
        revenue_forecast = pd.read_csv(os.path.join(current_path,'out_put',forecast_output))
        
    except:
        print('no forecating revenue data')
        
    list_smape = []
    col_name = 'Actual_Rev'
    
    revenue_forecast['Groups'] = [i.split('_')[0] for i in revenue_forecast['Period_Day']]
    revenue_forecast['Day'] = [i.split('_')[1] for i in revenue_forecast['Period_Day']]
        
    all_points_selected = revenue_forecast.copy()

    if period_dropdown_values == 'all' and weekday_dropdown_values == 'all':
        
        for col in ['Linear_Forecast', 'Max_Forecast', 'Tree_Forecast', 'Best_Forecast']:
        
        
            smape_1month_v1 = 1 - sMAPE(revenue_forecast[col_name][-30:], revenue_forecast[col][-30:])
            smape_3month_v1 = 1 - sMAPE(revenue_forecast[col_name][-90:], revenue_forecast[col][-90:])
            smape_6month_v1 = 1 - sMAPE(revenue_forecast[col_name][-180:], revenue_forecast[col][-180:])
            smape_12month_v1 = 1 - sMAPE(revenue_forecast[col_name][-365:], revenue_forecast[col][-365:])
            smape_all_points = 1 - sMAPE(all_points_selected[col_name], all_points_selected[col])
            
            
            result = pd.DataFrame()
            result['Models'] = [col]
            result['Acc_1m'] = ['{:,.2%}'.format(smape_1month_v1)]
            result['Acc_3m'] = ['{:,.2%}'.format(smape_3month_v1)]
            result['Acc_6m'] = ['{:,.2%}'.format(smape_6month_v1)]
            result['Acc_12m'] = ['{:,.2%}'.format(smape_12month_v1)]
            result['Acc_points_on_Graph'] = ['{:,.2%}'.format(smape_all_points)]
            
            
            list_smape.append(result.iloc[0].to_dict())
    
       
    elif period_dropdown_values != 'all' and period_dropdown_values != 'outlier' and weekday_dropdown_values == 'all':
        all_points_selected = all_points_selected[all_points_selected['Groups'] == str(period_dropdown_values)]
        for col in ['Linear_Forecast', 'Max_Forecast', 'Tree_Forecast', 'Best_Forecast']:
        
        
            smape_1month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_3month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_6month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_12month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_all_points = 1 - sMAPE(all_points_selected[col_name], all_points_selected[col])
            
            result = pd.DataFrame()
            result['Models'] = [col]
            result['Acc_1m'] = ['{:,.2%}'.format(smape_1month_v1)]
            result['Acc_3m'] = ['{:,.2%}'.format(smape_3month_v1)]
            result['Acc_6m'] = ['{:,.2%}'.format(smape_6month_v1)]
            result['Acc_12m'] = ['{:,.2%}'.format(smape_12month_v1)]
            result['Acc_points_on_Graph'] = ['{:,.2%}'.format(smape_all_points)]            

            list_smape.append(result.iloc[0].to_dict())
                                    
        
    elif period_dropdown_values == 'all' and weekday_dropdown_values != 'all':
        
        all_points_selected = all_points_selected[all_points_selected['Day'] == str(weekday_dropdown_values)]
        for col in ['Linear_Forecast', 'Max_Forecast', 'Tree_Forecast', 'Best_Forecast']:
        
        
            smape_1month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_3month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_6month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_12month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_all_points = 1 - sMAPE(all_points_selected[col_name], all_points_selected[col])
            
            result = pd.DataFrame()
            result['Models'] = [col]
            result['Acc_1m'] = ['{:,.2%}'.format(smape_1month_v1)]
            result['Acc_3m'] = ['{:,.2%}'.format(smape_3month_v1)]
            result['Acc_6m'] = ['{:,.2%}'.format(smape_6month_v1)]
            result['Acc_12m'] = ['{:,.2%}'.format(smape_12month_v1)]
            result['Acc_points_on_Graph'] = ['{:,.2%}'.format(smape_all_points)]                        

            list_smape.append(result.iloc[0].to_dict())
        
                              
    else:
        all_points_selected = all_points_selected[all_points_selected['Groups'] == str(period_dropdown_values)]
        all_points_selected = all_points_selected[all_points_selected['Day'] == str(weekday_dropdown_values)]
        for col in ['Linear_Forecast', 'Max_Forecast', 'Tree_Forecast', 'Best_Forecast']:
        
        
            smape_1month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_3month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_6month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_12month_v1 = 1 - sMAPE(revenue_forecast[col_name], revenue_forecast[col])
            smape_all_points = 1 - sMAPE(all_points_selected[col_name], all_points_selected[col])
            
            result = pd.DataFrame()
            result['Models'] = [col]
            result['Acc_1m'] = ['{:,.2%}'.format(smape_1month_v1)]
            result['Acc_3m'] = ['{:,.2%}'.format(smape_3month_v1)]
            result['Acc_6m'] = ['{:,.2%}'.format(smape_6month_v1)]
            result['Acc_12m'] = ['{:,.2%}'.format(smape_12month_v1)]
            result['Acc_points_on_Graph'] = ['{:,.2%}'.format(smape_all_points)]                                    

            list_smape.append(result.iloc[0].to_dict())    
    return list_smape   


       

