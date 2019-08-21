import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os
import dash_table
import plotly.graph_objs as go
import time

#from plotly.graph_objs import *
import plot_season_rev as plt_ss_rev

current_path = os.getcwd()

UPLOAD_DIRECTORY = os.path.join(current_path, 'input_data') 
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


dicts = {'Bristol' : '50_0_RV', 'Frokost' : '258_0_RV', 'Grillen' : '259_0_RV', 'Kurs & Konferanse' : '260_0_RV',
             'Minibar' : '261_0_RV', 'Bankett' : '266_0_RV', 'Lokaleie' : '269_0_RV', 'Parkering' : '270_0_RV', 'Annet Salg' : '273_0_RV' , 'Room Service' : '854_0_RV'}
       
dicts_weekday = {'Sunday' : '6' , 'Saturday' : '5', 'Friday' : '4', 'Thursday' : '3', 'Wednesday' : '2', 'Tuesday' : '1', 'Monday' : '0', 'outlier' : 'outlier', 'all' : 'all'}
             
             
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Markdown('''# __Demo System__'''),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Button('Upload File')
        ]),
        style={
            'width': '10%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '1px',
            'textAlign': 'center',
            'margin': '1px'
        },
       
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload', style = {'marginTop': '5px'}),
    html.Hr(),
    dcc.Dropdown(id='my-dropdown', style = {'width': '40%'}),
    html.Hr(),
    html.Div(id='output-dep_id-upload'),
    
    dcc.Markdown('''## __Seasonality__'''),
    dcc.ConfirmDialogProvider(
        children=html.Button(
            'Generate Season',
        ),
        id='generate_season',
        submit_n_clicks = 0,
        message='Are you sure you want to generate Season?'
    ),
    html.Div(id='output-provider_ss_finished', style = {'marginTop': '5px'}),
    html.Div(id='output-provider_ss', style = {'marginTop': '5px'}),
    
    html.Hr(),
    html.Div(id='data_season'),
    html.Div([
        #Boxplot
        html.Div([dcc.Markdown('Period'), dcc.Dropdown(id='s111')], style={'width': '10%', 'display': 'inline-block'}),
        html.Div(' ', style={'width': '5%', 'display': 'inline-block'}),
        html.Div([dcc.Markdown('Weekday'), dcc.Dropdown(id='s222')], style={'width': '10%', 'display': 'inline-block'})
    ], style={'marginBottom': 1, 'marginTop': 1}),
    html.Hr(),
    dcc.Graph(id='my-graph_ss'),

    dcc.Markdown('''## __Revenue Forecasting__'''),
    dcc.ConfirmDialogProvider(
        children=html.Button(
            'Generate Forecasted Revenue',
        ),
        id='generate_driver_rev',
        submit_n_clicks = 0,
        message='Are you sure you want to forecast revenue?'
    ),
    html.Div(id='output-provider_rev', style = {'marginTop': '5px'}),
#    html.Hr(),
##    dcc.Dropdown(id='periods'),
#    html.Div([
#        #Boxplot
#        html.Div([dcc.Markdown('Period'), dcc.Dropdown(id='r111')], style={'width': '10%', 'display': 'inline-block'}),
#        html.Div(' ', style={'width': '5%', 'display': 'inline-block'}),
#        html.Div([dcc.Markdown('Weekday'), dcc.Dropdown(id='r222')], style={'width': '10%', 'display': 'inline-block'})
#    ], style={'marginBottom': 1, 'marginTop': 1}),
#    html.Hr(),
    dcc.Graph(id='my-graph_rev'),

    dcc.Markdown('''## __Accuracy Report__'''),
    html.Div([
        html.Div(' ', style={'width': '35%', 'display': 'inline-block'}),  
        html.Div(dash_table.DataTable(
            id='sMAPE_table',
            columns=[{
                'name': '{}'.format(i),
                'id': '{}'.format(i)
            } for i in ['Models','Acc_1m','Acc_3m', 'Acc_6m', 'Acc_12m', 'Acc_points_on_Graph']],
            style_table={'width': '500px'},
            editable=True),
        style={'display': 'inline-block'})
    ]) 
])

def getKeysByValues(dictOfElements, listOfValues):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] in listOfValues:
            listOfKeys.append(item[0])
    return  listOfKeys        

def parse_contents(contents, filename, date):
    
    content_type, content_string = contents.split(',')
    global list_dep
    global list_col
    global name_file
    name_file = filename
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
#            del df['261_0_RV']
#            del df['854_0_RV']
            list_dep = [{'label' : i, 'value' : i} for i in df.columns if i != 'Date']
            list_col = [i for i in df.columns if i != 'Date']
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


    return html.Div([
            'Upload file {} successfully '.format(filename)
        ])


    
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    return children
    
     
@app.callback(Output('my-dropdown', 'options'),
              [Input('output-data-upload', 'children')])

def option_values(childrenss):
               
    name_list_dep = getKeysByValues(dicts,list_col)
    
    new_list_dep = [{'label' : i, 'value' : i} for i in name_list_dep]
                
    return new_list_dep

    

    
@app.callback(Output('my-dropdown','value'),
              [Input('my-dropdown','options')])

def set_value(available_options): 
    return available_options['value']

#@app.callback(Output('my-graph_ss','figure'),
#              [Input('generate_season','submit_n_clicks'),
#               Input('my-dropdown','value')])
#
#data_season
@app.callback(Output('s111', 'options'),
              [Input('generate_season','submit_n_clicks'),
               Input('my-dropdown','value')])
               

def take_period_ss(submit_n_clicks, selected_dropdown_values):
            
    filename = name_file
    selected_dropdown_values = dicts[str(selected_dropdown_values)]
    selected_dropdown_values = str(selected_dropdown_values)
    Mydep = selected_dropdown_values.split('_')[0]
    segment = selected_dropdown_values.split('_')[1]

    period = plt_ss_rev.list_period(filename, UPLOAD_DIRECTORY, current_path, Mydep,segment, submit_n_clicks)
    
    dict_period = [{'label' : i, 'value' : i} for i in period]
   
    return dict_period

    
@app.callback(Output('s111','value'),
              [Input('s111','options')])

def set_value_period(available_options): 
    
    return available_options[0]['value']

    
@app.callback(Output('s222', 'options'),
              [Input('generate_season','submit_n_clicks'),
               Input('my-dropdown','value'),
               Input('s111', 'value')])



def take_weekday_ss(submit_n_clicks, selected_dropdown_values, period_dropdown_values):
            
    filename = name_file
    selected_dropdown_values = dicts[str(selected_dropdown_values)]
    selected_dropdown_values = str(selected_dropdown_values)
    Mydep = selected_dropdown_values.split('_')[0]
    segment = selected_dropdown_values.split('_')[1]

    weekday = plt_ss_rev.list_weekday(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values)
    
    name_list_day = getKeysByValues(dicts_weekday,weekday.tolist())
    
    list_weekday = [{'label' : i, 'value' : i} for i in name_list_day]
   
    return list_weekday    

    
@app.callback(Output('s222','value'),
              [Input('s222','options'),
               Input('s111', 'value')])

def set_value_weekday(available_options,selected_dropdown_values):
#    if selected_dropdown_values != 'all':
#        return ''
#    else:
    return available_options[1]['value'] 

    
    
@app.callback(Output('my-graph_ss', 'figure'),
              [Input('generate_season','submit_n_clicks'),
               Input('my-dropdown','value'),
               Input('s111', 'value'),
               Input('s222', 'value')])


def convert_ss(submit_n_clicks, selected_dropdown_values, period_dropdown_values, weekday_dropdown_values):
            
    filename = name_file
    selected_dropdown_values = dicts[str(selected_dropdown_values)]
    weekday_dropdown_values = dicts_weekday[str(weekday_dropdown_values)]
                                     
                                     
    selected_dropdown_values = str(selected_dropdown_values)
    Mydep = selected_dropdown_values.split('_')[0]
    segment = selected_dropdown_values.split('_')[1]

    data = plt_ss_rev.plot_season(filename, UPLOAD_DIRECTORY, current_path, Mydep,segment, submit_n_clicks, period_dropdown_values,weekday_dropdown_values)
    
#    list_weekday = [{'label' : i, 'value' : i} for i in weekday]
   
    return data     
    
    


   
    
#@app.callback(Output('my-graph_ss','figure'),
#              [Input('generate_season','submit_n_clicks'),
#               Input('my-dropdown','value')])
#
#def convert_ss(submit_n_clicks, selected_dropdown_values):
#            
#    filename = name_file
#    selected_dropdown_values = dicts[str(selected_dropdown_values)]
#    selected_dropdown_values = str(selected_dropdown_values)
#    Mydep = selected_dropdown_values.split('_')[0]
#    segment = selected_dropdown_values.split('_')[1]
#
#    data = plt_ss_rev.plot_season(filename, UPLOAD_DIRECTORY, current_path, Mydep,segment, submit_n_clicks)
#
#   
#    return data

@app.callback(Output('generate_season','submit_n_clicks'),
              [Input('my-dropdown','value')])

def convert_ss_clicks(selected_dropdown_values):
    
    return 0

@app.callback(Output('generate_driver_rev','submit_n_clicks'),
              [Input('my-dropdown','value')])    
def convert_rev_clicks(selected_dropdown_values):
    
    return 0    
    
#@app.callback(Output('my-graph_rev','figure'),
#              [Input('generate_driver_rev','submit_n_clicks'),
#               Input('my-dropdown','value')])

@app.callback(Output('my-graph_rev', 'figure'),
              [Input('generate_driver_rev','submit_n_clicks'),
               Input('my-dropdown','value'),
               Input('s111', 'value'),
               Input('s222', 'value')])
def convert_rev(submit_n_clicks, selected_dropdown_values, period_dropdown_values, weekday_dropdown_values):
     
    selected_dropdown_values = dicts[str(selected_dropdown_values)]
    weekday_dropdown_values = dicts_weekday[str(weekday_dropdown_values)]

    filename = name_file
    Mydep = selected_dropdown_values.split('_')[0]
    segment = selected_dropdown_values.split('_')[1]
    
    data = plt_ss_rev.plot_revenue(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values, weekday_dropdown_values)
    
    return data


#@app.callback(Output('sMAPE_table','data'),
#              [Input('my-graph_rev','figure'),
#               Input('my-dropdown','value')])

@app.callback(Output('sMAPE_table','data'),
              [Input('generate_driver_rev','submit_n_clicks'),
               Input('my-dropdown','value'),
               Input('s111', 'value'),
               Input('s222', 'value')])
def smape_revenue(submit_n_clicks, selected_dropdown_values, period_dropdown_values, weekday_dropdown_values):
    weekday_dropdown_values = dicts_weekday[str(weekday_dropdown_values)]

    selected_dropdown_values = dicts[str(selected_dropdown_values)]
    
    filename = name_file
    Mydep = selected_dropdown_values.split('_')[0]
    segment = selected_dropdown_values.split('_')[1]
    
    data = plt_ss_rev.smape_revenue(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment, submit_n_clicks, period_dropdown_values, weekday_dropdown_values)
    
    return data

    
@app.callback(Output('output-provider_ss_finished','children'),
              [Input('generate_season','submit_n_clicks'),
               Input('my-dropdown','value')])
  


def generate_text_ss(submit_n_clicks,selected_dropdown_values):
    
    if submit_n_clicks == 0:
        return "Load historical resutls succesfully."
    else:
        return html.Div([dcc.Markdown("Generating Season {} times.".format(submit_n_clicks))], id='output-provider_ss')

@app.callback(Output('output-provider_ss','children'),
              [Input('my-graph_ss','figure'),
               Input('generate_season','submit_n_clicks')])
  
def generate_text_finished(figure,submit_n_clicks):
    if submit_n_clicks == 0:
        return ''
    else:      
        time.sleep(1)
        return 'generated Season succesfully ({} times).'.format(submit_n_clicks)
#def generate_text_ss(submit_n_clicks,selected_dropdown_values):
#    
#    if submit_n_clicks == 0:
#        return "Load historical resutls succesfully."
#    else:
#        return html.Div([dcc.Markdown("Generating Season {} times".format(submit_n_clicks))], id='output-provider_ss')        
        
        
@app.callback(Output('output-provider_rev','children'),
              [Input('generate_driver_rev','submit_n_clicks'),
               Input('my-dropdown','value')])        
        
def generate_text_rev(submit_n_clicks,selected_dropdown_values):
    
    if submit_n_clicks == 0:
        return "Load historical resutls succesfully."
    else:
        return "Generated Revenue Driver {} times".format(submit_n_clicks)        
        
#@app.callback(Output('generate_season','confirm'),
#               Input('my-dropdown','value')])


#def run_season(selected_dropdown_values):
#            
#    filename = name_file
#    selected_dropdown_values = dicts[str(selected_dropdown_values)]
#    selected_dropdown_values = str(selected_dropdown_values)
#    Mydep = selected_dropdown_values.split('_')[0]
#    segment = selected_dropdown_values.split('_')[1]
#
#    data = plt_ss_rev.plot_season(filename, UPLOAD_DIRECTORY, current_path, Mydep,segment)
#
#   
#    return data
#
#@app.callback(Output('my-graph_rev','figure'),
#              [Input('generate_driver_rev','confirm'),
#               Input('my-dropdown','value')])
#
#
#def run_revenue(selected_dropdown_values):
#            
#    selected_dropdown_values = dicts[str(selected_dropdown_values)]
#    
#    filename = name_file
#    Mydep = selected_dropdown_values.split('_')[0]
#    segment = selected_dropdown_values.split('_')[1]
#
#    data = plt_ss_rev.run_revenue(filename, UPLOAD_DIRECTORY, current_path, Mydep, segment)
#
#   
#    return data
    
    
if __name__ == '__main__':
#    app.run_server(debug=True, host = '127.0.0.1', port = 5000)
    app.run_server( host = '127.0.0.1', port = 4000)