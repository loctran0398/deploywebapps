import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta, date

FORMAT_DATE = "%Y-%m-%d"
# HOST = "http://172.16.0.51:8485"

def get_ip_address():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
    if s.getsockname()[0].split('.')[3] == '23':
        print('Production server')
        HOST = "http://10.50.10.4:8485"
    else:
        print('Test server')
        HOST = "http://172.16.0.51:8485"
    s.close()
    return HOST

HOST = get_ip_address()
print(HOST)
CLIENT_LINK             = "%s/Client/Clients" % (HOST)
HOTEL_IN_DATABASE_LINK  = "%s/Hierarchy/Properties/{client_id}" % (HOST)
DEPT_IN_HOTEL_LINK      = "%s/Hierarchy/Departments/{client_id}/?h_id_parent={h_id_parent}" % (HOST)
REVENUE_LINK            = "%s/Revenue/Departments/Days/{client_id}/?h_id_parent={h_id_parent}&total=true&segment=true&from={from_date}&to={to_date}" % (HOST)
NEW_REVENUE_LINK        = "%s/LiveForecast/Drivers/{client_id}/?h_id_parent={h_id_parent}&from={from_date}&to={to_date}" % (HOST)
DEPT_DEF_PARAMETER_LINK = "%s/Season/Auto/Settings/{client_id}/?h_id={h_id}" % (HOST)
DEPT_SEASONS_DEF        = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=0" % (HOST)
DEPT_SEASON_LINK        = "%s/Season/Auto/{client_id}/?h_id={h_id}&type=0" % (HOST)
GET_SEASON_AUTO         = "%s/Forecast/RevenueDriver/Auto/{client_id}/?h_id={h_id}" % (HOST)
ONE_DEPT_IN_HOTEL       = "%s/Hierarchy/Department/{client_id}/?h_id={h_id}" % (HOST)
# CRUISE_DAY              = "%s/Hierarchy/CruiseDays/{client_id}/?h_id_parent={h_id}&from={from_date}&to={to_date}" % (HOST)
# CRUISE_DAY              = "%s/Season/Auto/V2/Days/{client_id}/?h_id={h_id}&type=0&from={from_date}&to={to_date}" % (HOST)
CRUISE_DAY              = "%s/PMIAuto/Season/Days/{client_id}/?h_id={h_id}&type=0&from={from_date}&to={to_date}" % (HOST)

REVENUE_LINK_ONE_DEPT   = "%s/Revenue/Department/Days/{client_id}/?h_id={h_id}&total=true&segment=true&from={from_date}&to={to_date}" % (HOST)


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


def request_data(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response)

def request_data_no_index(link):
    response = requests.get(link)
    json_response = json.loads(response.text)
    return pd.DataFrame(json_response, index=[0])

def get_all_client():
    return request_data(CLIENT_LINK)

def get_hotels_in_database(client_id):
    link = HOTEL_IN_DATABASE_LINK.format(client_id=client_id)
    return request_data(link)

def get_depts_in_hotel(client_id, hotel_id):
    link = DEPT_IN_HOTEL_LINK.format(client_id=client_id, h_id_parent=hotel_id)
    return request_data(link)

def get_one_dept_in_hotel(client_id, hotel_id):
    link = ONE_DEPT_IN_HOTEL.format(client_id=client_id, h_id=hotel_id)
    return json.loads(requests.get(link).text)

def pivot_raw_data(raw_df):
    d = raw_df.rename(columns={"Date": "date"})
    d = d.set_index('date')
    d['rv'] = d['H_Id'].astype('str') + "_" + d['Segment_Id'].astype('str') + "_rv"
    d['rn'] = d['H_Id'].astype('str') + "_" + d['Segment_Id'].astype('str') + "_rn"
    d['gn'] = d['H_Id'].astype('str') + "_" + d['Segment_Id'].astype('str') + "_gn"
    l_df = []
    for type_data, values in [('rv', 'Revenue'), ('rn', 'Units'), ('gn', 'Guests')]:
        df_temp = d[[type_data, values]]
        df_pivot = df_temp.pivot(columns=type_data, values=values)
        l_df.append(df_pivot)

    new_df = pd.concat(l_df, axis=1)
    return new_df

#=======================================================================================================================
#mm
def v2_get_data(to_date, client_id, hotel_id, year_back = 2):
#    to_date = datetime.now().date()
    to_date = to_date.date()
    year_add = - year_back
    from_date = add_years(to_date, year_add)
    link = NEW_REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id, from_date=str(from_date), to_date=str(to_date))
    print(link)
    response = requests.get(link)
    json_response = json.loads(response.text)
    
    df_new = pd.DataFrame(json_response['Sources'])
    date_df = pd.DataFrame(json_response['Dates'], columns = ['Date'])
    temp_name = df_new['Type']
    temp_name = [str(temp_name[i]['Name']) for i in temp_name.index]
    df_new['Name'] = temp_name
    df_new['dep_seg'] = ['_'.join([str(df_new['H_Id'][i]), str(df_new['Segment_Id'][i])]) for i in df_new.index]
    
    #transform to old dataframe
    df = pd.DataFrame(columns = list(set(temp_name))+['Date', 'H_Id', 'Segment_Id'])
    for dep_seg in set(df_new['dep_seg']):
        dep = int(dep_seg.split('_')[0])
        seg = int(dep_seg.split('_')[1])
        df_temp = pd.DataFrame(columns = list(set(temp_name))+['Date', 'H_Id', 'Segment_Id'])
        df_temp['Date'] = date_df
        df_temp['H_Id'] = dep
        df_temp['Segment_Id'] = seg
        for col in set(temp_name):
            temp = df_new[(df_new['Name'] == col) & (df_new['H_Id'] == dep) & (df_new['Segment_Id'] == seg)]['Values']   
            if len(temp) > 0:
                df_temp[col] = temp.values[0]
        df = pd.concat([df, df_temp])
    df['H_Id'] = df['H_Id'].astype('int', copy=False)
    df['Segment_Id'] = df['Segment_Id'].astype('int', copy=False)
    df = df.reset_index(drop=True)
    
#    df_new = pd.DataFrame()
#    date_df = pd.DataFrame(json_response['Dates'], columns = ['Date'])
#    
#    typeName = {'Units': 'rn', 'Guests': 'gn', 'Revenue': 'rv'}
#
#    for s in json_response['Sources']:
#        if s['Type']['Name'] in typeName:
#            sourceDF = pd.DataFrame.from_dict(s['Values'])
#            sourceDF.columns = ['%s_%s_%s' % (s['H_Id'], s['Segment_Id'], typeName[s['Type']['Name']])]
#            df_new = pd.concat([df_new, sourceDF], axis= 1)
#    df_new = pd.concat([df_new, date_df], axis= 1)
#    df = df_new.set_index('Date')
    
    if len(df) == 0:
        has_data_flag = False
        dep_no_data = []
    else:
        has_data_flag = True
        h_id_full_list = np.unique(df['H_Id'])
        dep_no_data = []
        for h_id in h_id_full_list:
            temp_df = df.ix[(df['H_Id']==h_id)]
            #mm
            if (not v2_has_data(temp_df, h_id)):
                dep_no_data.append(h_id)
        h_id_has_data = set(h_id_full_list) - set(dep_no_data)
        df = df.loc[df['H_Id'].isin(h_id_has_data)]
        df = pivot_raw_data(df)
        df = df.dropna(axis = 1, how = 'all')
    return df, dep_no_data, has_data_flag


#    response = requests.get(link)
#    json_response = json.loads(response.text)
#
#    re_df = pd.DataFrame.from_dict(json_response["Dates"])
#    re_df.columns = ['date']
#    re_df = re_df.set_index('date')
#    typeName = {'Units': 'rn', 'Guests': 'gn', 'Revenue': 'rv'}
#
#    for s in json_response['Sources']:
#        if s['Type']['Name'] in typeName:
#            sourceDF = pd.DataFrame.from_dict(s['Values'])
#            sourceDF.columns = ['%s_%s_%s' % (s['H_Id'], s['Segment_Id'], typeName[s['Type']['Name']])]
#            re_df = pd.concat([re_df, sourceDF], axis= 1)
#
#    if len(re_df) == 0:
#        has_data_flag = False
#        dep_no_data = []
#    else:
#        has_data_flag = True
#
#        dep_no_data = []
#    print("Length of re_df " + str(len(re_df.columns)))
#    return re_df, dep_no_data, has_data_flag

#def v2_get_data(to_date, client_id, hotel_id, year_back = 2):
##    to_date = datetime.now().date()
#    to_date = to_date.date()
#    year_add = - year_back
#    from_date = add_years(to_date, year_add)
#    link = REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id, from_date=str(from_date), to_date=str(to_date))
#    df = request_data(link)
#    if len(df) == 0:
#        has_data_flag = False
#        dep_no_data = []
#    else:
#        has_data_flag = True
#        h_id_full_list = np.unique(df['H_Id'])
#        dep_no_data = []
#        for h_id in h_id_full_list:
#            temp_df = df.ix[(df['H_Id']==h_id)]
#            #mm
#            if (not v2_has_data(temp_df, h_id)):
#                dep_no_data.append(h_id)
#        h_id_has_data = set(h_id_full_list) - set(dep_no_data)
#        df = df.loc[df['H_Id'].isin(h_id_has_data)]
#        df = pivot_raw_data(df)
#    return df, dep_no_data, has_data_flag

def v2_get_data_intime(from_date, to_date, client_id, hotel_id):
    to_date = to_date.date()
    from_date = from_date.date()
    link = REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id, from_date=str(from_date), to_date=str(to_date))
    df = request_data(link)
    if len(df) == 0:
        has_data_flag = False
        dep_no_data = []
    else:
        has_data_flag = True
        h_id_full_list = np.unique(df['H_Id'])
        dep_no_data = []
        for h_id in h_id_full_list:
            temp_df = df.ix[(df['H_Id']==h_id)]
            if (not has_data(temp_df)):
                dep_no_data.append(h_id)
        h_id_has_data = set(h_id_full_list) - set(dep_no_data)
        df = df.loc[df['H_Id'].isin(h_id_has_data)]
        df = pivot_raw_data(df)
    return df, dep_no_data, has_data_flag

def v2_has_data(df, dep_id):
    """
    Check whether df has enough data or not
    :param df: dataframe
    :return: True or False
    """
    REQ_LEN = 14
    REQ_PERCENT_DATA = 80

    has_year = len(df) >= REQ_LEN
    if (not has_year):
        print('Data length of DEP ID {}: {} . Not enough data for calculation.'.format(dep_id, len(df)))
        return False

    stats = statistics(df)
    if (stats['Revenue']['percent_nan'] > REQ_PERCENT_DATA or stats['Revenue']['percent_zero'] > REQ_PERCENT_DATA):
        return False
    return True
#=======================================================================================================================

def get_data(client_id, hotel_id, year_back=2):
    to_date = datetime.now().date()
    year_add = - year_back
    from_date = add_years(to_date, year_add)
    link = NEW_REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id, from_date=str(from_date), to_date=str(to_date))
    response = requests.get(link)
    json_response = json.loads(response.text)
    
    df_new = pd.DataFrame(json_response['Sources'])
    date_df = pd.DataFrame(json_response['Dates'], columns = ['Date'])
    temp_name = df_new['Type']
    temp_name = [str(temp_name[i]['Name']) for i in temp_name.index]
    df_new['Name'] = temp_name
    df_new['dep_seg'] = ['_'.join([str(df_new['H_Id'][i]), str(df_new['Segment_Id'][i])]) for i in df_new.index]
    
    #transform to old dataframe
    df = pd.DataFrame(columns = list(set(temp_name))+['Date', 'H_Id', 'Segment_Id'])
    for dep_seg in set(df_new['dep_seg']):
        dep = int(dep_seg.split('_')[0])
        seg = int(dep_seg.split('_')[1])
        df_temp = pd.DataFrame(columns = list(set(temp_name))+['Date', 'H_Id', 'Segment_Id'])
        df_temp['Date'] = date_df
        df_temp['H_Id'] = dep
        df_temp['Segment_Id'] = seg
        for col in set(temp_name):
            temp = df_new[(df_new['Name'] == col) & (df_new['H_Id'] == dep) & (df_new['Segment_Id'] == seg)]['Values']   
            if len(temp) > 0:
                df_temp[col] = temp.values[0]
        df = pd.concat([df, df_temp])
    df['H_Id'] = df['H_Id'].astype('int', copy=False)
    df['Segment_Id'] = df['Segment_Id'].astype('int', copy=False)
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        has_data_flag = False
        dep_no_data = []
    else:
        has_data_flag = True
        h_id_full_list = np.unique(df['H_Id'])
        dep_no_data = []
        for h_id in h_id_full_list:
            temp_df = df.ix[(df['H_Id']==h_id)]
            #mm
            if (not has_data(temp_df)):
                dep_no_data.append(h_id)
        h_id_has_data = set(h_id_full_list) - set(dep_no_data)
        df = df.loc[df['H_Id'].isin(h_id_has_data)]
        df = pivot_raw_data(df)
        df = df.dropna(axis = 1, how = 'all')
    return df, dep_no_data, has_data_flag

#def get_data(client_id, hotel_id, year_back = 2):
#    to_date = datetime.now().date()
#    year_add = - year_back
#    from_date = add_years(to_date, year_add)
#    link = REVENUE_LINK.format(client_id=client_id, h_id_parent=hotel_id, from_date=str(from_date), to_date=str(to_date))
#    df = request_data(link)
#    if len(df) == 0:
#        has_data_flag = False
#        dep_no_data = []
#    else:
#        has_data_flag = True
#        h_id_full_list = np.unique(df['H_Id'])
#        dep_no_data = []
#        for h_id in h_id_full_list:
#            temp_df = df.ix[(df['H_Id']==h_id)]
#            if (not has_data(temp_df)):
#                dep_no_data.append(h_id)
#        h_id_has_data = set(h_id_full_list) - set(dep_no_data)
#        df = df.loc[df['H_Id'].isin(h_id_has_data)]
#        df = pivot_raw_data(df)
#    return df, dep_no_data, has_data_flag

def get_data_for_dep_season(client_id, hotel_id, year_back = 2):
    to_date = datetime.now().date()
    year_add = - year_back
    from_date = add_years(to_date, year_add)
    link = REVENUE_LINK_ONE_DEPT.format(client_id=client_id, h_id=hotel_id, from_date=str(from_date), to_date=str(to_date))
    df = request_data(link)
    if len(df) == 0:
        has_data_flag = False
    else:
        has_data_flag = has_data(df)
        df = pivot_raw_data(df)
    return df, has_data_flag


def post_api(link, json_data):
    response = requests.post(url=link, json=json_data)
    return response

def get_dept_def_parameter(client_id, dept_id):
    link = DEPT_DEF_PARAMETER_LINK.format(client_id=client_id, h_id=dept_id)
    return request_data_no_index(link)

def unpack(df, column, fillna=None):
    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
        del ret[column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
        del ret[column]
    return ret

def statistics(df):
  """
  Counts and computes some basic statistics of the dataframe.

  Parameters
  ----------
  df : pd.DataFrame

  Returns
  -------
  stat_df : pd.DataFrame
    Dataframe containing the statistics.

  Notes
  -----
  The rows in stat_df is {'num_nan', 'num_zero', 'percent_nan',
  'percent_zero', 'median','mean', 'standard_deviation', 'variance'}
  """
  statdict = {
    'num_nan':{},
    'num_zero':{},
    'percent_nan':{},
    'percent_zero':{},
    'median':{},
    'mean':{},
    'standard_deviation':{},
    'variance':{}
  }

  length   = len(df)
  num_nan  = df.isnull().sum().to_dict()
  num_zero = (df == 0).astype(float).sum(axis=0).to_dict()
  medians  = df.median().to_dict()
  means     = df.mean().to_dict()
  stds     = df.std().to_dict()
  _vars    = df.var().to_dict()

  statdict['num_nan']            = num_nan
  statdict['num_zero']           = num_zero
  statdict['percent_nan']        = {k:(v/float(length))*100.0 for k,v in num_nan.iteritems()}
  statdict['percent_zero']       = {k:(v/float(length))*100.0 for k,v in num_zero.iteritems()}
  statdict['median']             = medians
  statdict['mean']               = means
  statdict['standard_deviation'] = stds
  statdict['variance']           = _vars
  return pd.DataFrame.from_dict(statdict).T

def has_data(df):
    """
    Check whether df has enough data or not
    :param df: dataframe
    :return: True or False
    """
    REQ_LEN = 365
    REQ_PERCENT_DATA = 80

    has_year = len(df) >= REQ_LEN
    if (not has_year):
      return False

    stats = statistics(df)
    if (stats['Revenue']['percent_nan'] > REQ_PERCENT_DATA or stats['Revenue']['percent_zero'] > REQ_PERCENT_DATA):
      return False
    return True

def wrapper_request_data(func):
    def wrapper_func(*args):
        link = func(*args)
        return request_data(link)
    return wrapper_func

@wrapper_request_data
def get_all_client():
    return CLIENT_LINK

def get_season_data(client_id, dept_id):
    df = request_data(DEPT_SEASON_LINK.format(client_id=client_id, h_id=dept_id))
    unpack_df = unpack(df, 'Periods', 0)
    unpack_df.columns.values[0] = 'Idx'
#    df_new = unpack_df.groupby('Color').Dates.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis=1)
    df_temp = unpack_df.groupby('Color').Dates
    df_new = pd.DataFrame(columns = ['From', 'Id', 'To'])
    for name, group in df_temp:
        if name != '#000000':
            for i in group.index:
                df_new = pd.concat([df_new, pd.DataFrame(group[i])])
                
    df_new['From'] = pd.to_datetime(df_new['From'], format='%Y-%m-%dT%H:%M:%S')
    df_new['To'] = pd.to_datetime(df_new['To'], format='%Y-%m-%dT%H:%M:%S')
    df_new['Id'] = [int(x) for x in df_new['Id']]
    #mm
#    df_new['period_length'] = df_new['To'] - df_new['From']
#    df_new['period_length'] = df_new['period_length'].dt.days
#    df_new = df_new.ix[df_new['period_length'] >= 7, :-1]

#    df_new = df_new.rename(columns={'Id': 'period_id'})
    del unpack_df['Dates']
    df_merge = pd.merge(unpack_df, df_new, on='Id', how='left')
    df_merge = df_merge.rename(columns={'Id': 'period_id'})
    return df_merge


def get_season_data_v2(client, dep_id, years_back):
    to_date = datetime.now() - timedelta(days=2)
    year_add = - years_back
    from_date = add_years(to_date, year_add)
    from_date = datetime(year=from_date.year, month=from_date.month, day=from_date.day)
    to_date = datetime(year=to_date.year, month=to_date.month, day=to_date.day)

    link = CRUISE_DAY.format(client_id=client, h_id=dep_id, from_date=str(from_date), to_date=str(to_date))

    data = request_data(link)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = [datetime.strptime(datetime(year=i.year, month=i.month, day=i.day).strftime('%Y-%m-%d'), '%Y-%m-%d') for i in data['Date']]
    # if (max(data['Day'][data['Period_Type'] == 0]) == 7) and (0 not in data['Day'][data['Period_Type'] == 0].values):
    #     cruise_flag = False
    # else:
    #     cruise_flag = True
    data['Day'][data['Period_Type'] == 1] = 70
    new_df = data.copy()
    new_df = new_df[['Date', 'Day', 'Period_Id', 'Period_Type']]
    return new_df


def get_season_auto(client_id, hotel_id):
    link = GET_SEASON_AUTO.format(client_id=client_id,h_id=hotel_id)
    df = request_data(link)
    if df.empty:
        df = pd.DataFrame(columns=["H_Id", "Period", "Day", "Source_H_Id", "Source_Segment_Id", "Destination_Segment_Id",
                                "Priority", "Property", "Type", "SourceOffset", "DestinationOffset"])
    return df

def segments(client_id, hotel_id):
    dept_df = get_depts_in_hotel(client_id, hotel_id)
    dept_df = dept_df[dept_df['Revenue'] == True]
    l_df = []
    for i in dept_df.index:
        segs = dept_df.loc[i, 'Segments']
        segs.append({u'Id': 0, u'Name': u'Total'})
        df_t = pd.DataFrame(segs)
        df_t['hotel_id'] = str(hotel_id)
        df_t['name'] = dept_df.loc[i, 'Name'] + "_" + df_t['Name']
        df_t['id']   = str(dept_df.loc[i, 'H_Id']) + "_" + df_t['Id'].astype(str)
        l_df.append(df_t[['hotel_id', 'name', 'id']])
    return pd.concat(l_df, ignore_index=True)

class Cruise_day:
    def __init__(self, to_date, client_id, hotel_id, year_back = 2):
        self.get_cruise_day(to_date, client_id, hotel_id, year_back)

    def get_cruise_day(self, to_date, client_id, hotel_id, year_back = 2):
#        to_date = datetime.now().date()
        year_add = - year_back
        from_date = add_years(to_date, year_add)
        link = CRUISE_DAY.format(client_id=client_id, h_id=hotel_id, from_date=str(from_date), to_date=str(to_date))
        self.__cruise_day = request_data(link)
        # if (len(self.__cruise_day) == 0):
        #     self.__cruise_day['Date'] = [from_date + timedelta(days=x) for x in range((to_date - from_date).days + 1)]
        #     self.__cruise_day['Day'] = [np.round((i.weekday() + 8) % 7 + 1).astype(int) for i in self.__cruise_day['Date']]


        # if (max(self.__cruise_day['Day'][self.__cruise_day['Period_Type'] == 0]) == 7) and (0 not in self.__cruise_day['Day'][self.__cruise_day['Period_Type'] == 0].values):
        #     self.__cruise_day = self.__cruise_day[(__cruise_day['Period_Type'] == 0) and (__cruise_day['Day'] == 0)]

        # self.__cruise_day['Date'] = pd.to_datetime(self.__cruise_day['Date'])
        # self.__cruise_day = self.__cruise_day[['Date','Day']]
        # self.__cruise_day = self.__cruise_day.set_index('Date')

        self.__cruise_day_copy = self.__cruise_day.copy()


            # self.__cruise_day = self.__cruise_day.drop(self.__cruise_day[(self.__cruise_day['Period_Type'] == 0) & (self.__cruise_day['Day'] == 0)]).reset_index(drop=True)
        self.__cruise_day['Day'][self.__cruise_day['Period_Type'] == 1] = 70
        self.__cruise_day['Date'] = pd.to_datetime(self.__cruise_day['Date'])
        self.__cruise_day = self.__cruise_day[['Date','Day']]
        self.__cruise_day = self.__cruise_day.set_index('Date')

        if len(self.__cruise_day_copy[(self.__cruise_day_copy['Period_Type'] == 0) & (self.__cruise_day_copy['Day'] == 0)]) != 0:
            self.__cruise_day = self.__cruise_day.replace(0, np.nan)
        self.num_type_cruise_day = len(self.__cruise_day['Day'].unique())

    def cruiseday(self, date):
        date_l = pd.to_datetime([date])
        return self.__cruise_day.ix[date_l, 'Day'][0]

    def cruiseday_list(self, dates):
        return self.__cruise_day.ix[dates, 'Day']

    def get_data(self):
        return self.__cruise_day
    def get_data_original(self):
        return self.__cruise_day_copy

    def cruise_flag(self):
        self.__cruise_day

if __name__ == "__main__":
    print(CLIENT_LINK)
