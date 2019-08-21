import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from d2o_common.util import d2o_util as util
from d2o_common.hospitality.DepartmentSeason import DepartmentSeason as Season
from d2o_common.hospitality.NameManage import NameManage
from d2o_common.hospitality import column_names as col_name
from d2o_common.data_provider import DataProvider
from d2o_common.util import logger as log
import numpy as np

ACCURACY_LEADTIME = 15
LABOR_SEASON_ALIAS = 'labor'
AUTO_SEASON_ALIAS = 'normal'

DEFAULT_TYPE_NAME_MAP = [('rv', 'Revenue'), ('rn', 'Units'), ('gn', 'Guests'),
                         ('food', 'FoodRevenue')]

def pivot_raw_data(raw_df, type_name_map=DEFAULT_TYPE_NAME_MAP):
  """
  Pivot raw data from API to structed dataframe
  Args:
    raw_df (pd.DataFrame):
    | Date | H_Id | Segment_Id | Revenue | Units | Guests
    type_name_map list[tuple]: Define the mapping from Column values
    in raw dataframe to alias prefix.
      Example: ('rv', 'Revenue') mean 'Revenue' column have prefix 'rv'

  Returns:
    pd.DataFrame: pivoted data
    | Date | dep1_seg1_rv |...|dep1_seg2_rn|...|dep2_seg3_gn|
    (number of columns depend on number of department and segment of the hotel).
  """
  enriched_df = raw_df.set_index('Date')
  create_col_name = lambda col_type, df: (
    df['H_Id'].astype('str') + "_" + df['Segment_Id'].astype('str')
    + "_%s" % (col_type))
  list_pivot_df = []
  for type_data, value in type_name_map:
    enriched_df[type_data] = create_col_name(type_data, enriched_df)
    type_data_df = enriched_df[[type_data, value]]
    type_data_df_pivot = type_data_df.pivot(columns=type_data, values=value)
    list_pivot_df.append(type_data_df_pivot)
  pivoted_df = pd.concat(list_pivot_df, axis=1)
  return pivoted_df

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
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

def convert_new_revenue(json_data):
    new_var = pd.DataFrame(json_data['Sources'])
    new_var['Date'] = [json_data['Dates']] * new_var.shape[0]
    new_var['LaborSourceType'] = [new_var['Type'][i]['LaborSourceType'] for i in new_var.index]
    new_var['Name'] = [new_var['Type'][i]['Name'] for i in new_var.index]
    new_var['RevenueSourceType'] = [new_var['Type'][i]['RevenueSourceType'] for i in new_var.index]

    new_var2 = explode(new_var, ['Values', 'Date'])

    new_var_pv = new_var2.pivot_table(
        index=['Date', 'H_Id', 'Segment_Id'],
        columns=['Name'], values='Values').reset_index().fillna(0)

    full_col = ['Date', 'FoodRevenue', 'Guests', 'H_Id', 'Revenue', 'Segment_Id', 'Units', 'TotalFoodRevenue', 'TotalRevenue']
    remain_col = list(set(full_col) - set(new_var_pv.columns))
    if len(remain_col) == 0:
      new_var_pv = new_var_pv[full_col]
    else:
      for i in remain_col:
        new_var_pv[i] = 0
      new_var_pv = new_var_pv[full_col]
    return new_var_pv

class HotelData(object):
  """
  Data at hotel level. Contain all functions work with hotel data
  """
  def __init__(self, client_id, hotel_id, year_back, data_provider):
    """
    Initialize hotel data for hotel_id of client_id litmit time with year back.
    Note that only general data was prepared. Some specific data just downloand in need.
    Args:
      client_id (str): id of client
      hotel_id (int): id hot hotel
      year_back (int): number of years to go back to get data
    """
    # Basic data for hotel schema
    self.client_id = client_id
    self.hotel_id = hotel_id
    self.from_date, self.to_date = util.get_range_date_bounds(year_back)
    self.db = data_provider
    # self.all_clients = self.get_all_clients()
    # self.all_hotels = self.get_all_hotels()
    self.client_name = ""
    self.hotel_name = ""
    self.department_info = self.get_department_info()
    self.name_manager = NameManage(self.hotel_id, self.hotel_name,
                                   self.department_info)
    self.hotel_data = None
    self.hotel_total_data = None
    self.department_labor_data = {}
    self.department_accuracy_data = None
    self.department_season_data = {}


  def get_all_clients(self):
    json_data = self.db.get_all_clients()
    return pd.DataFrame(json_data)

  def get_all_hotels(self):
    json_data = self.db.get_hotel_ids_by_client_id(self.client_id)
    return pd.DataFrame(json_data)

  def get_department_info(self):
    """
    Get imformation about all department in a hotel
    Returns:
      pd.DataFrame: dataframe contain department infomation.
      None: if cannot get department info data

    """
    try:
      json_data = self.db.get_dept_ids_and_configuration(self.client_id, self.hotel_id)
      dep_info = pd.DataFrame(json_data)
    except Exception, e:
      debug_mess = '%s:%s %s:%s %s' % (col_name.CLIENT, self.client_id, col_name.H_ID, self.hotel_id, e.message)
      log.debug(debug_mess)
      dep_info = None

    return dep_info

  def set_hotel_data(self):
    """
    If hotel_data set up hotel data as dataframe that index is date, columns is values by
    department, segment, type of values
     Ex:     | Date | dep1_seg1_rv |...|dep1_seg2_rn|...|dep2_seg3_gn|
    """
    if self.hotel_data is None:
      data_in_json = self.db.get_hotel_data_in_a_specific_time_period(self.client_id, self.hotel_id,
                                                                         self.from_date, self.to_date)

      if len(data_in_json) > 0:
#        raw_df = pd.DataFrame(data_in_json)
        raw_df = convert_new_revenue(data_in_json)
        pivoted_df = pivot_raw_data(raw_df)
        pivoted_df.index = pd.to_datetime(pivoted_df.index)
        pivoted_df = pivoted_df.astype(float)
        pivoted_df[pivoted_df < 0] = 0
        self.hotel_data = pivoted_df
      else:
        debug_mess = '%s:%s %s:%s %s' % (col_name.CLIENT, self.client_id,
                                           col_name.H_ID, self.hotel_id, 'Data_empty')
        log.debug(debug_mess)

  def set_hotel_total_data(self):
    """
    if hotel total data (total is a special kind of department data) set up hotel_total_data as
    dataframe follows structure bellow:
    Date | DepId | Guest | Unit | Revenue | FoodRevenue | TotalFoodRevenue
    """
    if self.hotel_total_data is None:
      data_in_json = self.db.get_hotel_food_revenue_in_a_specific_time_period(self.client_id,
                                                                                       self.hotel_id,
                                                                                       self.from_date,
                                                                                       self.to_date)
      if len(data_in_json) > 0:
        total_rev_food_df = pd.DataFrame(data_in_json)
        total_rev_food_df['Date'] = pd.to_datetime(total_rev_food_df['Date'])
        total_rev_food_df.set_index('Date', inplace=True)
        total_rev_food_df[total_rev_food_df < 0] = 0
        self.hotel_total_data = total_rev_food_df
      else:
        warning_mess = '%s:%s %s:%s %s' % (col_name.CLIENT, self.client_id,
                                           col_name.H_ID, self.hotel_id, 'Total_data_empty:1')
        log.warning(warning_mess)



  def __get_hotel_name(self):
    return self.all_hotels[self.all_hotels['Id'] ==
                           self.hotel_id]['Name'].iloc[0]

  def __get_client_name(self):
    return self.all_clients[self.all_clients['Id']
                            == self.client_id]['Name'].iloc[0]

  #mm 09182018: change new API season to old
  def transform_labor_season_v2_to_v1(self, dep_id):
    season_v1 = self.db.get_labor_season_auto_by_client_and_department_id(self.client_id, dep_id)
    ss_v2_link1 = '%s/PMIAuto/Season/Days/%s/?h_id=%s&type=1&from=%s&to=%s' % (self.db._host, self.client_id, \
                                                                                  dep_id, self.from_date, self.to_date)
    
    response1 = requests.get(ss_v2_link1)
    season_df = pd.DataFrame(json.loads(response1.text))
    season_df = season_df.drop_duplicates('Period_Id')
    #Remove off-working dates
    season_df = season_df.loc[~((season_df['Day'] == 0) & (season_df['Period_Type'] != 1))]
    season_df = season_df[season_df['Period_Type'] != 1].reset_index(drop = True)
    #Remove outliers
#    season_df = season_df.loc[season_df['Period_Type'] != 2].reset_index(drop=True)
    
    ss_v2_link2 = '%s/PMIAuto/Season/%s/?h_id=%s&type=1' % (self.db._host, self.client_id, dep_id)
    response2 = requests.get(ss_v2_link2)
    season_v2 = pd.DataFrame(json.loads(response2.text))
    for i in range(len(season_v2)):
        if (season_v2['Periods'][i]['Type'] == 0) & (season_v2['Periods'][i]['Value1'] == 0) & (season_v2['Periods'][i]['Value2'] == 0):
            season_v2.drop(i, inplace = True)
            season_v2 = season_v2.reset_index(drop = True)
    
    
    for i in range(len(season_v2)):
        if (season_v2['Periods'][i]['Type'] == 1) & (season_v2['Periods'][i]['Value1'] == 0) & (season_v2['Periods'][i]['Value2'] == 0):
            season_v2.drop(i, inplace = True)
    season_v2 = season_v2.reset_index(drop = True)
    
    for i in range(len(season_v2)):
        if (season_v2['Periods'][i]['Type'] == 2) & (season_v2['Periods'][i]['Value1'] == 0) & (season_v2['Periods'][i]['Value2'] == 0):
            season_v2.drop(i, inplace = True)
    season_v2 = season_v2.reset_index(drop = True)
    
    
    if len(season_df[season_df['Period_Type'] == 2]) == 0:
       for i in range(len(season_v2)):
           if season_v2['Periods'][i]['Type'] == 2:
               season_v2.drop(i, inplace = True)
              
    season_v2 = season_v2.reset_index(drop = True)
          

               
    
    
    for idx, Periods_key in enumerate(season_v2['Periods']):
      color = 0
      name = 0
      if (Periods_key['Type'] == 0):
        Periods_key['Type'] = int(1)
        for i in season_df[season_df['Period_Type'] == 0]['Date']:
          temp_i = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')
          cond_from = [datetime.strptime(Periods_key['Dates'][x]['From'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
          cond_to = [datetime.strptime(Periods_key['Dates'][x]['To'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
          
          if any([cond_from[cond_idx] <= temp_i <= cond_to[cond_idx] for cond_idx in range(len(cond_from))]):
            Periods_key['Id'] = int(season_df[season_df['Date'] == i]['Period_Id'].values[0])
            Periods_key['Color'] = '#' + str(color)
            Periods_key['Name'] = str(name)
            color += 1
            name += 1
            for Dates_key in Periods_key['Dates']:
                Dates_key['Id'] = Periods_key['Id']
        print('normal', Periods_key['Id'])   
      elif (Periods_key['Type'] == 1):
        Periods_key['Type'] = int(1)
        for i in season_df[season_df['Period_Type'] == 1]['Date']:
          temp_i = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')
          cond_from = [datetime.strptime(Periods_key['Dates'][x]['From'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
          cond_to = [datetime.strptime(Periods_key['Dates'][x]['To'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
          
          if any([cond_from[cond_idx] <= temp_i <= cond_to[cond_idx] for cond_idx in range(len(cond_from))]):
            Periods_key['Id'] = int(season_df[season_df['Date'] == i]['Period_Id'].values[0])
            Periods_key['Color'] = '#' + str(color)
            Periods_key['Name'] = str(name)
            color += 1
            name += 1
            for Dates_key in Periods_key['Dates']:
                Dates_key['Id'] = Periods_key['Id']
        print('special_period', Periods_key['Id'])
      else:
        try:  
            Periods_key['Type'] = int(3)
            for i in season_df[season_df['Period_Type'] == 2]['Date']:
              temp_i = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')
              cond_from = [datetime.strptime(Periods_key['Dates'][x]['From'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
              cond_to = [datetime.strptime(Periods_key['Dates'][x]['To'], '%Y-%m-%dT%H:%M:%S') for x in range(len(Periods_key['Dates']))]
              
              if any([cond_from[cond_idx] <= temp_i <= cond_to[cond_idx] for cond_idx in range(len(cond_from))]):
                Periods_key['Id'] = int(season_df[season_df['Date'] == i]['Period_Id'].values[0])
                Periods_key['Color'] = '#000000'
                Periods_key['Name'] = 'Outlier (Auto)'
                Periods_key['Dates'] = []
    
            print('outlier_period', Periods_key['Id'])
        except:
            print('No ouliter in data')
#        season_v2 = season_v2.drop(season_v2.index[idx])

      map(lambda x: Periods_key.pop(x), ['Value1', 'Value2', 'Value3', 'SpecialPeriod'])    
    season_v1['Periods'] = season_v2['Periods'].tolist()
    
    return season_v1

  def set_dep_season(self, dep_id, type_season):
    """
    Set season data for department according type of season
    Args:
      dep_id (int): id of department
      type_season (str): type of season LABOR_SEASON_ALIAS = 'labor' or
      AUTO_SEASON_ALIAS = 'normal'

    """
    if dep_id not in self.department_season_data:
      if type_season == LABOR_SEASON_ALIAS:
        #mm 09182018: change new API season to old
#        json_data = self.db.get_labor_season_auto_by_client_and_department_id(self.client_id, dep_id)
        json_data = self.transform_labor_season_v2_to_v1(dep_id)
        import json
        with open('json_data.json','w') as fp:
            json.dump(json_data,fp)
        dep_season = Season(json_data)
        print('dep_season',dep_season)
        
        if dep_season.has_season:
          self.department_season_data[dep_id] = dep_season
          print(dep_season)
        else:
          json_data = self.db.get_labor_season_by_client_and_department_id(self.client_id, dep_id)
          dep_season = Season(json_data)
          dep_season = dep_season if dep_season.has_season else None
          self.department_season_data[dep_id] = dep_season
#        print(json_data)
      else:
        raise NotImplemented

  def check_labor_season_auto(self, dep_id, type_season):
    """
    Set season data for department according type of season
    Args:
      dep_id (int): id of department
      type_season (str): type of season LABOR_SEASON_ALIAS = 'labor' or
      AUTO_SEASON_ALIAS = 'normal'

    """
    if dep_id not in self.department_season_data:
      if type_season == LABOR_SEASON_ALIAS:
        #mm 09182018: change new API season to old
#        json_data = self.db.get_labor_season_auto_by_client_and_department_id(self.client_id, dep_id)
        json_data = self.transform_labor_season_v2_to_v1(dep_id)
        dep_season = Season(json_data)
        return dep_season.has_season
      else:
        raise NotImplemented
        
  def set_labor_data(self, dep_id):
    """
    Set up labor data for one department
    Args:
      dep_id (int): id of department
    """
    if dep_id not in self.department_labor_data:
      try:
        raw_json = self.db.get_department_labor_in_a_specific_time_period(self.client_id, dep_id,
                                                                              self.from_date, self.to_date)
        if raw_json == []:
          raise
        dep_labor_df = pd.DataFrame(raw_json)
        dep_labor_df['Date'] = pd.to_datetime(dep_labor_df['Date'])
        dep_labor_df.set_index('Date', inplace=True)
        dep_labor_df[dep_labor_df < 0] = 0
      except:
        print('Fail to get labor department %s' % (dep_id))
        return
      self.department_labor_data[dep_id] = dep_labor_df

  def set_accuracy_data(self):
    """
    Set up accuracy data for one hotel
    """
    if self.department_accuracy_data is None:
      dep_have_revenue_data = self.get_list_dept(has_revenue=True, no_include_hotel_id=False)
      l_df = []
      for dep_id in dep_have_revenue_data:
        try:
          dep_acc_data = self.db.get_department_forecast_accuracy_data(self.client_id, dep_id,
                                                      ACCURACY_LEADTIME)
        except:
          log.warning('%s:%s %s:%s Cannot_get_accuracy_data:1' % (col_name.CLIENT, self.client_id,
                                                                col_name.DEPT_ID, dep_id))
          continue

        if not dep_acc_data:
          log.warning('%s:%s %s:%s Cannot_get_accuracy_data:1' % (col_name.CLIENT, self.client_id,
                                                                col_name.DEPT_ID, dep_id))
#        dep_acc_data = pd.DataFrame(dep_acc_data, index=[0])
#        dep_acc_data['H_Id'] = dep_id
        #mm
        dep_acc_data = pd.DataFrame(dep_acc_data)
        dep_acc_data = dep_acc_data[dep_acc_data['H_Id'] == dep_id]
        
        l_df.append(dep_acc_data)

      self.department_accuracy_data = pd.concat(l_df, ignore_index=True)


  def transform_to_season_df(self, df, dep_id, type_season):
    self.set_dep_season(dep_id, type_season)  # make sure have season data for dep_id
    if self.department_season_data[dep_id] is None:
      return None
    return self.department_season_data[dep_id].transform_data_to_season_df(df)

  def get_accuracy_from_alias(self, alias):
    """
    Get department accuracy data base on alias
    Args:
      alias(str): alias of columns like 3_32_0_rv_lag1
    Returns:
      float : accuracy of department
    """
    self.set_accuracy_data()
    if self.department_accuracy_data is None:
      return None

    dep_id, seg_id, type_data, _ = self.name_manager.parse_id_from_alias(alias)
    # Hard code here because we don't have accuracy for food
    # if type_data == 'FoodRevenue':
    #   type_data = 'Revenue'
    try:
#      acc = self.department_accuracy_data[self.department_accuracy_data['H_Id']
#                                          == dep_id][type_data].iloc[0]
      #mm
      #Using new API accuracy link
      acc = self.department_accuracy_data[(self.department_accuracy_data['H_Id'] == dep_id) & \
                                          (self.department_accuracy_data['Segment_Id'] == seg_id)]['Accuracy'].iloc[0]
    except:
      acc = 1

    return acc

  def to_dataframe(self, use_total=False):
    """
    Return hotel data in dataframe form.
     Ex:     | Date | dep1_seg1_rv |...|dep1_seg2_rn|...|dep2_seg3_gn|
    Args:
      use_total (bool): If true return dataframe include total data.

    Returns:
      (pd.DataFrame): Data of hotel

    """
    self.set_hotel_data()
    if self.hotel_data is None:
      return None
    df = self.hotel_data.copy()
    if use_total:
      self.set_hotel_total_data()
      if self.hotel_total_data is not None:
        df['TotalRevenue'] = self.hotel_total_data['TotalRevenue']
        df['TotalFoodRevenue'] = self.hotel_total_data['TotalFoodRevenue']

    return df

  def get_list_dept(self, no_include_hotel_id=True, has_labor=None, has_revenue=None,
                    has_food_rev=None):
    """
    Get list deparment ids of hotel filterd by some condition
    Args:
      no_include_hotel_id (bool): True - result does not contain hotel id. False - otherwise
      has_labor (bool,None): None - ignore. True - result contain depts has labor data
      has_revenue (bool,None): None - ignore. True - result contain depts has revenue data
      has_food_rev (bool,None): None - ignore. True - result contain dept has food revenue data

    Returns:
      list[str]: list of department ids.
    """
    depts = set(self.department_info['H_Id'])
    if no_include_hotel_id:
      depts = depts - {self.hotel_id}
    if has_labor is not None:
      dep_labor = set(self.department_info[self.department_info['Labor']
                                           == has_labor]['H_Id'])
      depts = depts.intersection(dep_labor)
    if has_revenue is not None:
      dep_rev = set(self.department_info[self.department_info['Revenue']
                                         == has_revenue]['H_Id'])
      depts = depts.intersection(dep_rev)
    if has_food_rev is not None:
      dep_food_rev = set(self.department_info[self.department_info['FoodRevenue']
                                              == has_food_rev]['H_Id'])
      depts = depts.intersection(dep_food_rev)
    return list(depts)

  def get_dep_labor(self, dep_id):
    """
    Get labor data for department
    Args:
      dep_id (int): id of department

    Returns:
      pd.DataFrame
    """
    self.set_labor_data(dep_id)
    return self.department_labor_data.get(dep_id, None)

