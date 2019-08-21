import pandas as pd
from column_names import *


class NameManage(object):
  """
  Hotel Name contain function mapping between alias (use as column names in dataframe)
  to the full name of segment or department.
  """
  def __init__(self, hotel_id, hotel_name, dep_infos):
    self.hotel_id = hotel_id
    self.hotel_name = hotel_name
    self.dep_name_df, self.seg_name_df = self.extract_df_name_info(dep_infos)

  def extract_df_name_info(self, dep_info):
    """
    Extract department and segment name infomation from department infomations
    dataframe
    Args:
      dep_info(pd.DataFrame): Raw info of departments.
    Returns:
      pd.DataFrame, pd.DataFrame: Department name dataframe (index are dept_id)
       and Segment name dataframe (index are dept_id and seg_id)
    """
    if not isinstance(dep_info, pd.DataFrame):
      raise Exception('Invalid dep information. The type must be dataframe')
    dep_name_df = dep_info[['H_Id', 'Name']]
    dep_name_df = dep_name_df.set_index('H_Id')
    dep_name_df.index.name = 'Dep_Id'
    dep_have_seg = dep_info[dep_info['Segments'].apply(lambda x: x != [])]
    l_seg_df = []
    for idx in dep_have_seg.index:
      seg_df = pd.DataFrame(dep_have_seg.loc[idx, 'Segments'])
      seg_df = seg_df.rename(columns={"Id":"Seg_Id"})
      seg_df['Dep_Id'] = dep_have_seg.loc[idx,'H_Id']
      l_seg_df.append(seg_df)
    if len(l_seg_df) > 0:
      final_seg_name_df = pd.concat(l_seg_df, ignore_index=True)
      final_seg_name_df = final_seg_name_df.set_index(['Dep_Id', 'Seg_Id'])
    else:
      final_seg_name_df = pd.DataFrame([], columns=['Dep_Id', 'Seg_Id', 'Name'])
      final_seg_name_df = final_seg_name_df.set_index(['Dep_Id', 'Seg_Id'])

    return dep_name_df, final_seg_name_df

  def get_dep_name(self, dep_id):
    try:
      return self.dep_name_df.loc[dep_id]['Name']
    except:
      print('Dep id %s is not exist' % dep_id)
      return None

  def get_segment_name(self, dep_id, seg_id):
    try:
      return self.seg_name_df.loc[(dep_id, seg_id)]['Name']
    except:
      if dep_id == self.hotel_id and seg_id == 0:
        return 'hotel total (segment 0)'

      print('Department id %s or Segment id %s is not exist'
                       % (dep_id, seg_id))
      return None

  def get_name_from_alias(self, alias, type_name):
    """
    Get readable name from alias.
    Args:
      alias (str): alias . Ex: 19_0_42_lag1
      type_name (str): 'department' - get department name.
                        'full' - get full name

    Returns:
      str: name of alias.
    """
    if 'TotalRevenue' in alias:
      return alias
    elif 'TotalFoodRevenue' in alias:
      return alias

    if type_name == 'department':
      dep_id, _, _, _ = self.parse_id_from_alias(alias)
      return self.get_dep_name(dep_id)
    elif type_name == 'full':
      dep_id, seg_id, type_data, lag = self.parse_id_from_alias(alias)
      dep_name = self.get_dep_name(dep_id)
      seg_name = self.get_segment_name(dep_id, seg_id)
      if lag is None:
        name = '%s of %s %s' % (type_data, dep_name, seg_name)
        return name
      else:
        name = 'lag %s of %s of %s %s' % (lag, type_data, dep_name, seg_name)
        return name

  #TODO: need unit test
  def parse_id_from_alias(self, alias):
    """
    Parse alias to ids (department id, segment id, type data, lag day)
    Ex: '629_48_rv' -> 629, 48, Revenue, None
        '623_5_food_lag1' -> 623, 4, FoodRevenue, 1
    Args:
      alias (str): alias need to parse

    Returns:
      (int, int, str, int): department id, segment id, type of data, number of lag day.
      Some elements are None if it's information not in alias.
    """
    if 'TotalRevenue' in alias:
      if 'lag' in alias:
        return self.hotel_id, 0, 'Revenue', int(alias[-1])
      else:
        return self.hotel_id, 0, 'Revenue', None
    if 'TotalFoodRevenue' in alias:
      if 'lag' in alias:
        return self.hotel_id, 0, 'FoodRevenue', int(alias[-1])
      else:
        return self.hotel_id, 0, 'FoodRevenue', None
    type_data = None
    lag_day = None
    l_ele = alias.split('_')
    if len(l_ele) < 2:
      raise Exception('Invalid alias')
    dep_id = int(l_ele[0])
    seg_id = int(l_ele[1])
    if len(l_ele) > 2:
      type_data = ALIAS_TYPES[l_ele[2]]
    if len(l_ele) > 3:
      lag_day = int(alias[-1])
    return dep_id, seg_id, type_data, lag_day

  def get_alias(self, h_id, seg_id, source_type, lag=None):
    # Normalize values
    h_id, seg_id, source_type = int(h_id), int(seg_id), int(source_type)

    if source_type == 0:
      if (lag is None) or (lag == 0):
        return "TotalRevenue"
      else:
        return "TotalRevenue_lag%s" % (lag)
    elif source_type == 1:
      if (lag is None) or (lag == 0):
        return "TotalFoodRevenue"
      else:
        return "TotalFoodRevenue_lag%s" % (lag)

    if source_type == 2:
      str_source_type = 'rv'
    elif source_type == 3:
      str_source_type = 'food'
    elif source_type == 4:
      str_source_type = 'rn'
    elif source_type == 5:
      str_source_type = 'gn'

    if (lag is None) or (lag == 0):
      alias = "%s_%s_%s" % (h_id, seg_id, str_source_type)
    else:
      alias = "%s_%s_%s_lag%s" % (h_id, seg_id, str_source_type, lag)

    return alias

  def parse_alias_follow_d2o_format(self, alias):
    if 'TotalRevenue' in alias:
      dep_id, seg_id, type_id = self.hotel_id, 0, DATA_TYPE_ID['TotalRevenue']
      lag = int(alias[-1]) if ('lag' in alias) else 0
    elif 'TotalFoodRevenue' in alias:
      dep_id, seg_id, type_id = self.hotel_id, 0, DATA_TYPE_ID['TotalFoodRevenue']
      lag = int(alias[-1]) if ('lag' in alias) else 0
    else:
      dep_id, seg_id, type_data, lag_day = self.parse_id_from_alias(alias)
      type_id = DATA_TYPE_ID[type_data]
      lag = 0 if lag_day is None else lag_day
    return dep_id, seg_id, type_id, lag
