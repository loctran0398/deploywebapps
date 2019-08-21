"""
Copyright: TP7
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import time
import os
import pickle

import create_driver_df
import capture_arr
import project_lib as lib


def create_filled_list(df, hotel_id):
    '''
    create a filled list for a hotel
    :param df: a dataframe that has rn, gn, rv data
    :param hotel_id: hotel_id in string type
    :return: a list of filled data column names
    '''
    filled_list = []
    for i in df.columns:
        if hotel_id in i:
            filled_list.append(i)
    return filled_list


def add_total_to_df(df):
    '''
    add total columns for dataframe
    :param df: the dataframe need to add total columns
    :return: a data frame that have total columns for all departments
    '''
    new_df = df[df.columns]
    dep_list = set(i.split("_")[0] for i in new_df.columns) - {"date"}
    for dep in dep_list:
        other_seg = []
        if (dep + "_0") not in new_df.columns:
            other_seg = [i for i in new_df.columns if dep in i]
            new_df[dep + "_0"] = new_df[other_seg].sum(axis = 1)
    return new_df


def add_total_segname(segment_name):
    '''
    add a row for total segment of all departments for a dataframe
    :param segment_name: a dataframe that has the segment names data
    :return: a data frame that have total segment rows of all departments for a dataframe
    '''
    new_df = segment_name[segment_name.columns]
    dep_list = set(i.split("_")[0] for i in new_df["id"])
    for dep in dep_list:
        if (dep + "_0") not in list(new_df["id"]):
            print(dep)
            print((dep + "_0") in list(new_df["id"]))
            count = len(new_df)
            other_seg = [i for i in new_df["id"] if dep in i]
            copy_id =  new_df[new_df.id == other_seg[0]].index[0]
            new_df.loc[count] = new_df.loc[copy_id]
            new_df.loc[count, ["id", "name"]] = [dep + "_0", new_df.loc[copy_id, "name"].split("_")[0] + "_Total"]
    new_df = new_df.reset_index()
    return new_df


def shift_data(df, segment_name, list_shifted_names):
    '''
    we shift some columns of a dataframe to 1 day later
    :param df: a pandas dataframe
    :param segment_name: segment_name of a hotel
    :param list_names: a list of segment names needs to shift
    :return: dataframe that has shifted columns where needed
    '''
    new_df = df[df.columns]
    for name in list_shifted_names:
        list_col_names = [i for i in segment_name["name"] if name in i]        
        list_id = []
        for seg_name in list(segment_name["name"]):
            if name in seg_name:
                id_ = segment_name.ix[segment_name.name == seg_name]["id"].iloc[0]
                new_df[id_] = new_df[id_].shift(1)

    return new_df

def dominant_and_fill(file_df_driver, hotel_id, df_driver, df_insample):
    # # # WARNING: 
    # # # this code is no longer used, we use fill_capture_arr.fill_capture_arr instead
    df_driver['index_order'] = range(len(df_driver))

    df_driver['ARR'] = -1
    df_driver['ARR_unit'] = -1
    df_driver['capture_unit'] = -1
    df_driver['capture'] = -1

    idx = pd.IndexSlice
    #Find dominant driver
    cols = df_driver['col'].unique()
    dates_of_df = df_driver['date']
    df_driver = df_driver.set_index(['col', 'date']).sort_index()
    for col in cols:
        dep = col.split('_')[0]
        print col
        l_season_tf = lib.read_dep_season_timeframe(dep, hotel_id)
        for i, season_tf in enumerate(l_season_tf):
            dates_in_season = lib.get_season_dates_from_tf(season_tf, dates_of_df)
            for day_of_week in range(7):
                weekdays_in_season = lib.get_dates_by_dayWeek(dates_in_season, day_of_week)
                dominant_driver = df_driver.loc[idx[col, weekdays_in_season],'col_d'].value_counts().index[0]
                df_driver.loc[idx[col, weekdays_in_season],'col_d'] = dominant_driver
                lag = int(dominant_driver.split("_")[3])
                col_d = "_".join(dominant_driver.split("_")[:3])
                result = capture_arr.compute_capture_arr(col, col_d, lag, season_tf, num_day, day_of_week, df_insample)
                df_driver.loc[idx[col, weekdays_in_season], 'ARR'] = result['ARR']
                df_driver.loc[idx[col, weekdays_in_season], 'ARR_unit'] = result['ARR_unit']
                df_driver.loc[idx[col, weekdays_in_season], 'capture_unit'] = result['capture_unit']
                df_driver.loc[idx[col, weekdays_in_season], 'capture'] = result['capture']

    df_driver = df_driver.reset_index()
    df_driver = df_driver.set_index('index_order').sort_index()
    df_driver.to_csv(os.path.join('data', '%s'%hotel_id,'filled_{}_0_{}'.format(hotel_id, file_df_driver)), index=False)


def create_dominant_df(driver_df, period_dict):
    new_df = driver_df[driver_df.columns]
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df['index_order'] = range(len(new_df))
    idx = pd.IndexSlice
    #Find dominant driver
    cols = new_df['col'].unique()
    dates_of_df = new_df['date']
    new_df = new_df.set_index(['col', 'date']).sort_index()
    for col in cols:
        dep = col.split('_')[0]
        l_season_tf = period_dict[dep]
        for i, season_tf in enumerate(l_season_tf):
            dates_in_season = lib.get_season_dates_from_tf(season_tf, dates_of_df)
            for day_of_week in range(7):
                try:
                    weekdays_in_season = lib.get_dates_by_dayWeek(dates_in_season, day_of_week)
                    dominant_driver = new_df.loc[idx[col, weekdays_in_season],'col_d'].value_counts().index[0]
                    new_df.loc[idx[col, weekdays_in_season],'col_d'] = dominant_driver
                    lag = int(dominant_driver.split("_")[3])
                    col_d = "_".join(dominant_driver.split("_")[:3])
                except:
                    pass
    new_df = new_df.reset_index()
    new_df = new_df.set_index('index_order').sort_index()
    return new_df

def convert_date_df(file_df_driver, period_dict, temp_year):
    # # # WARNING: 
    # # # this code is no longer used, because we do not need to 
    # # # change date of driver df anymore
    df_driver = pd.read_csv(os.path.join('data', '%s'%hotel_id, file_df_driver), index_col=False)
    df_driver['date'] = pd.to_datetime(df_driver['date'])
    ### add more
    df_driver["season_tf"] = 1
    df_driver["day_of_week"] = df_driver['date'].dt.dayofweek

    year_target = [int(temp_year)]

    df_target = df_driver[df_driver.columns]
    df_target['date'] = df_driver['date'] + pd.DateOffset(years= (year_target[0] - year_input[0]))
    df_target["day_of_week"] = df_target['date'].dt.dayofweek

    print("ok, will predict for year %s" %temp_year)
    for id in df_driver.index:
        col = df_driver.iloc[id]['col']
        ids = col.split("_")
        if id % 100 == 0:
            print(id)        
        try:
            day_cv = df_driver.iloc[id]['date'] + pd.DateOffset(years= (2016 - df_driver.iloc[id]['date'].year))
            season_tf_driver, day_of_week_driver = lib.find_out_season(period_dict[ids[0]], day_cv)
            season_tf_target, day_of_week_target = lib.find_out_season(period_dict[ids[0]], day_cv)
        except:
            season_tf_driver, day_of_week_driver = [("01-01", "12-31")], df_driver.iloc[id]['date'].weekday()
            season_tf_target, day_of_week_target = [("01-01", "12-31")], df_target.iloc[id]['date'].weekday()
        
        df_driver.loc[id, "season_tf"] = str(season_tf_driver)
        df_target.loc[id, "season_tf"] = str(season_tf_target)

    for id in df_target.index:
        try:
            driver_id = df_driver[(df_driver.col == df_target.iloc[id]["col"]) & \
                                (df_driver.season_tf == df_target.iloc[id]["season_tf"]) & \
                                (df_driver.day_of_week == df_target.iloc[id]["day_of_week"])].index[0]
            df_target.loc[id] = [df_target.loc[id][df_target.columns[0]]] + \
                                list(df_target.loc[id][["date"]]) + \
                                [df_target.loc[id][df_target.columns[2]]] + \
                                list(df_driver.loc[driver_id][df_driver.columns[3:9]]) + \
                                list(df_target.loc[id][["season_tf", "day_of_week"]])
            if id % 100 == 0:
                print(id)
        except:
            pass
    df_target.to_csv(os.path.join('data', '%s'%hotel_id,'year_{}'.format(file_df_driver)))


def predict_df(file_df_driver_union, df, model = "error"):
    '''
    In case of out_sample predict type, we create a dataframe that is filled by
    our predict method and export it in .csv file as output
    :param file_df_driver_union: name of a .csv file that represents the dataframe
    that has dominant driver and filled arr and capture.
    :param df: a dataframe that has filled data
    :param model: (default "error") type of model, choose either "error", "corr" 
    or "regression" ("corr" and "error" give the same result)
    '''
    
    df_driver_union = pd.read_csv(os.path.join('data', '%s'%hotel_id, file_df_driver_union), index_col=False)
    df_driver_union['date'] = pd.to_datetime(df_driver_union['date'])
    df_driver_union['filled_values'] = -1
    df_driver_union['values'] = -1
    df_driver_union['col_d_values'] = -1
    df_new_data = df[df.columns]
    df_new_data.index = df_new_data["date"]
    for id in df_driver_union.index:
        if id % 100 == 0:
            print(id)
        col = df_driver_union.iloc[id]['col']
        col_d = df_driver_union.iloc[id]['col_d']
        lag = int(col_d.split("_")[3])
        col_d = "_".join(col_d.split("_")[:3])
        type_col = col.split("_")[-1]
        date = df_driver_union.iloc[id]['date']

        col_d_values = df_new_data[df_new_data['date'] == (date-timedelta(days=lag))][col_d]
        col_values = df[df['date'] == date][col]
        if ((model == "error") or (model == "corr")):
            if type_col == 'rv':
                ARR = df_driver_union.loc[id]['ARR']
                capture = df_driver_union.loc[id]['capture']
            else:
                ARR = df_driver_union.loc[id]['ARR_unit']
                capture = df_driver_union.loc[id]['capture_unit']
            filled_values = float(ARR*capture*col_d_values)
        elif model == "regression":
            coef = df_driver_union.loc[id]['coef']
            intercept = df_driver_union.loc[id]['intercept']
            filled_values = float(coef * col_d_values + intercept)
        df_driver_union.loc[id, 'filled_values'] = filled_values
        df_driver_union.loc[id, 'values'] = float(col_values)
        df_driver_union.loc[id, 'col_d_values'] = float(col_d_values)
        df_new_data.loc[date, col] = filled_values
    df_driver_union.to_csv(os.path.join('data', '%s'%hotel_id, 'predicted_{}'.format(file_df_driver_union)), index=False)
    df_new_data.to_csv(os.path.join('data', '%s'%hotel_id, 'store_df_{}'.format(file_df_driver_union)), index=False)


def predict_in_sample(file_df_driver_union, df, model = "error"):
    '''
    In case of in_sample predict type, we create a dataframe that is filled by
    our predict method and export it in .csv file as output
    :param file_df_driver_union: name of a .csv file that represents the dataframe
    that has dominant driver and filled arr and capture.
    :param df: a dataframe that has filled data
    :param model: (default "error") type of model, choose either "error", "corr" 
    or "regression" ("corr" and "error" give the same result)
    '''
    df_driver_union = pd.read_csv(os.path.join('data', '%s'%hotel_id, file_df_driver_union), index_col=False)

    df_driver_union['date'] = pd.to_datetime(df_driver_union['date'])
    df_driver_union['filled_values'] = -1
    df_driver_union['values'] = -1
    df_driver_union['col_d_values'] = -1
    for id in df_driver_union.index:
        if id % 100 == 0:
            print(id)
        col = df_driver_union.iloc[id]['col']
        col_d = df_driver_union.iloc[id]['col_d']
        lag = int(col_d.split("_")[3])
        col_d = "_".join(col_d.split("_")[:3])
        type_col = col.split("_")[-1]
        date = df_driver_union.iloc[id]['date']
        col_d_values = df[df['date'] == (date-timedelta(days=lag))][col_d]
        col_values = df[df['date'] == date][col]
        if ((model == "error") or (model == "corr")):
            if type_col == 'rv':
                ARR = df_driver_union.loc[id]['ARR']
                capture = df_driver_union.loc[id]['capture']
            else:
                ARR = df_driver_union.loc[id]['ARR_unit']
                capture = df_driver_union.loc[id]['capture_unit']
            filled_values = float(ARR*capture*col_d_values)
        elif model == "regression":
            coef = df_driver_union.loc[id]['coef']
            intercept = df_driver_union.loc[id]['intercept']
            filled_values = float(coef * col_d_values + intercept)
        df_driver_union.loc[id, 'filled_values'] = filled_values
        df_driver_union.loc[id, 'values'] = float(col_values)
        df_driver_union.loc[id, 'col_d_values'] = float(col_d_values)
    df_driver_union.to_csv(os.path.join('data', '%s'%hotel_id, 'predicted_{}'.format(file_df_driver_union)), index=False)


def prepare_plot_df(file_plot_df):
    '''
    load data from csv file and do some preprocessing works
    :param file_plot_df: name of .csv file
    :return: a preprocessed dataframe
    '''
    if type(file_plot_df) == type(None):
        return None
    else:
        plot_df = pd.read_csv(os.path.join('data', '%s'%hotel_id, file_plot_df), index_col=False)
        plot_df["date"] = pd.to_datetime(plot_df['date'])
        plot_df["day_of_week"] = plot_df['date'].dt.dayofweek
        return plot_df


def cv_period_df(temp_df, current_time):
    new_df = temp_df[temp_df["date_from"].dt.year == current_time.year]
    new_df = new_df.sort_values('date_from', ascending=True)
    new_df["from"] = new_df["date_from"].dt.strftime("%m-%d")
    new_df["to"] = new_df["date_to"].dt.strftime("%m-%d")
    new_df["season"] = new_df["period_id"]
    target_df = new_df[["from", "to", "season"]]
    return target_df


def cv_period_id_dict(temp_df, current_time):
    new_df = temp_df[temp_df["date_from"].dt.year == current_time.year]
    new_df = new_df.sort_values('date_from', ascending=True)
    new_df["from"] = new_df["date_from"].dt.strftime("%m-%d")
    new_df["to"] = new_df["date_to"].dt.strftime("%m-%d")
    new_df["season"] = new_df["period_id"]
    target_df = new_df[["from", "to", "season"]]
    seasons = target_df['season'].unique()
    target_df = target_df.set_index('season')
    ss_list = [zip(pd.Series(target_df.loc[season, 'from']),pd.Series(target_df.loc[season, 'to'])) for season in seasons]
    period_df = pd.DataFrame(["period_id", "timeframe"])
    count = 0
    for season in seasons:
        period_df.loc[count, "period_id"] = season
        period_df.loc[count, "timeframe"] = [zip(pd.Series(target_df.loc[season, 'from']),pd.Series(target_df.loc[season, 'to']))]
        count += 1
    return period_df


def cv_top_df(top_df, period_id_dict):
    feature_list = ["h_id", "period_id", "day", "source_h_id", \
        "source_segment_id", "destination_segment_id", "priority", "property", \
        "driver_type", "source_day_offset", "destination_day_offset"]
    driver_type = {"rv": 3,
                    "rn": 1,
                    "gn": 2}
    new_top_df = top_df[top_df.columns]
    new_df = pd.DataFrame(0, index=np.arange(len(new_top_df)), columns=feature_list)
    new_df["priority"]    = new_df.index % 3 + 1
    new_top_df['date']    = pd.to_datetime(new_top_df['date'])
    new_top_df["day"]     = new_top_df['date'].dt.dayofweek
    new_df["day"]         = new_top_df["day"]
    new_df["property"]    = new_top_df["cov_value"]
    new_df["h_id"]        = new_top_df["col"].apply(lambda x: int(x.split("_")[0]))
    new_df["source_segment_id"]      = new_top_df["cov_col"].apply(lambda x: int(x.split("_")[1]))
    new_df["destination_segment_id"] = new_top_df["col"].apply(lambda x: int(x.split("_")[1]))
    new_df["driver_type"]            = new_top_df["cov_col"].apply(lambda x: driver_type[x.split("_")[2]])
    new_df["source_day_offset"]      = new_top_df["lag"]
    new_df["key"]                    = new_top_df["col"].apply(lambda x: int(x.split("_")[1]))
    for ind in new_df.index:
        if ind % 100 == 0:
            print(ind)
        col = df_driver_union.iloc[ind]['col']
        dep = col.split("_")[0]
        new_df.loc[ind, "source_h_id"] = dep
        new_df.loc[ind, "period_id"] = lib.find_out_period_id(period_id_dict[dep], new_df.loc(ind, "date"))
    new_df = new_df
    return new_df[feature_list]


if __name__ == "__main__":

    # ===================== PREPARE DATA ========================
    # ===========================================================

    # # ========= INPUT ===============
    # parser = argparse.ArgumentParser("Generates drivers and writes result to database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-H", "--host", help="Database host", required=True)
    # parser.add_argument("-U", "--user", help="Database user name", required=True)
    # parser.add_argument("-P", "--password", help="Database password", required=True)
    # parser.add_argument("-d", "--database", help="Database", required=True)
    # parser.add_argument("-s", "--schema", help="Database schema", default="dbo")
    # parser.add_argument("-i", "--hotel-ids", help="Comma separated list of hotel id's or 'all'", default="all")
    # parser.add_argument("-Y", "--years", help="Number of years of data to include", default=3)
    # parser.add_argument("-N", "--no-write", help="Run, but do not write results to database", action="store_true", default=False)


    # args = parser.parse_args()

    # hotel_id = args.hotel_ids
    hotel_id = "223"
    print(hotel_id)
    
    # # choose: "in_sample" / "out_sample"
    # type_predict = "in_sample"
    # ## choose: "error" / "regression" / "corr"
    model = "error"
    otb_run = "no"

    years_back = 3
    # years_back = int(args.years)

    current_time = datetime.now()
    # tf_input_raw = "01-01-2012, 12-31-2015"
    # season_target = [("01-01-2015", "12-31-2015")]
    start_str = (current_time - timedelta(days = 365 * years_back + 2)).strftime('%m-%d-%Y')
    end_str = (current_time - timedelta(days = 2)).strftime('%m-%d-%Y')
    curr_str = current_time.strftime('%m-%d-%Y')
    predict_str = (current_time + timedelta(days = 365 * 1 - 1)).strftime('%m-%d-%Y')
    tf_input_raw = "%s, %s" % (start_str, end_str)
    season_target = [(curr_str, predict_str)]

    # list_shifted_names = ["Breakfast"]

    username = "tp7"
    password = "reHj456"
    host = "172.16.0.50"
    database = "pmi_tp7"
    schema = None

    # host = args.host
    # user = args.user
    # password = args.password
    # dbname = args.database
    # schema = args.schema

    # # # # ========== SOME WORK WITH INPUT =========
    start_time1 = time.time()
    start_time = time.time()
    # ==================== CREATE DRIVER DF =====================
    # ===========================================================
    start_time = time.time()
    driver_ver_09_cl = create_driver_df.create_driver_df_cl(df, tf_input, year_input, \
                        filled_list, significance_level, segment_name, period_dict, num_day = num_day, model = model)

    try:
        if model == "error":
            with open(os.path.join('data', '%s'%hotel_id,'pickle_dict_df_error'), 'rb') as handle:
                dict_store = pickle.load(handle)
                driver_ver_09_cl.df_seg_dict = dict_store
            with open(os.path.join('data', '%s'%hotel_id,'pickle_dict_df_error_week'), 'rb') as handle:
                dict_store_week = pickle.load(handle)
                driver_ver_09_cl.df_seg_dict_week = dict_store_week
        elif model == "regression" or model == "corr":
            with open(os.path.join('data', '%s'%hotel_id,'pickle_dict_df_corr'), 'rb') as handle:
                dict_store = pickle.load(handle)
                driver_ver_09_cl.df_seg_dict = dict_store
            with open(os.path.join('data', '%s'%hotel_id,'pickle_dict_df_corr_week'), 'rb') as handle:
                dict_store_week = pickle.load(handle)
                driver_ver_09_cl.df_seg_dict_week = dict_store_week
    except:
        driver_ver_09_cl.prepare_dict()

    # driver_ver_09, top_ver_09 = driver_ver_09_cl.create_both_driver_top_ver_2(season_target)
    driver_ver_09 = driver_ver_09_cl.create_driver_ver_2(season_target)
    print("--- %s seconds ---" % (time.time() - start_time))
    dominant_df = create_dominant_df(driver_ver_09, period_dict)
    print("--- %s seconds ---" % (time.time() - start_time))
    top_df_01 = driver_ver_09_cl.create_top_all(season_target, dominant_df)

    # top_df_db = cv_top_df(top_ver_09, period_id_dict)
    # if (write):
    #     db_01.write_topdriver(top_df_db)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time1))
    os.system('say "your program has finished"')



