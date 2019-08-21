"""
Copyright: TP7
"""

import pandas as pd
from datetime import timedelta, datetime
from datetime import date as ddate
import time

import d2o.revenue_lib.department_correlation
import d2o.revenue_lib.project_lib as lib


class create_driver_df_cl:
    def __init__(self, df, tf_input, year_input, filled_list, significance_level, segment_name, period_dict, num_day = None, model = "corr"):
        self.start_date = tf_input[0]
        self.end_date = tf_input[1]
        self.df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        self.year_input = year_input
        self.filled_list = filled_list
        self.significance_level = significance_level
        self.segment_name = segment_name
        self.num_day = num_day
        self.model = model
        self.hotel_id = str(self.segment_name["hotel_id"].iloc[0])
        self.all_col = set(df.columns) - {'date', 'day_of_week'}
        self.all_rv_col = list(self.all_col)
        self.temp_list = list(self.all_col)
        self.all_rv_col_without_hid = list(self.all_rv_col)
        for i in self.temp_list:
            if (("rn" in i) or ("gn" in i)):
                self.all_rv_col.remove(i)
                self.all_rv_col_without_hid.remove(i)
            if (self.hotel_id == i.split("_")[0]):
                try:
                    self.all_rv_col_without_hid.remove(i)
                except:
                    pass
        self.all_dep = set([i.split("_")[0] for i in self.all_col])
        self.period_dict_ver2 = period_dict
        self.df_seg_dict = {}
        self.df_seg_dict_week = {}
        self.df_seg_key = []
        for i, key in enumerate(self.df_seg_dict):
            key_short = key.split("_")
            self.df_seg_key.append(key_short[0] + "_" + key_short[1] + "_" + key_short[2])
        self.df_seg_key = list(set(self.df_seg_key))
        self.change_dict = {}
    

    def prepare_dict(self):
        '''
        create correlation dictionary for period segment in df_seg_dict
        It takes self.all_rv_col, self.model as input, and change the 
        self.df_seg_dict as the output
        '''
        for i, df_seg in enumerate(set(self.all_rv_col) - set(self.df_seg_key)):
            ids = df_seg.split("_")
            season_all_tf = self.period_dict_ver2[ids[0]]
            for index in xrange(0, len(season_all_tf)):
                for day_of_week in xrange(0, 7):
                    try:
                        if self.model == "error":
                            self.df_seg_dict[df_seg + "_" + str(season_all_tf[index]) + "_" + str(day_of_week)] = \
                                                    department_correlation.compute_corr_ver_2(ids[0], ids[1], season_all_tf[index],\
                                                    self.num_day, day_of_week, self.significance_level, self.df)
                        elif ((self.model == "regression") or (self.model == "corr")):
                            self.df_seg_dict[df_seg + "_" + str(season_all_tf[index]) + "_" + str(day_of_week)] = \
                                                    department_correlation.compute_corr_ver_3(ids[0], ids[1], season_all_tf[index],\
                                                    self.num_day, day_of_week, self.significance_level, self.df)
                    except:
                        pass


    def prepare_dict_week(self):
        # # # WARNING: 
        # # # this code is no longer used
        for i, df_seg in enumerate(set(self.all_rv_col) - set(self.df_seg_key)):
            ids = df_seg.split("_")
            season_all_tf = self.period_dict_ver2[ids[0]]
            for index in xrange(0, len(season_all_tf)):
                try:
                    if self.model == "error":
                        self.df_seg_dict[df_seg + "_" + str(season_all_tf[index])] = \
                                                department_correlation.compute_corr_ver_2_week(ids[0], ids[1], season_all_tf[index],\
                                                self.num_day, self.significance_level, self.df, self.year_input)
                    elif ((self.model == "regression") or (self.model == "corr")):
                        self.df_seg_dict[df_seg + "_" + str(season_all_tf[index])] = \
                                                department_correlation.compute_corr_ver_3_week(ids[0], ids[1], season_all_tf[index],\
                                                self.num_day, self.significance_level, self.df)
                except:
                    pass


    def check_total(self, col_1, col_2):
        '''
        check 2 segments to determine we should keep the correlation data or not
        :param col_1: the 1st segment
        :param col_2: the 2nd segment
        :return: True if 2 segment is in same department, and 1 segment is total of department. Else return False
        '''

        ids_1 = col_1.split("_")
        ids_2 = col_2.split("_")
        if ids_1[0] != ids_2[0]:
            return False
        else:
            if ((int(ids_1[1]) * int(ids_2[1])) == 0) and ((int(ids_1[1]) + int(ids_2[1])) != 0):
                return True
            else:
                return False


    def create_all_corr_df(self, month, day, year_target):
        '''
        create correlation DataFrame from df_seg_dict for a specific date of a year
        :param month: month of date
        :param day: day of date
        :param year_target: year of date
        :return: a DataFrame that contains all correlation data of a date
        '''
        df_seg_key = self.df_seg_key
        all_rv_col_without_hid = self.all_rv_col_without_hid
        df = self.df
        start_time = time.time()

        df_top_all = pd.DataFrame(columns=['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col"])
        count = 0
        day_of_week = str(datetime(year_target, month, day).weekday())
        for i,col in enumerate(all_rv_col_without_hid):
            ids = col.split("_")
            if col in df_seg_key:
                df_seg = self.df_seg_dict[col + "_" + day_of_week]
                start_time = time.time()
                df_top_all = pd.concat([df_top_all, df_seg], ignore_index=True)
            else:
                try:
                # maybe this code is slow
                    day_cv = datetime(year_target, month, day)
                    season_and_dow = lib.find_out_season(self.period_dict_ver2[ids[0]], day_cv)
                    df_seg = self.df_seg_dict[col + "_" + str(season_and_dow[0]) + "_" + day_of_week]
                except:
                    try:
                        day_cv = datetime(year_target, month, day)
                        season_and_dow = lib.find_out_season(self.period_dict_ver2[ids[0]], day_cv)
                        if self.model == "error":
                            df_seg = department_correlation.compute_corr_ver_2(ids[0], ids[1], season_and_dow[0], self.num_day, season_and_dow[1], self.significance_level, df)
                        elif ((self.model == "regression") or (self.model == "corr")):
                            df_seg = department_correlation.compute_corr_ver_3(ids[0], ids[1], season_and_dow[0], self.num_day, season_and_dow[1], self.significance_level, df)
                    except:
                        season_and_dow = [[("01-01", "12-31")], datetime(year_target, month, day).weekday()]
                        if self.model == "error":
                            df_seg = department_correlation.compute_corr_ver_2(ids[0], ids[1], season_and_dow[0], self.num_day, season_and_dow[1], self.significance_level, df)
                        elif ((self.model == "regression") or (self.model == "corr")):
                            df_seg = department_correlation.compute_corr_ver_3(ids[0], ids[1], season_and_dow[0], self.num_day, season_and_dow[1], self.significance_level, df)
                df_top_all = pd.concat([df_top_all, df_seg], ignore_index=True)
        return df_top_all


    def create_all_corr_df_week(self, month, day, year_target):
        ''' will add comment later
        '''
        df_seg_key = self.df_seg_key
        all_rv_col_without_hid = self.all_rv_col_without_hid
        df = self.df
        df_top_all = pd.DataFrame(columns=['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col"])
        count = 0
        for i,col in enumerate(all_rv_col_without_hid):
            ids = col.split("_")
            if col in df_seg_key:
                df_seg = self.df_seg_dict_week[col]                
                df_top_all = pd.concat([df_top_all, df_seg], ignore_index=True)
            else:
                try:
                    # maybe this code is slow
                    day_cv = datetime(year_target, month, day)
                    season_and_dow = lib.find_out_season(self.period_dict_ver2[ids[0]], day_cv)
                    df_seg = self.df_seg_dict_week[col + "_" + str(season_and_dow[0])]
                except:
                    try:
                        day_cv = datetime(year_target, month, day)
                        season_and_dow = lib.find_out_season(self.period_dict_ver2[ids[0]], day_cv)
                        if self.model == "error":
                            df_seg = department_correlation.compute_corr_ver_2_week(ids[0], ids[1], season_and_dow[0], self.num_day, self.significance_level, df, self.year_input)
                        elif ((self.model == "regression") or (self.model == "corr")):
                            df_seg = department_correlation.compute_corr_ver_3_week(ids[0], ids[1], season_and_dow[0], self.num_day, self.significance_level, df)                        
                    except:
                        season_and_dow = [[("01-01", "12-31")], datetime(year[0], month, day).weekday()]
                        if self.model == "error":
                            df_seg = department_correlation.compute_corr_ver_2_week(ids[0], ids[1], season_and_dow[0], self.num_day, self.significance_level, df, self.year_input)
                        elif ((self.model == "regression") or (self.model == "corr")):
                            df_seg = department_correlation.compute_corr_ver_3_week(ids[0], ids[1], season_and_dow[0], self.num_day, self.significance_level, df)                        
                df_top_all = pd.concat([df_top_all, df_seg], ignore_index=True)
        return df_top_all


    def Change_df_seg(self, all_corr_df):
        # # # WARNING: 
        # # # this code is no longer used
        combine_list = []
        for i,col in enumerate(self.all_rv_col):
            ids = col.split("_")
            new_df = all_corr_df.ix[all_corr_df.col == col]
            # choose 1 lag row
            new_df = new_df.groupby('cov_col').first().reset_index()
            temp_df = pd.DataFrame(columns = new_df.columns)
            # choose only 1 rv/rn/gn for each dep_seg
            for segment_and_type in self.all_rv_col:
                ids = segment_and_type.split("_")
                dep_and_seg = ids[0] + "_" + ids[1]
                # create a dumb df to store the rv + rn + gn data for 1 dep_seg
                trans_df = pd.DataFrame(columns = new_df.columns)
                for index in xrange(0, len(new_df)):
                    if dep_and_seg in new_df["cov_col"][index]:
                        trans_df = trans_df.append(new_df.iloc[[index]], ignore_index=True)
                trans_df = trans_df.sort_values(['cov_col','cov_value'], ascending=[False,False])
                trans_df = trans_df.reset_index(drop=True)
                if len(trans_df["col"]) == 0:
                    continue
                elif len(trans_df["col"]) < 3:
                    temp_df = temp_df.append(trans_df.iloc[[0]], ignore_index=True)
                else:
                    # if rv is way higher than others, so choose it
                    if ((trans_df["cov_value"][0] * 0.9 > trans_df["cov_value"][1]) and (trans_df["cov_value"][0] * 0.9 > trans_df["cov_value"][2])):
                        temp_df = temp_df.append(trans_df.iloc[[0]], ignore_index=True)
                    elif (trans_df["cov_value"][1] > trans_df["cov_value"][2]):
                        temp_df = temp_df.append(trans_df.iloc[[1]], ignore_index=True)
                    else:
                        temp_df = temp_df.append(trans_df.iloc[[2]], ignore_index=True)
            combine_list.append(temp_df.sort_values('cov_value', ascending=False))
        combine_df = pd.concat(combine_list, axis=0)
        combine_df = combine_df.reset_index(drop = True)
        return combine_df


    def Change_df_seg_ver_2_old(self, all_corr_df):
        # # # WARNING: 
        # # # this code is no longer used
        combine_list = []
        for i,col in enumerate(self.all_col): 
            ids = col.split("_")
            new_df = all_corr_df.ix[all_corr_df.col == col]
            # choose 1 lag row
            new_df = new_df.groupby('cov_col').first().reset_index()
            temp_df = pd.DataFrame(columns = new_df.columns)
            # choose only 1 rv/rn/gn for each dep_seg
            for segment_and_type in self.all_rv_col:
                ids = segment_and_type.split("_")
                dep_and_seg = ids[0] + "_" + ids[1]
                # create a dumb df to store the rv + rn + gn data for 1 dep_seg
                trans_df = pd.DataFrame(columns = new_df.columns)
                for index in xrange(0, len(new_df)):
                    if dep_and_seg in new_df["cov_col"][index]:
                        trans_df = trans_df.append(new_df.iloc[[index]], ignore_index=True)
                if len(trans_df["col"]) == 0:
                    continue
                elif len(trans_df["col"]) < 3:
                    temp_df = temp_df.append(trans_df.iloc[[0]], ignore_index=True)
                else:
                    # if rv is way higher than others, so choose it
                    rv_corr_con_1 = trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")].conf_interval_0.iloc[0]
                    rn_corr = trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")].cov_value.iloc[0]
                    gn_corr = trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")].cov_value.iloc[0]
                    if self.model == "error":
                        if ((rv_corr_con_1 > rn_corr) and (rv_corr_con_1 > gn_corr)):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")], ignore_index=True)
                        elif (rn_corr > gn_corr):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")], ignore_index=True)
                        else:
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)
                    elif ((self.model == "regression") or (self.model == "corr")):
                        if ((rv_corr_con_1 > rn_corr) and (rv_corr_con_1 > gn_corr)):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")], ignore_index=True)
                        elif (rn_corr > gn_corr):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")], ignore_index=True)
                        else:
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)

            combine_list.append(temp_df.sort_values('cov_value', ascending=False))
        combine_df = pd.concat(combine_list, ignore_index=True)
        combine_df = combine_df.reset_index(drop = True)
        return combine_df


    def Change_df_seg_ver_2_old_2(self, all_corr_df):
        '''
        apply some rule to remove reduntdant data
        :rule: choose only 1 lag, and choose only 1 between rv vs rn, then sort
        :param all_corr_df: a DataFrame that contains correlation data of a date
        :return: a DataFrame that contains correlation data of a date
        '''
        combine_list = []
        for i,col in enumerate(self.all_col): 
            ids = col.split("_")
            new_df = all_corr_df.ix[all_corr_df.col == col]
            # choose 1 lag row
            new_df = new_df.groupby('cov_col').first().reset_index()
            temp_df = pd.DataFrame(columns = new_df.columns)
            # choose only 1 rv/rn/gn for each dep_seg
            for segment_and_type in self.all_rv_col:
                ids = segment_and_type.split("_")
                dep_and_seg = ids[0] + "_" + ids[1]
                if self.check_total(col, segment_and_type):
                    continue

                # create a dumb df to store the rv + rn + gn data for 1 dep_seg
                trans_df = pd.DataFrame(columns = new_df.columns)
                for index in xrange(0, len(new_df)):
                    if dep_and_seg in new_df["cov_col"][index]:
                        trans_df = trans_df.append(new_df.iloc[[index]], ignore_index=True)
                if len(trans_df["col"]) == 0:
                    continue
                elif len(trans_df["col"]) < 3:
                    temp_df = temp_df.append(trans_df.iloc[[0]], ignore_index=True)
                else:
                    # if rv is way higher than others, so choose it
                    rv_corr_con_1 = trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")].conf_interval_0.iloc[0]
                    rn_corr = trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")].cov_value.iloc[0]
                    if self.model == "error":
                        if ((rv_corr_con_1 > rn_corr)):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")], ignore_index=True)
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)
                        else:
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")], ignore_index=True)
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)
                    elif ((self.model == "regression") or (self.model == "corr")):
                        if ((rv_corr_con_1 > rn_corr)):
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rv")], ignore_index=True)
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)
                        else:
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "rn")], ignore_index=True)
                            temp_df = temp_df.append(trans_df.ix[trans_df.cov_col == (ids[0] + "_" + ids[1] + "_" + "gn")], ignore_index=True)

            combine_list.append(temp_df.sort_values('cov_value', ascending=False))
        combine_df = pd.concat(combine_list, ignore_index=True)
        combine_df = combine_df.reset_index(drop = True)
        return combine_df


    def Change_df_seg_ver_2(self, all_corr_df):
        '''
        apply some rule to remove reduntdant data
        :rule: choose only 1 lag, and choose only 1 between rv vs rn, then sort
        :param all_corr_df: a DataFrame that contains correlation data of a date
        :return: a DataFrame that contains correlation data of a date
        '''
        def col_check_rn_rv(row):
            ids = row["cov_col"].split("_")
            if ("rn" == ids[2]) or ("rv"  == ids[2]):
                return ids[0] + "_" + ids[1]
            else:
                return ids[0] + "_" + ids[1] + "_" + ids[2]
        new_corr_df = all_corr_df[all_corr_df.columns]
        new_corr_df["check_col"] = new_corr_df.apply(col_check_rn_rv, axis=1)
        combine_list = []
        start_time = time.time()
        for i,col in enumerate(self.all_col):
            ids = col.split("_")
            new_df = new_corr_df.ix[new_corr_df.col == col]
            # choose 1 lag row
            new_df = new_df.groupby('cov_col').first().reset_index()
            # choose 1 of rn/rv
            new_df = new_df.sort_values('conf_interval_0', ascending=False)
            new_df = new_df.groupby('check_col').first().reset_index()
            combine_list.append(new_df.sort_values('cov_value', ascending=False))
        combine_df = pd.concat(combine_list, ignore_index=True)
        combine_df = combine_df.reset_index(drop = True)
        combine_df = combine_df[['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col"]]
        return combine_df


    def Change_df_seg_ver_3(self, all_corr_df):
        '''
        apply some rule to remove reduntdant data
        :rule: choose only 1 lag, and choose only 1 between rv vs rn, then sort
        :param all_corr_df: a DataFrame that contains correlation data of a date
        :return: a DataFrame that contains correlation data of a date
        '''
        def col_check_rn_rv(row):
            ids = row["cov_col"].split("_")
            if ("rn" == ids[2]) or ("rv"  == ids[2]):
                return ids[0] + "_" + ids[1]
            else:
                return ids[0] + "_" + ids[1] + "_" + ids[2]
        new_corr_df = all_corr_df[all_corr_df.columns]
        new_corr_df["check_col"] = new_corr_df.apply(col_check_rn_rv, axis=1)
        combine_list = []
        start_time = time.time()
        for i,col in enumerate(self.all_col):
            ids = col.split("_")
            new_df = new_corr_df.ix[new_corr_df.col == col]
            # choose 1 of rn/rv
            new_df = new_df.sort_values('conf_interval_0', ascending=False)
            new_df = new_df.groupby('check_col').first().reset_index()
            combine_list.append(new_df.sort_values('cov_value', ascending=False))
            # choose 1 lag row
            new_df = new_df.groupby('cov_col').first().reset_index()
        combine_df = pd.concat(combine_list, ignore_index=True)
        combine_df = combine_df.reset_index(drop = True)
        combine_df = combine_df[['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col"]]
        return combine_df


    def shorten_corr_df_rule_2(self, change_df_seg_df, all_corr_df):
        '''
        a strict rule to keep driver along with hotel rn/gn/rv
        :rule: the multiple of direct and indirect relationship between a RV 
        data must larger than benchmark relationship
        :param change_df_seg_df: a DataFrame that contains correlation data of a date
        :param all_corr_df: a DataFrame that contains ALL correlation data of a date
        :return: a DataFrame that contains correlation data of a date
        '''
        top_names = [col for col in self.all_col if col.split("_")[0] == self.hotel_id] #hotel_id
        combine_list = []
        for i, col in enumerate(self.all_rv_col_without_hid):
            new_df = change_df_seg_df[change_df_seg_df.col == col].reset_index(drop = True) # df of values of each seg_dep
            temp_df = pd.DataFrame(columns = new_df.columns) # append selection
            top_names_update = list(top_names + [col]) # rn/gn/rv and itself
            for index in xrange(0, len(new_df)):
                corr_direct = new_df["cov_value"][index]            
                dep_sep_type = new_df["cov_col"][index]
                if dep_sep_type in top_names_update:
                    temp_df = temp_df.append(new_df.iloc[[index]], ignore_index=True) 
                else: 
                    # dep_sep_type is not hotel_id then study indirect
                    corr_indirect = all_corr_df.query('col in "%s" and cov_col in %s'% (dep_sep_type, str(tuple(top_names_update))))["cov_value"].iloc[0]
                    corr_benchmark = new_df.query('col in "%s" and cov_col in %s'% (col, str(tuple(top_names_update))))["cov_value"].iloc[0]
                    if corr_direct * corr_indirect > corr_benchmark:
                        temp_df = temp_df.append(new_df.iloc[[index]], ignore_index=True)
            combine_list.append(temp_df)
        combine_df = pd.concat(combine_list, axis=0)
        combine_df = combine_df.reset_index(drop = True)
        return combine_df


    def shorten_corr_df_rule_3_week(self, change_df_seg_df, all_corr_df):
        # # # WARNING: 
        # # # this code is no longer used
        top_names = [col for col in self.all_col if col.split("_")[0] == self.hotel_id]
        
        combine_list = []
        for i, col in enumerate(self.all_rv_col_without_hid):
            new_df = change_df_seg_df[change_df_seg_df.col == col].reset_index(drop = True)
            temp_df = pd.DataFrame(columns = new_df.columns)
            top_names_update = list(top_names + [col])
            for index in xrange(0, len(new_df)):
                corr_direct = new_df["cov_value"][index]            
                dep_sep_type = new_df["cov_col"][index]
                if dep_sep_type in top_names:
                    temp_df = temp_df.append(new_df.iloc[[index]], ignore_index=True)
                else:
                    corr_indirect = all_corr_df.query('col in "%s" and cov_col in %s'% (dep_sep_type, str(tuple(top_names))))["cov_value"].iloc[0]
                    corr_benchmark = new_df.query('col in "%s" and cov_col in %s'% (col, str(tuple(top_names))))["cov_value"].iloc[0]
                    if corr_direct * corr_indirect > corr_benchmark:
                        temp_df = temp_df.append(new_df.iloc[[index]], ignore_index=True)
            combine_list.append(temp_df)
        combine_df = pd.concat(combine_list, axis=0)
        combine_df = combine_df.reset_index(drop = True)
        return combine_df


    def create_corr_rela_df(self, all_corr_df, str_day, year_target):
        '''
        final rule to create driver DataFrame for a specific date
        :rule: we apply the prevent circulation rule
        :param all_corr_df: a DataFrame that contains correlation data of a date
        :param str_day: input date in string type (mm-dd)
        :param year_target: year of date
        :return: driver DataFrame for this date
        '''
        filled_list = list(self.filled_list)
        corr_df =  pd.DataFrame(columns = ["col", "col_d", "corr", "date"])
        corr_remain_list = list(set(all_corr_df["col"])) # all of dept_seg except Hotel_ID, remove gradually the filled dept_segid
        corr_remain_list_temp = list(corr_remain_list)
        missing_data_list = []
        times_loops = len(set(all_corr_df["col"]))
        count = 0
        while len(corr_remain_list) > 0:
            for top_x in xrange(0, times_loops): # number of down step move for searching filled connection
                for dep_seg in corr_remain_list:
                    # to do: loop for top1 again
                    for x in xrange(0, top_x + 1):
                        next_dep_seg = all_corr_df.ix[all_corr_df.col == dep_seg].adj_corr_col.iloc[x]
                        next_dep_seg_corr = all_corr_df.ix[all_corr_df.col == dep_seg].cov_value.iloc[x]
                        if next_dep_seg_corr != - 99:
                            if next_dep_seg.split("_")[3] != "0":
                                if dep_seg not in filled_list:
                                    corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                    count += 1
                                    try:
                                        corr_remain_list_temp.remove(dep_seg)
                                    except:
                                        pass
                                    filled_list.append(dep_seg)
                            else:
                                ids = next_dep_seg.split("_")
                                short_next_ds = ids[0] + "_" + ids[1] + "_" + ids[2]
                                if short_next_ds in filled_list:
                                    if dep_seg not in filled_list:
                                        corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                        count += 1
                                        try:
                                            corr_remain_list_temp.remove(dep_seg)
                                        except:
                                            pass
                                        filled_list.append(dep_seg)
                        else:
                            missing_data_list.append(dep_seg)
                            try:
                                corr_remain_list_temp.remove(dep_seg)
                            except:
                                pass
                corr_remain_list = list(corr_remain_list_temp)
                to_sort_df = pd.DataFrame(columns=["col", "cov_val"])
                count_sort = 0
                for dep_seg_type in corr_remain_list:
                    to_sort_df.loc[count_sort] = [dep_seg_type] + [all_corr_df.ix[all_corr_df.col == dep_seg].cov_value.iloc[x + 1]]
                    count_sort += 1
                to_sort_df = to_sort_df.sort_values(['cov_val'], ascending=[False])
                corr_remain_list = list(to_sort_df.col)
                for dep_seg in missing_data_list:
                    for x in xrange(0, top_x + 1):
                        next_dep_seg = all_corr_df.ix[all_corr_df.col == dep_seg].adj_corr_col.iloc[x]
                        next_dep_seg_corr = all_corr_df.ix[all_corr_df.col == dep_seg].cov_value.iloc[x]
                        if next_dep_seg.split("_")[3] != "0":
                            if dep_seg not in filled_list:
                                corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                count += 1
                                try:
                                    corr_remain_list_temp.remove(dep_seg)
                                except:
                                    pass
                                filled_list.append(dep_seg)
                        else:
                            ids = next_dep_seg.split("_")
                            short_next_ds = ids[0] + "_" + ids[1] + "_" + ids[2]
                            if short_next_ds in filled_list:
                                if dep_seg not in filled_list:
                                    corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                    count += 1
                                    try:
                                        corr_remain_list_temp.remove(dep_seg)
                                    except:
                                        pass
                                    filled_list.append(dep_seg)
        # return corr_df
        corr_df["ind"] = corr_df.index
        new_df = corr_df.groupby('col').first().reset_index()
        new_df = new_df.sort_values(by=["ind"])
        del new_df["ind"]
        new_df = new_df.reset_index(drop = True)
        return new_df


    def create_corr_rela_df_old(self, all_corr_df, str_day, year_target):
        '''
        final rule to create driver DataFrame for a specific date
        :rule: we apply the prevent circulation rule
        :param all_corr_df: a DataFrame that contains correlation data of a date
        :param str_day: input date in string type (mm-dd)
        :param year_target: year of date
        :return: driver DataFrame for this date
        '''
        filled_list = list(self.filled_list)
        corr_df =  pd.DataFrame(columns = ["col", "col_d", "corr", "date"])
        corr_remain_list = list(set(all_corr_df["col"])) # all of dept_seg except Hotel_ID, remove gradually the filled dept_segid
        corr_remain_list_temp = list(corr_remain_list)
        times_loops = len(set(all_corr_df["col"]))
        count = 0
        while len(corr_remain_list) > 0:
            for top_x in xrange(0, times_loops): # number of down step move for searching filled connection
                for dep_seg in corr_remain_list:
                    # to do: loop for top1 again
                    for x in xrange(0, top_x + 1):
                        next_dep_seg = all_corr_df.ix[all_corr_df.col == dep_seg].adj_corr_col.iloc[x]
                        next_dep_seg_corr = all_corr_df.ix[all_corr_df.col == dep_seg].cov_value.iloc[x]
                        if next_dep_seg_corr != - 99:
                            if next_dep_seg.split("_")[3] != "0":
                                if dep_seg not in filled_list:
                                    corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                    count += 1
                                    try:
                                        corr_remain_list_temp.remove(dep_seg)
                                    except:
                                        pass
                                    filled_list.append(dep_seg)
                            else:
                                ids = next_dep_seg.split("_")
                                short_next_ds = ids[0] + "_" + ids[1] + "_" + ids[2]
                                if short_next_ds in filled_list:
                                    if dep_seg not in filled_list:
                                        corr_df.loc[count] = [dep_seg, next_dep_seg, next_dep_seg_corr, str_day + "-" + str(year_target)]
                                        count += 1
                                        try:
                                            corr_remain_list_temp.remove(dep_seg)
                                        except:
                                            pass
                                        filled_list.append(dep_seg)
                        else:
                            try:
                                corr_remain_list_temp.remove(dep_seg)
                            except:
                                pass
                corr_remain_list = list(corr_remain_list_temp)
                to_sort_df = pd.DataFrame(columns=["col", "cov_val"])
                count_sort = 0
                for dep_seg_type in corr_remain_list:
                    to_sort_df.loc[count_sort] = [dep_seg_type] + [all_corr_df.ix[all_corr_df.col == dep_seg].cov_value.iloc[x + 1]]
                    count_sort += 1
                to_sort_df = to_sort_df.sort_values(['cov_val'], ascending=[False])
                corr_remain_list = list(to_sort_df.col)
        return corr_df


    def perdelta(self, start, end, delta = timedelta(days=1)):
        '''
        create a list of dates that are equally separated
        :param start: start date
        :param end: end date
        :param delta: (default 1) the separated time you want
        :return: a list of dates that are equally separated
        '''
        curr = start
        while curr <= end:
            yield curr
            curr += delta


    def create_driver_df(self, season_tf, all_year = False):
        # # # WARNING: 
        # # # this code is no longer used
        '''
        create a driver dataframe for all date in a timeframe
        :param season_tf: a season timeframe
        :param segment_name: segment_name of a hotel
        :param list_names: a list of segment names needs to shift
        :return: dataframe that has shifted columns where needed
        '''
        year = self.year_input
        start_month_date = season_tf[0].split("-")
        end_month_date = season_tf[1].split("-")
        time_period = self.perdelta(ddate(year[0], int(start_month_date[0]), int(start_month_date[1])),\
                            ddate(year[0], int(end_month_date[0]), int(end_month_date[1])), timedelta(days = 1))
        result_df = pd.DataFrame(columns = ["col", "col_d", "corr", "date"])
        for date in time_period:
            all_corr_df = self.create_all_corr_df(date.month, date.day)
            change_df_seg = self.Change_df_seg(all_corr_df)
            result_dict = self.create_corr_rela_df(change_df_seg, date.strftime('%m-%d'))
            result_df = pd.concat([result_df, result_dict])
            print("done for day %s" % date.strftime('%m-%d'))
        if all_year:
            all_corr_df = self.create_all_corr_df(12, 31)
            change_df_seg = self.Change_df_seg(all_corr_df)
            result_dict = self.create_corr_rela_df(change_df_seg, "12-31")
            result_df = pd.concat([result_df, result_dict])
            print("done for day 12-31")
            print("all year is done")
        return result_df


    def create_top_1_day(self, corr_df, change_df_seg):
        '''
        create top DataFrame for a specigic day
        :param corr_df: a driver dataframe
        :param change_df_seg: a DataFrame create from Change_df_seg_ver_2 function 
        that contains correlation data of a date
        :return: dataframe that has shifted columns where needed
        '''
        l_df = []
        for i,col in enumerate(self.all_rv_col_without_hid):
            # try:
            driver_seg = corr_df.ix[corr_df.col == col].col_d.iloc[0]
            temp_df = change_df_seg.ix[change_df_seg.adj_corr_col == driver_seg]
            start_index = temp_df.ix[temp_df.col == col].index[0]
            l_df.append(change_df_seg.iloc[start_index : (3 + start_index)])
            # except:
            #     print(col, driver_seg)
        df_top_all = pd.concat(l_df, axis=0)
        return df_top_all


    def create_top_all(self, season_tf, driver_df):
        # # # will add comment later
        # # # 
        start_str = map(int, season_tf[0][0].split("-"))
        end_str = map(int, season_tf[0][1].split("-"))
        start_date = ddate(start_str[2], start_str[0], start_str[1])
        end_date = ddate(end_str[2], end_str[0], end_str[1])
        time_period = self.perdelta(start_date, end_date)
        top_df = pd.DataFrame(columns=['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col", "date"])
        for date in time_period:
            try:
                corr_df = driver_df[driver_df["date"] == date]
                change_df_seg_df = self.change_dict[date.strftime('%m-%d-%Y')]
                # temp_df equal result_dict in create_driver_df
                top_dict = self.create_top_1_day(corr_df, change_df_seg_df)
                top_dict["date"] = date.strftime('%m-%d-%Y')
                top_df = pd.concat([top_df, top_dict])
            except:
                print(date.strftime('%m-%d-%Y'))
        return top_df
        
    def shorten_corr_df_rule_b_week_old(self, change_df_seg_df, date):
        '''
        will add comment later
        '''
        start_time = time.time()
        all_week = self.create_all_corr_df_week(date.month, date.day, date.year)
        print("--- %s seconds ---" % (time.time() - start_time))
        combine_list = []
        for i, col in enumerate(self.all_rv_col_without_hid):
            new_df = change_df_seg_df[change_df_seg_df.col == col].reset_index(drop = True)
            temp_df = pd.DataFrame(columns = new_df.columns)
            top_week = all_week[all_week.col == col].iloc[[0]]
            top_col = top_week.adj_corr_col.iloc[0]
            day_check = new_df[new_df.adj_corr_col == top_col]
            print(col, date)
            try:
                if top_week.cov_value.iloc[0] < day_check.cov_value.iloc[0]:
                    combine_list.append(new_df)
                else:
                    day_id = day_check.index[0]
                    new_df.loc[day_id] = list(top_week.iloc[0])
                    new_df_id = map(int, new_df.index)
                    iloc_track = new_df.index.get_loc(day_id)
                    new_df_id[iloc_track] = -1
                    new_df.index = new_df_id
                    temp_df = pd.DataFrame(new_df.sort_index(axis = 0))
                    combine_list.append(temp_df)
            except:
                pass
        combine_df = pd.concat(combine_list, axis=0)
        combine_df = combine_df.reset_index(drop = True)
        print("--- %s seconds ---" % (time.time() - start_time))
        return combine_df

    def Rule_week(self, all_corr_df, date):
        ''' will add comment later
        '''
        all_week = self.create_all_corr_df_week(date.month, date.day, date.year)
        combine_list = []
        for i, col in enumerate(self.all_rv_col_without_hid):
            new_df = all_corr_df[all_corr_df.col == col].reset_index(drop=True)
            temp_df = pd.DataFrame(columns=new_df.columns)
            top_week = all_week[all_week.col == col].iloc[[0]]
            top_col = top_week.adj_corr_col.iloc[0]
            day_check = new_df[new_df.adj_corr_col == top_col]
            if top_week.cov_value.iloc[0] < day_check.cov_value.iloc[0]:
                combine_list.append(new_df)
            elif day_check.cov_value.iloc[0] <= 0:
                combine_list.append(new_df)
            else:
                day_id = day_check.index[0]
                new_df.loc[day_id] = list(top_week.iloc[0])
                combine_list.append(new_df)
        combine_df = pd.concat(combine_list, axis=0)
        combine_df = combine_df.reset_index(drop=True)
        return combine_df


    def create_both_driver_top(self, season_tf, rule_a = True, rule_b = True, rule_c = False):
        '''
        create a driver dataframe and a top dataframe for all date in a timeframe
        :param season_tf: a season timeframe
        :param year_target: year of dates
        :param rule_a: (default True) choose apply rule A (Change_df_seg_ver_2) or not
        :param rule_b: (default False) choose apply rule B () or not
        :param rule_c: (default False) choose apply rule C () or not
        :return: a driver dataframe and a top dataframe for all date in a timeframe
        '''
        start_str = map(int, season_tf[0][0].split("-"))
        end_str = map(int, season_tf[0][1].split("-"))
        start_date = ddate(start_str[2], start_str[0], start_str[1])
        end_date = ddate(end_str[2], end_str[0], end_str[1])

        time_period = self.perdelta(start_date, end_date)
        result_df = pd.DataFrame(columns = ["col", "col_d", "corr", "date"])
        top_df = pd.DataFrame(columns=['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col", "date"])
        for date in time_period:
            # try:
            start_time = time.time()
            all_corr_df = self.create_all_corr_df(date.month, date.day, date.year)
            if rule_a:
                change_df_seg_df = self.Change_df_seg_ver_2(all_corr_df)
            else:
                change_df_seg_df = all_corr_df
            if rule_b:
                rule_b_week_df = self.shorten_corr_df_rule_b_week(change_df_seg_df, date)
            else:
                rule_b_week_df = change_df_seg_df
            if rule_c:
                rule_c_df = self.shorten_corr_df_rule_2(rule_b_week_df, all_corr_df)
            else:
                rule_c_df = rule_b_week_df
            # temp_df equal result_dict in create_driver_df
            temp_df = self.create_corr_rela_df(rule_c_df, date.strftime('%m-%d'), date.year)
            top_dict = self.create_top_1_day(temp_df, change_df_seg_df)
            result_df = pd.concat([result_df, temp_df])
            top_dict["date"] = date.strftime('%m-%d-%Y')
            top_df = pd.concat([top_df, top_dict])
            print("done for day %s" % date.strftime('%m-%d-%y'))
            # except:
                # print("can not calculate for %s-%s-%s"%(date.month, date.day, date.year))
        return result_df, top_df

    def create_both_driver_top_ver_2(self, season_tf, rule_a=True, rule_week=True, rule_c=False):
        '''
        create a driver dataframe and a top dataframe for all date in a timeframe
        :param season_tf: a season timeframe
        :param year_target: year of dates
        :param rule_a: (default True) choose apply rule A (Change_df_seg_ver_2) or not
        :param rule_b: (default False) choose apply rule B () or not
        :param rule_c: (default False) choose apply rule C () or not
        :return: a driver dataframe and a top dataframe for all date in a timeframe
        '''
        start_str = map(int, season_tf[0][0].split("-"))
        end_str = map(int, season_tf[0][1].split("-"))
        start_date = ddate(start_str[2], start_str[0], start_str[1])
        end_date = ddate(end_str[2], end_str[0], end_str[1])

        time_period = self.perdelta(start_date, end_date)
        result_df = pd.DataFrame(columns = ["col", "col_d", "corr", "date"])
        top_df = pd.DataFrame(columns=['col', 'cov_col', 'lag', 'cov_value',"conf_interval_0", "conf_interval_1", "adj_corr_col", "date"])
        for date in time_period:
            try:
                all_corr_df = self.create_all_corr_df(date.month, date.day, date.year)
                if rule_week:
                    rule_b_week_df = self.Rule_week(all_corr_df, date)
                else:
                    rule_b_week_df = all_corr_df
                if rule_a:
                    change_df_seg_df = self.Change_df_seg_ver_3(rule_b_week_df)
                else:
                    change_df_seg_df = rule_b_week_df
                if rule_c:
                    rule_c_df = self.shorten_corr_df_rule_2(rule_b_week_df, all_corr_df)
                else:
                    rule_c_df = change_df_seg_df

                self.change_dict[date.strftime('%m-%d-%Y')] = change_df_seg_df
                # temp_df equal result_dict in create_driver_df
                temp_df = self.create_corr_rela_df(rule_c_df, date.strftime('%m-%d'), date.year)
                top_dict = self.create_top_1_day(temp_df, change_df_seg_df)
                result_df = pd.concat([result_df, temp_df])

                top_dict["date"] = date.strftime('%m-%d-%Y')
                top_df = pd.concat([top_df, top_dict])
                if date.day == 1:
                    print("done for day %s" % date.strftime('%m-%d-%y'))
            except:
                print("can not calculate for %s-%s-%s"%(date.month, date.day, date.year))
        return result_df, top_df

    def create_driver_ver_2(self, season_tf, rule_a = True, rule_week = True, rule_c = False):
        '''
        create a driver dataframe and a top dataframe for all date in a timeframe
        :param season_tf: a season timeframe
        :param year_target: year of dates
        :param rule_a: (default True) choose apply rule A (Change_df_seg_ver_2) or not
        :param rule_b: (default False) choose apply rule B () or not
        :param rule_c: (default False) choose apply rule C () or not
        :return: create_driver_df_cla driver dataframe and a top dataframe for all date in a timeframe
        '''
        start_str = map(int, season_tf[0][0].split("-"))
        end_str = map(int, season_tf[0][1].split("-"))
        start_date = ddate(start_str[2], start_str[0], start_str[1])
        end_date = ddate(end_str[2], end_str[0], end_str[1])

        time_period = self.perdelta(start_date, end_date)
        result_df = pd.DataFrame(columns=["col", "col_d", "corr", "date"])
        for date in time_period:
            try:
                all_corr_df = self.create_all_corr_df(date.month, date.day, date.year)
                if rule_week:
                    rule_b_week_df = self.Rule_week(all_corr_df, date)
                else:
                    rule_b_week_df = all_corr_df
                if rule_a:
                    change_df_seg_df = self.Change_df_seg_ver_3(rule_b_week_df)
                else:
                    change_df_seg_df = rule_b_week_df
                if rule_c:
                    rule_c_df = self.shorten_corr_df_rule_2(rule_b_week_df, all_corr_df)
                else:
                    rule_c_df = change_df_seg_df
                self.change_dict[date.strftime('%m-%d-%Y')] = change_df_seg_df
                try:
                    temp_df = self.create_corr_rela_df(rule_c_df, date.strftime('%m-%d'), date.year)
                except:
                    temp_df = self.create_corr_rela_df_old(rule_c_df, date.strftime('%m-%d'), date.year)
                result_df = pd.concat([result_df, temp_df])
                if date.day == 1:
                    print("done for day %s" % date.strftime('%m-%d-%y'))
            except:
                print("can not calculate for %s-%s-%s"%(date.month, date.day, date.year))
        return result_df
