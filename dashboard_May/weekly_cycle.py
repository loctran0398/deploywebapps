import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers(df, col):
    '''
    df: dataframe ['Actual_FoodRevenue', 'Actual_Purchase']
    replace outliers of column 'col' of dataframe df by 0.6745*(i-med)/mad
    '''
    
    df_new = df.copy()    
    x = np.asarray(df[col])
    med = np.nanmedian(x)
    mad = np.nanmedian(np.fabs(x - med))
    outliers = filter(lambda i: 0.6745*(i-med)/mad > 3, x)    
    df_new.loc[df_new[col].isin(outliers), col] = med 
    df_new.reset_index(drop = True)
    return(df)
      
#def rolling_corr(df):
#    '''
#    compute corr of sum(rev) and sum(pur) in a weekly cycle
#    df: dataframe['Date', 'weekday', 'Actual_Purchase', 'Actual_FoodRevenue'], 'Date': datetime
#    return corr_df: dataframe['weekday', 'corr']
#    '''    
##    wd = df['weekday']
#    n = df.shape[0]
#    corr_df = pd.DataFrame(columns = ["weekday", "corr"])
#    list_wd = []
#    list_cor = []
#    for k in range(7):
#        pur = []
#        rev = []
#        i = k
#        while (i<=n-7):
#            pur.append(sum(df["Actual_Purchase"][i:i+7]))
#            rev.append(sum(df["Actual_FoodRevenue"][i:i+7]))
#            i = i+7
#        t =  stats.pearsonr(rev, pur)
#        cor = t[0]
##        pval = t[1]
#        list_wd.append(k)
#        list_cor.append(cor)
#    corr_df['weekday'] = pd.Series(list_wd, index = range(7))
#    corr_df["corr"] = pd.Series(list_cor, index = range(7))
#    corr_df = corr_df.sort_values(by = "corr", ascending = False)
#    return(corr_df)
 
def rolling_corr(df, weekday_begin, weekday_end):
    '''
    compute corr of sum(rev) and sum(pur) in a weekly cycle
    df: dataframe['Date', 'weekday', 'Actual_Purchase', 'Actual_FoodRevenue'], 'Date': datetime
    return corr_df: dataframe['weekday', 'corr']
    '''    
#    wd = df['weekday']
    n = df.shape[0]
    corr_df = pd.DataFrame(columns = ["weekday", "corr"])
    list_wd = []
    list_cor = []
    nb_days = (weekday_end - weekday_begin + 1)
    
    for k in range(weekday_begin, weekday_end +1):
        pur = []
        rev = []
        i = k
        while (i<=n-nb_days):
            pur.append(sum(df["Actual_Purchase"][i:i+nb_days]))
            rev.append(sum(df["Actual_FoodRevenue"][i:i+nb_days]))
            i = i+nb_days
        t =  stats.pearsonr(rev, pur)
        cor = t[0]
#        pval = t[1]
        list_wd.append(k)
        list_cor.append(cor)
    corr_df['weekday'] = pd.Series(list_wd, index = range(nb_days))
    corr_df["corr"] = pd.Series(list_cor, index = range(nb_days))
    corr_df = corr_df.sort_values(by = "corr", ascending = False)
    return(corr_df)    
    
    
    
def volatility(df, weekday_begin, weekday_end):
    '''
    compute std dev of purchase for each weekday
    df: dataframe['weekday', 'Actual_Purchase']
    return df_group['weekday', 'std_purchase']
    '''
    
    df_group = pd.DataFrame(columns = ["weekday", "std_purchase"])
    nb_days = (weekday_end - weekday_begin + 1)
    wdlist = range(nb_days)
    df_group['weekday'] = pd.Series(range(weekday_begin, weekday_end + 1), index = range(nb_days))
    df_group['std_purchase'] = pd.Series([0]*nb_days, index = range(nb_days))
    for d in range(weekday_begin, weekday_end + 1):
        df_group.loc[df_group['weekday']==d,'std_purchase'] = np.std(np.asarray(df.loc[df['weekday']==d,'Actual_Purchase']))
    df_group = df_group.sort_values(by = 'std_purchase', ascending=True)
    
    return(df_group)
    
def starting_day1(df_vol, df_corr):
    '''
    not consider notworking day condition
    df_vol: dataframe['weekday', 'std_dev']
    df_corr: dataframe['weekday', 'corr']
    '''
    
#    wdlist = range(7)    
    #df_vol = volatility(df)    
    df_vol = df_vol.reset_index(drop=True)[0:2]    
    #df_corr = rolling_corr(df)
    df_corr = df_corr.reset_index(drop=True)[0:3]    
    
    l = list(df_corr['weekday'].apply(lambda x: int(x)))
    for id,row in df_vol.iterrows():
        wd = int(row['weekday'])
        if(wd in l):
            return wd
    return(int(df_vol['weekday'][0]))

def weekly_cycle_of_hotel(df_data, weekday_begin = 0, weekday_end = 6):
    '''
    add columns: weekday and cycle_day to df_data
    df_data: dataframe ['Date','Actual_Purchase','Actual_FoodRevenue'], 'Date': datetime '%Y-%m-%d'
    return df_data: dataframe['cycle_day']
    '''
    
    df = df_data.copy()
    df = remove_outliers(df, 'Actual_Purchase')
    df = remove_outliers(df, 'Actual_FoodRevenue')
        
    df_corr = rolling_corr(df, weekday_begin, weekday_end)
    df_vol = volatility(df, weekday_begin, weekday_end)
    start = starting_day1(df_vol, df_corr)
    nb_days = (weekday_end - weekday_begin + 1)
    
    df['cycle_day'] = df['weekday'].map(lambda d: (d- start) if (d>=start) else (d + nb_days- start))
    return df




