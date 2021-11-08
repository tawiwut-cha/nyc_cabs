'''
Module for cleaning data. (raw --> interim)

Can be run as a script to generate interim data (To be run from main directory only)
'''
import pandas as pd
import numpy as np


NYC_MIN_LON, NYC_MAX_LON = -74.4, -73.4 # approx from google map
NYC_MIN_LAT, NYC_MAX_LAT = float(40.0), 41.6

def train_data_df_format():
    # returns empty df with the correct column names for training data
    raise NotImplementedError

def test_data_df_format():
    # returns empty df with the correct column names for test data
    raise NotImplementedError

def drop_zero_records(df, cols:list=None):
    # return read_raw_data()
    # drop records with 0 in the columns specified 
    if not cols:
        cols = df.select_dtypes('number').columns
    temp_df = df.copy()
    dtypes = dict(df.dtypes)
    for col in cols:
        temp_df.loc[temp_df[col]==0, [col]] = np.nan
        temp_df.dropna(axis=0, subset=[col], inplace=True)
    return temp_df.astype(dtypes) # int cols will get converted to float when replacing nulls

def drop_statistical_outliers(df, cols:list=None):
    # drop records with outlying values in the columns specified
    # outliers are further than 3 SDs from the mean
    if not cols:
        cols = df.select_dtypes('number').columns
    temp_df = df.copy()
    col_stats = {col:(temp_df[col].mean(), temp_df[col].std()) for col in cols}
    for col in cols:
        col_lower = col_stats[col][0] - 3*col_stats[col][1] 
        col_upper = col_stats[col][0] + 3*col_stats[col][1] 
        temp_df = temp_df[(temp_df[col] < col_upper) & (temp_df[col] > col_lower)]
    return temp_df

def drop_minmax(df, col, min_to_keep, max_to_keep):
    # drop values above max_to_keep or below min_to_keep
    return df[(df[col] <= max_to_keep) & (df[col] >= min_to_keep)]

def main():
    # import inside main() to prevent error when notebook imports function from this module
    # from notebook it doesn't see the correct path
    # but when we execute this script it will import read raw data
    from read import read_raw_data 

    print('Reading raw data....')
    df = read_raw_data() # must be executed from main directory for the default filepath to work

    print('Dropping zero records....')
    df = drop_zero_records(df, ['passenger_count'])

    print('Dropping outliers by minmax values....')
    df = drop_minmax(df, 'pickup_latitude', NYC_MIN_LAT, NYC_MAX_LAT)
    df = drop_minmax(df, 'pickup_longitude', NYC_MIN_LON, NYC_MAX_LON)
    df = drop_minmax(df, 'dropoff_latitude', NYC_MIN_LAT, NYC_MAX_LAT)
    df = drop_minmax(df, 'dropoff_longitude', NYC_MIN_LON, NYC_MAX_LON)

    print('Dropping statistical outliers....')
    df = drop_statistical_outliers(df)

    print('Saving data to data/interim/train.pkl')
    # df.to_csv('data/interim/train.csv', index=False)
    df.to_pickle('data/interim/train.pkl')

    print('DONE')

if __name__ == '__main__':
    main()