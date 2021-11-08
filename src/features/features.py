'''
Module for creating features for the model (interim --> processed)

'''
import pandas as pd
import numpy as np
from joblib import load

def features_cols():
    # returns correct column names for features
    X_cols_num = [
        'trip_distance',
        'pickup_datetime_date',
        'pickup_datetime_day_of_week',
        'pickup_datetime_hour',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude',
    ]

    X_cols_cat = []
    X_cols = X_cols_num + X_cols_cat
    return X_cols, X_cols_num, X_cols_cat

def get_target(df):
    # Get target variable (y) from df
    return np.log(df['trip_duration']).ravel()

def get_features(df, pl_fpath='models/preprocessing_pl.joblib'):
    # Get features (X) from df via a saved preprocessing pl
    loaded_pl = load(pl_fpath)
    return loaded_pl.transform(df)

def combine_features_target(X, y, cols):
    # Combine X and y to df to be saved as data/processed/train.pkl
    assert X.shape[0] == y.shape[0], 'Number of rows are not equal'
    preprocessed = pd.DataFrame(data=X, columns=cols[:-1])
    preprocessed[cols[-1]] = y
    return preprocessed

def decompose_pickup_datetime_features(df, pickup_datetime_col='pickup_datetime'):
    '''
    Decompose pickup datetime features into features (date, day of week, hour)
    '''
    # prevent mutating original df
    temp_df = df.copy()

    # Pickup
    # df['pickup_datetime_month'] = df[pickup_datetime_col].dt.month
    temp_df['pickup_datetime_date'] = temp_df[pickup_datetime_col].dt.day
    temp_df['pickup_datetime_day_of_week'] = temp_df[pickup_datetime_col].dt.day_of_week
    temp_df['pickup_datetime_hour'] = temp_df[pickup_datetime_col].dt.hour

    return temp_df

def decompose_dropoff_datetime_features(df, dropoff_datetime_col='dropoff_datetime'):
    '''
    Decompose dropoff datetime features into features (date, day of week, hour)
    '''
    # prevent mutating original df
    temp_df = df.copy()

    # Dropoff
    # df['dropoff_datetime_month'] = df[dropoff_datetime_col].dt.month
    temp_df['dropoff_datetime_date'] = temp_df[dropoff_datetime_col].dt.day
    temp_df['dropoff_datetime_day_of_week'] = temp_df[dropoff_datetime_col].dt.day_of_week
    temp_df['dropoff_datetime_hour'] = temp_df[dropoff_datetime_col].dt.hour

    return temp_df


def compute_trip_distance(df, lat1='pickup_latitude', lat2='dropoff_latitude', lon1='pickup_longitude', lon2='dropoff_longitude'):
    '''
    Calculate trip_distance column
    
    Using the haversine formula assuming angle is so small since we are only in new york city
    
    lat1, lat2, lon1, lon2 are col names in df
    returns distance in km
    '''
    # prevent mutating original df
    temp_df = df.copy()

    # Convert to radians
    lat1_rad = np.vectorize(np.math.radians)(df[lat1])
    lat2_rad = np.vectorize(np.math.radians)(df[lat2])
    lon1_rad = np.vectorize(np.math.radians)(df[lon1])
    lon2_rad = np.vectorize(np.math.radians)(df[lon2])

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.vectorize(np.math.sin)(dlat / 2)**2 + \
        np.vectorize(np.math.cos)(lat1_rad) * np.vectorize(np.math.cos)(lat2_rad) * np.vectorize(np.math.sin)(dlon / 2)**2
 
    c = 2 * np.vectorize(np.math.asin)(np.sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    temp_df['trip_distance'] = c*r 
    return temp_df