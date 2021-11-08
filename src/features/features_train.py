'''
Module for creating features for training the model (interim --> processed)

Can be run as a script to generate preprocessing pipeline and processed data from interim data for model training
(To be run from main directory only)
'''
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def create_preprocessing_pl(df, features_cols):
    '''
    Create preprocessing pl for features that will be feed to the model

    df: pd.DataFrame
        - dataframe of data which the pl will be trained from (must be based on training data)
    features_cols: list
        - list of list X_cols, X_cols_num, X_cols_cat
    '''
    _, X_cols_num, _ = features_cols

    num_pipeline = Pipeline(
        steps=[
            ('median_imputer', SimpleImputer(strategy='median')),
            ('standard_scaler', StandardScaler()),
        ])

    preprocessing_pl = Pipeline(
        steps=[
            ('column_transformer', ColumnTransformer(
                [('num', num_pipeline, X_cols_num),] 
                ) 
            ),
        ])        

    preprocessing_pl.fit(df)

    return preprocessing_pl

def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from src.data.read import read_interim_data
    from src.data.preprocess import drop_zero_records
    from src.features import (
        decompose_pickup_datetime_features,
        decompose_dropoff_datetime_features,
        compute_trip_distance,
        get_features,
        get_target,
        features_cols,
        combine_features_target
    )
    from joblib import dump
    
    print('Reading interim data....')
    df = read_interim_data()
    
    print('Applying feature engineering....')
    df = decompose_pickup_datetime_features(df)
    df = decompose_dropoff_datetime_features(df)
    df = compute_trip_distance(df)
    df = drop_zero_records(df, ['trip_distance'])

    print('Training preprocessing pipeline....')
    preprocessing_pl = create_preprocessing_pl(df, features_cols())
    dump(preprocessing_pl, 'models/preprocessing_pl.joblib') 

    print('Extracting features and target....')
    X = get_features(df, 'models/preprocessing_pl.joblib')
    y = get_target(df)

    # print('Saving features....')
    # print('Saving target....')
    print('Saving training data....')
    X_cols = features_cols()[0]
    preprocessed = combine_features_target(X, y, X_cols+['log_trip_duration'])
    preprocessed.to_pickle('data/processed/train.pkl')

    print('Done')

if __name__ == '__main__':
    main()
