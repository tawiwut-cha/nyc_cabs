'''
Module for creating features for making predictions from model (raw --> processed)

Can be run as a script to generate preprocessing pipeline and processed data from interim data for model training
(To be run from main directory only)
'''
import pandas as pd


def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from src.data.read import read_test_data
    from src.features import (
        decompose_pickup_datetime_features,
        compute_trip_distance,
        get_features,
        features_cols
    )

    print('Reading raw test data....')
    df = read_test_data()
    
    print('Applying feature engineering....')
    df = decompose_pickup_datetime_features(df)
    df = compute_trip_distance(df)


    print('Extracting features....')
    X = get_features(df, 'models/preprocessing_pl.joblib')

    pd.DataFrame(data=X, columns=features_cols()[0]).to_pickle('data/processed/test.pkl')

    print('Done')

if __name__ == '__main__':
    main()
