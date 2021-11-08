'''
Module for generating predictions from model

Can be run as to generate model predictions (To be run from main directory only)
'''
import pandas as pd
import numpy as np
from joblib import load

def make_predictions(model, test):
    '''
    Make predictions with model and convert back to seconds

    model: sklearn regressor
        - loaded model from models dir
    test: pd.DataFrame
        - features to make predictions
    '''
    y_pred = model.predict(test)
    return np.exp(y_pred)

def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from src.data.read import read_processed_test_data
    
    # Read processed test data
    print('Downloading processed test data....')
    test = read_processed_test_data()

    # Feed to model to make predictions
    print('Making predictions....')
    loaded_model = load('models/best_estimator.joblib')
    predictions = make_predictions(loaded_model, test)

    # Save results to model_predictions
    print('Saving results....')
    submission = pd.DataFrame()
    submission['trip_duration'] = pd.Series(predictions)
    submission.to_pickle('models/predictions.pkl')

    print('Done')

if __name__ == '__main__':
    main()

