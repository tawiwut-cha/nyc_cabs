'''
Module for creating model from processed data

Can be run as a script to generate model (To be run from main directory only)
'''
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import dump

def evaluate_regressor_skl(regressor, X_train, y_train, X_val, y_val):
    '''
    Evaluate regressor from sklearn models by computing RMSE.
    '''
    regressor.fit(X_train, y_train)
    # make predictions on training data
    y_pred_train = regressor.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # make predictions on test data
    y_pred_val = regressor.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    print(f'Predicting with {regressor}')
    print('Training error:')
    print(f'Root mean squared error of log(trip_duration) = {rmse_train}')
    
    print('Validation error:')
    print(f'Root mean squared error of log(trip_duration) = {rmse_val}')

    return regressor, y_pred_train, y_pred_val

def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from src.data.read import read_processed_data
    
    print('Reading processed data....')
    df = read_processed_data()
    X = df.drop(['log_trip_duration'], axis=1)
    y = df['log_trip_duration'] 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    print('Training model....')
    training_results = evaluate_regressor_skl(
        HistGradientBoostingRegressor(max_leaf_nodes=300),
        X_train,
        y_train,
        X_val,
        y_val
    )

    print('Saving model....')
    dump(training_results[0], 'models/best_estimator.joblib')

    print('Done')

if __name__ == '__main__':
    main()
