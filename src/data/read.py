'''
Module for reading data.
'''
import pandas as pd


def read_raw_data(fname='data/raw/train.csv'):
    return pd.read_csv(fname, parse_dates=['pickup_datetime', 'dropoff_datetime'])

def read_test_data(fname='data/raw/test.csv'):
    return pd.read_csv(fname, parse_dates=['pickup_datetime'])
    
def read_interim_data(fname='data/interim/train.pkl'):
    return pd.read_pickle(fname)

def read_processed_data(fname='data/processed/train.pkl'):
    return pd.read_pickle(fname)
 
def read_processed_test_data(fname='data/processed/test.pkl'):
    return pd.read_pickle(fname)

