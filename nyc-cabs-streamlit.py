'''
Streamlit application
'''
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

from src.data.preprocess import (NYC_MIN_LON, NYC_MAX_LON, NYC_MIN_LAT, NYC_MAX_LAT)
from src.features import features_cols, compute_trip_distance, get_features
from src.models.predict import make_predictions

# Header
st.write('''
# NYC Taxi trip duration prediction!

Predict your taxi trip duration by entering:
- pickup datetime
- pickup location
- dropoff location

Training data is from **NYC Trip Duration** [Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration)

''')


# Get user input
def user_input():
    pickup_date = st.date_input('Pickup date')
    pickup_time = st.time_input('Pickup time')
    # Default pickup at Columbia University
    pickup_lat = st.number_input('pickup latitude', min_value=NYC_MIN_LAT, max_value=NYC_MAX_LAT, value=40.8075)
    pickup_lon = st.number_input('pickup longitude', min_value=NYC_MIN_LON, max_value=NYC_MAX_LON, value=-73.9626)
    # Default dropoff at Time square
    dropoff_lat = st.number_input('dropoff latitude', min_value=NYC_MIN_LAT, max_value=NYC_MAX_LAT, value=40.7589)
    dropoff_lon = st.number_input('dropoff longitude', min_value=NYC_MIN_LON, max_value=NYC_MAX_LAT, value=-73.9851)

    # for making predictions
    input_df = pd.DataFrame(columns=features_cols()[0]) 

    input_df['pickup_datetime_date'] = [pickup_date.day]
    input_df['pickup_datetime_day_of_week'] = [pickup_date.weekday()]
    input_df['pickup_datetime_hour'] = [pickup_time.hour]
    input_df['pickup_latitude'] = [pickup_lat]
    input_df['pickup_longitude'] = [pickup_lon]
    input_df['dropoff_latitude'] = [dropoff_lat]
    input_df['dropoff_longitude'] = [dropoff_lon]
    input_df = compute_trip_distance(input_df, 'pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude')

    # for showing trip on map
    trip_df = pd.DataFrame({'lat':[pickup_lat, dropoff_lat], 'lon':[pickup_lon,dropoff_lon]})
    return input_df, trip_df

input_df, trip_df = user_input()

# Show trip on map
st.map(trip_df)

# Extract features from input data using our trained features pipeline
X = get_features(input_df)
X_df = pd.DataFrame(X, columns=features_cols()[0])

# Use trained model to predict
if st.button('Estimate trip duration'):
    # Load model
    model = load('models/best_estimator.joblib')
    # Predict using input data --> log scale
    predictions = make_predictions(model, X_df)
    # Convert predictions to minutes (rounded up)
    predicted_duration = int(predictions[0] // 60 + 1)
    st.write(f'Your trip will take about {predicted_duration} minutes')
    