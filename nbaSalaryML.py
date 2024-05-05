import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
 
# Load the trained Random Forest Regressor model
model = load('nbaSalaryRandomForestPredictor.joblib')

# Define the columns used by the model
features_to_scale = ['AGE', 'GP', 'MIN', 'PTS', 'NBA_FANTASY_PTS']
all_features = ['AGE', 'GP', 'MIN', 'PTS', 'NBA_FANTASY_PTS', 'GP_RANK', 'MIN_RANK', 'FG3A_RANK', 
                'FTM_RANK', 'TOV_RANK', 'BLK_RANK', 'BLKA_RANK', 'PTS_RANK', 'NBA_FANTASY_PTS_RANK', 'WNBA_FANTASY_PTS_RANK']

# Load the CSV file containing x_train
x_train = pd.read_csv('x_train.csv')

# Create sliders for user input
st.sidebar.title('Input Parameters')
inputs = {}
for feature in all_features:
    min_val = float(x_train[feature].min())
    max_val = float(x_train[feature].max())
    mean_val = float(x_train[feature].mean())
    inputs[feature] = st.sidebar.slider(f'{feature}', min_val, max_val, mean_val)

# Convert inputs to a numpy array
input_array = np.array(list(inputs.values())).reshape(1, -1)

# Split the dataset into two parts: one containing the features to scale and the other containing the remaining features
x_train_to_scale = x_train[features_to_scale]
x_train_remaining = x_train.drop(columns=features_to_scale)

# Create a new scaler that scales only the selected features
scaler = MinMaxScaler()

# Fit the scaler on the subset of features in your training data
scaler.fit(x_train_to_scale)

# Scale the input data using the fitted scaler
scaled_input_to_scale = scaler.transform(input_array[:, :len(features_to_scale)])
scaled_input_remaining = input_array[:, len(features_to_scale):]

# Concatenate the scaled features and the remaining features
scaled_input = np.concatenate((scaled_input_to_scale, scaled_input_remaining), axis=1)

# Make predictions
prediction = model.predict(scaled_input)

# Display the prediction
st.title('NBA Salary Prediction')
st.write(f'Predicted Salary: {prediction[0]:.2f}')
