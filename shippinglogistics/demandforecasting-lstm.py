import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 3. Import necessary libraries

# Import sklearn library for machine learning tasks
import sklearn

# Import random module for generating random numbers
import random

# Import pandas for data manipulation and analysis
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import tensorflow for deep learning tasks
import tensorflow as tf

# Import MinMaxScaler from sklearn.preprocessing for feature scaling
from sklearn.preprocessing import MinMaxScaler

# Import mean_squared_error from sklearn.metrics for model evaluation
from sklearn.metrics import mean_squared_error

# Import matplotlib.pyplot for data visualization
import matplotlib.pyplot as plt


# 4. Set the seed for reproducibility across different libraries
SEED = 42

# Seed for numpy's random number generator
np.random.seed(SEED)

# Seed for TensorFlow's random number generator
tf.random.set_seed(SEED)

# Seed for Python's built-in random module
random.seed(SEED)

# 1. Generate a date range from 1970 to 2024
dates = pd.date_range(start="1970-01-01", periods=55, freq="Y")

# 2. Create synthetic logistics data
data = {
    "year": dates.year,

    # Simulate inventory turnover with a sinusoidal trend and some noise
    "inventory_turnover": np.round(np.abs(np.sin(np.linspace(0, 10, 55)) + np.random.normal(0, 0.2, 55)), 2),

    # Generate random on-time delivery percentages (between 85% and 99%)
    "on_time_in_full": np.round(np.random.uniform(85, 99, 55), 1),

    # Simulate lead time values in days (between 2 and 10 days)
    "lead_time": np.round(np.random.uniform(2, 10, 55), 1),

    # Generate transportation costs in dollars (ranging from $1000 to $5000)
    "transport_cost": np.round(np.random.uniform(1000, 5000, 55), 2)
}

# 3. Create a DataFrame
df = pd.DataFrame(data)


# 4. Check the shape of the DataFrame to understand its dimensions
df_shape = df.shape

# Print the number of rows and columns in the dataset
print(f"The dataset has {df_shape[0]} rows and {df_shape[1]} columns.")

# Display the first 5 rows of the DataFrame to get a quick overview of the data
df_head = df.head()
print("First 5 rows of the dataset:")
print(df_head)

# Display the last 5 rows of the DataFrame to check the end of the data
df_tail = df.tail()
print("\\nLast 5 rows of the dataset:")
print(df_tail)

# 7. Define our time series data
inventory_turnover = df['inventory_turnover'].values.reshape(-1, 1)

# Display the inventory turnover values
inventory_turnover

# 8. Function to create the dataset for the LSTM model with look_back

# The look_back parameter defines the number of time steps the model will use to make each prediction
def prepare_lstm_data(data, time_steps=1):
    
    # 8.a Initialize empty lists X and Y to store input sequences and target values, respectively
    X, Y = [], []
    
    # 8.b Iterate through the data up to the point where adding time_steps won't exceed the data length
    for i in range(len(data) - time_steps):
        # Collect a sequence of data of size time_steps starting at index i
        seq = data[i:(i + time_steps), 0]
        
        # Append the sequence to list X
        X.append(seq)
        
        # Append the value immediately after the time_steps sequence to list Y
        # This will be our target value
        Y.append(data[i + time_steps, 0])
    
    # 8.c Convert X and Y to numpy arrays for compatibility with most machine learning libraries
    return np.array(X), np.array(Y)

# 9. Split the data into training and testing sets (preserving the chronological order of the data)
split_index = int(len(inventory_turnover) * 0.8)

# 9.a Extract the training set from the beginning up to the split index
training_data = inventory_turnover[0:split_index, :]
print("Training Data:")
print(training_data)

# 9.b Extract the testing set from the split index to the end of the dataset
testing_data = inventory_turnover[split_index:len(inventory_turnover), :]
print("\\nTesting Data:")
print(testing_data)

# 10. Normalize the data (requirement for neural networks)
scaler = MinMaxScaler(feature_range=(0, 1))

# 11. Fit the scaler to the training data and transform both the training and testing data
training_data_normalized = scaler.fit_transform(training_data)
testing_data_normalized = scaler.transform(testing_data)

print("Normalized Training Data:")
print(", ".join([str(x[0]) for x in training_data_normalized]))

print("\\nNormalized Testing Data:")
print(", ".join([str(x[0]) for x in testing_data_normalized]))

# 11. Create the datasets for the LSTM model
time_steps = 1

# Prepare the training dataset
X_train, y_train = prepare_lstm_data(training_data_normalized, time_steps)

# Prepare the testing dataset
X_test, y_test = prepare_lstm_data(testing_data_normalized, time_steps)

print("Training Data X:")
print(", ".join([str(x[0]) for x in X_train]))

print("\\nTraining Data Y:")
print(", ".join([str(y) for y in y_train]))

print("\\nTesting Data X:")
print(", ".join([str(x[0]) for x in X_test]))

print("\\nTesting Data Y:")
print(", ".join([str(y) for y in y_test]))

# 12. Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)
])

# 13.Compile the model with an appropriate loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# 14. Display the model summary to understand its architecture
model.summary()

# 13. Train the LSTM model
history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# Plot the training loss to visualize the model's learning process
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 14. Make predictions with the LSTM model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 15. Inverse transform the predictions to get them back to the original scale
train_predictions_rescaled = scaler.inverse_transform(train_predictions)
test_predictions_rescaled = scaler.inverse_transform(test_predictions)

# Also inverse transform the actual values for training and testing sets
y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print the first few rescaled training predictions and actual values for verification
print("First few rescaled training predictions:")
print(", ".join([str(x[0]) for x in train_predictions_rescaled[:5]]))

print("\\nFirst few rescaled training actual values:")
print(", ".join([str(x[0]) for x in y_train_rescaled[:5]]))

# Print the first few rescaled testing predictions and actual values for verification
print("\\nFirst few rescaled testing predictions:")
print(", ".join([str(x[0]) for x in test_predictions_rescaled[:5]]))

print("\\nFirst few rescaled testing actual values:")
print(", ".join([str(x[0]) for x in y_test_rescaled[:5]]))

# 16. Calculate the Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate the RMSE for the training set
train_score = np.sqrt(mean_squared_error(y_train_rescaled[:, 0], train_predictions_rescaled[:, 0]))
print(f"\\nRMSE in Training: {train_score:.2f}")

# Calculate the RMSE for the testing set
test_score = np.sqrt(mean_squared_error(y_test_rescaled[:, 0], test_predictions_rescaled[:, 0]))
print(f"RMSE in Testing: {test_score:.2f}")

# 17. Format data for visualization

# 17.b Create indices for original and predicted data
import plotly.graph_objects as go

original_train_index = df['year'][time_steps:time_steps + len(y_train_rescaled)]
original_test_index = (
    df['year'][len(y_train_rescaled) + 2 * time_steps:
               len(y_train_rescaled) + 2 * time_steps + len(y_test_rescaled)]
)
predicted_train_index = df['year'][time_steps:time_steps + len(train_predictions_rescaled)]
predicted_test_index = (
    df['year'][len(y_train_rescaled) + 2 * time_steps:
               len(y_train_rescaled) + 2 * time_steps + len(test_predictions_rescaled)]
)

# Create the Plotly figure
fig = go.Figure()

# Add traces for each dataset with updated colors
fig.add_trace(go.Scatter(
    x=original_train_index,
    y=y_train_rescaled[:, 0],
    mode='lines',
    name='Original Training Data',
    line=dict(color='#1f77b4', dash='solid')  # Dark blue
))

fig.add_trace(go.Scatter(
    x=predicted_train_index,
    y=train_predictions_rescaled[:, 0],
    mode='lines',
    name='Training Predictions',
    line=dict(color='#ff7f0e', dash='dash')   # Orange
))

fig.add_trace(go.Scatter(
    x=original_test_index,
    y=y_test_rescaled[:, 0],
    mode='lines',
    name='Original Testing Data',
    line=dict(color='#2ca02c', dash='solid')  # Green
))

fig.add_trace(go.Scatter(
    x=predicted_test_index,
    y=test_predictions_rescaled[:, 0],
    mode='lines',
    name='Testing Predictions',
    line=dict(color='#d62728', dash='dash')   # Red
))

# Enhance plot appearance
fig.update_layout(
    title="Real vs. Predicted Inventory Turnover Index",
    xaxis_title="Year",
    yaxis_title="Inventory Turnover Index",
    legend_title="Legend",
    font=dict(family="Arial", size=14),
    hovermode="x unified"
)

# Show the plot
fig.show()

# 18. Forecast Module for Predictions with the Trained Model

# We use the last entry from the original test series to make the next prediction
last_data = testing_data_normalized[-time_steps:]
last_data = np.reshape(last_data, (1, time_steps, 1))

# 19. List
forecast_list = []

# 20. Prediction Loop to forecast 2 years (2025 and 2026)
for _ in range(2):
    # 20.a Make a prediction using the model (we use the normalized data)
    prediction = model.predict(last_data)

    # 20.b Add the prediction to the list of predictions
    forecast_list.append(prediction[0, 0])
   
    # 20.c Update the data to include the new prediction and remove the oldest value
    # This means we will use the 2025 prediction to predict the value for 2026
    last_data = np.roll(last_data, shift=-1)
    last_data[0, -1, 0] = prediction

# 21. Transform back to the original scale
forecast_list_rescaled = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1))

# 21.a Print the forecasts for 2025 and 2026
print(f"\\nInventory Turnover Index Forecast for 2025: {forecast_list_rescaled[0, 0]:.2f}")
print(f"Inventory Turnover Index Forecast for 2026: {forecast_list_rescaled[1, 0]:.2f}")