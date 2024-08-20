"""
This module trains a RandomForestRegressor model to predict placement package based on CGPA.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("placement.csv")

# Feature and target variable
X = df[['cgpa']]
y = df['package']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model performance on the training and testing data
train_score = model.score(X_train, y_train) * 100
test_score = model.score(X_test, y_test) * 100

# Predictions and evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model to a file
joblib.dump(model, 'model.joblib')

# Optional: Print the metrics
print(f"Training Score: {train_score:.2f}%")
print(f"Testing Score: {test_score:.2f}%")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")
