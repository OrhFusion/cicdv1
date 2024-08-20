import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, f1_score, mean_squared_error
import joblib
# Read the data from a CSV file (replace 'your_data.csv' with your actual file path)
df = pd.read_csv("placement.csv")
X= df[['cgpa']]
y = df['package']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
score = model.score(X_train, y_train)*100, model.score(X_test, y_test)*100
y_pred = model.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
# Save the model
joblib.dump(model, 'model.joblib')
