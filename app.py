"""
This module defines a Flask web application for predicting placement packages
based on CGPA.
"""
from flask import Flask, render_template, request
import joblib
import pandas as pd
model = joblib.load('model.joblib') 
app = Flask(__name__)
@app.route('/')
def index():
    """
    Renders the homepage of the application.
    """
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """
    result on the homepage.
    """
    if request.method == 'POST':
        try:
            # Extract CGPA from the form data
            cgpa = float(request.form['cgpa'])

            # Prepare the data for prediction
            data = pd.DataFrame([[cgpa]], columns=['cgpa'])

            # Make prediction using the loaded model
            prediction = model.predict(data)[0]

            # Render the result on the homepage
            return render_template(
                'index.html', prediction=prediction
            )
        except ValueError:
            # Handle cases where CGPA is not a valid float
            return render_template(
                'index.html', prediction="Invalid"
            )
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

