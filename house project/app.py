# import the libraies
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# create the Flask app
app = Flask(__name__)

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the route for the home page and prediction
@app.route('/')
def index():
    return render_template('index.html')

# define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'prediction': float(prediction[0])})
# Run the app
if __name__ == '__main__':
    app.run(debug=True)