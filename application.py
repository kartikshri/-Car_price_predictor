from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the trained model
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
except FileNotFoundError:
    print("Error: LinearRegressionModel.pkl file not found.")
    exit()

# Load the car data
try:
    car = pd.read_csv('Cleaned_Car_data.csv')
except FileNotFoundError:
    print("Error: Cleaned_Car_data.csv file not found.")
    exit()


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get the input values from the form
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Predict the price
    try:
        prediction = model.predict(
            pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                         data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))
        )
        return str(np.round(prediction[0], 2))
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
