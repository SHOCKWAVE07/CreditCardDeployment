
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__, template_folder='template')
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    gender = int(request.form.get('gender', '0'))
    age = int(request.form.get('age', 20))
    ethnicity = int(request.form.get('ethnicity','0'))
    citizen = int(request.form.get('citizen','0'))
    industry = int(request.form.get('industry','0'))
    debt = int(request.form.get('debt', '0'))
    married = int(request.form.get('married', ''))
    bankCustomer = int(request.form.get('bankCustomer', '0'))
    yearsEmployed = int(request.form.get('yearsEmployed', '0'))
    priorDefault = int(request.form.get('priorDefault', '0'))
    employed = int(request.form.get('employed', '0'))
    creditScore = int(request.form.get('creditScore', '0'))
    driversLicense = int(request.form.get('driversLicense', '0'))
    zipCode = int(request.form.get('zipCode', '0'))
    income = int(request.form.get('income', '0'))

    # Perform any necessary data preprocessing or transformations
    # Use the loaded model for prediction
    # {'Gender':[gender],'Age':[age],'Debt':[debt],'Married':[married],'BankCustomer':[bankCustomer],'Industry':[industry],'Ethnicity':[ethnicity],'YearsEmployed':[yearsEmployed],'PriorDefault':[priorDefault],'Employed':[employed],'CreditScore':[creditScore],'DriversLicense':[driversLicense],'Citizen':[citizen],'ZipCode':[zipCode],'Income':[income]}
    
    input_1=[[0.661438,	-0.057723,	-0.956613,	0.560612,	0.556146,	7.0,	4.0,	-0.291083,	0.954650,	1.157144,	-0.288101,	-0.919195,	0.0,	0.123399,	-0.195413]]
    # default_input = [[gender, age, debt, married, bankCustomer, industry, ethnicity, yearsEmployed, priorDefault, employed, creditScore, driversLicense, citizen, zipCode, income]]
    prediction = model.predict(input_1)
 
    if prediction[0][0] >=0.5:
        output="Rejected"
    else:
        output="Accepted"

    print(prediction)
    
    prediction_text = f'Your application would be {output}'

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
