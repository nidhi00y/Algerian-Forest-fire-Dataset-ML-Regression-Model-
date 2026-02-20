from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge = pickle.load(open('Models/ridge.pkl','rb'))
scaler = pickle.load(open('Models/scaler.pkl','rb'))

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Convert into array for model
        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge.predict(new_data)
        return render_template('home.html', result=prediction[0])
    else:
        return render_template('home.html')

@app.route("/")
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")