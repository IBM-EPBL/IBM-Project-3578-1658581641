import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import  flask, request, jsonify, render_template

app=flask(__name__)
model=pickle.load(open('xxxxxxxxxxxxxxxxxxxxxxxxxxxx'))
scale=pickle.load(open('xxxxxxxxxxxxxxxxxxxxxxxxxxxx'))

@app.route('/')#route to display the homepage
def home():
    return render_template('index.html')# rendering the homepage

@app.route('\predict',methods=["POST","GET"]) # route to show the prediction in a web UI
def predict():
    input_feature=[x for x in request.form.values() ]
    features_values=[np.array(input_feature)]
    names=[['Location','MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am',
            'WindSpeed3pm','Humidity9am','Humidity3pm',
            'Pressure9am','Pressure3pm','Temp9am','Temp3pm','RainToday',
            'WindGustDir','WindDir9am','WindDir3pm','year','month','day']]
    data=pandas.DataFrame(features_values,columns=names)
    data=scale.fit_transform(data)
    data=pandas.DataFrame(data,columns=names)
    # predictions Using the loaded model file
    prediction=model.predict(data)
    pred_prob=model.predict_proba(data)
    print(prediction)
    if prediction=="yes":
        return render_template("chance.html")
    else:
        return render_template("nochance.html")
    # showing the prediction results in a UI
if __name__=="__main__":
