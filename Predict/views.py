from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from django.conf import settings
import os

# Create your views here.
def home(r):
    return render(r, 'base.html')

def predict(r):
    return render(r, 'Predict/predict.html')

def result(r):
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'Predict/diabetes.csv'))
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    #get input data
    pregnancies = float(r.GET.get("pregnancies"))
    glucose	 = float(r.GET.get("glucose"))
    bloodPressure = float(r.GET.get("bloodPressure"))
    skinThickness = float(r.GET.get("skinThickness"))
    insulin = float(r.GET.get("insulin"))
    bmi = float(r.GET.get("bmi"))
    diabetesPedigreeFunction = float(r.GET.get("diabetesPedigreeFunction"))
    age = float(r.GET.get("age"))

    pred = model.predict(np.array([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]).reshape(1,-1))

    result = ""
    if pred==[1]:
        result = "Positive"
    else:
        result = "Negative"

    print("Result: "+result)

    return render(r, 'Predict/predict.html', {"result":result})