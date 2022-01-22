# Library imports
import uvicorn
from fastapi import FastAPI, Header
from starlette.responses import FileResponse
import numpy as np 
import pickle
import pandas as pd

# Create the app object
app = FastAPI()

@app.get('/favicon.ico', include_in_schema=False)

# Index route, opens automatically http://127.0.0.1:8000
@app.get("/")
def home():
    return {'message': 'Hello, World'}

# Route with a single parameter, returns the parameter within a message 
# Located at: http://127.0.0.1:800/AnyNameHere

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome to diabetes web app prediction': f'{name}'}

# Expose the prediction functionality, make a prediction from the pass JSON data and return the predicted Diabetes with the confidence
# Expose the prediction functionality, make a prediction from the pass JSON data and return the predicted Diabetes with the confidence
@app.post('/predict')
def predict_diabetes(Pregnancies: float = Header("0-17"), Glucose: float = Header("0-199"), BloodPressure: float = Header("0-122"), SkinThickness: float = Header("0-99"), Insulin: float=Header("0-846"), BMI: float=Header("0-67.1"), DiabetesPedigreeFunction: float=Header("0.078-2.42"),Age: float=Header("21-81")):
    classifier = pickle.load(open("model.pk", "rb"))
    prediction = classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction,Age]])
#   print(classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction,Age]]))
    if prediction[0]>(0.5):
        prediction = "Your chance of having diabetes is very high based on this model (This is only a predictive model, please consult a certified doctor for any medical advice)" 
    else:
        prediction = "You have a low chance of diabetes which is currently consider safe (This is only a predictive model, please consult a certified doctor for any medical advice)"
    return {
        "prediction": prediction

    }


# Run the API with uvicorn
# Will run on http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)