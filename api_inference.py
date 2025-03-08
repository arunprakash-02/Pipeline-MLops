#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load the trained model
with open("trained_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the input schema for the API
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(input_data: ModelInput):
    # Convert input data to a NumPy array
    features = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4]])
    prediction = model.predict(features)
    return {"class": int(prediction[0])}

