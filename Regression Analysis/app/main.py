# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:31:21 2020

@author: Sarfo Senior
"""

# Data Handling
import logging
import pickle
import numpy as np
from pydantic import BaseModel

# Server
import uvicorn
from fastapi import FastAPI

# Modeling
#import lightgbm

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Initialize files
clf = pickle.load(open('data/model.pickle', 'rb'))
features = pickle.load(open('data/features.pickle', 'rb'))


class Data(BaseModel):
    temperature: float
    pH: float
        
@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = [data_dict[feature] for feature in features];

        # Convert inputs into to appropriate data structure
        to_predict = np.array(to_predict)

        # Create and return prediction
        prediction = clf.predict(to_predict.reshape(1, -1))
        return {"Predicted Dissolved oxygen": float(prediction[0])}
    
    except:
        my_logger.error("Something went wrong!")
        return {"Prediction": "error"}

if __name__ == "__main__":
    uvicorn.run('main:app')