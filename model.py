import pandas as pd 
from pydantic import BaseModel
import pickle

class Passenger(BaseModel):
    pclass: float 
    sex: int 
    age: float 

class ClfModel:

    def __init__(self):
        self.model_fname_ = 'ReallySimpleModel.pkl'
        try:
            self.model = pickle.load(open(self.model_fname_, "rb" ))
        except Exception as _:
            print('No Model found!')

    def predict_survival(self, pclass, sex, age):
        data_in = [[pclass, sex, age]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability