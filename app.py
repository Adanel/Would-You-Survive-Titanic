import uvicorn
from fastapi import FastAPI
from model import Passenger, ClfModel

app = FastAPI()
model = ClfModel()

@app.post('/predict')
def predict_species(titanic: Passenger):
    data = titanic.dict()
    prediction, probability = model.predict_survival(
        data['pclass'], data['sex'], data['age']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)