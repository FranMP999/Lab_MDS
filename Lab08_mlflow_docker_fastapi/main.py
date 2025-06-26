from joblib import load
import os
import pandas as pd
import pickle

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from make_prediction import make_prediction

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    ph:                 float
    Hardness:           float
    Solids:             float
    Chloramines:        float
    Sulfate:            float
    Conductivity:       float
    Organic_carbon:     float
    Trihalomethanes:    float
    Turbidity:          float

# init app
app = FastAPI()

@app.get('/') # ruta
async def home(): 
    '''
    Ruta tipo *home* que describe brevemente el modelo, el problema que intenta
    resolver, su entrada y salida.
     '''
    descr_str = (
        "Modelo de árbol clasificador XGBoost optimizado con optuna para\n"
            "predecir si el agua es potable en base a\n"
            "mediciones de múltiples sensores IOT."
    )

    entrada_str = (
        "1. pH value\n"
            "2. Hardness\n"
            "3. Solids (Total dissolved solids - TDS)\n"
            "4. Chloramines\n"
            "5. Sulfate\n"
            "6. Conductivity\n"
            "7. Organic_carbon\n"
            "8. Trihalomethanes\n"
            "9. Turbidity\n"
    )
    salida_str = "Potability (1 si es potable, 0 no potable)"

    dic_respuesta = {
        "Descripción": descr_str,
        "Entrada": entrada_str,
        "Salida": salida_str,
    }


    return dic_respuesta


# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/potabilidad/')
def predict(data: request_body):
    features = [[
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanes,
        data.Turbidity,
    ]]

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    label = model.predict(features).item()

    return {"potabilidad": int(label)}

if __name__ == '__main__':
    uvicorn.run('main:app', port = 8000)


