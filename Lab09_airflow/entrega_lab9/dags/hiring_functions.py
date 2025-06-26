'''
Archivo guardado en el directorio `dags` que contiene las funciones `create_folders()`, `split_data()`,`preprocess_and_train()`, `gradio_interface`

1. (3 puntos) Una función llamada `create_folders()` que cree una carpeta, la cual utilice la fecha de ejecución como nombre. Adicionalmente, dentro de esta carpeta debe crear las siguientes subcarpetas:
  - raw
  - splits
  - models

  `Hint`: Puede hacer uso de kwargs para obtener la fecha de ejecución mediante el DAG. El siguiente [Enlace](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html) le puede ser útil.

2. (3 puntos) Una función llamada `split_data()` que lea el archivo `data_1.csv` de la carepta `raw` y a partir de este, aplique un *hold out*, generando un dataset de entrenamiento y uno de prueba. Luego debe guardar estos nuevos conjuntos de datos en la carpeta `splits`. `Nota:` Utilice un 20% para el conjunto de prueba, mantenga la proporción original en la variable objetivo y fije una semilla.

3. (8 puntos) Cree una función llamada `preprocess_and_train()` que:
  - Lea los set de entrenamiento y prueba de la carpeta `splits`.
  - Cree y aplique un `Pipeline` con una etapa de preprocesamiento. Utilice `ColumnTransformers` para aplicar las transformaciones que estime convenientes. Puede apoyarse del archivo `data_1_report.html` para justificar cualquier paso del preprocesamiento.
  
  - Añada una etapa de entrenamiento utilizando el modelo `RandomForest`.
  
  Esta función **debe crear un archivo `joblib` (análogo a `pickle`) con el pipeline entrenado** en la carepta `models`, además debe **imprimir** el accuracy en el conjunto de prueba y el f1-score de la clase positiva (contratado).

4. (1 punto) Incorpore la función `gradio_interface` en su script, modificando la ruta de acceso a su modelo, de forma que pueda leerlo desde la carepta `models`. Puede realizar las modificaciones que estime necesarias.

`NOTA:` Se permite la creación de funciones auxiliares si lo estiman conveniente.

'''
import os
import subprocess
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier

import gradio as gr

def create_folders(dir_name):
    '''
    Función llamada que crea una carpeta, la cual utilice la fecha de ejecución como nombre. 
    Adicionalmente, dentro de esta carpeta debe crear las siguientes subcarpetas:
      - raw
      - splits
      - models
      `Hint`: Puede hacer uso de kwargs para obtener la fecha de ejecución mediante el DAG. El siguiente [Enlace](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html) le puede ser útil.
    '''
    dir_path = Path(dir_name)
    os.makedirs(dir_path, exist_ok=True)
    for subdirectory_name in [ "raw", "splits", "models",]: 
        os.makedirs(dir_path / subdirectory_name, exist_ok=True)

def download_dataset(dir_name):
    '''
    Función que descarga el dataset.
    ''' 
    dir_path = Path(dir_name)
    subprocess.run([
        "curl", "-o",
        str(dir_path/"raw"/"data_1.csv"),
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    ])

def split_data(dir_name, random_seed=99):
    '''
    Función que lee el archivo `data_1.csv` de la carepta `raw` y a partir de 
    este, aplica un *hold out*, generando un dataset de entrenamiento y uno de 
    prueba.
    Luego guarda estos nuevos conjuntos de datos en la carpeta `splits`. 
    `Nota:` Utilice un 20% para el conjunto de prueba, mantenga la proporción 
    original en la variable objetivo y fije una semilla.
    '''
    
    dir_path = Path(dir_name)
    for subdir in ["raw", "splits"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."
    df_path = dir_path / "raw/data_1.csv"

    X, y = (
        lambda df, label_name:
            (df.drop(columns=label_name), df[label_name])
        )( pd.read_csv(df_path), "HiringDecision")

    split_data_names = [
        "data_1_X_train.csv",
        "data_1_X_test.csv",
        "data_1_y_train.csv",
        "data_1_y_test.csv",
    ]
    for name, data in zip (
        split_data_names,
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)
        ):
        data.to_csv(dir_path / "splits" / name, index=False)


def preprocess_and_train(dir_name, random_seed=99):
    '''
    Función que lee los set de entrenamiento y prueba de la carpeta `splits`.
    Luego crea y aplica un `Pipeline` con una etapa de preprocesamiento. 
    (Utilice `ColumnTransformers` para aplicar las transformaciones que estime 
    convenientes. Puede apoyarse del archivo `data_1_report.html` para 
    justificar cualquier paso del preprocesamiento.)
    Finalmente añade una etapa de entrenamiento utilizando el modelo `RandomForest`.
  
    Esta función **debe crear un archivo `joblib` (análogo a `pickle`) con el 
    pipeline entrenado** en la carepta `models`, además debe **imprimir** el 
    accuracy en el conjunto de prueba y el f1-score de la clase positiva 
    (contratado).
    '''
    dir_path = Path(dir_name)
    split_data_names = [
        "data_1_X_train.csv",
        "data_1_X_test.csv",
        "data_1_y_train.csv",
        "data_1_y_test.csv",
    ]
    X_train, X_test, y_train, y_test = [
        pd.read_csv(dir_path / "splits" / name)
        for name in split_data_names
    ]

    # Por indicación de enunciado se creará un pipeline, pero no es necesario
    # (ni recomendable) aplicar escalamiento a variables numéricas (u ordinales)
    # al entrenar un random forest, mientras que la categórica (Gender) al ser 
    # binaria no tiene sentido aplicar OneHot.
    # Tampoco hay nulos que imputar.

    pipeline = Pipeline([
        ("regressor", RandomForestClassifier(random_state=random_seed)),
    ])

    y_predict= pipeline.fit(X_train, y_train).predict(X_test)
    acc_value = accuracy_score(y_test, y_predict)
    f1_value = f1_score(y_test, y_predict, pos_label=1)

    model_path = dir_path / "models" / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"accuracy en el cjto. de prueba: {round(acc_value, 2)}")
    print(f"f1-score en la clase positiva (contratado): : {round(f1_value, 2)}")



def predict(file, model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(dir_name):

    dir_path = Path(dir_name)
    model_path = dir_path / "models" / "model.joblib"

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)

if __name__ == "__main__":
    dir_name = datetime.datetime.today().strftime("%Y-%m-%d")
    dir_path = Path(dir_name)

    create_folders(dir_name)

    subprocess.run([
        "curl", "-o",
        str(dir_path/"raw"/"data_1.csv"),
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    ])

    split_data(dir_name)
    preprocess_and_train(dir_name)
    gradio_interface(dir_name)

    subprocess.run([
        "rm",
        "-r",
        f"{dir_name}",
    ])

