'''
Módulo que debe contener:
1. (2 puntos) Una función llamada `create_folders()` que cree una carpeta, la cual utilice la fecha de ejecución como nombre. Adicionalmente, dentro de esta carpeta debe crear las siguientes subcarpetas:
  - raw
  - preprocessed
  - splits
  - models
2. (2 puntos) Una función llamada `load_ands_merge()` que lea desde la carpeta `raw` los archivos `data_1.csv`y `data_2.csv` en caso de estar disponible. Luego concatene estos y genere un nuevo archivo resultante, guardándolo en la carpeta `preprocessed`.

3. (2 puntos) Una función llamada `split_data()` que lea la data guardada en la carpeta `preprocessed` y realice un hold out sobre esta data. Esta función debe crear un conjunto de entrenamiento y uno de prueba. Mantenga una semilla y 20% para el conjunto de prueba. Guarde los conjuntos resultantes en la carpeta `splits`.

4. (6 puntos) Una función llamada `train_model()` que reciba un modelo de clasificación.
    - La función debe comenzar leyendo el conjunto de entrenamiento desde la carpeta `spits`.
    - Esta debe crear y aplicar un `Pipeline` con una etapa de preprocesamiento. Utilice `ColumnTransformers` para aplicar las transformaciones que estime convenientes.
    - Añada una etapa de entrenamiento utilizando un modelo que ingrese a la función.
  
  Esta función **debe crear un archivo joblib con el pipeline entrenado**. Guarde el modelo con un nombre que le permita una facil identificación dentro de la carpeta `models`.

5. (3 puntos) Una función llamada `evaluate_models()` que reciba sus modelos entrenados desde la carpeta `models`, evalúe su desempeño mediante `accuracy` en el conjunto de prueba y seleccione el mejor modelo obtenido. Luego guarde el mejor modelo como archivo `.joblib`. Su función debe imprimir el nombre del modelo seleccionado y el accuracy obtenido.
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
from sklearn.preprocessing import  RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import gradio as gr


def create_folders(dir_name):
    '''
    función que crea una carpeta, la cual utilice la fecha de ejecución como nombre. Adicionalmente, dentro de esta carpeta debe crear las siguientes subcarpetas:
      - raw
      - preprocessed
      - splits
      - models
    '''
    dir_path = Path(dir_name)
    os.makedirs(dir_path, exist_ok=True)
    for subdirectory_name in [ "raw", "preprocessed", "splits", "models",]: 
        os.makedirs(dir_path / subdirectory_name, exist_ok=True)


def download_dataset(dir_name, which=1):
    '''
    Función que descarga el dataset.
    ''' 
    dir_path = Path(dir_name)
    assert (dir_path).exists(), f"El directorio no ha sido creado."
    if which == 1:
        subprocess.run([
            "curl", "-o",
            str(dir_path/"raw"/"data_1.csv"),
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ])
    elif which == 2:
        subprocess.run([
            "curl", "-o",
            str(dir_path/"raw"/"data_2.csv"),
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        ])


def load_and_merge(dir_name):
    '''
    función que lea desde la carpeta `raw` los archivos `data_1.csv`y `data_2.csv` en caso de estar disponible. Luego concatene estos y genere un nuevo archivo resultante, guardándolo en la carpeta `preprocessed`.
    '''
    dir_path = Path(dir_name)
    for subdir in ["raw", "preprocessed"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."
    df1_path = dir_path / "raw" / "data_1.csv"
    df2_path = dir_path / "raw" / "data_2.csv"
    out_path = dir_path / "preprocessed" / "preprocessed_data.csv"

    df1 = pd.read_csv(df1_path)
    if df2_path.exists():
        df2 = pd.read_csv(df2_path)
        pd.concat([df1, df2]).to_csv(out_path)
        output_msg = "Concatenación exitosa"
        print(output_msg)
        return output_msg

    df1.to_csv(out_path)
    output_msg = "data_2.csv no disponible, se almacena df1 en preprocessed."
    print(output_msg)
    return output_msg
    

def split_data(dir_name, random_seed=99):
    '''
    función que lee la data guardada en la carpeta `preprocessed` y realice un hold out sobre esta data. Esta función debe crear un conjunto de entrenamiento y uno de prueba. Mantenga una semilla y 20% para el conjunto de prueba. Guarde los conjuntos resultantes en la carpeta `splits`.
    '''
    dir_path = Path(dir_name)
    for subdir in ["preprocessed", "splits"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."
    df_path = dir_path / "preprocessed" / "preprocessed_data.csv"

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


def train_model(dir_name, classifier_model):
    '''
     función que recibe un modelo de clasificación.
        - La función debe comenzar leyendo el conjunto de entrenamiento desde la carpeta `splits`.
        - Esta debe crear y aplicar un `Pipeline` con una etapa de preprocesamiento. Utilice `ColumnTransformers` para aplicar las transformaciones que estime convenientes.
        - Añada una etapa de entrenamiento utilizando un modelo que ingrese a la función.
      
      Esta función **debe crear un archivo joblib con el pipeline entrenado**. Guarde el modelo con un nombre que le permita una facil identificación dentro de la carpeta `models`.
    '''
    dir_path = Path(dir_name)
    for subdir in ["splits", "models"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."
    train_data_names = [
        "data_1_X_train.csv",
        "data_1_y_train.csv",
    ]
    X_train,  y_train = [
        pd.read_csv(dir_path / "splits" / name)
        for name in train_data_names
    ]

    #Para esta parte de la tarea se realizará un escalamiento de las variables
    #numéricas y ordinales, la variable categórica Gender se dejará intacta pues
    #es binaria (onehot natural)

    numericas = X_train.columns.drop("Gender")
    column_transformers= ColumnTransformer([
        ('numerical', RobustScaler(), numericas),
    ], verbose_feature_names_out=False).set_output(transform='pandas')

    pipeline = Pipeline([
        ("preprocess", column_transformers),
        ("classifier", classifier_model),
    ])

    pipeline.fit(X_train, y_train)

    model_id = datetime.datetime.today().strftime("%H%m%s")
    model_name = type(classifier_model).__name__
    model_path = dir_path / "models" / f"pipe_{model_name}_{model_id}.joblib"
    joblib.dump(pipeline, model_path)

def evaluate_models(dir_name):
    '''
    función que recibe sus modelos entrenados desde la carpeta `models`, evalúe su desempeño mediante `accuracy` en el conjunto de prueba y seleccione el mejor modelo obtenido. Luego guarde el mejor modelo como archivo `.joblib`. Su función debe imprimir el nombre del modelo seleccionado y el accuracy obtenido.
    '''
    dir_path = Path(dir_name)
    for subdir in ["splits", "models"]:
        assert (dir_path/ subdir).exists(), f"El directorio {subdir} no ha sido creado."
    test_data_names = [
        "data_1_X_test.csv",
        "data_1_y_test.csv",
    ]
    X_test, y_test = [
        pd.read_csv(dir_path / "splits" / name)
        for name in test_data_names
    ]

    acc_dic = {}
    pipe_dic = {}
    for pipeline_path in dir_path.glob("models/pipe*.joblib"):
        pipeline = joblib.load(pipeline_path)
        model_name = type(pipeline["classifier"]).__name__
        y_predict = pipeline.predict(X_test)
        acc_dic[model_name] = accuracy_score(y_test, y_predict)
        pipe_dic[model_name] = pipeline
    best_pipe_name = pd.Series(acc_dic).idxmax()

    model_path = dir_path / "models" / "best_model.joblib"
    joblib.dump(pipe_dic[best_pipe_name], model_path)
    print(f"mejor modelo: {best_pipe_name}")
    print(f"accuracy del mejor modelo: {acc_dic[best_pipe_name]}")
    return acc_dic


if __name__ == "__main__":
    dir_name = datetime.datetime.today().strftime("%Y-%m-%d")
    dir_path = Path(dir_name)
    classifier_model1 = RandomForestClassifier()
    classifier_model2 = DummyClassifier()

    create_folders(dir_name)
    download_dataset(dir_name)
    load_and_merge(dir_name)
    split_data(dir_name)
    train_model(dir_name, classifier_model1)
    train_model(dir_name, classifier_model2)
    evaluate_models(dir_name)

    subprocess.run([
        "rm",
        "-r",
        f"{dir_name}",
    ])

