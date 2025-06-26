'''
 Este script debe seguir la siguiente estructura (Ver imagen de referencia):

    0. Inicialice un DAG con fecha de inicio el 1 de octubre de 2024, ejecución manual y **sin backfill**. Asigne un `dag_id` que pueda reconocer facilmente, como `hiring_lineal`, etc.
    1. Debe comenzar con un marcador de posición que indique el inicio del pipeline.
    2. Cree una carpeta correspondiente a la ejecución del pipeline y cree las subcarpetas `raw`, `splits` y `models` mediante la función `create_folders()`.
    3. Debe descargar el archivo `data_1.csv` del siguiente [enlace](https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv). Debe guardar el archivo en la carpeta raw de la ejecución correspondiente.`Hint:` Le puede ser útil el comando `curl -o <path de guardado> <enlace con los datos>`.
    4. Debe aplicar un hold out mediante la función `split_data()` de su archivo creado en la subsección anterior.
    5. Debe aplicar el preprocesamiento y el entrenamiento del modelo mediante la función `preprocess_and_train()`.
    6. Finalmente, debe montar una interfaz en gradio donde pueda cargar un archivo ``json``.


- (3 puntos) Cree un `DockerFile` para montar un contenedor que contenga Airflow. Adicionalmente, cree una carpeta llamada dags donde guardará el script.py creado anteriormente.

    `Nota:` Para la imagen, se recomienda utilizar python 3.10-slim. Adicionalmente, puede instalar `curl` mediante la siguiente linea de código: `RUN apt-get update && apt-get install -y curl`.

- Construya el contenedor en Docker y acceda a la aplicación web de Airflow mediante el siguiente [enlace](http://localhost:8080/). Inicie sesión, acceda al DAG creado y ejecute de forma manual su pipeline.

- (2 puntos) Acceda a la URL pública de Gradio e ingrese el archivo `vale_data.json` a su modelo. ¿Que predicción entregó el modelo para Vale? Adjunte imágenes de su resultado. `Hint:` Puede acceder a los `logs` para obtener los prints y la URL pública.

`Hint:` Recuerde que puede entregar `kwargs` a sus funciones, como por ejemplo la fecha de ejecución `ds`.

**Para esta sección, debe adjuntar todos los scripts creados junto a su notebook en la entrega, ya que serán ejecutados para validar el funcionamiento. Para justificar sus respuestas, adicionaslmente puede utilizar imágenes de apoyo, como screenshots.**
'''
import pandas as pd
from datetime import timedelta

from airflow import DAG
#Operadores de airflow, encargados de ejecutar una tarea atómica, una Task es 
#una instancia de un operador
from airflow.operators.empty import EmptyOperator #clase dummy para iniciar o terminar el DAG
from airflow.operators.bash import BashOperator #para ejecutar comandos de consola
from airflow.operators.python import PythonOperator #para ejecutar funciones python
from airflow.utils.dates import days_ago

from hiring_functions import (
    create_folders, split_data, preprocess_and_train,
    gradio_interface, download_dataset
)

args = {
    "owner": "Francisco Maldonado",
    "retries": 1,
    "retry_delay": timedelta(seconds=10)
}

with DAG(
    dag_id="hiring_lineal",
    default_args=args,
    description="MLops pipeline Lab 9",
    start_date = pd.to_datetime("20241001"),
    schedule=None
) as dag:

    #Task 1 - Un simple print
    dummy_task = EmptyOperator(task_id="Iniciando_proceso", retries=2)

    # Task 2 - Creación de directorios
    task_create_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Task 3 - Descargar el dataset
    task_download_dataset = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Task 4 - Hold Out
    task_hold_out = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Task 5 - Preprocesamiento y entrenamiento
    task_preprocess_and_train = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Task 6 - Interfaz Gradio
    task_gradio_interface = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={"dir_name": "{{ ds }}"},
    )


    dummy_task >> task_create_folders >> task_download_dataset >> task_hold_out >> task_preprocess_and_train >> task_gradio_interface 


