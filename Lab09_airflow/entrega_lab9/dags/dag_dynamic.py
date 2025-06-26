'''
Este script debe contener la siguiente estructura:

1. (1 punto) Inicialice un DAG con fecha de inicio el 1 de octubre de 2024, el cual se debe ejecutar el día 5 de cada mes a las 15:00 UTC. Utilice un `dag_id` interpretable para identificar fácilmente. **Habilite el backfill** para que pueda ejecutar tareas programadas desde fechas pasadas.
2. (1 punto) Comience con un marcador de posición que indique el inicio del pipeline.
3. (2 puntos) Cree una carpeta correspondiente a la ejecución del pipeline y cree las subcarpetas `raw`, `preprocessed`, `splits` y `models` mediante la función `create_folders()`.
4. (2 puntos) Implemente un `Branching`que siga la siguiente lógica:
  - Fechas previas al 1 de noviembre de 2024: Se descarga solo `data_1.csv`
  - Desde el 1 de noviembre del 2024: descarga `data_1.csv` y `data_2.csv`.
  En el siguiente [enlace](https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv) puede descargar `data_2.csv`.
5. (1 punto) Cree una tarea que concatene los datasets disponibles mediante la función `load_and_merge()`. Configure un `Trigger` para que la tarea se ejecute si encuentra disponible **como mínimo** uno de los archivos.
6. (1 punto) Aplique el hold out al dataset mediante la función `split_data()`, obteniendo un conjunto de entrenamiento y uno de prueba.
7. (2 puntos) Realice 3 entrenamientos en paralelo:
  - Un modelo Random Forest.
  - 2 modelos a elección.
  Asegúrese de guardar sus modelos entrenados con nombres distintivos. Utilice su función `train_model()` para ello.
8. (2 puntos) Mediante la función `evaluate_models()`, evalúe los modelos entrenados, registrando el accuracy de cada modelo en el set de prueba. Luego debe imprimir el mejor modelo seleccionado y su respectiva métrica. Configure un `Trigger` para que la tarea se ejecute solamente si los 3 modelos fueron entrenados y guardados.
'''
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from airflow import DAG
#Operadores de airflow, encargados de ejecutar una tarea atómica, una Task es 
#una instancia de un operador
from airflow.operators.empty import EmptyOperator #clase dummy para iniciar o terminar el DAG
from airflow.operators.bash import BashOperator #para ejecutar comandos de consola
from airflow.operators.python import PythonOperator #para ejecutar funciones python
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils.dates import days_ago

from hiring_dynamic_functions import (
    create_folders,
    download_dataset,
    load_and_merge,
    split_data,
    train_model,
    train_model,
    evaluate_models,
)

args = {
    "owner": "Francisco Maldonado",
    "retries": 1,
    "retry_delay": timedelta(seconds=10)
}

with DAG(
    dag_id="hiring_dynamic",
    default_args=args,
    description="MLops pipeline Lab 9",
    start_date = pd.to_datetime("20241001"),
    schedule="0 15 5 * *",
    catchup=True,
) as dag:

    #Task 1 - Un simple print
    dummy_task = EmptyOperator(task_id="Iniciando_proceso", retries=2)

    # Task 2 - Creación de directorios
    task_create_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Función para determinar qué rama se ejecutará
    def choose_branch(ds):
        '''
      - Fechas previas al 1 de noviembre de 2024: Se descarga solo `data_1.csv`
      - Desde el 1 de noviembre del 2024: descarga `data_1.csv` y `data_2.csv`.
        '''
        if pd.to_datetime(ds) < pd.to_datetime("2024-11-01"):
            return "download_dataset1"
        else:
            return ["download_dataset1", "download_dataset2"]

    # Branching task
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_branch,
        provide_context=True,
    )

    # Task 3 - Descarga dataset_1.csv
    task_download_dataset1 = PythonOperator(
        task_id="download_dataset1",
        python_callable=download_dataset,
        op_kwargs={
            "dir_name": "{{ ds }}",
            "which": 1,
                   },
    )
    # Task 4 - Descarga dataset_3.csv
    task_download_dataset2 = PythonOperator(
        task_id="download_dataset2",
        python_callable=download_dataset,
        op_kwargs={
            "dir_name": "{{ ds }}",
            "which": 2,
                   },
    )

    # Task 5 - Load and Merge the Dataset
    task_load_and_merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"dir_name": "{{ ds }}"},
        trigger_rule="one_success",
    )

    # Task 6 - Hold Out
    task_hold_out = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"dir_name": "{{ ds }}"},
    )

    # Task 7.1 - Preprocesamiento y entrenamiento Dummy
    task_train_model_Dummy = PythonOperator(
        task_id="train_model_dummy",
        python_callable=train_model,
        op_kwargs={
            "dir_name": "{{ ds }}",
            "classifier_model": DummyClassifier(),
        },
    )
    # Task 7.2 - Preprocesamiento y entrenamiento Random Forest
    task_train_model_RF = PythonOperator(
        task_id="train_model_RF",
        python_callable=train_model,
        op_kwargs={
            "dir_name": "{{ ds }}",
            "classifier_model": RandomForestClassifier(),
        },
    )
    # Task 7.3 - Preprocesamiento y entrenamiento XGBoost
    task_train_model_XGB = PythonOperator(
        task_id="train_model_XGB",
        python_callable=train_model,
        op_kwargs={
            "dir_name": "{{ ds }}",
            "classifier_model": XGBClassifier(),
        },
    )

    # Task 8 - Evaluación de Modelos Entrenados
    task_evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"dir_name": "{{ ds }}"},
        trigger_rule="all_success", #default
    )

    dummy_task >> task_create_folders >> \
        branch_task >> [task_download_dataset1, task_download_dataset2] >> \
        task_load_and_merge >> task_hold_out >> \
        [task_train_model_RF, task_train_model_XGB, task_train_model_Dummy ] >> task_evaluate_models

