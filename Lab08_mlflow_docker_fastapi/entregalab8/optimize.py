#Dado que un experimento agrupa runs y cada entrenamiento en un estudio
#se asocia a una run, se ignorará la indicación de registrar cada 
#entrenamiento en un experimento nuevo, pues no tiene sentido.
#Se preferirá en cambio asociar cada optimización a un experimento y cada entrenamiento 
#realizado en un estudio será asociado a un run.
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import mlflow # importar mlflow
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances


def optimize_model():
    # Importación e imputación de los datos
    df = pd.read_csv("water_potability.csv")
    df = pd.DataFrame(si.fit_transform(df), columns=df.columns)
    X = df.drop(columns="Potability")
    y = df["Potability"]

    #split de datos de validación
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    #Para Omptimización
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    #Función objetivo para optuna
    def objective(trial):
        # Defininmos los hiperparámetros a tunear
        xgb_params = {
            "n_estimators" : trial.suggest_int("n_estimators", 50, 1000),
            "max_depth"    : trial.suggest_int("max_depth",    3, 10),
            "max_leaves"   : trial.suggest_int("max_leaves",   0, 100),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 5),
            "learning_rate" : trial.suggest_float("learning_rate", 0.001, 0.1),
            "reg_alpha"     : trial.suggest_float("reg_alpha",     0, 1),
            "reg_lambda"    : trial.suggest_float("reg_lambda",    0, 1),
        }


        mlflow.autolog() # registrar automáticamente información del entrenamiento
        with mlflow.start_run(
            run_name=f"XGB con params = {xgb_params}",
            nested= True
        ): # delimita inicio y fin del run

            # aquí comienza el run
            model = XGBClassifier(**xgb_params)
            model.fit(
                X_train, y_train,
            )
            # aquí termina el run
            #Guardamos el pipeline en el trial
            trial.set_user_attr("model", model)
            # Obtenemos la predicción de validación
            yhat = model.predict(X_val)
            metric = f1_score(y_val, yhat)
            mlflow.log_metric("valid_f1", metric)
            mlflow.log_params(xgb_params)

        return metric

# Initiate the parent run and call the hyperparameter tuning child run logic
    experiment_id = mlflow.create_experiment(
        name= f"Lab 8: Optimización XGBoost {datetime.datetime.today().strftime("%Y-%m-%d")}"
    )

    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="parent_run para lab8",
        nested=True
    ):
        study = optuna.create_study( direction="maximize")
        study.optimize(objective, timeout=5*60)
        #plots
        history_plot = plot_optimization_history(study)
        parallel_plot = plot_parallel_coordinate(study)
        importances_plot = plot_param_importances(study)
        mlflow.log_figure(figure=history_plot, artifact_file="plots/optimization_history.png")
        mlflow.log_figure(figure=parallel_plot, artifact_file="plots/parallel_cordinate.png")
        mlflow.log_figure(figure=importances_plot, artifact_file="plots/feature_importances.png")

        #Obtenemos el mejor modelo y lo alimentamos con todos los datos
        best_model = get_best_model(experiment_id)
        best_model.fit(X, y)
        with open('model.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        mlflow.xgboost.log_model(
          xgb_model=best_model,
          artifact_path="model",
          input_example=X_val.iloc[[0]],
          model_format="ubj",
          metadata={"model_data_version": 1},
        )
        # Log tags
        mlflow.set_tags(
            tags={
                "project": "Optimización XGB para Lab 8",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )
