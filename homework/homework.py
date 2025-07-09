# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)


class CreditDefaultModel:
    def __init__(self):
        self.input_dir = Path("files/input/")
        self.model_dir = Path("files/models/")
        self.output_dir = Path("files/output/")
        self.model_file = self.model_dir / "model.pkl.gz"
        self.metrics_file = self.output_dir / "metrics.json"
        self.train_file = self.input_dir / "train_data.csv.zip"
        self.test_file = self.input_dir / "test_data.csv.zip"
        self.categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
        self.random_state = 42
        self.cv_folds = 10

    def load_data(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, compression="zip", index_col=False)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(columns={"default payment next month": "default"}, inplace=True)
        df.drop(columns=["ID"], inplace=True)
        df = df[df["MARRIAGE"] != 0]
        df = df[df["EDUCATION"] != 0]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
        return df

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df.drop(columns=["default"]), df["default"]

    def build_pipeline(self) -> Pipeline:
        transformer = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)],
            remainder="passthrough"
        )
        model = RandomForestClassifier(random_state=self.random_state)
        return Pipeline([("preprocessing", transformer), ("classifier", model)])

    def get_grid_search(self, pipeline: Pipeline) -> GridSearchCV:
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=self.cv_folds,
            n_jobs=-1,
            verbose=2
        )

    def save_model(self, model: GridSearchCV):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.model_file, "wb") as f:
            pickle.dump(model, f)

    def compute_metrics(self, name: str, y_true, y_pred) -> Dict:
        return {
            "type": "metrics",
            "dataset": name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
        }

    def compute_conf_matrix(self, name: str, y_true, y_pred) -> Dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": name,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }

    def save_metrics(self, metrics: List[Dict]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            for entry in metrics:
                f.write(json.dumps(entry) + "\n")

    def execute(self):
        print("Cargando y limpiando datos...")
        train_df = self.clean_data(self.load_data(self.train_file))
        test_df = self.clean_data(self.load_data(self.test_file))

        x_train, y_train = self.split_xy(train_df)
        x_test, y_test = self.split_xy(test_df)

        print("Entrenando modelo...")
        pipeline = self.build_pipeline()
        grid = self.get_grid_search(pipeline)
        grid.fit(x_train, y_train)

        print("Guardando modelo...")
        self.save_model(grid)

        print("Calculando métricas...")
        metrics = [
            self.compute_metrics("train", y_train, grid.predict(x_train)),
            self.compute_metrics("test", y_test, grid.predict(x_test)),
            self.compute_conf_matrix("train", y_train, grid.predict(x_train)),
            self.compute_conf_matrix("test", y_test, grid.predict(x_test)),
        ]

        print("Guardando métricas...")
        self.save_metrics(metrics)

        print("Proceso finalizado exitosamente.")
        print(f"Mejor precisión balanceada: {grid.best_score_:.4f}")
        print(f"Hiperparámetros óptimos: {grid.best_params_}")


if __name__ == "__main__":
    CreditDefaultModel().execute()