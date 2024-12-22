import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=5,error_score=np.nan)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            report[model_name] = {
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "best_params": gs.best_params_
            }

        # Select the best model based on test R-squared score
        best_model_name = max(report, key=lambda x: report[x]["test_r2"])
        print(f"Best model: {best_model_name}")
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    