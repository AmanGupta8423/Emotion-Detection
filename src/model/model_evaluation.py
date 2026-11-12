import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import logging
import os

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    'Load the trained model from a pickle file.'
    try:
        model = pickle.load(open(model_path, 'rb'))
        logger.debug('Model loaded successfully from %s', model_path)
        return model
    except FileNotFoundError as e:
        logger.error('Model file not found: %s', model_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', model_path)
        raise

def load_data(file_path : str) -> pd.DataFrame:
    'Load data from csv'
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing CSV file: %s', file_path)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data: %s', file_path)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    'Evaluate the model and return evaluation metrics.'
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dic = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        logger.debug('Model evaluation metrics calculated successfully.')
        return metrics_dic
    except Exception as e:
        logger.error('Error occurred during model evaluation: %s', e)
        raise

def save_metrics(metrics_dic: dict, file_path: str) -> None:
    'Save the evaluation metrics to a JSON file.'
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics_dic, f, indent=4)
        logger.debug('Evaluation metrics saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving metrics: %s', e)
        raise

def main():
    try:
        model = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model,X_test,y_test)

        save_metrics(metrics,'./reports/metrics.json')
        logger.info('Model Evaluation successfully.')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

    
if __name__ == "__main__":
    main()