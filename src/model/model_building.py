import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import yaml
import logging
import os

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    'Load Parameters from a YAML file.'
    try:
        params = yaml.safe_load(open(params_path,'r'))
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError as e:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', params_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred: %s', params_path)
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

def train_model(X_train:np.ndarray, y_train:np.ndarray, params: dict) -> xgb.XGBClassifier:
    'Train the XGBOOST model'
    try:
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                        n_estimators=params['n_estimators'], max_depth = params['max_depth'], learning_rate=params['learning_rate'])

        clf.fit(X_train,y_train)
        logger.debug('Model trained successfully')
        return clf
    except Exception as e:
        logger.error('Error occurred while training the model: %s', e)
        raise

def save_model(model, file_path:str)-> None:
    'Save the model to a file'
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        save_model(clf, 'models/model.pkl')

        logger.debug('Model build Successfully.')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()