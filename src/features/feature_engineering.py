import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    'Load parameters from a YAML file.'
    try:
        max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
        logger.debug("Max features loaded: %s", max_features)
        return max_features
    except FileNotFoundError as e:
        logger.error("Parameters file not found: %s", e)
        raise 
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while loading parameters: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    'Load data'
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while loading data: %s", e)
        raise


def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    'Apply TF-IDF vectorization to the data'
    try:
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        vectorizer = CountVectorizer(max_features=max_features)
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df['label'] = y_test

        logger.debug("TF-IDF vectorization applied successfully")
        return train_df, test_df
    except Exception as e:
        logger.error("Error occurred while applying TF-IDF: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    'Save the dataframe to a CSV file'
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved successfully to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving data to %s: %s", file_path, e)
        raise

def main():
    try:
        max_features = load_params('params.yaml')
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
        logger.debug("Feature engineering process completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        raise
if __name__ == "__main__":
    main()