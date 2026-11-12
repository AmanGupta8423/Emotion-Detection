import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    'Load parameters from a YAML file.'
    try:
        test_size = yaml.safe_load(open(params_path,'r'))['data_ingestion']['test_size']
        logger.debug(f"Test size loaded: {test_size}")
        return test_size
    except Exception as e:
        logger.error(f"Error occurred while loading parameters: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    """Read data from a CSV file."""
    try:
        df = pd.read_csv(url)
        logger.debug(f"Data read successfully from {url}")
        return df
    except Exception as e:
        logger.error(f"Error occurred while reading data from {url}: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    'Process the data by filtering and mapping sentiment values.'
    try:
        df.drop(columns = ['tweet_id'], inplace = True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        return final_df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def split_data(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    'Split the data into training and testing sets.'
    train_data , test_data = train_test_split(df, test_size = test_size, random_state = 42)
    return train_data, test_data

def save_path(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    'Save the training and testing data to CSV files.'
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path,'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'), index=False)
        logger.debug('Train and test data saved to %s', data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = split_data(final_df, test_size)
        data_path = os.path.join('data', 'raw')
        save_path(data_path, train_data, test_data)
        print("Data ingestion completed successfully!")

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
