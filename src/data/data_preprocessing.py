import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

logger= logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error('Error occurred while lemmatizing text: %s', e)
        raise

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error('Error occurred while removing stop words: %s', e)
        raise

def removing_numbers(text: str) -> str:
    """Remove numbers from the text."""
    try:
        text = ''.join([char for char in text if not char.isdigit()])
        return text
    except Exception as e:
        logger.error('Error occurred while removing numbers: %s', e)
        raise

def lower_case(text: str) -> str:
    """Convert text to lower case."""
    try:
        text = text.split()
        text = [word.lower() for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error('Error occurred while converting to lower case: %s', e)
        raise

def removing_punctuations(text: str) -> str:
    """Remove punctuations from the text."""
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error('Error occurred while removing punctuations: %s', e)
        raise

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error('Error occurred while removing URLs: %s', e)
        raise

def remove_small_sentences(df: pd.DataFrame) -> None:
    """Remove sentences with less than 3 words."""
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logger.error('Error occurred while removing small sentences: %s', e)
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the text data in the dataframe."""
    try:
        df['content'] = df.content.apply(lambda content : lower_case(content))
        logger.debug('Lowercasing completed.')
        df['content'] = df['content'].apply(lambda content : remove_stop_words(content))
        logger.debug('Stop words removal completed.')
        df['content'] = df['content'].apply(lambda content : removing_numbers(content))
        logger.debug('Number removal completed.')
        df['content'] = df['content'].apply(lambda content : removing_punctuations(content))
        logger.debug('Punctuation removal completed.')
        df['content'] = df['content'].apply(lambda content : removing_urls(content))
        logger.debug('URL removal completed.')
        df['content'] = df['content'].apply(lambda content : lemmatization(content))
        logger.debug('Lemmatization completed.')
        return df
    except Exception as e:
        logger.error('Error occurred while normalizing text: %s', e)
        raise
    
def main():
    """Main function to normalize text data and save processed data."""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully.')
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        logger.debug('Text normalization completed successfully.')

        data_path = os.path.join('data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'))
        logger.debug('Processed train data saved successfully.')
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'))
        logger.debug('Processed test data saved successfully.')
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while loading raw data: %s', e)
        raise

if __name__ == "__main__":
    main()