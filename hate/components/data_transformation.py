import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

# Ensure stopwords are available once
nltk.download("stopwords", quiet=True)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/urls/numbers, drop stopwords, and stem."""
    stemmer = nltk.SnowballStemmer("english")
    stopword = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    tokens = [word for word in text.split() if word not in stopword]
    stems = [stemmer.stem(word) for word in tokens]
    return " ".join(stems)


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        logging.info("Entered the initiate_data_transformation method of DataTransformation class")
        try:
            raw_data_path = self.data_ingestion_artifacts.raw_data_path
            df = pd.read_csv(raw_data_path)

            # Drop noisy columns when present
            columns_to_drop = [col for col in self.data_transformation_config.DROP_COLUMNS if col in df.columns]
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)

            # Map classes to binary label: 1 = hate/offensive, 0 = neutral
            df[self.data_transformation_config.CLASS] = df[self.data_transformation_config.CLASS].replace({0: 1, 1: 1, 2: 0})
            df.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL}, inplace=True)

            # Clean tweets
            df[self.data_transformation_config.TWEET] = (
                df[self.data_transformation_config.TWEET]
                .fillna("")
                .astype(str)
                .apply(clean_text)
            )

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("Completed data transformation")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    