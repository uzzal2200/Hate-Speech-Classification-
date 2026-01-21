import os
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import TWEET, LABEL
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.ml.model import ModelArchitecture
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def split_data(self, csv_path):
        try:
            logging.info("Reading transformed data for train/test split")
            df = pd.read_csv(csv_path, index_col=False)
            x = df[TWEET].fillna("")
            y = df[LABEL]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=self.model_trainer_config.RANDOM_STATE,
                stratify=y,
            )
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenize(self, x_train):
        try:
            texts = x_train.fillna("").astype(str)
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer")
        try:
            x_train, x_test, y_train, y_test = self.split_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            sequences_matrix, tokenizer = self.tokenize(x_train)

            logging.info("Starting model training")
            model.fit(
                sequences_matrix,
                y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
                verbose=1,
            )

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            # Save tokenizer
            with open(self.model_trainer_config.TOKENIZER_PATH, "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model and splits
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH, index=False)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH, index=False)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH, index=False)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                tokenizer_path=self.model_trainer_config.TOKENIZER_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
                x_train_path=self.model_trainer_config.X_TRAIN_DATA_PATH,
            )
            logging.info("Model training complete")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
