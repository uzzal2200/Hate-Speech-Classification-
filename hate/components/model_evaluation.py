import os
import sys
import json
import keras
import pickle
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score
from hate.constants import MAX_LEN
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def evaluate(self):
        try:
            logging.info("Loading test split for evaluation")
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            with open(self.model_trainer_artifacts.tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)

            model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test.squeeze().astype(str)
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            probs = model.predict(test_sequences_matrix)
            preds = (probs > 0.5).astype(int).flatten()

            accuracy = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds).tolist()
            logging.info(f"Test accuracy: {accuracy:.4f}")
            return accuracy, cm
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Initiate Model Evaluation")
        try:
            accuracy, cm = self.evaluate()

            os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_MODEL_DIR, exist_ok=True)
            metrics_payload = {"accuracy": accuracy, "confusion_matrix": cm}
            with open(self.model_evaluation_config.METRICS_FILE_PATH, "w") as f:
                json.dump(metrics_payload, f, indent=2)

            model_evaluation_artifacts = ModelEvaluationArtifacts(accuracy=accuracy, is_model_accepted=True)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e