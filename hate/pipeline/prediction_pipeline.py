import os
import sys
import keras
import pickle
from hate.logger import logging
from hate.constants import MAX_LEN, SAVED_MODEL_DIR, SAVED_MODEL_NAME, TOKENIZER_NAME
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.components.data_transformation import clean_text


class PredictionPipeline:
    def __init__(self):
        self.model_dir = SAVED_MODEL_DIR
        self.model_path = os.path.join(self.model_dir, SAVED_MODEL_NAME)
        self.tokenizer_path = os.path.join(self.model_dir, TOKENIZER_NAME)

    def load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Saved model not found. Please run training first.")
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError("Tokenizer not found. Please run training first.")

        model = keras.models.load_model(self.model_path)
        with open(self.tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer

    def predict(self, text: str) -> str:
        logging.info("Running the predict function")
        try:
            model, tokenizer = self.load_artifacts()
            cleaned = clean_text(text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = model.predict(padded)[0][0]
            return "hate and abusive" if pred > 0.5 else "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text: str) -> str:
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            return self.predict(text)
        except Exception as e:
            raise CustomException(e, sys) from e