import os
from datetime import datetime


# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
LABEL = "label"
TWEET = "tweet"
MODEL_NAME = "model.h5"
TOKENIZER_NAME = "tokenizer.pickle"
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Local data source
DATA_SOURCE_PATH = os.path.join("Data", "labeled_data.csv")

# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_FILE_NAME = "labeled_data.csv"

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME = "final.csv"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMNS = ["count", "hate_speech", "offensive_language", "neither"]
CLASS = "class"


# Model training constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = "trained_model"
TRAINED_MODEL_NAME = MODEL_NAME
X_TEST_FILE_NAME = "x_test.csv"
Y_TEST_FILE_NAME = "y_test.csv"
X_TRAIN_FILE_NAME = "x_train.csv"

RANDOM_STATE = 42
EPOCH = 3
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# Model architecture constants
MAX_WORDS = 50000
MAX_LEN = 300
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
ACTIVATION = "sigmoid"


# Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
BEST_MODEL_DIR = "best_Model"
MODEL_EVALUATION_FILE_NAME = "metrics.csv"

# Model pusher constants
SAVED_MODEL_DIR = os.path.join("saved_model")
SAVED_MODEL_NAME = MODEL_NAME