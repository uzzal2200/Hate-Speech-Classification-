from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
	raw_data_path: str


@dataclass
class DataTransformationArtifacts:
	transformed_data_path: str


@dataclass
class ModelTrainerArtifacts:
	trained_model_path: str
	tokenizer_path: str
	x_test_path: str
	y_test_path: str
	x_train_path: str


@dataclass
class ModelEvaluationArtifacts:
	accuracy: float
	is_model_accepted: bool


@dataclass
class ModelPusherArtifacts:
	saved_model_path: str