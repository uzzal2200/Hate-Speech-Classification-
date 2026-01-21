# 🛡️ Hate Speech Classification

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.2-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Research](https://img.shields.io/badge/Research-NLP-blueviolet)](https://github.com)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%2FLSTM-red)](https://github.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io)

</div>

A deep learning-based Natural Language Processing research project for detecting hate speech and abusive content in text. This project uses TensorFlow for model training and provides both a FastAPI REST API and a Streamlit web interface for predictions.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Author](#author)
- [License](#license)

## 🎯 Project Overview

This project implements a hate speech classification system that can analyze text data and classify it into different categories of hate speech or abusive content. The system is built using a modular architecture with separate components for data ingestion, transformation, model training, evaluation, and deployment.

## ✨ Features

- **Modular Architecture**: Organized into components for easy maintenance and scalability
- **Data Pipeline**: Automated data ingestion and transformation pipelines
- **Deep Learning Model**: TensorFlow-based neural network for text classification
- **Model Evaluation**: Comprehensive evaluation metrics and model performance tracking
- **REST API**: FastAPI-based API for programmatic access
- **Web Interface**: Interactive Streamlit application for easy predictions
- **Artifact Management**: Organized storage of training artifacts and model versions
- **Docker Support**: Containerized deployment ready

## 📁 Project Structure

```
Hate-Speech-Classification/
│
├── app.py                      # Main training script
├── demo.py                     # Demo/testing script
├── streamlit_app.py            # Streamlit web interface
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup file
├── template.py                 # Project template generator
│
├── Data/                       # Raw data directory
│   └── labeled_data.csv        # Dataset
│
├── artifacts/                  # Training artifacts
│   └── [timestamp]/            # Timestamped runs
│       ├── DataIngestionArtifacts/
│       └── DataTransformationArtifacts/
│
├── hate/                       # Main package
│   ├── components/             # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/          # Configuration modules
│   │   └── gcloud_syncer.py
│   │
│   ├── constants/              # Constants and configurations
│   │   └── __init__.py
│   │
│   ├── entity/                 # Data entities
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   │
│   ├── exception/              # Custom exceptions
│   │   └── __init__.py
│   │
│   ├── logger/                 # Logging utilities
│   │   └── __init__.py
│   │
│   ├── ml/                     # ML model modules
│   │   └── model.py
│   │
│   └── pipeline/               # Training and prediction pipelines
│       ├── train_pipeline.py
│       └── prediction_pipeline.py
│
├── logs/                       # Application logs
└── notebook/                   # Jupyter notebooks
    └── Expirement.ipynb        # Experimental analysis
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Hate-Speech-Classification
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
conda create -n hate python=3.8.20 -y

conda activate hate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install the Package
```bash
pip install -e .
```

## 💻 Usage

### Model Training

To train the model from scratch, run:
```bash
python app.py
```

This will:
1. Ingest data from the `Data/` directory
2. Transform and preprocess the data
3. Train the neural network model
4. Evaluate model performance
5. Save artifacts to the `artifacts/` directory

### Prediction

#### Using Streamlit Web App
```bash
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501` to access the interactive interface.


#### Programmatic Prediction
```python
from hate.pipeline.prediction_pipeline import PredictionPipeline

predictor = PredictionPipeline()
text = "Your text here"
prediction = predictor.predict(text)
print(f"Prediction: {prediction}")
```

## 👤 Author

**MD UZZAL MIA**
- Email: uzzal.220605@s.pust.ac.bd

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- Dataset: `labeled_data.csv` containing labeled hate speech samples
- Built as part of a Natural Language Processing research project

## 📝 Notes

- Ensure that the `Data/labeled_data.csv` file is present before training
- Model artifacts are saved with timestamps for version control
- Training logs are stored in the `logs/` directory
- For production deployment, consider using the provided Dockerfile

## 🐛 Issues and Contributions

For issues, questions, or contributions, please contact the author or create an issue in the repository.

---

**Happy Coding! 🚀**