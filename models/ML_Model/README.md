# Breast Cancer Recurrence Prediction - ML Model

## Overview

This project implements a classical machine learning model to predict breast cancer recurrence using the METABRIC dataset.
It includes all steps from data preprocessing to model training, evaluation, and inference.

## Features and Targets

### Features (used for training)

- Age at diagnosis
- Tumor size
- Tumor stage
- Lymph nodes positive
- ER / PR / HER2
- PAM50 subtype
- Nottingham Prognostic Index
- Treatment (Chemo / Radio / Hormone)

### Targets (not allowed in training)

- Overall survival months
- Disease-free survival
- Vital status

## Folder Structure

### `data/`

- `train.csv` : Training dataset including only allowed features
- `test.csv` : Testing dataset including only allowed features

### `preprocessing/`

- `preprocessing.py` : Data cleaning, feature engineering, and scaling
- `__init__.py` : Makes the folder a Python package

### `models/`

- `breast_cancer_model.pkl` : The trained ML model (RandomForest or XGBoost)

### `training/`

- `train.py` : Script to train the ML model on preprocessed data

### `evaluation/`

- `evaluate.py` : Script to evaluate model performance using metrics such as accuracy, F1-score, and recall

### `inference/`

- `predict.py` : Script to make predictions on new patient data using the trained model

## How to Use

1. Prepare your dataset with only the allowed features and put it in the `data/` folder.
2. Run `preprocessing/preprocessing.py` to clean and prepare data.
3. Run `training/train.py` to train the model.
4. Run `evaluation/evaluate.py` to evaluate the model's performance.
5. Run `inference/predict.py` to make predictions on new data.

## Notes

- Ensure you have all required libraries installed (`pandas`, `scikit-learn`, etc.).
- This structure keeps the project clean, organized, and ready for expansion in the future.
