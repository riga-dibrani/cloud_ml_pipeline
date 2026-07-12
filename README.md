# Cloud-Based Machine Learning Pipeline for Predictive Maintenance

## Overview

This project implements a cloud-based machine learning pipeline for predictive maintenance using Remaining Useful Life (RUL) prediction. The goal is to develop a complete end-to-end machine learning workflow that processes sensor data from industrial equipment, performs feature engineering, trains a predictive model and evaluates the model's ability to estimate the remaining operational lifetime of a system.

---

## Problem Description

Predictive maintenance aims to reduce unexpected equipment failures by estimating the future condition of machinery based on historical operational data. A key task in predictive maintenance is Remaining Useful Life (RUL) prediction, which estimates how many operational cycles remain before equipment failure.

This project addresses this problem by applying machine learning techniques to time-series sensor data collected from turbofan engines.

---

## Dataset

The project uses the **NASA C-MAPSS FD001 Turbofan Engine Degradation Dataset**.

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) provides simulated engine degradation data containing multiple sensor measurements recorded throughout engine operation.

Dataset characteristics:
- Dataset subset: FD001
- Operating condition: Single operating condition
- Failure mode: High-pressure compressor (HPC) degradation
- Training trajectories: 100 engines
- Testing trajectories: 100 engines
- Sensor measurements: 21

The dataset contains:
- Training data with complete degradation histories
- Test data with partial operational histories
- True RUL values for test engines

---

## Machine Learning Pipeline

The implemented pipeline consists of the following stages:

### 1. Data Ingestion
- Loading raw sensor data and RUL information
- Assigning meaningful column names
- Preparing structured datasets for processing

### 2. Data Preprocessing
- Handling time-series engine data
- Calculating Remaining Useful Life values
- Applying RUL capping to improve model stability
- Scaling numerical features using StandardScaler

### 3. Feature Engineering
Additional temporal features were created to capture degradation patterns:

- Rolling mean features
- Rolling standard deviation features
- Sensor trend features
- Delta/change-based features

These features help the model identify changes in equipment behavior over time.

### 4. Model Training

A Random Forest Regression model was implemented as the predictive model.

Model characteristics:
- Algorithm: Random Forest Regressor
- Number of estimators: 100
- Maximum depth: 10
- Validation strategy: GroupShuffleSplit based on engine IDs

Grouping by engine prevents data leakage between training and validation samples from the same engine.

### 5. Model Evaluation

The model was evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Final validation results:

| Metric | Value |
|--------|-------|
| RMSE | 18.01 |
| MAE | 13.06 |
| R² Score | 0.83 |

---

## Cloud Implementation

The pipeline was extended to support cloud-based execution using Amazon Web Services.

Implemented cloud components:

- **Amazon S3**
  - Storage of raw and processed datasets
  - Management of machine learning artifacts

- **Amazon SageMaker**
  - Cloud-based development environment
  - Pipeline execution and model experimentation

The cloud implementation enables a more scalable workflow by separating data storage, processing, and model development.

---

## Technologies Used

### Programming Languages
- Python

### Machine Learning & Data Science
- Scikit-learn
- NumPy
- Pandas
- SciPy
- Matplotlib

### Cloud & Data Platforms
- AWS S3
- AWS SageMaker

### Development Tools
- Git
- Jupyter Notebook

---

## Repository Structure
The project follows a modular machine learning pipeline structure:

cloud-ml-pipeline/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ models/
|
├─ data/
│  ├─ raw/
│  └─ preprocessed/
│  └─ processed/
|
├─ notebooks/
|
├─ src/
│  ├─ ingest.py
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ infer.py
│  └─ run_pipeline.py


The source code is organized into independent modules for:
- data ingestion
- preprocessing
- feature engineering
- model training
- evaluation
- inference
- pipeline execution

---

## Future Improvements

Possible extensions of this project include:

- Testing advanced time-series models such as LSTM or Transformer-based architectures
- Incorporating additional C-MAPSS datasets with multiple operating conditions
- Improving uncertainty estimation for RUL predictions
- Deploying the trained model as a cloud-based prediction service

---

## Author

Riga Dibrani  
Bachelor Thesis Project  
Computer Engineering
