# Heart Failure ML System

A machine learning system for predicting the risk of heart failure using clinical data. The project combines a reproducible training pipeline, model explainability with SHAP, a FastAPI inference service, and optional deployment components such as Docker and a Streamlit dashboard.

---

## Overview

This project demonstrates how a machine learning model can be developed and deployed as a production-style application. It includes data preprocessing, model training, explainability, API serving, testing, and deployment support within a modular project structure.

---

## Features

- End-to-end machine learning pipeline
- Heart failure risk prediction
- Model explainability using SHAP
- REST API built with FastAPI
- Unit, integration, and API tests
- Docker support
- Optional Streamlit dashboard
- Monitoring and drift detection utilities

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- FastAPI
- Streamlit
- Docker

---

## Project Structure

```
heart-failure-ml-system/
│
├── app/               # FastAPI application
├── config/            # Configuration files
├── dashboard/         # Streamlit dashboard
├── data/              # Dataset
├── models/            # Trained models and artifacts
├── scripts/           # Training and utility scripts
├── src/               # ML pipeline and feature engineering
├── tests/             # Unit and integration tests
│
├── requirements.txt
└── README.md
```

---

## Workflow

The project follows a complete machine learning workflow:

1. Load and preprocess clinical data.
2. Engineer and transform features.
3. Train the prediction model.
4. Evaluate model performance.
5. Generate SHAP explanations.
6. Save the trained model and artifacts.
7. Serve predictions through a FastAPI application.

> Add the workflow diagram here if available.

```md
![Workflow](assets/workflow.png)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gopal092003/heart-failure-ml-system.git

cd heart-failure-ml-system
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training pipeline:

```bash
python scripts/train.py
```

The training process will:

- preprocess the data
- train the model
- evaluate performance
- generate SHAP explanations
- save the trained model and artifacts

---

## Running the API

Start the FastAPI server:

```bash
python run_api.py
```

The API will be available at:

```
http://localhost:8000
```

Interactive documentation:

```
http://localhost:8000/docs
```

API base path:

```
http://localhost:8000/api/v1
```

---

## Example Prediction Request

```http
POST /api/v1/predict
```

```json
{
  "age": 60,
  "anaemia": 0,
  "creatinine_phosphokinase": 250,
  "diabetes": 0,
  "ejection_fraction": 35,
  "high_blood_pressure": 1,
  "platelets": 250000,
  "serum_creatinine": 1.2,
  "serum_sodium": 135,
  "sex": 1,
  "smoking": 0
}
```

---

## Testing

Run all tests with:

```bash
pytest
```

The project includes:

- Unit tests
- Integration tests
- API tests

---

## Docker

Build the Docker image:

```bash
docker build -t heart-failure-api -f app/Dockerfile .
```

Run the container:

```bash
docker run -p 8000:8000 heart-failure-api
```

---

## Streamlit Dashboard

Launch the optional dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Dataset

The project uses the **Heart Failure Clinical Records Dataset**, available on Kaggle.

---

## Configuration

Application settings can be managed through the project configuration files and environment variables.

---

## Future Improvements

- Model versioning
- Hyperparameter optimization
- CI/CD pipeline
- Cloud deployment
- Authentication and rate limiting
- Experiment tracking with MLflow

---

## Author

**Gopal Gupta**

GitHub: https://github.com/gopal092003

---

## License

This project is licensed under the MIT License.
