# ❤️ Heart Failure ML System

A production-style machine learning system to predict heart failure risk using clinical data, with explainability and API deployment.

---

## 👨‍💻 Author

**Gopal Gupta**
🔗 GitHub: https://github.com/gopal092003/heart-failure-ml-system

---

## 🚀 Features

* 📊 End-to-end ML pipeline (data → model → evaluation)
* 🔍 Explainable predictions using SHAP
* 🌐 REST API built with FastAPI
* 🧪 Unit + integration + API tests
* 📦 Dockerized for deployment
* 📈 Monitoring & drift detection
* 🖥️ Optional Streamlit dashboard

---

## 🧠 Tech Stack

* Python, Pandas, NumPy
* Scikit-learn
* SHAP
* FastAPI
* Streamlit
* Docker

---

## 📁 Project Structure

```bash
heart-failure-ml-system/
│
├── src/              # ML system (pipelines, models, features)
├── app/              # FastAPI application
├── data/             # datasets
├── models/           # trained models + artifacts
├── scripts/          # CLI scripts
├── tests/            # unit + integration tests
├── dashboard/        # Streamlit UI (optional)
├── config/           # YAML configs
```

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone https://github.com/gopal092003/heart-failure-ml-system.git
cd heart-failure-ml-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train Model

```bash
python scripts/train.py
```

This will:

* train the model
* save artifacts
* generate SHAP plots

---

## 🌐 Run API

```bash
python run_api.py
```

Open:

* Swagger Docs → http://localhost:8000/docs
* API Base → http://localhost:8000/api/v1

---

## 🔍 Example API Request

### Predict

```json
POST /api/v1/predict/

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

## 🧪 Run Tests

```bash
pytest
```

---

## 🐳 Run with Docker

```bash
docker build -t heart-failure-api -f app/Dockerfile .
docker run -p 8000:8000 heart-failure-api
```

---

## 🖥️ Dashboard (Optional)

```bash
streamlit run dashboard/app.py
```

---

## 📊 Dataset

Heart Failure Clinical Records Dataset
(Source: Kaggle)

---

## 🔐 Environment Variables

Configured via `.env`

---

## 🧭 Roadmap

* [ ] Model versioning
* [ ] Cloud deployment (AWS/GCP)
* [ ] CI/CD pipeline
* [ ] Authentication & rate limiting

---

## 📄 License

This project is for educational and demonstration purposes.

---

## ⭐ Acknowledgements

* Scikit-learn
* SHAP
* FastAPI

---

## 💡 Final Note

This is not just a model—it’s a **complete ML system** with:

* training pipeline
* explainability
* API
* deployment setup

---
