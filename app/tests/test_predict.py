from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def sample_input():
    return {
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


def test_predict_success():
    response = client.post("/api/v1/predict/", json=sample_input())

    assert response.status_code == 200

    data = response.json()

    assert "death_risk" in data
    assert 0 <= data["death_risk"] <= 1


def test_predict_invalid_input():
    bad_input = sample_input()
    bad_input["age"] = -10  # invalid

    response = client.post("/api/v1/predict/", json=bad_input)

    assert response.status_code == 422  # validation error