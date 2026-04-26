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


def test_explain_success():
    response = client.post("/api/v1/explain/", json=sample_input())

    assert response.status_code == 200

    data = response.json()

    assert "death_risk" in data
    assert "explanation" in data

    assert isinstance(data["explanation"], dict)


def test_explain_feature_count():
    response = client.post("/api/v1/explain/", json=sample_input())

    data = response.json()

    # should match number of features
    assert len(data["explanation"]) == 11