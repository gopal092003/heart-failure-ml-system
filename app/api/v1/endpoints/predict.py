from fastapi import APIRouter, Depends
from app.schemas.prediction import PatientData
from app.api.deps import get_model, get_feature_order, get_schema
from src.models.predict import prepare_input, predict_proba
from src.validation.input_validation import full_input_validation

router = APIRouter()


@router.post("/")
def predict(
    data: PatientData,
    model=Depends(get_model),
    feature_order=Depends(get_feature_order),
    schema=Depends(get_schema)
):
    input_data = data.dict()

    # Validation
    full_input_validation(input_data, feature_order, schema)

    # Prepare input
    X = prepare_input(input_data, feature_order)

    # Predict
    prob = predict_proba(model, X)[0]

    return {
        "death_risk": float(prob)
    }