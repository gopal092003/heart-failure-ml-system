from typing import Dict

from src.models.predict import prepare_input, predict_proba
from src.validation.input_validation import full_input_validation


class ModelService:
    def __init__(self, model, feature_order, schema):
        self.model = model
        self.feature_order = feature_order
        self.schema = schema

    def predict(self, input_data: Dict) -> float:
        """
        Full prediction pipeline.
        """
        # Validate input
        full_input_validation(input_data, self.feature_order, self.schema)

        # Prepare input
        X = prepare_input(input_data, self.feature_order)

        # Predict probability
        prob = predict_proba(self.model, X)[0]

        return float(prob)