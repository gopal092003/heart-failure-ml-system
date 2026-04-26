from typing import Dict

from src.models.predict import prepare_input, predict_proba
from src.validation.input_validation import full_input_validation


class ExplainService:
    def __init__(self, model, explainer, feature_order, schema):
        self.model = model
        self.explainer = explainer
        self.feature_order = feature_order
        self.schema = schema

    def explain(self, input_data: Dict) -> Dict:
        """
        Prediction + SHAP explanation.
        """
        # Validate input
        full_input_validation(input_data, self.feature_order, self.schema)

        # Prepare input
        X = prepare_input(input_data, self.feature_order)

        # Prediction
        prob = predict_proba(self.model, X)[0]

        # SHAP values
        shap_values = self.explainer.shap_values(X)[1][0]

        explanation = dict(
            zip(self.feature_order, shap_values.tolist())
        )

        return {
            "death_risk": float(prob),
            "explanation": explanation
        }