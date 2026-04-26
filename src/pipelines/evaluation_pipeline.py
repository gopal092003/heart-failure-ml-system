from src.utils.config_loader import load_config
from src.models.train import load_model
from src.models.evaluate import evaluate_model, save_metrics
from src.visualization.evaluation_plots import plot_roc_curve, plot_confusion_matrix


def run_evaluation_pipeline():
    config = load_config()

    # Load model
    model = load_model("models/trained/model_v1.pkl")

    # Load processed test data (you can extend this)
    import pandas as pd
    df_test = pd.read_parquet("data/processed/test.parquet")

    X_test = df_test.drop(config["data"]["target_column"], axis=1)
    y_test = df_test[config["data"]["target_column"]]

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    save_metrics(metrics, "reports/evaluation/metrics.json")

    # Plots
    plot_roc_curve(y_test, y_prob, "reports/evaluation/roc_curve.png")
    plot_confusion_matrix(model, X_test, y_test, "reports/evaluation/confusion_matrix.png")

    print("✅ Evaluation pipeline completed!")


if __name__ == "__main__":
    run_evaluation_pipeline()