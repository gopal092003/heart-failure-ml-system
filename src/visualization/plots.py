import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_plot(fig, path: str):
    """
    Save matplotlib figure.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution(df, column: str, path: str):
    """
    Plot distribution of a feature.
    """
    fig, ax = plt.subplots()

    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")

    save_plot(fig, path)


def plot_correlation_matrix(df, path: str):
    """
    Plot correlation heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    ax.set_title("Correlation Matrix")

    save_plot(fig, path)