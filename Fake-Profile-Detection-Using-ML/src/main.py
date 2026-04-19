from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "results"
GENUINE_USERS_FILE = DATA_DIR / "users.csv"
FAKE_USERS_FILE = DATA_DIR / "fusers.csv"


def read_datasets() -> tuple[pd.DataFrame, pd.Series]:
    genuine_users = pd.read_csv(GENUINE_USERS_FILE)
    fake_users = pd.read_csv(FAKE_USERS_FILE)

    combined = pd.concat([genuine_users, fake_users], ignore_index=True)
    labels = pd.Series(
        [1] * len(genuine_users) + [0] * len(fake_users),
        name="is_genuine",
    )
    return combined, labels


def encode_language_column(df: pd.DataFrame) -> pd.Series:
    categories = sorted(df["lang"].fillna("unknown").unique())
    language_lookup = {name: index for index, name in enumerate(categories)}
    return df["lang"].fillna("unknown").map(language_lookup).astype(int)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = pd.DataFrame(
        {
            "statuses_count": df["statuses_count"].fillna(0),
            "followers_count": df["followers_count"].fillna(0),
            "friends_count": df["friends_count"].fillna(0),
            "favourites_count": df["favourites_count"].fillna(0),
            "listed_count": df["listed_count"].fillna(0),
            "lang_code": encode_language_column(df),
        }
    )
    return feature_frame.astype(float)


def plot_learning_curve(
    estimator: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=5,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    plt.figure(figsize=(8, 5))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training")
    plt.plot(
        train_sizes,
        np.mean(validation_scores, axis=1),
        "o-",
        label="Cross-validation",
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix_figure(y_test: pd.Series, y_pred: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Fake", "Genuine"],
        cmap=plt.cm.Blues,
        ax=axis,
        colorbar=False,
    )
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def plot_roc_curve_figure(y_test: pd.Series, y_scores: np.ndarray) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_features, labels = read_datasets()
    feature_frame = extract_features(raw_features)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        labels,
        test_size=0.2,
        random_state=44,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=40,
        random_state=44,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("Fake Profile Detection Using Machine Learning")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Cross-validation mean score: {cv_scores.mean():.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, target_names=["Fake", "Genuine"]))

    plot_learning_curve(model, X_train, y_train)
    plot_confusion_matrix_figure(y_test, predictions)
    plot_roc_curve_figure(y_test, probabilities)


if __name__ == "__main__":
    main()
