from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data


@dataclass(frozen=True)
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float | int]
    report_text: str
    confusion: pd.DataFrame
    classes: tuple[str, ...]
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class PredictionArtifacts:
    label: str
    probabilities: dict[str, float]
    factors: list[dict[str, float | str]]


def _split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
    categorical = [column for column in features.columns if column not in numeric]
    return categorical, numeric


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical, numeric = _split_feature_types(features)

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _clean_feature_name(name: str) -> str:
    name = name.replace("categorical__", "").replace("numeric__", "")
    name = name.replace("__", ": ")
    return name.replace("_", " ")


def _class_coefficients(classifier: LogisticRegression, label: str) -> np.ndarray:
    classes = list(classifier.classes_)
    if label not in classes:
        raise ValueError(f"Unknown class '{label}'")

    class_index = classes.index(label)
    if classifier.coef_.shape[0] == 1 and len(classes) == 2:
        # Binary logistic regression stores coefficients for the positive class only.
        return classifier.coef_[0] if class_index == 1 else -classifier.coef_[0]

    return classifier.coef_[class_index]


def _balanced_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(target.dropna().unique()):
        label_indices = np.flatnonzero(target.to_numpy() == label).tolist()
        if not label_indices:
            continue

        rng.shuffle(label_indices)
        if len(label_indices) == 1:
            train_indices.extend(label_indices)
            continue

        desired_test = max(1, int(round(len(label_indices) * test_size)))
        desired_test = min(desired_test, len(label_indices) - 1)
        test_indices.extend(label_indices[:desired_test])
        train_indices.extend(label_indices[desired_test:])

    train_indices = sorted(set(train_indices))
    test_indices = sorted(set(test_indices))

    if not test_indices:
        raise ValueError("Need more than one example per class to create a test split")

    X_train = features.iloc[train_indices].reset_index(drop=True)
    X_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = target.iloc[train_indices].reset_index(drop=True)
    y_test = target.iloc[test_indices].reset_index(drop=True)

    if y_train.nunique() < 2:
        raise ValueError("Need at least two outcome classes in the training split")

    return X_train, X_test, y_train, y_test


def _split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)

    if len(features) < 2:
        raise ValueError("Need at least two rows to train a classifier")

    class_counts = target.value_counts(dropna=False)
    if class_counts.min() >= 2:
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )

    return _balanced_holdout(features, target, test_size=test_size, random_state=random_state)


def train_model(features: pd.DataFrame, target: pd.Series) -> TrainingArtifacts:
    if target.nunique() < 2:
        raise ValueError("Need at least two outcome classes to train a classifier")

    X_train, X_test, y_train, y_test = _split_train_test(features, target)

    pipeline = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    classes = tuple(pipeline.named_steps["classifier"].classes_)
    confusion = confusion_matrix(y_test, y_pred, labels=list(classes))
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(classes),
        target_names=list(classes),
        zero_division=0,
    )

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return TrainingArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        report_text=report_text,
        confusion=pd.DataFrame(confusion, index=classes, columns=classes),
        classes=classes,
        feature_columns=tuple(features.columns),
    )


def explain_prediction(
    pipeline: Pipeline,
    match_frame: pd.DataFrame,
    label: str,
    *,
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(match_frame)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    if dense.ndim == 1:
        dense = dense.reshape(1, -1)

    feature_names = preprocessor.get_feature_names_out()
    coefficients = _class_coefficients(classifier, label)
    contributions = dense[0] * coefficients

    ranked = sorted(
        zip(feature_names, contributions),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:top_n]

    factors: list[dict[str, float | str]] = []
    for feature_name, contribution in ranked:
        value = float(contribution)
        factors.append(
            {
                "feature": _clean_feature_name(str(feature_name)),
                "contribution": value,
                "direction": "supports" if value >= 0 else "pushes away",
            }
        )
    return factors


def predict_with_explanation(
    artifacts: TrainingArtifacts,
    match_frame: pd.DataFrame,
    *,
    top_n: int = 5,
) -> PredictionArtifacts:
    pipeline = artifacts.pipeline
    probabilities = pipeline.predict_proba(match_frame)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    label = classes[int(np.argmax(probabilities))]
    probability_map = {class_name: float(probability) for class_name, probability in zip(classes, probabilities)}
    factors = explain_prediction(pipeline, match_frame, label, top_n=top_n)
    return PredictionArtifacts(label=label, probabilities=probability_map, factors=factors)
