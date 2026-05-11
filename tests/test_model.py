import pandas as pd
import pytest

import data
import model


def _build_training_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.DataFrame(
        [
            {
                "home_team": "United States",
                "away_team": "Korea Republic",
                "home_score": 2,
                "away_score": 1,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2026,
            },
            {
                "home_team": "France",
                "away_team": "Brazil",
                "home_score": 1,
                "away_score": 1,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
            {
                "home_team": "Croatia",
                "away_team": "Argentina",
                "home_score": 1,
                "away_score": 2,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
            {
                "home_team": "United States",
                "away_team": "Korea Republic",
                "home_score": 1,
                "away_score": 3,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "USA",
                "Year": 1994,
            },
            {
                "home_team": "France",
                "away_team": "Brazil",
                "home_score": 3,
                "away_score": 3,
                "home_penalty": 4,
                "away_penalty": 2,
                "Host": "France",
                "Year": 1998,
            },
            {
                "home_team": "Brazil",
                "away_team": "France",
                "home_score": 0,
                "away_score": 1,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "France",
                "Year": 1998,
            },
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "team": "USA",
                "team_code": "USA",
                "association": "CONCACAF",
                "rank": 13,
                "previous_rank": 14,
                "points": 1650.0,
                "previous_points": 1640.0,
            },
            {
                "team": "KOR",
                "team_code": "KOR",
                "association": "AFC",
                "rank": 28,
                "previous_rank": 27,
                "points": 1500.0,
                "previous_points": 1495.0,
            },
            {
                "team": "France",
                "team_code": "FRA",
                "association": "UEFA",
                "rank": 3,
                "previous_rank": 2,
                "points": 1830.0,
                "previous_points": 1820.0,
            },
            {
                "team": "Brazil",
                "team_code": "BRA",
                "association": "CONMEBOL",
                "rank": 1,
                "previous_rank": 1,
                "points": 1840.0,
                "previous_points": 1837.0,
            },
            {
                "team": "Croatia",
                "team_code": "CRO",
                "association": "UEFA",
                "rank": 7,
                "previous_rank": 8,
                "points": 1710.0,
                "previous_points": 1700.0,
            },
            {
                "team": "Argentina",
                "team_code": "ARG",
                "association": "CONMEBOL",
                "rank": 2,
                "previous_rank": 3,
                "points": 1780.0,
                "previous_points": 1773.0,
            },
        ]
    )
    return matches, ranking


def test_train_model_returns_metrics_report_and_confusion_matrix() -> None:
    matches, ranking = _build_training_dataset()
    prepared = data.prepare_training_frame(matches, ranking)

    artifacts = model.train_model(prepared.features, prepared.target)

    assert set(artifacts.classes) == set(data.TARGET_VALUES)
    assert artifacts.feature_columns == tuple(prepared.features.columns)
    assert 0.0 <= artifacts.metrics["accuracy"] <= 1.0
    assert 0.0 <= artifacts.metrics["macro_f1"] <= 1.0
    assert artifacts.metrics["train_rows"] + artifacts.metrics["test_rows"] == len(prepared.features)
    assert artifacts.confusion.shape == (3, 3)
    assert "precision" in artifacts.report_text
    assert "recall" in artifacts.report_text


def test_predict_with_explanation_returns_ranked_factors() -> None:
    matches, ranking = _build_training_dataset()
    prepared = data.prepare_training_frame(matches, ranking)
    artifacts = model.train_model(prepared.features, prepared.target)
    match = data.build_prediction_frame("United States", "Korea Republic", "Qatar", 2026, ranking)

    outcome = model.predict_with_explanation(artifacts, match, top_n=3)

    assert outcome.label in data.TARGET_VALUES
    assert set(outcome.probabilities) == set(artifacts.classes)
    assert pytest.approx(sum(outcome.probabilities.values()), rel=1e-6, abs=1e-6) == 1.0
    assert len(outcome.factors) == 3
    assert all(set(factor) == {"feature", "contribution", "direction"} for factor in outcome.factors)
    assert all(factor["feature"] for factor in outcome.factors)
    assert all(factor["direction"] in {"supports", "pushes away"} for factor in outcome.factors)
    magnitudes = [abs(float(factor["contribution"])) for factor in outcome.factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_train_model_rejects_single_class_training_data() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Argentina",
                "away_team": "France",
                "home_score": 3,
                "away_score": 1,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
            {
                "home_team": "Brazil",
                "away_team": "Croatia",
                "home_score": 2,
                "away_score": 0,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
        ]
    )

    prepared = data.prepare_training_frame(frame)

    with pytest.raises(ValueError, match="at least two outcome classes"):
        model.train_model(prepared.features, prepared.target)
