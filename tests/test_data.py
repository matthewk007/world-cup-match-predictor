import pandas as pd
import pytest

import data


def test_derive_match_result_handles_scores_and_penalties() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Argentina",
                "away_team": "France",
                "home_score": 3,
                "away_score": 3,
                "home_penalty": 4,
                "away_penalty": 2,
                "Host": "Qatar",
                "Year": 2022,
            },
            {
                "home_team": "Spain",
                "away_team": "Germany",
                "home_score": 1,
                "away_score": 1,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
            {
                "home_team": "Brazil",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 2,
                "home_penalty": None,
                "away_penalty": None,
                "Host": "Qatar",
                "Year": 2022,
            },
        ]
    )

    labels = [data.derive_match_result(row) for _, row in frame.iterrows()]

    assert labels == ["home_win", "draw", "away_win"]


def test_prepare_training_frame_adds_ranking_and_drops_leakage_columns() -> None:
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
            }
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
        ]
    )

    prepared = data.prepare_training_frame(matches, ranking)

    assert prepared.target.tolist() == ["home_win"]
    assert "home_score" not in prepared.features.columns
    assert "away_score" not in prepared.features.columns
    assert prepared.features.loc[0, "home_rank"] == 13
    assert prepared.features.loc[0, "away_rank"] == 28
    assert prepared.features.loc[0, "home_team"] == "United States"
    assert prepared.features.loc[0, "away_team"] == "Korea Republic"
    assert prepared.features.loc[0, "home_association"] == "CONCACAF"
    assert prepared.features.loc[0, "away_association"] == "AFC"


def test_build_prediction_frame_matches_training_schema() -> None:
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
            }
        ]
    )
    match = data.build_prediction_frame("United States", "Korea Republic", "Qatar", 2026, ranking)

    assert list(match.columns) == [
        "home_team",
        "away_team",
        "Host",
        "Year",
        "home_rank",
        "home_previous_rank",
        "home_points",
        "home_previous_points",
        "home_association",
        "away_rank",
        "away_previous_rank",
        "away_points",
        "away_previous_points",
        "away_association",
    ]


def test_prepare_training_frame_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "Argentina",
                "away_team": "France",
                "home_score": 3,
                "away_score": 3,
                "home_penalty": 4,
                "away_penalty": 2,
                "Host": "Qatar",
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing required columns: Year"):
        data.prepare_training_frame(frame)
