from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import pandas as pd

REQUIRED_MATCH_COLUMNS = (
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_penalty",
    "away_penalty",
    "Host",
    "Year",
)

BASE_FEATURE_COLUMNS = ("home_team", "away_team", "Host", "Year")
RANKING_FEATURE_COLUMNS = (
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
)
TARGET_VALUES = ("home_win", "draw", "away_win")
LABEL_DISPLAY = {
    "home_win": "Home win",
    "draw": "Draw",
    "away_win": "Away win",
}
TEAM_ALIAS_MAP = {
    "USA": "UNITED STATES",
    "UNITED STATES OF AMERICA": "UNITED STATES",
    "KOR": "KOREA REPUBLIC",
    "SOUTH KOREA": "KOREA REPUBLIC",
    "IRAN": "IR IRAN",
    "IVORY COAST": "COTE D IVOIRE",
    "COTE DIVOIRE": "COTE D IVOIRE",
}


@dataclass(frozen=True)
class PreparedDataset:
    features: pd.DataFrame
    target: pd.Series


def load_match_csv(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a DataFrame."""
    return pd.read_csv(uploaded_file)


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MATCH_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_number(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_team_key(value) -> str | None:
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    if not text:
        return None
    return TEAM_ALIAS_MAP.get(text, text)


def derive_match_result(row: pd.Series) -> str:
    """Derive a win/draw/loss label from raw result columns.

    Penalty shootouts are handled before ordinary score comparison so that a
    tied match resolved on penalties is not mis-labeled as a draw.
    """

    home_penalty = _coerce_number(row["home_penalty"])
    away_penalty = _coerce_number(row["away_penalty"])
    home_score = _coerce_number(row["home_score"])
    away_score = _coerce_number(row["away_score"])

    if home_penalty is not None and away_penalty is not None and home_penalty != away_penalty:
        return "home_win" if home_penalty > away_penalty else "away_win"

    if home_score is None or away_score is None:
        raise ValueError("home_score and away_score are required to derive the target")

    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def build_ranking_lookup(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the ranking snapshot into a lookup indexed by canonical team key."""

    required = ("team", "team_code", "association", "rank", "previous_rank", "points", "previous_points")
    missing = [column for column in required if column not in ranking_df.columns]
    if missing:
        raise ValueError(f"Missing ranking columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in ranking_df.iterrows():
        payload = {
            "rank": row["rank"],
            "previous_rank": row["previous_rank"],
            "points": row["points"],
            "previous_points": row["previous_points"],
            "association": row["association"],
        }
        for candidate in (normalize_team_key(row["team"]), normalize_team_key(row["team_code"])):
            if candidate:
                rows.append({"team_key": candidate, **payload})

    lookup = pd.DataFrame(rows)
    if lookup.empty:
        return pd.DataFrame(
            columns=["rank", "previous_rank", "points", "previous_points", "association"],
            index=pd.Index([], name="team_key"),
        )

    lookup = lookup.drop_duplicates(subset=["team_key"], keep="first").set_index("team_key")
    return lookup


def enrich_match_features(base_frame: pd.DataFrame, ranking_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach ranking features to the base pre-match frame."""

    enriched = base_frame.copy()
    if ranking_df is None or ranking_df.empty:
        return enriched

    lookup = build_ranking_lookup(ranking_df)
    rank_map = lookup["rank"].to_dict()
    previous_rank_map = lookup["previous_rank"].to_dict()
    points_map = lookup["points"].to_dict()
    previous_points_map = lookup["previous_points"].to_dict()
    association_map = lookup["association"].to_dict()

    for side in ("home", "away"):
        key_col = f"{side}_team_key"
        enriched[key_col] = enriched[f"{side}_team"].map(normalize_team_key)
        enriched[f"{side}_rank"] = enriched[key_col].map(rank_map)
        enriched[f"{side}_previous_rank"] = enriched[key_col].map(previous_rank_map)
        enriched[f"{side}_points"] = enriched[key_col].map(points_map)
        enriched[f"{side}_previous_points"] = enriched[key_col].map(previous_points_map)
        enriched[f"{side}_association"] = enriched[key_col].map(association_map)
        enriched = enriched.drop(columns=[key_col])

    return enriched


def prepare_training_frame(match_df: pd.DataFrame, ranking_df: pd.DataFrame | None = None) -> PreparedDataset:
    """Validate the uploaded data, derive labels, and return feature/target frames."""

    _ensure_required_columns(match_df)
    working = match_df.copy()
    working["target"] = working.apply(derive_match_result, axis=1)

    base = working.loc[:, list(BASE_FEATURE_COLUMNS)].copy()
    base["Year"] = pd.to_numeric(base["Year"], errors="coerce")
    base = enrich_match_features(base, ranking_df)

    return PreparedDataset(features=base, target=working["target"].astype("string"))


def build_prediction_frame(
    home_team: str,
    away_team: str,
    host: str,
    year: int | float,
    ranking_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a single-row feature frame suitable for inference."""

    base = pd.DataFrame(
        [
            {
                "home_team": home_team,
                "away_team": away_team,
                "Host": host,
                "Year": year,
            }
        ]
    )
    base["Year"] = pd.to_numeric(base["Year"], errors="coerce")
    return enrich_match_features(base, ranking_df)
