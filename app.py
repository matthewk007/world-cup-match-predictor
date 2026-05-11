from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st

import data
import model


st.set_page_config(page_title="2026 World Cup Match Predictor", layout="wide")
st.title("2026 World Cup Match Outcome Predictor")
st.caption("Upload historical FIFA match data, train on demand, and inspect the top factors behind a prediction.")


@st.cache_data(show_spinner=False)
def load_ranking_snapshot() -> pd.DataFrame:
    path = Path("fifa_ranking_2022-10-06.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _options(series: pd.Series) -> list[str]:
    return sorted({str(value) for value in series.dropna().astype(str).unique()})


def _display_label(label: str) -> str:
    return data.LABEL_DISPLAY.get(label, label)


def _render_metrics(metrics: dict[str, float | int]) -> None:
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    cols[2].metric("Train rows", f"{metrics['train_rows']}")
    cols[3].metric("Test rows", f"{metrics['test_rows']}")


def _render_factors(factors: list[dict[str, float | str]]) -> None:
    factor_frame = pd.DataFrame(factors)
    st.dataframe(factor_frame, use_container_width=True, hide_index=True)


if "training_artifacts" not in st.session_state:
    st.session_state.training_artifacts = None
if "ranking_df" not in st.session_state:
    st.session_state.ranking_df = load_ranking_snapshot()
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

ranking_df = st.session_state.ranking_df
if ranking_df.empty:
    st.info("Ranking snapshot not found; predictions will use only match metadata.")

uploaded_file = st.file_uploader("Upload a historical FIFA match CSV", type=["csv"])
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_signature = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.upload_signature != file_signature:
        try:
            st.session_state.raw_df = load_uploaded_csv(file_bytes)
            st.session_state.training_artifacts = None
            st.session_state.upload_signature = file_signature
        except Exception as exc:  # pragma: no cover - Streamlit surface
            st.session_state.raw_df = None
            st.session_state.training_artifacts = None
            st.session_state.upload_signature = None
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()

raw_df = st.session_state.raw_df

if raw_df is not None:
    st.subheader("Uploaded data preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.write("Train the model from the uploaded file, then use the form below to predict a single match.")
    train_clicked = st.button("Train model", type="primary")
    if train_clicked:
        try:
            prepared = data.prepare_training_frame(raw_df, ranking_df)
            st.session_state.training_artifacts = model.train_model(prepared.features, prepared.target)
            st.success("Model trained successfully.")
        except Exception as exc:  # pragma: no cover - Streamlit surface
            st.session_state.training_artifacts = None
            st.error(str(exc))

    artifacts = st.session_state.training_artifacts
    if artifacts is not None:
        st.subheader("Evaluation metrics")
        _render_metrics(artifacts.metrics)
        st.text_area("Classification report", artifacts.report_text, height=240)
        st.dataframe(artifacts.confusion, use_container_width=True)

        team_series = pd.concat([raw_df["home_team"], raw_df["away_team"]], ignore_index=True)
        teams = _options(team_series)
        hosts = _options(raw_df["Host"])
        year_values = pd.to_numeric(raw_df["Year"], errors="coerce").dropna()
        default_year = int(year_values.max()) if not year_values.empty else 2026

        st.subheader("Predict one match")
        with st.form("predict_form"):
            home_team = st.selectbox("Home team", teams)
            away_team = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
            host = st.selectbox("Host", hosts)
            year = st.number_input("Year", min_value=1930, max_value=2100, value=default_year, step=1)
            submitted = st.form_submit_button("Predict")

        if submitted:
            if home_team == away_team:
                st.error("Home team and away team must be different.")
            else:
                try:
                    match_frame = data.build_prediction_frame(home_team, away_team, host, int(year), ranking_df)
                    outcome = model.predict_with_explanation(artifacts, match_frame)
                    st.success(f"Prediction: {_display_label(outcome.label)}")

                    probability_frame = pd.DataFrame(
                        [
                            {"Outcome": _display_label(label), "Probability": probability}
                            for label, probability in outcome.probabilities.items()
                        ]
                    ).sort_values("Probability", ascending=False)
                    st.dataframe(probability_frame, use_container_width=True, hide_index=True)

                    st.subheader("Top factors")
                    _render_factors(outcome.factors)
                except Exception as exc:  # pragma: no cover - Streamlit surface
                    st.error(str(exc))
    else:
        st.info("Click Train model to fit a classifier and unlock prediction inputs.")
else:
    st.info("Upload a FIFA match CSV to preview the data and train a model.")
