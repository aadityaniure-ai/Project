import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Multi-Sport Prediction", layout="wide")

PAGES = ["NBA", "Football", "Cricket"]

with st.sidebar:
    st.title("Multi-Sport Predictor")
    page = st.radio("Sport", PAGES)


MODEL_PATH = "models/xgb_nba.json"
LOGS_PATH = "data/nba_game_logs.parquet"
AVAIL_PATH = "data/nba_availability_today.parquet"

# Features must match training schema exactly
FEATURE_COLS = [
    "home_avg_pts", "home_avg_reb", "home_avg_ast", "home_avg_plus_minus",
    "home_available_ratio", "home_win_pct_last10",
    "away_avg_pts", "away_avg_reb", "away_avg_ast", "away_avg_plus_minus",
    "away_available_ratio", "away_win_pct_last10",
]


@st.cache_resource(show_spinner="Loading XGBoost model...")
def _load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data(ttl=300, show_spinner="Loading game logs...")
def _load_logs() -> pd.DataFrame:
    return pd.read_parquet(LOGS_PATH)


@st.cache_data(ttl=60, show_spinner="Loading availability...")
def _load_availability() -> pd.DataFrame:
    return pd.read_parquet(AVAIL_PATH)


def _team_features(team: str, logs: pd.DataFrame, avail: pd.DataFrame, prefix: str) -> dict:
    team_logs = logs[logs["TEAM_ABBREVIATION"] == team].copy()
    team_logs["GAME_DATE"] = pd.to_datetime(team_logs["GAME_DATE"])
    recent = team_logs.nlargest(10, "GAME_DATE")

    team_avail = avail[avail["team"] == team]
    total = len(team_avail)
    available = team_avail["model_available"].sum() if total else 0
    available_ratio = available / total if total else 0.0

    wins_last10 = (recent["WL"] == "W").sum() / max(len(recent), 1)

    return {
        f"{prefix}_avg_pts": recent["PTS"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_reb": recent["REB"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_ast": recent["AST"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_plus_minus": recent["PLUS_MINUS"].mean() if len(recent) else 0.0,
        f"{prefix}_available_ratio": available_ratio,
        f"{prefix}_win_pct_last10": wins_last10,
    }


def _predict(home_team: str, away_team: str) -> tuple[float, float, pd.DataFrame]:
    """Returns (p_home_win, p_away_win, availability_df)."""
    model = _load_model()
    logs = _load_logs()
    avail = _load_availability()

    feats = {}
    feats.update(_team_features(home_team, logs, avail, "home"))
    feats.update(_team_features(away_team, logs, avail, "away"))

    X = pd.DataFrame([feats])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]  # [p_away_win, p_home_win] for binary label home=1
    p_home = float(proba[1])
    p_away = 1.0 - p_home

    game_avail = avail[avail["team"].isin([home_team, away_team])]
    return p_home, p_away, game_avail


# ── Shared dialogs ───────────────────────────────────────────────────────────
@st.dialog("Create Pull Request")
def _create_pr_dialog():
    pr_title = st.text_input("PR Title")
    pr_branch = st.text_input("Target Branch", value="main")
    pr_desc = st.text_area("Description")
    if st.button("Submit PR"):
        if not pr_title:
            st.error("PR title is required.")
        else:
            st.success(f"PR created: {pr_title} → {pr_branch}")
            st.balloons()


# ── NBA ──────────────────────────────────────────────────────────────────────
def render_nba():
    st.header("NBA Predictions")

    def american_to_implied(odds: int) -> float:
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        return 100 / (odds + 100)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Game Selection")
        game_input = st.text_input("Game (e.g. LAL vs GSW)", placeholder="LAL vs GSW")
        odds_home = st.number_input("Home Moneyline Odds (American)", value=-110, step=1)
        odds_away = st.number_input("Away Moneyline Odds (American)", value=+100, step=1)

    with col2:
        st.subheader("Model Output")

        home_implied = american_to_implied(odds_home)
        away_implied = american_to_implied(odds_away)
        vig = home_implied + away_implied - 1.0
        fair_home = home_implied - vig / 2
        fair_away = away_implied - vig / 2

        parts = [p.strip() for p in game_input.upper().replace("VS", "").split()] if game_input else []
        model_ready = len(parts) == 2

        if model_ready:
            home_team, away_team = parts[0], parts[1]
            try:
                p_win, p_lose, avail_df = _predict(home_team, away_team)
                inference_error = None
            except FileNotFoundError as exc:
                inference_error = str(exc)
                p_win, p_lose, avail_df = 0.5, 0.5, pd.DataFrame()
            except Exception as exc:
                inference_error = str(exc)
                p_win, p_lose, avail_df = 0.5, 0.5, pd.DataFrame()

            if inference_error:
                st.error(f"Inference failed: {inference_error}")
        else:
            st.info("Enter a game (e.g. LAL vs GSW) to run live inference.")
            p_win, p_lose, avail_df = 0.5, 0.5, pd.DataFrame()

        ev_home = (p_win * (1 / fair_home - 1)) - p_lose
        ev_away = (p_lose * (1 / fair_away - 1)) - p_win

        st.metric("Home Win Probability", f"{p_win:.1%}", delta=f"{p_win - home_implied:+.1%} vs implied")
        st.metric("Away Win Probability", f"{p_lose:.1%}", delta=f"{p_lose - away_implied:+.1%} vs implied")
        st.metric("Home EV", f"{ev_home:+.3f}")
        st.metric("Away EV", f"{ev_away:+.3f}")

        if ev_home > 0.02:
            st.success(f"Positive EV on HOME: {ev_home:+.3f}")
        elif ev_away > 0.02:
            st.success(f"Positive EV on AWAY: {ev_away:+.3f}")
        else:
            st.warning("No strong EV edge detected.")

    st.divider()
    if st.button("Create PR"):
        _create_pr_dialog()

    st.divider()
    st.subheader("Player Availability")

    if avail_df.empty:
        st.info("Run `python data/fetch_nba.py` to load today's availability data.")
    else:
        def highlight_injured(row):
            if row["availability"] == "INJURED":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            avail_df.style.apply(highlight_injured, axis=1),
            use_container_width=True,
        )
        injured_count = (avail_df["availability"] == "INJURED").sum()
        unavailable_count = (~avail_df["model_available"]).sum()
        st.caption(
            f"{injured_count} player(s) flagged injured — "
            f"{unavailable_count} excluded from model feature vector"
        )


# ── Football ─────────────────────────────────────────────────────────────────
def render_football():
    st.header("Football Predictions")
    st.info("Football data ingestion not yet implemented. See data/fetch_football.py.")


# ── Cricket ──────────────────────────────────────────────────────────────────
def render_cricket():
    st.header("Cricket Predictions")
    st.info("Cricket data ingestion not yet implemented. See data/fetch_cricket.py.")


if page == "NBA":
    render_nba()
elif page == "Football":
    render_football()
elif page == "Cricket":
    render_cricket()
