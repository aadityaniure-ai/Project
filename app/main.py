import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Multi-Sport Prediction", layout="wide")

PAGES = ["NBA", "Football", "Cricket"]

with st.sidebar:
    st.title("Multi-Sport Predictor")
    page = st.radio("Sport", PAGES)


# ── NBA paths / features ──────────────────────────────────────────────────────
NBA_MODEL_PATH = "models/xgb_nba.json"
NBA_LOGS_PATH = "data/nba_game_logs.parquet"
NBA_AVAIL_PATH = "data/nba_availability_today.parquet"

NBA_FEATURE_COLS = [
    "home_avg_pts", "home_avg_reb", "home_avg_ast", "home_avg_plus_minus",
    "home_available_ratio", "home_win_pct_last10",
    "away_avg_pts", "away_avg_reb", "away_avg_ast", "away_avg_plus_minus",
    "away_available_ratio", "away_win_pct_last10",
]

# ── Football paths / features ─────────────────────────────────────────────────
NFL_MODEL_PATH = "models/xgb_football.json"
NFL_LOGS_PATH = "data/football_game_logs.parquet"
NFL_AVAIL_PATH = "data/football_availability_today.parquet"

NFL_FEATURE_COLS = [
    "home_avg_pts_for", "home_avg_pts_against", "home_avg_margin",
    "home_win_pct", "home_win_streak", "home_rest_days",
    "away_avg_pts_for", "away_avg_pts_against", "away_avg_margin",
    "away_win_pct", "away_win_streak", "away_rest_days",
    "spread_line",
]

ROLLING_WINDOW = 6


# ── Shared helpers ────────────────────────────────────────────────────────────
def american_to_implied(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def ev(p_win: float, fair_prob: float) -> float:
    return (p_win * (1 / fair_prob - 1)) - (1 - p_win)


def _highlight_injured(row):
    if row.get("availability", "") != "AVAILABLE":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)


def _render_ev_block(
    p_win: float,
    p_lose: float,
    home_implied: float,
    away_implied: float,
    fair_home: float,
    fair_away: float,
):
    ev_home = ev(p_win, fair_home)
    ev_away = ev(p_lose, fair_away)

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


def _render_availability_table(avail_df: pd.DataFrame, fetch_cmd: str):
    if avail_df.empty:
        st.info(f"Run `{fetch_cmd}` to load today's availability data.")
        return
    st.dataframe(
        avail_df.style.apply(_highlight_injured, axis=1),
        use_container_width=True,
    )
    injured_count = (avail_df["availability"] != "AVAILABLE").sum()
    unavailable_count = (~avail_df["model_available"]).sum()
    st.caption(
        f"{injured_count} player(s) flagged injured/inactive — "
        f"{unavailable_count} excluded from model feature vector"
    )


# ── NBA model & inference ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NBA model...")
def _load_nba_model() -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(NBA_MODEL_PATH)
    return m


@st.cache_data(ttl=300, show_spinner="Loading NBA game logs...")
def _load_nba_logs() -> pd.DataFrame:
    return pd.read_parquet(NBA_LOGS_PATH)


@st.cache_data(ttl=60, show_spinner="Loading NBA availability...")
def _load_nba_availability() -> pd.DataFrame:
    return pd.read_parquet(NBA_AVAIL_PATH)


def _nba_team_features(team: str, logs: pd.DataFrame, avail: pd.DataFrame, prefix: str) -> dict:
    tl = logs[logs["TEAM_ABBREVIATION"] == team].copy()
    tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"])
    recent = tl.nlargest(10, "GAME_DATE")

    ta = avail[avail["team"] == team]
    total = len(ta)
    avail_ratio = ta["model_available"].sum() / total if total else 0.0
    wins = (recent["WL"] == "W").sum() / max(len(recent), 1)

    return {
        f"{prefix}_avg_pts": recent["PTS"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_reb": recent["REB"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_ast": recent["AST"].mean() if len(recent) else 0.0,
        f"{prefix}_avg_plus_minus": recent["PLUS_MINUS"].mean() if len(recent) else 0.0,
        f"{prefix}_available_ratio": avail_ratio,
        f"{prefix}_win_pct_last10": wins,
    }


def _predict_nba(home_team: str, away_team: str) -> tuple[float, float, pd.DataFrame]:
    model = _load_nba_model()
    logs = _load_nba_logs()
    avail = _load_nba_availability()

    feats = {}
    feats.update(_nba_team_features(home_team, logs, avail, "home"))
    feats.update(_nba_team_features(away_team, logs, avail, "away"))

    X = pd.DataFrame([feats])[NBA_FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    p_home = float(proba[1])
    return p_home, 1.0 - p_home, avail[avail["team"].isin([home_team, away_team])]


# ── Football model & inference ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Football model...")
def _load_football_model() -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(NFL_MODEL_PATH)
    return m


@st.cache_data(ttl=300, show_spinner="Loading Football game logs...")
def _load_football_logs() -> pd.DataFrame:
    return pd.read_parquet(NFL_LOGS_PATH)


@st.cache_data(ttl=60, show_spinner="Loading Football availability...")
def _load_football_availability() -> pd.DataFrame:
    return pd.read_parquet(NFL_AVAIL_PATH)


def _nfl_team_features(team: str, logs: pd.DataFrame, avail: pd.DataFrame, prefix: str) -> dict:
    logs = logs.copy()
    logs["gameday"] = pd.to_datetime(logs["gameday"], errors="coerce")

    home_rows = logs[logs["home_team"] == team].assign(
        pts_for=logs["home_score"], pts_against=logs["away_score"]
    )
    away_rows = logs[logs["away_team"] == team].assign(
        pts_for=logs["away_score"], pts_against=logs["home_score"]
    )
    tl = pd.concat([home_rows, away_rows]).sort_values("gameday")
    recent = tl.tail(ROLLING_WINDOW)

    if recent.empty:
        pts_for = pts_against = margin = win_pct = win_streak = rest = 0.0
    else:
        pts_for = recent["pts_for"].mean()
        pts_against = recent["pts_against"].mean()
        margin = (recent["pts_for"] - recent["pts_against"]).mean()
        wins = (recent["pts_for"] > recent["pts_against"]).astype(int)
        win_pct = wins.mean()

        streak = 0
        last = None
        for w in wins:
            if w == last:
                streak += w if w else -1
            else:
                streak = w if w else -1
            last = w
        win_streak = float(streak)

        rest_col = "home_rest" if prefix == "home" else "away_rest"
        rest = float(recent[rest_col].iloc[-1]) if rest_col in recent.columns else 7.0

    ta = avail[avail["team"] == team]
    total = len(ta)
    avail_ratio = ta["model_available"].sum() / total if total else 0.0

    return {
        f"{prefix}_avg_pts_for": pts_for,
        f"{prefix}_avg_pts_against": pts_against,
        f"{prefix}_avg_margin": margin,
        f"{prefix}_win_pct": win_pct,
        f"{prefix}_win_streak": win_streak,
        f"{prefix}_rest_days": rest,
        f"{prefix}_available_ratio": avail_ratio,
    }


def _predict_football(home_team: str, away_team: str, spread: float = 0.0) -> tuple[float, float, pd.DataFrame]:
    model = _load_football_model()
    logs = _load_football_logs()
    avail = _load_football_availability()

    feats = {}
    feats.update(_nfl_team_features(home_team, logs, avail, "home"))
    feats.update(_nfl_team_features(away_team, logs, avail, "away"))
    feats["spread_line"] = spread

    X = pd.DataFrame([feats])[NFL_FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    p_home = float(proba[1])
    return p_home, 1.0 - p_home, avail[avail["team"].isin([home_team, away_team])]


# ── Shared dialogs ────────────────────────────────────────────────────────────
@st.dialog("Create Pull Request")
def _create_pr_dialog():
    pr_title = st.text_input("PR Title")
    pr_branch = st.text_input("Target Branch", value="main")
    st.text_area("Description")
    if st.button("Submit PR"):
        if not pr_title:
            st.error("PR title is required.")
        else:
            st.success(f"PR created: {pr_title} → {pr_branch}")
            st.balloons()


# ── NBA page ──────────────────────────────────────────────────────────────────
def render_nba():
    st.header("NBA Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Game Selection")
        game_input = st.text_input("Game (e.g. LAL vs GSW)", placeholder="LAL vs GSW")
        odds_home = st.number_input("Home Moneyline Odds (American)", value=-110, step=1, key="nba_odds_home")
        odds_away = st.number_input("Away Moneyline Odds (American)", value=+100, step=1, key="nba_odds_away")

    with col2:
        st.subheader("Model Output")
        home_imp = american_to_implied(odds_home)
        away_imp = american_to_implied(odds_away)
        vig = home_imp + away_imp - 1.0
        fair_home = home_imp - vig / 2
        fair_away = away_imp - vig / 2

        parts = [p.strip() for p in game_input.upper().replace("VS", "").split()] if game_input else []
        avail_df = pd.DataFrame()

        if len(parts) == 2:
            try:
                p_win, p_lose, avail_df = _predict_nba(parts[0], parts[1])
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                p_win, p_lose = 0.5, 0.5
        else:
            st.info("Enter a game (e.g. LAL vs GSW) to run live inference.")
            p_win, p_lose = 0.5, 0.5

        _render_ev_block(p_win, p_lose, home_imp, away_imp, fair_home, fair_away)

    st.divider()
    if st.button("Create PR", key="nba_pr"):
        _create_pr_dialog()

    st.divider()
    st.subheader("Player Availability")
    _render_availability_table(avail_df, "python data/fetch_nba.py")


# ── Football page ─────────────────────────────────────────────────────────────
def render_football():
    st.header("Football Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Game Selection")
        game_input = st.text_input("Game (e.g. KC vs SF)", placeholder="KC vs SF")
        spread = st.number_input(
            "Spread (home perspective, negative = home favoured)",
            value=0.0, step=0.5, format="%.1f",
        )
        odds_home = st.number_input("Home Moneyline Odds (American)", value=-110, step=1, key="nfl_odds_home")
        odds_away = st.number_input("Away Moneyline Odds (American)", value=+100, step=1, key="nfl_odds_away")

    with col2:
        st.subheader("Model Output")
        home_imp = american_to_implied(odds_home)
        away_imp = american_to_implied(odds_away)
        vig = home_imp + away_imp - 1.0
        fair_home = home_imp - vig / 2
        fair_away = away_imp - vig / 2

        parts = [p.strip() for p in game_input.upper().replace("VS", "").split()] if game_input else []
        avail_df = pd.DataFrame()

        if len(parts) == 2:
            try:
                p_win, p_lose, avail_df = _predict_football(parts[0], parts[1], spread)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                p_win, p_lose = 0.5, 0.5
        else:
            st.info("Enter a game (e.g. KC vs SF) to run live inference.")
            p_win, p_lose = 0.5, 0.5

        _render_ev_block(p_win, p_lose, home_imp, away_imp, fair_home, fair_away)

    st.divider()
    if st.button("Create PR", key="nfl_pr"):
        _create_pr_dialog()

    st.divider()
    st.subheader("Player Availability")
    _render_availability_table(avail_df, "python data/fetch_football.py")


# ── Cricket page ──────────────────────────────────────────────────────────────
def render_cricket():
    st.header("Cricket Predictions")
    st.info("Cricket data ingestion not yet implemented. See data/fetch_cricket.py.")


if page == "NBA":
    render_nba()
elif page == "Football":
    render_football()
elif page == "Cricket":
    render_cricket()
