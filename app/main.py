import streamlit as st
import pandas as pd

st.set_page_config(page_title="Multi-Sport Prediction", layout="wide")

PAGES = ["NBA", "Football", "Cricket"]

with st.sidebar:
    st.title("Multi-Sport Predictor")
    page = st.radio("Sport", PAGES)


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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Game Selection")
        game_input = st.text_input("Game (e.g. LAL vs GSW)", placeholder="LAL vs GSW")
        odds_home = st.number_input("Home Moneyline Odds (American)", value=-110, step=1)
        odds_away = st.number_input("Away Moneyline Odds (American)", value=+100, step=1)

    with col2:
        st.subheader("Model Output")

        def american_to_implied(odds: int) -> float:
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            return 100 / (odds + 100)

        home_implied = american_to_implied(odds_home)
        away_implied = american_to_implied(odds_away)
        vig = home_implied + away_implied - 1.0

        # placeholders — replace with XGBoost model output
        model_home_prob = st.slider("Model Win Probability (Home)", 0.0, 1.0, 0.52, 0.01)
        model_away_prob = 1.0 - model_home_prob

        fair_home = home_implied - vig / 2
        fair_away = away_implied - vig / 2

        ev_home = (model_home_prob * (1 / fair_home - 1)) - (1 - model_home_prob)
        ev_away = (model_away_prob * (1 / fair_away - 1)) - (1 - model_away_prob)

        st.metric("Home Win Probability", f"{model_home_prob:.1%}")
        st.metric("Away Win Probability", f"{model_away_prob:.1%}")
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

    try:
        avail_df = pd.read_parquet("data/nba_availability_today.parquet")
        if game_input:
            parts = [p.strip() for p in game_input.upper().replace("VS", "").split()]
            if len(parts) == 2:
                avail_df = avail_df[avail_df["team"].isin(parts)]

        def highlight_injured(row):
            if row["availability"] == "INJURED":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            avail_df.style.apply(highlight_injured, axis=1),
            use_container_width=True,
        )

        injured_count = (avail_df["availability"] == "INJURED").sum()
        st.caption(f"{injured_count} player(s) flagged as injured/unavailable")
    except FileNotFoundError:
        st.info("Run `python data/fetch_nba.py` to load today's availability data.")


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
