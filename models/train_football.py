import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

LOGS_PATH = "data/football_game_logs.parquet"
MODEL_OUT = "models/xgb_football.json"
ROLLING_WINDOW = 6   # NFL regular season is short; 6-game window ≈ ~38% of season


def _team_rolling(df: pd.DataFrame, team_col: str, score_col: str, opp_score_col: str, prefix: str) -> pd.DataFrame:
    grp = df[[team_col, "gameday", score_col, opp_score_col, "rest_days"]].copy()
    grp = grp.rename(columns={team_col: "team", score_col: "pts_for", opp_score_col: "pts_against"})
    grp["margin"] = grp["pts_for"] - grp["pts_against"]
    grp["win"] = (grp["margin"] > 0).astype(int)
    grp = grp.sort_values(["team", "gameday"])

    grp[f"{prefix}_avg_pts_for"] = (
        grp.groupby("team")["pts_for"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    grp[f"{prefix}_avg_pts_against"] = (
        grp.groupby("team")["pts_against"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    grp[f"{prefix}_avg_margin"] = (
        grp.groupby("team")["margin"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    grp[f"{prefix}_win_pct"] = (
        grp.groupby("team")["win"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    grp[f"{prefix}_win_streak"] = grp.groupby("team")["win"].transform(
        lambda x: x.shift(1).groupby((x.shift(1) != x.shift(1).shift(1)).cumsum()).cumcount() * x.shift(1)
    )
    grp[f"{prefix}_rest_days"] = grp["rest_days"]

    feat_cols = [
        "team", "gameday",
        f"{prefix}_avg_pts_for", f"{prefix}_avg_pts_against",
        f"{prefix}_avg_margin", f"{prefix}_win_pct",
        f"{prefix}_win_streak", f"{prefix}_rest_days",
    ]
    return grp[feat_cols]


def engineer_features(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "gameday"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # rest days: days since last game per team
    home_rest = df[["home_team", "gameday"]].rename(columns={"home_team": "team"})
    away_rest = df[["away_team", "gameday"]].rename(columns={"away_team": "team"})
    all_games = pd.concat([home_rest, away_rest]).sort_values(["team", "gameday"])
    all_games["rest_days"] = all_games.groupby("team")["gameday"].diff().dt.days.fillna(7)

    home_rest_map = all_games.set_index(["team", "gameday"])["rest_days"]
    df["home_rest_days"] = df.apply(
        lambda r: home_rest_map.get((r["home_team"], r["gameday"]), 7), axis=1
    )
    df["away_rest_days"] = df.apply(
        lambda r: home_rest_map.get((r["away_team"], r["gameday"]), 7), axis=1
    )

    home_feats = _team_rolling(
        df.assign(team=df["home_team"], rest_days=df["home_rest_days"]),
        "home_team", "home_score", "away_score", "home",
    ).rename(columns={"team": "home_team", "gameday": "gameday_h"})

    away_feats = _team_rolling(
        df.assign(team=df["away_team"], rest_days=df["away_rest_days"]),
        "away_team", "away_score", "home_score", "away",
    ).rename(columns={"team": "away_team", "gameday": "gameday_a"})

    df = df.merge(home_feats, on="home_team", suffixes=("", "_dup"))
    df = df[df["gameday"] == df["gameday_h"]].drop(columns=["gameday_h"])

    df = df.merge(away_feats, on="away_team", suffixes=("", "_dup"))
    df = df[df["gameday"] == df["gameday_a"]].drop(columns=["gameday_a"])

    df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

    if "spread_line" in df.columns:
        df["spread_line"] = df["spread_line"].fillna(0.0)
    else:
        df["spread_line"] = 0.0

    return df.dropna(subset=FEATURE_COLS)


FEATURE_COLS = [
    "home_avg_pts_for", "home_avg_pts_against", "home_avg_margin",
    "home_win_pct", "home_win_streak", "home_rest_days",
    "away_avg_pts_for", "away_avg_pts_against", "away_avg_margin",
    "away_win_pct", "away_win_streak", "away_rest_days",
    "spread_line",
]


def train(df: pd.DataFrame) -> XGBClassifier:
    df = df.sort_values("gameday").reset_index(drop=True)
    X = df[FEATURE_COLS].astype(float)
    y = df["home_win"]

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"  Fold {fold + 1} — val logloss: {log_loss(y_val, oof_preds[val_idx]):.4f}")

    valid_mask = oof_preds > 0
    print(f"\nOOF log-loss : {log_loss(y[valid_mask], oof_preds[valid_mask]):.4f}")
    print(f"OOF ROC-AUC  : {roc_auc_score(y[valid_mask], oof_preds[valid_mask]):.4f}")

    # final fit on full data
    model.fit(X, y, verbose=False)
    return model


def run():
    print("Loading football game logs...")
    logs = pd.read_parquet(LOGS_PATH)
    print(f"  {len(logs)} raw game records")

    print("Engineering features...")
    df = engineer_features(logs)
    print(f"  {len(df)} games after feature engineering")

    print("Training XGBClassifier...")
    model = train(df)

    model.save_model(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    run()
