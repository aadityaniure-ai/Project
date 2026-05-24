import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

SEASONS = [2022, 2023, 2024]
CURRENT_SEASON = 2024

# Injury designations that make a player unavailable for modeling
UNAVAILABLE_STATUSES = {"Out", "Doubtful", "IR", "PUP-R", "PUP-P", "NFI-R", "NFI-A", "Suspended"}


def fetch_historical_game_logs(seasons: list[int] = SEASONS) -> pd.DataFrame:
    schedules = nfl.import_schedules(seasons)
    cols = [
        "season", "week", "game_id", "game_type",
        "home_team", "away_team",
        "home_score", "away_score",
        "home_qb_name", "away_qb_name",
        "spread_line", "total_line",
        "result",                          # home_score - away_score
        "home_rest", "away_rest",
        "home_moneyline", "away_moneyline",
    ]
    existing = [c for c in cols if c in schedules.columns]
    df = schedules[existing].copy()
    df = df[df["game_type"] == "REG"].reset_index(drop=True)
    return df


def _current_week(season: int) -> int:
    schedules = nfl.import_schedules([season])
    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="coerce")
    today = pd.Timestamp(datetime.utcnow().date())
    past = reg[reg["gameday"] <= today]
    if past.empty:
        return 1
    return int(past["week"].max())


def fetch_active_rosters(season: int = CURRENT_SEASON) -> pd.DataFrame:
    week = _current_week(season)
    rosters = nfl.import_weekly_rosters(years=[season])
    latest = rosters[rosters["week"] == week].copy()

    keep = [
        "player_id", "player_name", "position", "team",
        "status", "injury_designation",
        "week", "season",
    ]
    existing = [c for c in keep if c in latest.columns]
    return latest[existing].reset_index(drop=True)


def fetch_injury_report(season: int = CURRENT_SEASON) -> pd.DataFrame:
    injuries = nfl.import_injuries(years=[season])
    week = _current_week(season)
    latest = injuries[injuries["week"] == week].copy()

    keep = [
        "player_id", "full_name", "team",
        "position", "report_status", "practice_status",
        "primary_injury", "week", "season",
    ]
    existing = [c for c in keep if c in latest.columns]
    return latest[existing].reset_index(drop=True)


def build_availability_report(
    rosters: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-reference active rosters against the weekly injury report.
    Each player is stamped AVAILABLE, INJURED, or IR/EXEMPT.
    model_available=False for Out, Doubtful, IR, PUP, NFI, Suspended.
    """
    inj_map: dict[str, dict] = {}
    if not injuries.empty:
        id_col = "player_id" if "player_id" in injuries.columns else None
        name_col = "full_name" if "full_name" in injuries.columns else None
        for _, row in injuries.iterrows():
            key = str(row[id_col]) if id_col and pd.notna(row.get(id_col)) else (
                str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else None
            )
            if key:
                inj_map[key] = row.to_dict()

    rows = []
    for _, player in rosters.iterrows():
        pid = str(player.get("player_id", ""))
        name = str(player.get("player_name", ""))

        inj = inj_map.get(pid) or inj_map.get(name)
        roster_status = str(player.get("status", "")).strip()
        inj_designation = str(player.get("injury_designation", "")).strip()
        report_status = str(inj.get("report_status", "")) if inj else ""

        effective_status = report_status or inj_designation or roster_status

        if roster_status in UNAVAILABLE_STATUSES or effective_status in UNAVAILABLE_STATUSES:
            availability = "INJURED" if effective_status not in {"IR", "PUP-R", "PUP-P", "NFI-R", "NFI-A"} else effective_status
        else:
            availability = "AVAILABLE"

        model_available = availability == "AVAILABLE" or effective_status == "Questionable"

        rows.append({
            "player_id": pid,
            "player_name": name,
            "team": player.get("team", ""),
            "position": player.get("position", ""),
            "availability": availability,
            "report_status": report_status or inj_designation or roster_status,
            "primary_injury": inj.get("primary_injury", "") if inj else "",
            "model_available": bool(model_available),
        })

    return pd.DataFrame(rows)


def run():
    print("Fetching historical NFL game logs...")
    logs = fetch_historical_game_logs()
    logs.to_parquet("data/football_game_logs.parquet", index=False)
    print(f"  Saved {len(logs)} games to data/football_game_logs.parquet")

    print("Fetching active NFL rosters...")
    rosters = fetch_active_rosters()
    print(f"  {len(rosters)} players on active rosters")

    print("Fetching NFL injury report...")
    try:
        injuries = fetch_injury_report()
        print(f"  {len(injuries)} injury report entries")
    except Exception as e:
        print(f"  Injury report unavailable ({e}); continuing with empty report")
        injuries = pd.DataFrame()

    availability = build_availability_report(rosters, injuries)
    availability.to_parquet("data/football_availability_today.parquet", index=False)
    print(f"  Saved {len(availability)} players to data/football_availability_today.parquet")

    injured_count = (availability["availability"] != "AVAILABLE").sum()
    unavailable_count = (~availability["model_available"]).sum()
    print(f"  {injured_count} flagged injured/inactive — {unavailable_count} excluded from model")

    return logs, availability


if __name__ == "__main__":
    run()
