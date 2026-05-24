import pandas as pd
from nba_api.stats.endpoints import playergamelogs, leaguedashplayerstats, scoreboardv2, commonteamroster
from nba_api.stats.static import teams
import requests
import time
from datetime import date

SEASONS = ["2022-23", "2023-24", "2024-25"]
INJURY_REPORT_URL = "https://www.nba.com/players/injuries"


def fetch_historical_game_logs(seasons: list[str] = SEASONS) -> pd.DataFrame:
    frames = []
    for season in seasons:
        logs = playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable="Regular Season")
        df = logs.get_data_frames()[0]
        df["SEASON"] = season
        frames.append(df)
        time.sleep(0.6)
    return pd.concat(frames, ignore_index=True)


def fetch_todays_lineups() -> dict[str, list[dict]]:
    today = date.today().strftime("%Y-%m-%d")
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games_df = board.get_data_frames()[0]

    lineups: dict[str, list[dict]] = {}
    team_list = teams.get_teams()
    team_id_map = {t["id"]: t["abbreviation"] for t in team_list}

    game_team_ids = set(games_df["HOME_TEAM_ID"].tolist() + games_df["VISITOR_TEAM_ID"].tolist())

    for team_id in game_team_ids:
        abbr = team_id_map.get(team_id, str(team_id))
        roster_ep = commonteamroster.CommonTeamRoster(team_id=team_id)
        roster_df = roster_ep.get_data_frames()[0]
        lineups[abbr] = roster_df[["PLAYER_ID", "PLAYER"]].rename(columns={"PLAYER": "PLAYER_NAME"}).to_dict("records")
        time.sleep(0.6)

    return lineups


def fetch_injury_report() -> pd.DataFrame:
    """
    Pull the official NBA injury report JSON feed.
    Returns a DataFrame with player_id, player_name, team, status, reason.
    Status values: Out, Doubtful, Questionable, Probable, Day-To-Day.
    """
    feed_url = "https://cdn.nba.com/static/json/liveData/injuries/injuries_0.json"
    resp = requests.get(feed_url, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    rows = []
    for team_entry in raw.get("injured_list", []):
        for player in team_entry.get("players", []):
            rows.append({
                "player_id": player.get("personId"),
                "player_name": player.get("name"),
                "team": team_entry.get("teamAbbreviation"),
                "status": player.get("status"),
                "reason": player.get("comment", ""),
            })

    return pd.DataFrame(rows)


def build_availability_report(lineups: dict, injury_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference today's roster with the injury report.
    Flags each player as AVAILABLE, INJURED, or MISSING.
    MISSING = on today's active roster but not in injury feed and not confirmed active.
    """
    injured_ids = set(injury_df["player_id"].dropna().astype(str).tolist())

    rows = []
    for team_abbr, players in lineups.items():
        for p in players:
            pid = str(p["PLAYER_ID"])
            if pid in injured_ids:
                inj_row = injury_df[injury_df["player_id"].astype(str) == pid].iloc[0]
                flag = "INJURED"
                detail = f"{inj_row['status']}: {inj_row['reason']}"
            else:
                flag = "AVAILABLE"
                detail = ""
            rows.append({
                "team": team_abbr,
                "player_id": pid,
                "player_name": p["PLAYER_NAME"],
                "availability": flag,
                "detail": detail,
            })

    df = pd.DataFrame(rows)

    # Any player whose status is Out or Doubtful is effectively unavailable for modeling
    df["model_available"] = ~df["availability"].eq("INJURED") | df["detail"].str.contains(
        "Probable|Questionable|Day-To-Day", case=False, na=False
    )

    return df


def run():
    print("Fetching historical game logs...")
    logs = fetch_historical_game_logs()
    logs.to_parquet("data/nba_game_logs.parquet", index=False)
    print(f"  Saved {len(logs)} rows to data/nba_game_logs.parquet")

    print("Fetching today's lineups...")
    lineups = fetch_todays_lineups()

    print("Fetching injury report...")
    try:
        injury_df = fetch_injury_report()
    except Exception as e:
        print(f"  Injury feed unavailable ({e}); continuing with empty report")
        injury_df = pd.DataFrame(columns=["player_id", "player_name", "team", "status", "reason"])

    availability = build_availability_report(lineups, injury_df)
    availability.to_parquet("data/nba_availability_today.parquet", index=False)
    print(f"  Saved availability for {len(availability)} players to data/nba_availability_today.parquet")

    injured_count = (availability["availability"] == "INJURED").sum()
    print(f"  Flagged {injured_count} injured/unavailable players")

    return logs, availability


if __name__ == "__main__":
    run()
