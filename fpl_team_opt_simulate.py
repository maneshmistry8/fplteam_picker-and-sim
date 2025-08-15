import sys
from pathlib import Path
import pandas as pd
import joblib
import pulp
from sklearn.preprocessing import LabelEncoder

# ---------------- Config ----------------
positions = ["FWD", "MID", "DEF", "GKP"]
position_map = {
    "GKP": {"min_start": 1, "max_start": 1, "squad": 2},
    "DEF": {"min_start": 3, "max_start": 5, "squad": 5},
    "MID": {"min_start": 2, "max_start": 5, "squad": 5},
    "FWD": {"min_start": 1, "max_start": 3, "squad": 3},
}
POINTS_WEIGHT = 10.0
INITIAL_BANK = 0.5
MAX_TRANSFERS_PER_GW = 1

# Models expect these features
position_features = {
    "FWD": [
        "minutes_roll3","goals_scored_roll3","assists_roll3","bonus_roll3",
        "bps_roll3","influence_roll3","creativity_roll3","threat_roll3",
        "ict_index_roll3","total_points_roll3","total_points_std3",
        "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
        "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
        "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
    ],
    "MID": [
        "minutes_roll3","goals_scored_roll3","assists_roll3","bonus_roll3",
        "bps_roll3","influence_roll3","creativity_roll3","threat_roll3",
        "ict_index_roll3","total_points_roll3","total_points_std3",
        "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
        "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
        "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
    ],
    "DEF": [
        "minutes_roll3","clean_sheets_roll3","yellow_cards_roll3","bonus_roll3",
        "bps_roll3","influence_roll3","total_points_roll3","total_points_std3",
        "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
        "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
        "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
    ],
    "GKP": [
        "minutes_roll3","clean_sheets_roll3","saves_roll3","bonus_roll3",
        "bps_roll3","influence_roll3","total_points_roll3","total_points_std3",
        "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
        "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
        "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
    ],
}
ALL_FEATURE_COLS = sorted(set(sum(position_features.values(), [])))

# ---------------- Helpers ----------------
def _ensure_cols(df: pd.DataFrame, cols, default=0):
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df

def load_models():
    Path("data/processed/multi_season").mkdir(parents=True, exist_ok=True)
    models = {}
    for pos in positions:
        p = Path(f"data/processed/multi_season/fpl_rf_model_{pos}.joblib")
        if not p.exists():
            raise FileNotFoundError(f"Missing model for {pos}: {p}")
        models[pos] = joblib.load(p)
    return models

def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "opponent_id" not in out.columns:
        out["opponent_id"] = 0
    if "was_home" not in out.columns:
        out["was_home"] = 0
    out["position_enc"] = LabelEncoder().fit_transform(out["position"])
    out["team_enc"]     = LabelEncoder().fit_transform(out["team_id"])
    out["opponent_enc"] = LabelEncoder().fit_transform(out["opponent_id"])
    out["was_home"]     = out["was_home"].astype(int)
    return out

def predict_gw_points(df_gw: pd.DataFrame, models: dict) -> pd.DataFrame:
    df_gw = encode_for_model(df_gw)
    df_gw = _ensure_cols(df_gw, ALL_FEATURE_COLS, default=0)
    if "status" not in df_gw.columns:
        df_gw["status"] = "a"
    df_gw["status"] = df_gw["status"].fillna("a")

    df_gw["predicted_points"] = 0.0
    for pos in positions:
        mask = df_gw["position"] == pos
        if not mask.any():
            continue
        X = df_gw.loc[mask, position_features[pos]].fillna(0)
        df_gw.loc[mask, "predicted_points"] = models[pos].predict(X)
    return df_gw

def optimize_squad(df_gw: pd.DataFrame):
    """Pick 15-man squad + starting XI for this GW."""
    df = df_gw[df_gw["status"] == "a"].copy()
    if df.empty:
        print("No active players for optimization.")
        return None

    prob = pulp.LpProblem("FPL_Backtest_Squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("Squad", df["player_id"], cat="Binary")
    start = pulp.LpVariable.dicts("Start", df["player_id"], cat="Binary")

    # Objective: maximize starting XI predicted points
    prob += pulp.lpSum([POINTS_WEIGHT * df.loc[i, "predicted_points"] * start[df.loc[i, "player_id"]] for i in df.index])

    # Constraints
    prob += pulp.lpSum([df.loc[i, "value"] * squad[df.loc[i, "player_id"]] for i in df.index]) <= 100, "BudgetMax"
    prob += pulp.lpSum([squad[pid] for pid in df["player_id"]]) == 15, "SquadSize"

    for pos, lim in position_map.items():
        prob += pulp.lpSum([squad[df.loc[i, "player_id"]] for i in df.index if df.loc[i, "position"] == pos]) == lim["squad"], f"Squad_{pos}"
        prob += pulp.lpSum([start[df.loc[i, "player_id"]] for i in df.index if df.loc[i, "position"] == pos]) >= lim["min_start"], f"MinStart_{pos}"
        prob += pulp.lpSum([start[df.loc[i, "player_id"]] for i in df.index if df.loc[i, "position"] == pos]) <= lim["max_start"], f"MaxStart_{pos}"

    prob += pulp.lpSum([start[pid] for pid in df["player_id"]]) == 11, "StartingXI"

    # Max 3 per team
    for team_id in df["team_id"].unique():
        prob += pulp.lpSum([squad[df.loc[i, "player_id"]] for i in df.index if df.loc[i, "team_id"] == team_id]) <= 3, f"MaxTeam_{team_id}"

    # Must start only if selected in squad
    for pid in df["player_id"]:
        prob += start[pid] <= squad[pid], f"StartInSquad_{pid}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        print("No optimal solution returned for this GW.")
        return None

    rows, starters = [], []
    total_cost = 0.0
    for i in df.index:
        pid = df.loc[i, "player_id"]
        if squad[pid].value() > 0.5:
            r = {
                "player_id": pid,
                "name": df.loc[i, "name"],
                "position": df.loc[i, "position"],
                "team_id": df.loc[i, "team_id"],
                "predicted_points": df.loc[i, "predicted_points"],
                "value": df.loc[i, "value"],
                "starting": start[pid].value() > 0.5,
                "premium": df.loc[i, "value"] >= 8.0
            }
            rows.append(r)
            total_cost += r["value"]
            if r["starting"]:
                starters.append(r)

    print(f"Optimized squad (Cost £{total_cost:.1f}m)")
    return pd.DataFrame(rows)

# ---- Sim-mode helpers (when no real GW stats exist) ----
def build_fixture_lookup(fixtures_2425: pd.DataFrame):
    """Return: lookup[team_id][gw] -> (opponent_id, was_home, fdr) and sorted GW list."""
    lookup = {}
    gws = set()
    for _, row in fixtures_2425.dropna(subset=["gameweek"]).iterrows():
        gw = int(row["gameweek"])
        gws.add(gw)
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        hf, af = float(row["home_fdr"]), float(row["away_fdr"])
        lookup.setdefault(h, {})[gw] = (a, 1, hf)
        lookup.setdefault(a, {})[gw] = (h, 0, af)
    return lookup, sorted(gws)

def make_gw_frame(base_players: pd.DataFrame, gw: int, lookup: dict) -> pd.DataFrame:
    """Clone snapshot of players; attach opponent_id/was_home/fdr_roll3 from fixtures for this gw."""
    df = base_players.copy()
    df["gameweek"] = gw
    df["opponent_id"] = df["team_id"].apply(lambda t: lookup.get(int(t), {}).get(gw, (0, 0, 3))[0])
    df["was_home"] = df["team_id"].apply(lambda t: lookup.get(int(t), {}).get(gw, (0, 0, 3))[1])
    df["fdr_roll3"] = df["team_id"].apply(lambda t: lookup.get(int(t), {}).get(gw, (0, 0, 3))[2])
    # ensure status present
    if "status" not in df.columns:
        df["status"] = "a"
    df["status"] = df["status"].fillna("a")
    return df

# ---------------- Main ----------------
def main():
    # Load cleaned 24/25 snapshot
    try:
        df_2425 = pd.read_csv("data/processed/fpl_2024-25_cleaned.csv")
    except FileNotFoundError:
        print("Error: data/processed/fpl_2024-25_cleaned.csv not found. Run clean_data.py first.")
        sys.exit(1)

    # Budget needs player prices
    if "value" not in df_2425.columns:
        raise ValueError("Expected 'value' column in cleaned data for budget optimization.")

    # Determine mode: real or simulated
    valid_gws = sorted([int(gw) for gw in df_2425["gameweek"].dropna().unique() if gw != 0])
    simulated_mode = len(valid_gws) == 0

    models = load_models()

    if simulated_mode:
        print("Backtest running in SIMULATED mode (no per-GW history available). Scoring uses predicted points.")
        # Build fixture-driven per-GW frames
        try:
            fixtures = pd.read_csv("data/raw/fpl_fixtures.csv")
        except FileNotFoundError:
            raise FileNotFoundError("data/raw/fpl_fixtures.csv not found. Run datadownload.py first.")

        # Use rows tagged as 2024-25 (labels may be synthetic, but we only need a 1..38 sequence)
        fixtures_2425 = fixtures[fixtures["season"] == "2024-25"].copy()
        if fixtures_2425.empty:
            # fallback: use all fixtures we have
            fixtures_2425 = fixtures.copy()

        lookup, gw_list = build_fixture_lookup(fixtures_2425)
        if not gw_list:
            # final fallback: assume 1..38 even if fixtures missing
            gw_list = list(range(1, 39))

        # Base snapshot: one row per player (drop any gw duplicates)
        base_cols = ["player_id","name","position","team_id","value","status"]
        base_players = df_2425[base_cols].drop_duplicates("player_id")

        # Initial squad from GW1 predictions
        df_gw1 = predict_gw_points(make_gw_frame(base_players, gw_list[0], lookup), models)
        initial_squad_df = optimize_squad(df_gw1)
        if initial_squad_df is None or initial_squad_df.empty:
            raise ValueError("Failed to build an initial squad for GW1.")
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        initial_squad_df.to_csv("data/processed/optimized_squad_2425_gw1.csv", index=False)

        # Sim loop (predicted points as truth)
        current_squad_df = initial_squad_df.copy()
        bank = INITIAL_BANK
        gw_scores, transfers_log = [], []
        total_score = 0

        for gw in gw_list:
            print(f"\n=== GW{gw} (sim) ===")
            df_gw = predict_gw_points(make_gw_frame(base_players, gw, lookup), models)

            # Transfers: replace bottom 20% by predicted points (no minutes rule in sim)
            if gw != gw_list[0] and MAX_TRANSFERS_PER_GW > 0:
                squad_metrics = current_squad_df.merge(
                    df_gw[["player_id","predicted_points","status"]],
                    on="player_id", how="left"
                ).fillna({"status":"a"})
                squad_metrics = squad_metrics[squad_metrics["status"] == "a"]
                thr = squad_metrics["predicted_points"].quantile(0.2) if not squad_metrics["predicted_points"].isna().all() else 0
                low_pred = squad_metrics[squad_metrics["predicted_points"] < thr]

                made = 0
                for _, low in low_pred.iterrows():
                    pos = low["position"]
                    candidates = df_gw[
                        (df_gw["position"] == pos) &
                        (~df_gw["player_id"].isin(current_squad_df["player_id"])) &
                        (df_gw["status"] == "a") &
                        (df_gw["value"] <= low["value"] + bank)
                    ]
                    if candidates.empty:
                        continue
                    best = candidates.loc[candidates["predicted_points"].idxmax()]
                    new_cost = current_squad_df["value"].sum() - low["value"] + best["value"]
                    if new_cost <= 100 + bank:
                        current_squad_df = current_squad_df[current_squad_df["player_id"] != low["player_id"]]
                        current_squad_df = pd.concat([current_squad_df, pd.DataFrame([{
                            "player_id": best["player_id"],
                            "name": best["name"],
                            "position": best["position"],
                            "team_id": best["team_id"],
                            "predicted_points": best["predicted_points"],
                            "value": best["value"],
                            "starting": low.get("starting", False),
                            "premium": best["value"] >= 8.0
                        }])], ignore_index=True)
                        bank += low["value"] - best["value"]
                        transfers_log.append({
                            "gameweek": gw,
                            "out_player_id": low["player_id"], "out_player_name": low["name"],
                            "in_player_id": best["player_id"], "in_player_name": best["name"],
                            "bank_after": bank
                        })
                        print(f"Transfer: {low['name']} ➜ {best['name']} (Bank £{bank:.1f}m)")
                        made += 1
                    if made >= MAX_TRANSFERS_PER_GW:
                        break

            # Pick XI (greedy by predicted points with formation limits)
            merged = current_squad_df.merge(
                df_gw[["player_id","predicted_points","status"]],
                on="player_id", how="left"
            ).fillna({"status":"a"})
            merged = merged[merged["status"] == "a"].sort_values("predicted_points", ascending=False)

            starting_ids, counts = [], {"GKP":0,"DEF":0,"MID":0,"FWD":0}
            for _, row in merged.iterrows():
                pos = row["position"]
                if ((pos == "GKP" and counts["GKP"] < 1) or
                    (pos == "DEF" and counts["DEF"] < 5) or
                    (pos == "MID" and counts["MID"] < 5) or
                    (pos == "FWD" and counts["FWD"] < 3)):
                    starting_ids.append(row["player_id"])
                    counts[pos] += 1
                    if len(starting_ids) == 11:
                        break

            # Score GW using predicted points (captain = top predicted)
            xi = merged[merged["player_id"].isin(starting_ids)]
            if xi.empty:
                gw_pts = 0
                captain_name = "None"
            else:
                xi_sorted = xi.sort_values("predicted_points", ascending=False)
                captain = xi_sorted.iloc[0]
                captain_name = captain["name"]
                # captain doubled: add one extra captain predicted_points
                gw_pts = float(xi_sorted["predicted_points"].sum()) + float(captain["predicted_points"])

            gw_scores.append({"gameweek": gw, "points": gw_pts, "captain_name": captain_name})
            total_score += gw_pts
            print(f"GW{gw}: {gw_pts:.2f} pts (Captain: {captain_name})")

        # Save sim outputs
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gw_scores).to_csv("data/processed/2425_simulation.csv", index=False)
        pd.DataFrame(transfers_log).to_csv("data/processed/2425_transfers.csv", index=False)
        # No subs log in sim-mode (we skipped the minutes-based auto-subs)
        print(f"\nTotal Season Points (Simulated 2024/25): {total_score:.2f}")
        print("Saved simulation to data/processed/2425_simulation.csv")
        print("Saved transfers to data/processed/2425_transfers.csv")

    else:
        # -------- REAL MODE (if you ever provide true per-GW rows with total_points) --------
        print("Backtest running in REAL mode (per-GW history found).")
        models = load_models()

        # Initial GW from the smallest GW present
        first_gw = min(valid_gws)
        df_gw1 = df_2425[df_2425["gameweek"] == first_gw].copy()
        df_gw1 = predict_gw_points(df_gw1, models)
        initial_squad_df = optimize_squad(df_gw1)
        if initial_squad_df is None or initial_squad_df.empty:
            raise ValueError(f"Failed to build an initial squad for GW{first_gw}.")
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        initial_squad_df.to_csv("data/processed/optimized_squad_2425_gw1.csv", index=False)

        current_squad_df = initial_squad_df.copy()
        bank = INITIAL_BANK
        gw_scores, transfers_log, subs_log = [], [], []
        total_score = 0

        for gw in sorted(valid_gws):
            print(f"\n=== GW{gw} ===")
            df_gw = predict_gw_points(df_2425[df_2425["gameweek"] == gw].copy(), models)

            # Transfers (same as before but keep minutes rule if you have it)
            if gw > first_gw and MAX_TRANSFERS_PER_GW > 0:
                squad_metrics = current_squad_df.merge(
                    df_gw[["player_id","predicted_points","minutes_roll3","status"]],
                    on="player_id", how="left"
                ).fillna({"status":"a"})
                squad_metrics = squad_metrics[squad_metrics["status"] == "a"]
                thr = squad_metrics["predicted_points"].quantile(0.2) if not squad_metrics["predicted_points"].isna().all() else 0
                low_pred = squad_metrics[(squad_metrics["predicted_points"] < thr) | (squad_metrics["minutes_roll3"] < 60)]

                made = 0
                for _, low in low_pred.iterrows():
                    pos = low["position"]
                    cands = df_gw[
                        (df_gw["position"] == pos) &
                        (~df_gw["player_id"].isin(current_squad_df["player_id"])) &
                        (df_gw["status"] == "a") &
                        (df_gw["minutes_roll3"] >= 60) &
                        (df_gw["value"] <= low["value"] + bank)
                    ]
                    if cands.empty:
                        continue
                    best = cands.loc[cands["predicted_points"].idxmax()]
                    new_cost = current_squad_df["value"].sum() - low["value"] + best["value"]
                    if new_cost <= 100 + bank:
                        current_squad_df = current_squad_df[current_squad_df["player_id"] != low["player_id"]]
                        current_squad_df = pd.concat([current_squad_df, pd.DataFrame([{
                            "player_id": best["player_id"],
                            "name": best["name"],
                            "position": best["position"],
                            "team_id": best["team_id"],
                            "predicted_points": best["predicted_points"],
                            "value": best["value"],
                            "starting": low.get("starting", False),
                            "premium": best["value"] >= 8.0
                        }])], ignore_index=True)
                        bank += low["value"] - best["value"]
                        transfers_log.append({
                            "gameweek": gw,
                            "out_player_id": low["player_id"], "out_player_name": low["name"],
                            "in_player_id": best["player_id"], "in_player_name": best["name"],
                            "bank_after": bank
                        })
                        print(f"Transfer: {low['name']} ➜ {best['name']} (Bank £{bank:.1f}m)")
                        made += 1
                    if made >= MAX_TRANSFERS_PER_GW:
                        break

            # XI selection and REAL scoring using total_points + captain double
            merged = current_squad_df.merge(
                df_gw[["player_id","predicted_points","minutes_roll3","status"]],
                on="player_id", how="left"
            ).fillna({"status":"a"})
            merged = merged[merged["status"] == "a"].sort_values("predicted_points", ascending=False)

            starting_ids, counts = [], {"GKP":0,"DEF":0,"MID":0,"FWD":0}
            for _, row in merged.iterrows():
                pos = row["position"]
                if ((pos == "GKP" and counts["GKP"] < 1) or
                    (pos == "DEF" and counts["DEF"] < 5) or
                    (pos == "MID" and counts["MID"] < 5) or
                    (pos == "FWD" and counts["FWD"] < 3)):
                    starting_ids.append(row["player_id"])
                    counts[pos] += 1
                    if len(starting_ids) == 11:
                        break

            starting = df_2425[(df_2425["player_id"].isin(starting_ids)) & (df_2425["gameweek"] == gw)].copy()
            starting = starting.merge(df_gw[["player_id","predicted_points","minutes_roll3","status"]], on="player_id", how="left").fillna({"status":"a"})

            if not starting.empty:
                srt = starting.sort_values("predicted_points", ascending=False)
                captain_id = srt.iloc[0]["player_id"]
                captain_name = srt.iloc[0]["name"]
            else:
                captain_id = None
                captain_name = "None"

            cap_pts = starting.loc[starting["player_id"] == captain_id, "total_points"]
            cap_pts = int(cap_pts.iloc[0]) if not cap_pts.empty else 0
            gw_points = int(starting["total_points"].sum()) + cap_pts
            gw_scores.append({"gameweek": gw, "points": gw_points, "captain_name": captain_name})
            total_score += gw_points
            print(f"GW{gw}: {gw_points} pts (Captain: {captain_name}, {cap_pts*2} pts)")

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gw_scores).to_csv("data/processed/2425_simulation.csv", index=False)
        pd.DataFrame(transfers_log).to_csv("data/processed/2425_transfers.csv", index=False)
        print(f"\nTotal Season Points (2024/25 Backtest): {total_score}")
        print("Saved simulation to data/processed/2425_simulation.csv")
        print("Saved transfers to data/processed/2425_transfers.csv")

if __name__ == "__main__":
    main()

