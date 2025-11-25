"""
clutch_score.py
----------------
Pulls NBA player stats via nba_api (2020-21 → present), merges regular/clutch/playoff
splits, and computes a simple "clutch score" per player-season

Usage (from repo root):
    pip install nba_api pandas numpy matplotlib scikit-learn
    python clutch_score.py

Notes:
- The NBA site can rate-limit: we sleep between calls.
- We rely on PLAYER_ID to merge (avoid name collisions)
- Clutch window uses NBA default: last 5 min, score within 5
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashPlayerClutch

# -----------------------
# Config
# -----------------------
SEASONS: List[str] = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# Weights for the clutch score
WEIGHTS: Dict[str, float] = {
    "REG": 1.0,   # Regular season
    "CLU": 2.0,   # Clutch (last 5 min, +- 5 pts)
    "PO":  3.0,   # Playoffs
}
# Small boost for players who actually take shots in clutch (volume adjustment)
CLUTCH_VOLUME_WEIGHT = 0.10  # adds up to +0.10 * (clutch FGA share)

REQUEST_PAUSE = 1.0  # seconds between API calls
OUT_DIR = Path("data/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)

# Keep a common set of columns across endpoints
COMMON_KEEP = [
    "PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION",
    "GP","MIN","FGA","FGM","FG_PCT",
    "FG3A","FG3M","FG3_PCT",
    "FTA","FTM","FT_PCT",
    "PTS","AST","TOV","REB","OREB","DREB",
    "PLUS_MINUS","USG_PCT","TS_PCT","EFG_PCT"
]

# -----------------------
# Cache helpers
# -----------------------
def _cache_path(name: str) -> Path:
    safe = name.replace(" ", "_").replace("-", "")
    return RAW_DIR / f"{safe}.csv"

def _read_cache(name: str) -> Optional[pd.DataFrame]:
    fp = _cache_path(name)
    return pd.read_csv(fp) if fp.exists() else None

def _write_cache(df: pd.DataFrame, name: str) -> pd.DataFrame:
    fp = _cache_path(name)
    df.to_csv(fp, index=False)
    return df

# -----------------------
# Data pulls
# -----------------------
def fetch_regular(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    key = f"regular_{season}_{per_mode}"
    cached = _read_cache(key)
    if cached is not None:
        return cached
    time.sleep(REQUEST_PAUSE)
    resp = LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed=per_mode
    )
    df = resp.get_data_frames()[0]
    return _write_cache(df, key)

def fetch_playoffs(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    key = f"playoffs_{season}_{per_mode}"
    cached = _read_cache(key)
    if cached is not None:
        return cached
    time.sleep(REQUEST_PAUSE)
    resp = LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Playoffs",
        per_mode_detailed=per_mode
    )
    df = resp.get_data_frames()[0]
    return _write_cache(df, key)

def fetch_clutch(
    season: str,
    per_mode: str = "PerGame",
    season_type: str = "Regular Season",
    clutch_time: str = "Last 5 Minutes",
    point_diff: int = 5,
    ahead_behind: str = "Ahead or Behind",
) -> pd.DataFrame:
    """
    LeagueDashPlayerClutch parameter names are a bit quirky in nba_api.
    We use the most common ones; if your installed version differs, adjust below.
    """
    key = f"clutch_{season}_{season_type}_{per_mode}_{clutch_time}_{point_diff}"
    cached = _read_cache(key)
    if cached is not None:
        return cached

    time.sleep(REQUEST_PAUSE)

    # Try common param names first; fallback if needed for older nba_api versions.
    try:
        resp = LeagueDashPlayerClutch(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed=per_mode,
            clutch_time=clutch_time,
            ahead_behind=ahead_behind,
            point_diff=point_diff,
        )
    except TypeError as e:
        print(f"First attempt failed with: {e}")
        # Fallback signatures (older libs used different kw names)
        try:
            resp = LeagueDashPlayerClutch(
                season=season,
                season_type_all_star=season_type,
                per_mode_time=per_mode,
                clutch_time=clutch_time,
                ahead_behind=ahead_behind,
                point_diff=point_diff,
            )
        except TypeError as e2:
            print(f"Second attempt failed with: {e2}")
            # Last resort - minimal params
            resp = LeagueDashPlayerClutch(
                season=season,
                season_type_all_star=season_type,
            )

    df = resp.get_data_frames()[0]
    return _write_cache(df, key)

# -----------------------
# Merging and scoring
# -----------------------
def calculate_efficiency_metrics(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Calculate TS%, eFG%, and FG% if they don't exist in the dataframe."""
    df = df.copy()
    
    # Calculate True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
    ts_col = f"TS_PCT{suffix}"
    if ts_col not in df.columns or df[ts_col].isna().all():
        pts = df.get(f"PTS{suffix}", 0)
        fga = df.get(f"FGA{suffix}", 0)
        fta = df.get(f"FTA{suffix}", 0)
        df[ts_col] = pts / (2 * (fga + 0.44 * fta))
        df[ts_col] = df[ts_col].replace([np.inf, -np.inf], np.nan)
    
    # Calculate Effective FG%: (FGM + 0.5 * FG3M) / FGA
    efg_col = f"EFG_PCT{suffix}"
    if efg_col not in df.columns or df[efg_col].isna().all():
        fgm = df.get(f"FGM{suffix}", 0)
        fg3m = df.get(f"FG3M{suffix}", 0)
        fga = df.get(f"FGA{suffix}", 0)
        df[efg_col] = (fgm + 0.5 * fg3m) / fga
        df[efg_col] = df[efg_col].replace([np.inf, -np.inf], np.nan)
    
    # Calculate FG%: FGM / FGA
    fg_col = f"FG_PCT{suffix}"
    if fg_col not in df.columns or df[fg_col].isna().all():
        fgm = df.get(f"FGM{suffix}", 0)
        fga = df.get(f"FGA{suffix}", 0)
        df[fg_col] = fgm / fga
        df[fg_col] = df[fg_col].replace([np.inf, -np.inf], np.nan)
    
    return df

def _prune(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [c for c in COMMON_KEEP if c in df.columns]
    out = df[cols].copy()
    perf_cols = [c for c in out.columns if c not in ["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION"]]
    out = out.rename(columns={c: f"{c}{suffix}" for c in perf_cols})
    return out

def merge_frames(reg: pd.DataFrame, clu: pd.DataFrame, po: Optional[pd.DataFrame]) -> pd.DataFrame:
    # Calculate efficiency metrics first (before pruning)
    reg = calculate_efficiency_metrics(reg, "")
    clu = calculate_efficiency_metrics(clu, "")
    if po is not None and len(po):
        po = calculate_efficiency_metrics(po, "")
    
    left = _prune(reg, "_REG")
    mid  = _prune(clu, "_CLU")
    merged = left.merge(mid, on=["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION"], how="outer")
    if po is not None and len(po):
        right = _prune(po, "_PO")
        merged = merged.merge(right, on=["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION"], how="outer")
    return merged

def _eff(row: pd.Series, suffix: str) -> float:
    """Prefer TS% > eFG% > FG% (returns NaN if none present)."""
    for col in (f"TS_PCT_{suffix}", f"EFG_PCT_{suffix}", f"FG_PCT_{suffix}"):
        val = row.get(col, np.nan)
        if pd.notna(val):
            return float(val)
    return np.nan

def compute_clutch_score(row: pd.Series) -> float:
    """
    Simple, interpretable clutch statistic:
    - Weighted average of TS% (or eFG%/FG%) across REG, CLU, PO using WEIGHTS.
    - Adds a mild volume bonus based on clutch FGA share (so 1/1 doesn't beat 50/40).
    """
    reg = _eff(row, "REG")
    clu = _eff(row, "CLU")
    po  = _eff(row, "PO")

    vals, wts = [], []
    if pd.notna(reg): vals.append(reg); wts.append(WEIGHTS["REG"])
    if pd.notna(clu): vals.append(clu); wts.append(WEIGHTS["CLU"])
    if pd.notna(po):  vals.append(po);  wts.append(WEIGHTS["PO"])

    if not vals:
        return np.nan

    base = float(np.average(vals, weights=wts))

    # Light clutch volume adjustment: clutch FGA / (reg FGA + tiny eps)
    fga_reg = row.get("FGA_REG", np.nan)
    fga_clu = row.get("FGA_CLU", np.nan)
    volume_adj = 0.0
    if pd.notna(fga_reg) and pd.notna(fga_clu) and (fga_reg > 0):
        share = min(float(fga_clu) / float(fga_reg + 1e-9), 1.0)   # cap to [0,1]
        volume_adj = CLUTCH_VOLUME_WEIGHT * share

    return base + volume_adj

def standardize_clutch_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CLUTCH_SCORE to -10 to +10 scale using MinMaxScaler.
    This creates a new column CLUTCH_SCORE_STD that Brian can use for modeling.
    """
    df = df.copy()
    
    # Only standardize rows with valid clutch scores
    valid_mask = df["CLUTCH_SCORE"].notna()
    
    if valid_mask.sum() == 0:
        df["CLUTCH_SCORE_STD"] = np.nan
        return df
    
    # Fit scaler on valid scores
    scaler = MinMaxScaler(feature_range=(-10, 10))
    scores = df.loc[valid_mask, "CLUTCH_SCORE"].values.reshape(-1, 1)
    
    # Transform and assign
    df.loc[valid_mask, "CLUTCH_SCORE_STD"] = scaler.fit_transform(scores).flatten()
    df.loc[~valid_mask, "CLUTCH_SCORE_STD"] = np.nan
    
    return df

# -----------------------
# Runner
# -----------------------
def run_one_season(season: str) -> pd.DataFrame:
    print(f"\n=== {season} ===")
    reg = fetch_regular(season, per_mode="PerGame")
    clu = fetch_clutch(season, per_mode="PerGame", season_type="Regular Season")
    try:
        po  = fetch_playoffs(season, per_mode="PerGame")
    except Exception:
        po = pd.DataFrame()

    merged = merge_frames(reg, clu, po)
    # Filter out tiny-minute players so small samples don't dominate
    if "MIN_REG" in merged.columns:
        merged = merged[merged["MIN_REG"].fillna(0) >= 10].copy()

    merged["CLUTCH_SCORE"] = merged.apply(compute_clutch_score, axis=1)
    merged = standardize_clutch_scores(merged)  # Add standardized -10 to +10 score
    merged["SEASON"] = season

    # Save
    out_fp = OUT_DIR / f"clutch_scores_{season.replace('-', '')}.csv"
    merged.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}")

    # Show quick top-10 with both scores
    preview_cols = ["PLAYER_NAME","TEAM_ABBREVIATION","CLUTCH_SCORE","CLUTCH_SCORE_STD",
                    "TS_PCT_REG","TS_PCT_CLU","TS_PCT_PO","FGA_REG","FGA_CLU"]
    for c in preview_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    print(merged[preview_cols].sort_values("CLUTCH_SCORE", ascending=False).head(10))

    return merged

def run_all(seasons: List[str]) -> pd.DataFrame:
    frames = [run_one_season(s) for s in seasons]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT_DIR / "clutch_scores_all_seasons.csv", index=False)
    print(f"\nSaved combined: {OUT_DIR / 'clutch_scores_all_seasons.csv'}")
    return all_df

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Tip: if you get rate-limit errors, increase REQUEST_PAUSE or run one season at a time.
    run_all(SEASONS)