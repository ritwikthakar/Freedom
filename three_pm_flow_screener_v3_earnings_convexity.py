"""
3PM Options Flow Screener v2
----------------------------
Adds three confirmation datasets to the existing 3PM workflow:
1) % increase in volatility
2) % decrease in volatility
3) % increase/decrease in open interest

Designed for Barchart CSV exports:
- options-flow-*.csv
- unusual-stock-options-activity-*.csv
- stocks-highest-implied-volatility-*.csv
- stocks-increase-percent-change-in-volatility-*.csv
- stocks-decrease-percent-change-in-volatility-*.csv
- stocks-increase-change-in-open-interest-*.csv
- stocks-decrease-change-in-open-interest-*.csv

Run:
    python three_pm_flow_screener_v2.py --data-dir "C:/Users/ritwi/Downloads" --top 25

Outputs:
    3pm_screener_ranked.csv
    3pm_screener_bullish.csv
    3pm_screener_bearish.csv
    3pm_screener_vol_expansion.csv
    3pm_screener_earnings_convexity.csv
    3pm_screener_put_speculation.csv
    3pm_screener_put_hedging.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

DEFAULT_PATTERNS = {
    "flow": "options-flow-*.csv",
    "unusual": "unusual-stock-options-activity-*.csv",
    "highest_iv": "stocks-highest-implied-volatility-*.csv",
    "iv_up": "stocks-increase-percent-change-in-volatility-*.csv",
    "iv_down": "stocks-decrease-percent-change-in-volatility-*.csv",
    "oi_up": "stocks-increase-change-in-open-interest-*.csv",
    "oi_down": "stocks-decrease-change-in-open-interest-*.csv",
}

DEFAULT_AVOID_EARNINGS_DTE = 3


@dataclass
class LoadedFiles:
    flow: pd.DataFrame
    unusual: pd.DataFrame
    highest_iv: pd.DataFrame
    iv_up: pd.DataFrame
    iv_down: pd.DataFrame
    oi_up: pd.DataFrame
    oi_down: pd.DataFrame
    paths: Dict[str, str]


# -----------------------------
# Helpers
# -----------------------------

def latest_file(data_dir: str, pattern: str) -> Optional[str]:
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_csv_safe(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return pd.DataFrame()


def clean_symbol(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def to_number(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "").replace("+", "")
    s = s.replace("%", "")
    if s in {"", "nan", "None", "--"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return df[col].map(to_number).fillna(default).astype(float)


def normalize_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if s.empty or s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    # percentile rank is more robust than min/max because Barchart rows often have huge outliers
    return s.rank(pct=True).fillna(0) * 100


def weighted_avg(value: pd.Series, weight: pd.Series) -> float:
    value = pd.to_numeric(value, errors="coerce")
    weight = pd.to_numeric(weight, errors="coerce").clip(lower=0).fillna(0)
    mask = value.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return float(value.mean()) if value.notna().any() else np.nan
    return float(np.average(value[mask], weights=weight[mask]))


def infer_aggressor_sign(df: pd.DataFrame) -> pd.Series:
    """
    Barchart flow often has Side as bid/ask/mid and Type as Call/Put.
    We convert to directional premium:
      ask-side call = bullish, ask-side put = bearish
      bid-side call = bearish/sell call, bid-side put = bullish/sell put
      mid = weaker signal, direction by option type only at 35% weight
    """
    opt_type = df.get("Type", pd.Series("", index=df.index)).astype(str).str.lower()
    side = df.get("Side", pd.Series("", index=df.index)).astype(str).str.lower()

    sign = pd.Series(0.0, index=df.index)

    call = opt_type.str.contains("call", na=False)
    put = opt_type.str.contains("put", na=False)
    ask = side.str.contains("ask", na=False)
    bid = side.str.contains("bid", na=False)
    mid = side.str.contains("mid", na=False) | side.eq("")

    sign.loc[call & ask] = 1.0
    sign.loc[put & ask] = -1.0
    sign.loc[call & bid] = -0.8
    sign.loc[put & bid] = 0.8
    sign.loc[call & mid] = 0.35
    sign.loc[put & mid] = -0.35
    return sign


def bucket_dte(dte: float) -> str:
    if pd.isna(dte):
        return "unknown"
    if dte <= 7:
        return "0-7d"
    if dte <= 21:
        return "8-21d"
    if dte <= 60:
        return "22-60d"
    if dte <= 180:
        return "61-180d"
    return "180d+"


# -----------------------------
# Load data
# -----------------------------

def load_files(data_dir: str, patterns: Dict[str, str] = DEFAULT_PATTERNS) -> LoadedFiles:
    paths = {name: latest_file(data_dir, pat) for name, pat in patterns.items()}
    for name, path in paths.items():
        print(f"{name:10s}: {os.path.basename(path) if path else 'NOT FOUND'}")
    return LoadedFiles(
        flow=read_csv_safe(paths["flow"]),
        unusual=read_csv_safe(paths["unusual"]),
        highest_iv=read_csv_safe(paths["highest_iv"]),
        iv_up=read_csv_safe(paths["iv_up"]),
        iv_down=read_csv_safe(paths["iv_down"]),
        oi_up=read_csv_safe(paths["oi_up"]),
        oi_down=read_csv_safe(paths["oi_down"]),
        paths={k: v or "" for k, v in paths.items()},
    )


# -----------------------------
# Aggregation blocks
# -----------------------------

def aggregate_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    x = df.copy()
    x["Symbol"] = clean_symbol(x["Symbol"])
    x["Premium_num"] = num_series(x, "Premium")
    x["Volume_num"] = num_series(x, "Volume")
    x["OpenInt_num"] = num_series(x, "Open Int")
    x["Delta_num"] = num_series(x, "Delta")
    x["AbsDelta_num"] = x["Delta_num"].abs()
    x["DTE_num"] = num_series(x, "DTE")
    x["direction_sign"] = infer_aggressor_sign(x)
    x["signed_premium"] = x["Premium_num"] * x["direction_sign"]

    opt_type = x.get("Type", pd.Series("", index=x.index)).astype(str).str.lower()
    side = x.get("Side", pd.Series("", index=x.index)).astype(str).str.lower()
    call = opt_type.str.contains("call", na=False)
    put = opt_type.str.contains("put", na=False)
    ask = side.str.contains("ask", na=False)
    bid = side.str.contains("bid", na=False)
    mid = side.str.contains("mid", na=False) | side.eq("")
    short_dte = x["DTE_num"].le(14) | x["DTE_num"].eq(0)
    very_short_dte = x["DTE_num"].le(7) | x["DTE_num"].eq(0)
    otmish = x["AbsDelta_num"].le(0.35)
    atmish = x["AbsDelta_num"].between(0.35, 0.65, inclusive="both")

    x["call_premium"] = np.where(call, x["Premium_num"], 0)
    x["put_premium"] = np.where(put, x["Premium_num"], 0)
    x["ask_call_premium"] = np.where(call & ask, x["Premium_num"], 0)
    x["bid_call_premium"] = np.where(call & bid, x["Premium_num"], 0)
    x["ask_put_premium"] = np.where(put & ask, x["Premium_num"], 0)
    x["bid_put_premium"] = np.where(put & bid, x["Premium_num"], 0)
    x["mid_put_premium"] = np.where(put & mid, x["Premium_num"], 0)

    # Convexity / catalyst-style sub-buckets.
    # Short-DTE ask-side calls are the key input for bullish earnings/gap risk candidates.
    # Short-DTE ask-side puts with OTM deltas are more likely speculative downside bets.
    # ATM/bid-side puts are more likely protective hedges or monetization/unwind activity.
    x["short_dte_call_ask_premium"] = np.where(call & ask & short_dte, x["Premium_num"], 0)
    x["very_short_dte_call_ask_premium"] = np.where(call & ask & very_short_dte, x["Premium_num"], 0)
    x["short_dte_put_ask_premium"] = np.where(put & ask & short_dte, x["Premium_num"], 0)
    x["very_short_dte_put_ask_premium"] = np.where(put & ask & very_short_dte, x["Premium_num"], 0)
    x["otm_put_ask_premium"] = np.where(put & ask & otmish, x["Premium_num"], 0)
    x["atm_put_premium"] = np.where(put & atmish, x["Premium_num"], 0)
    x["atm_put_bid_or_mid_premium"] = np.where(put & atmish & (bid | mid), x["Premium_num"], 0)
    x["otm_call_ask_premium"] = np.where(call & ask & otmish, x["Premium_num"], 0)
    x["atm_call_ask_premium"] = np.where(call & ask & atmish, x["Premium_num"], 0)
    x["to_open"] = x.get("*", pd.Series("", index=x.index)).astype(str).str.lower().str.contains("open", na=False).astype(int)

    g = x.groupby("Symbol", as_index=False).agg(
        flow_trades=("Symbol", "size"),
        total_premium=("Premium_num", "sum"),
        net_signed_premium=("signed_premium", "sum"),
        call_premium=("call_premium", "sum"),
        put_premium=("put_premium", "sum"),
        ask_call_premium=("ask_call_premium", "sum"),
        bid_call_premium=("bid_call_premium", "sum"),
        ask_put_premium=("ask_put_premium", "sum"),
        bid_put_premium=("bid_put_premium", "sum"),
        mid_put_premium=("mid_put_premium", "sum"),
        short_dte_call_ask_premium=("short_dte_call_ask_premium", "sum"),
        very_short_dte_call_ask_premium=("very_short_dte_call_ask_premium", "sum"),
        short_dte_put_ask_premium=("short_dte_put_ask_premium", "sum"),
        very_short_dte_put_ask_premium=("very_short_dte_put_ask_premium", "sum"),
        otm_put_ask_premium=("otm_put_ask_premium", "sum"),
        atm_put_premium=("atm_put_premium", "sum"),
        atm_put_bid_or_mid_premium=("atm_put_bid_or_mid_premium", "sum"),
        otm_call_ask_premium=("otm_call_ask_premium", "sum"),
        atm_call_ask_premium=("atm_call_ask_premium", "sum"),
        total_volume=("Volume_num", "sum"),
        avg_delta=("Delta_num", "mean"),
        min_dte=("DTE_num", "min"),
        avg_dte=("DTE_num", "mean"),
        to_open_trades=("to_open", "sum"),
    )
    g["call_put_premium_ratio"] = g["call_premium"] / g["put_premium"].replace(0, np.nan)
    g["ask_call_share"] = g["ask_call_premium"] / g["call_premium"].replace(0, np.nan)
    g["ask_put_share"] = g["ask_put_premium"] / g["put_premium"].replace(0, np.nan)
    g["short_dte_call_ask_share"] = g["short_dte_call_ask_premium"] / g["call_premium"].replace(0, np.nan)
    g["short_dte_put_ask_share"] = g["short_dte_put_ask_premium"] / g["put_premium"].replace(0, np.nan)
    g["put_ask_to_bid_ratio"] = g["ask_put_premium"] / g["bid_put_premium"].replace(0, np.nan)
    g["flow_bias"] = np.where(g["net_signed_premium"] > 0, "Bullish", np.where(g["net_signed_premium"] < 0, "Bearish", "Neutral"))
    return g

def aggregate_unusual(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    x = df.copy()
    x["Symbol"] = clean_symbol(x["Symbol"])
    x["Volume_num"] = num_series(x, "Volume")
    x["OpenInt_num"] = num_series(x, "Open Int")
    x["VolOI_num"] = num_series(x, "Vol/OI")
    x["Delta_num"] = num_series(x, "Delta")
    x["DTE_num"] = num_series(x, "DTE")
    x["call_volume"] = np.where(x["Type"].astype(str).str.lower().str.contains("call"), x["Volume_num"], 0)
    x["put_volume"] = np.where(x["Type"].astype(str).str.lower().str.contains("put"), x["Volume_num"], 0)

    g = x.groupby("Symbol", as_index=False).agg(
        unusual_rows=("Symbol", "size"),
        unusual_volume=("Volume_num", "sum"),
        unusual_call_volume=("call_volume", "sum"),
        unusual_put_volume=("put_volume", "sum"),
        max_vol_oi=("VolOI_num", "max"),
        avg_unusual_delta=("Delta_num", "mean"),
        min_unusual_dte=("DTE_num", "min"),
    )
    g["unusual_call_put_ratio"] = g["unusual_call_volume"] / g["unusual_put_volume"].replace(0, np.nan)
    return g


def aggregate_iv_change(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    x = df.copy()
    x["Symbol"] = clean_symbol(x["Symbol"])
    x["Volume_num"] = num_series(x, "Volume")
    x["IVChg_num"] = num_series(x, "IV %Chg")
    x["IV_num"] = num_series(x, "IV")
    x["Delta_num"] = num_series(x, "Delta")
    x["DTE_num"] = num_series(x, "DTE")
    x["call_rows"] = x["Type"].astype(str).str.lower().str.contains("call", na=False).astype(int)
    x["put_rows"] = x["Type"].astype(str).str.lower().str.contains("put", na=False).astype(int)

    g = x.groupby("Symbol", as_index=False).agg(
        **{
            f"{prefix}_rows": ("Symbol", "size"),
            f"{prefix}_volume": ("Volume_num", "sum"),
            f"{prefix}_avg_iv_chg": ("IVChg_num", "mean"),
            f"{prefix}_max_abs_iv_chg": ("IVChg_num", lambda s: float(np.nanmax(np.abs(s))) if len(s) else np.nan),
            f"{prefix}_avg_iv": ("IV_num", "mean"),
            f"{prefix}_avg_delta": ("Delta_num", "mean"),
            f"{prefix}_min_dte": ("DTE_num", "min"),
            f"{prefix}_call_rows": ("call_rows", "sum"),
            f"{prefix}_put_rows": ("put_rows", "sum"),
        }
    )
    return g


def aggregate_oi_change(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    x = df.copy()
    x["Symbol"] = clean_symbol(x["Symbol"])
    x["Volume_num"] = num_series(x, "Volume")
    x["OpenInt_num"] = num_series(x, "Open Int")
    x["OIChg_num"] = num_series(x, "OI %Chg")
    x["VolOI_num"] = num_series(x, "Vol/OI")
    x["Delta_num"] = num_series(x, "Delta")
    x["DTE_num"] = num_series(x, "DTE")
    x["call_rows"] = x["Type"].astype(str).str.lower().str.contains("call", na=False).astype(int)
    x["put_rows"] = x["Type"].astype(str).str.lower().str.contains("put", na=False).astype(int)

    g = x.groupby("Symbol", as_index=False).agg(
        **{
            f"{prefix}_rows": ("Symbol", "size"),
            f"{prefix}_volume": ("Volume_num", "sum"),
            f"{prefix}_open_interest": ("OpenInt_num", "sum"),
            f"{prefix}_avg_oi_chg": ("OIChg_num", "mean"),
            f"{prefix}_max_abs_oi_chg": ("OIChg_num", lambda s: float(np.nanmax(np.abs(s))) if len(s) else np.nan),
            f"{prefix}_max_vol_oi": ("VolOI_num", "max"),
            f"{prefix}_avg_delta": ("Delta_num", "mean"),
            f"{prefix}_min_dte": ("DTE_num", "min"),
            f"{prefix}_call_rows": ("call_rows", "sum"),
            f"{prefix}_put_rows": ("put_rows", "sum"),
        }
    )
    return g


def aggregate_highest_iv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    x = df.copy()
    x["Symbol"] = clean_symbol(x["Symbol"])
    x["Volume_num"] = num_series(x, "Volume")
    x["IV_num"] = num_series(x, "IV")
    x["ImpVol_num"] = num_series(x, "Imp Vol")
    x["DTE_num"] = num_series(x, "DTE")
    g = x.groupby("Symbol", as_index=False).agg(
        highest_iv_rows=("Symbol", "size"),
        highest_iv_volume=("Volume_num", "sum"),
        max_iv=("IV_num", "max"),
        avg_high_iv=("IV_num", "mean"),
        min_high_iv_dte=("DTE_num", "min"),
    )
    return g


def merge_blocks(blocks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out: Optional[pd.DataFrame] = None
    for block in blocks:
        if block is None or block.empty or "Symbol" not in block.columns:
            continue
        out = block if out is None else out.merge(block, on="Symbol", how="outer")
    if out is None:
        return pd.DataFrame()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(0)
    return out.fillna("")


# -----------------------------
# Scoring
# -----------------------------

def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()

    # Core directional scores
    x["flow_score"] = normalize_0_100(x.get("net_signed_premium", 0).abs())
    x["premium_quality_score"] = normalize_0_100(x.get("total_premium", 0))
    x["unusual_score"] = normalize_0_100(x.get("unusual_volume", 0)) * 0.6 + normalize_0_100(x.get("max_vol_oi", 0)) * 0.4

    # New confirmation layers
    x["iv_expansion_score"] = normalize_0_100(x.get("iv_up_volume", 0)) * 0.45 + normalize_0_100(x.get("iv_up_avg_iv_chg", 0).abs()) * 0.55
    x["iv_crush_score"] = normalize_0_100(x.get("iv_down_volume", 0)) * 0.45 + normalize_0_100(x.get("iv_down_avg_iv_chg", 0).abs()) * 0.55
    x["oi_accumulation_score"] = normalize_0_100(x.get("oi_up_open_interest", 0)) * 0.5 + normalize_0_100(x.get("oi_up_volume", 0)) * 0.25 + normalize_0_100(x.get("oi_up_avg_oi_chg", 0).abs()) * 0.25
    x["oi_unwind_score"] = normalize_0_100(x.get("oi_down_open_interest", 0)) * 0.5 + normalize_0_100(x.get("oi_down_volume", 0)) * 0.25 + normalize_0_100(x.get("oi_down_avg_oi_chg", 0).abs()) * 0.25
    x["high_iv_risk_score"] = normalize_0_100(x.get("max_iv", 0)) * 0.55 + normalize_0_100(x.get("highest_iv_volume", 0)) * 0.45

    # Directional bias from flow and unusual call/put volume
    x["flow_direction"] = np.sign(pd.to_numeric(x.get("net_signed_premium", 0), errors="coerce").fillna(0))
    unusual_call = pd.to_numeric(x.get("unusual_call_volume", 0), errors="coerce").fillna(0)
    unusual_put = pd.to_numeric(x.get("unusual_put_volume", 0), errors="coerce").fillna(0)
    x["unusual_direction"] = np.sign(unusual_call - unusual_put)

    oi_call = pd.to_numeric(x.get("oi_up_call_rows", 0), errors="coerce").fillna(0)
    oi_put = pd.to_numeric(x.get("oi_up_put_rows", 0), errors="coerce").fillna(0)
    x["oi_direction"] = np.sign(oi_call - oi_put)

    x["direction_score"] = (x["flow_direction"] * 0.60) + (x["unusual_direction"] * 0.25) + (x["oi_direction"] * 0.15)
    x["bias"] = np.where(x["direction_score"] > 0.15, "Bullish", np.where(x["direction_score"] < -0.15, "Bearish", "Vol/Neutral"))

    # -----------------------------
    # Earnings convexity + put classifier
    # -----------------------------
    # Bullish earnings convexity tries to detect DELL-style setups:
    # large ask-side call premium, short-DTE calls, OTM/ATM call demand,
    # IV expansion, OI accumulation, and call-heavy unusual activity.
    x["call_sweep_score"] = normalize_0_100(x.get("ask_call_premium", 0)) * 0.45 + normalize_0_100(x.get("short_dte_call_ask_premium", 0)) * 0.35 + normalize_0_100(x.get("otm_call_ask_premium", 0)) * 0.20
    x["call_flow_quality_score"] = (
        pd.to_numeric(x.get("ask_call_share", 0), errors="coerce").fillna(0).clip(0, 1) * 55
        + pd.to_numeric(x.get("short_dte_call_ask_share", 0), errors="coerce").fillna(0).clip(0, 1) * 45
    )
    x["bullish_earnings_convexity_score"] = (
        x["call_sweep_score"] * 0.35
        + x["call_flow_quality_score"] * 0.20
        + x["iv_expansion_score"] * 0.15
        + x["oi_accumulation_score"] * 0.15
        + x["unusual_score"] * 0.10
        + np.where(x["bias"].eq("Bullish"), 5, 0)
    ).clip(lower=0, upper=100)

    # Put speculation = aggressive downside buying: ask-side, short-DTE, OTM-ish puts,
    # bearish bias, IV expansion, and OI accumulation.
    x["put_speculation_score"] = (
        normalize_0_100(x.get("ask_put_premium", 0)) * 0.25
        + normalize_0_100(x.get("short_dte_put_ask_premium", 0)) * 0.25
        + normalize_0_100(x.get("otm_put_ask_premium", 0)) * 0.20
        + x["iv_expansion_score"] * 0.12
        + x["oi_accumulation_score"] * 0.10
        + np.where(x["bias"].eq("Bearish"), 8, 0)
    ).clip(lower=0, upper=100)

    # Put hedge = large put premium that looks less directional: ATM puts, bid/mid execution,
    # high IV / event protection, but without clear bearish directional confirmation.
    x["put_hedging_score"] = (
        normalize_0_100(x.get("atm_put_premium", 0)) * 0.25
        + normalize_0_100(x.get("atm_put_bid_or_mid_premium", 0)) * 0.25
        + normalize_0_100(x.get("bid_put_premium", 0) + x.get("mid_put_premium", 0)) * 0.15
        + x["high_iv_risk_score"] * 0.15
        + x["iv_expansion_score"] * 0.10
        + np.where(~x["bias"].eq("Bearish"), 10, 0)
    ).clip(lower=0, upper=100)

    x["put_flow_classification"] = np.select(
        [
            (x["put_speculation_score"] >= 60) & (x["put_speculation_score"] >= x["put_hedging_score"] + 8),
            (x["put_hedging_score"] >= 60) & (x["put_hedging_score"] >= x["put_speculation_score"] + 8),
            (x.get("put_premium", 0) > 0) & (x["put_speculation_score"] >= 45) & (x["put_hedging_score"] >= 45),
        ],
        ["Speculative bearish puts", "Likely protective hedging", "Mixed put flow"],
        default="No major put signal",
    )

    x["earnings_convexity_flag"] = np.select(
        [
            x["bullish_earnings_convexity_score"] >= 75,
            x["bullish_earnings_convexity_score"].between(60, 75, inclusive="left"),
            x["put_speculation_score"] >= 70,
        ],
        ["High bullish convexity", "Watch bullish convexity", "High bearish convexity"],
        default="No convexity alert",
    )

    # Main 3PM score: accumulation + unusual + flow, with penalties for IV crush/unwind/high IV tail risk.
    # Earnings convexity is deliberately a small additive layer so normal scanner behavior is preserved.
    x["setup_score"] = (
        x["flow_score"] * 0.23
        + x["premium_quality_score"] * 0.09
        + x["unusual_score"] * 0.18
        + x["oi_accumulation_score"] * 0.20
        + x["iv_expansion_score"] * 0.12
        + x["bullish_earnings_convexity_score"] * 0.05
        + x["put_speculation_score"] * 0.05
        - x["oi_unwind_score"] * 0.06
        - x["iv_crush_score"] * 0.04
        - x["high_iv_risk_score"] * 0.02
    )

    # Trade archetype
    conditions = [
        (x["bullish_earnings_convexity_score"] >= 75),
        (x["put_speculation_score"] >= 70) & (x["put_flow_classification"].eq("Speculative bearish puts")),
        (x["put_hedging_score"] >= 70) & (x["put_flow_classification"].eq("Likely protective hedging")),
        (x["bias"].eq("Bullish") & (x["iv_expansion_score"] >= 60) & (x["oi_accumulation_score"] >= 50)),
        (x["bias"].eq("Bullish") & (x["iv_expansion_score"] < 60) & (x["oi_accumulation_score"] >= 50)),
        (x["bias"].eq("Bearish") & (x["iv_expansion_score"] >= 50) & (x["oi_accumulation_score"] >= 40)),
        (x["bias"].eq("Bearish") & (x["iv_crush_score"] >= 50)),
        (x["bias"].eq("Vol/Neutral") & (x["iv_expansion_score"] >= 65)),
    ]
    choices = [
        "Bullish earnings convexity: call debit spread / call diagonal",
        "Bearish put speculation: bear put spread / put diagonal",
        "Likely hedge flow: avoid shorting unless price confirms",
        "Bullish expansion: call diagonal / bull call spread",
        "Bullish accumulation: call diagonal / debit spread",
        "Bearish expansion: put diagonal / bear put spread",
        "Bearish/unwind: bear call spread or avoid longs",
        "Vol expansion: ratio backspread / defined-risk strangle",
    ]
    x["preferred_structure"] = np.select(conditions, choices, default="Watchlist only / needs confirmation")

    # Useful flags
    x["confirmation"] = np.select(
        [
            x["bullish_earnings_convexity_score"] >= 75,
            x["put_flow_classification"].eq("Speculative bearish puts") & (x["put_speculation_score"] >= 60),
            x["put_flow_classification"].eq("Likely protective hedging") & (x["put_hedging_score"] >= 60),
            (x["oi_accumulation_score"] >= 60) & (x["iv_expansion_score"] >= 50),
            (x["oi_accumulation_score"] >= 60) & (x["iv_expansion_score"] < 50),
            (x["oi_unwind_score"] >= 55) | (x["iv_crush_score"] >= 55),
        ],
        ["Bullish convexity + call sweep", "Speculative bearish puts", "Put hedge / protection", "OI + IV expansion", "OI accumulation only", "Unwind / IV crush risk"],
        default="Mixed",
    )

    x["rank"] = x["setup_score"].rank(ascending=False, method="dense").astype(int)
    x = x.sort_values(["setup_score", "total_premium", "unusual_volume"], ascending=False)
    return x

# -----------------------------
# Main
# -----------------------------

def build_screener(data_dir: str, output_dir: str, top: int = 50, min_premium: float = 0) -> pd.DataFrame:
    loaded = load_files(data_dir)
    blocks = [
        aggregate_flow(loaded.flow),
        aggregate_unusual(loaded.unusual),
        aggregate_highest_iv(loaded.highest_iv),
        aggregate_iv_change(loaded.iv_up, "iv_up"),
        aggregate_iv_change(loaded.iv_down, "iv_down"),
        aggregate_oi_change(loaded.oi_up, "oi_up"),
        aggregate_oi_change(loaded.oi_down, "oi_down"),
    ]
    merged = merge_blocks(blocks)
    ranked = score_candidates(merged)

    if ranked.empty:
        os.makedirs(output_dir, exist_ok=True)
        empty_outputs = [
            "3pm_screener_ranked.csv",
            "3pm_screener_bullish.csv",
            "3pm_screener_bearish.csv",
            "3pm_screener_vol_expansion.csv",
            "3pm_screener_earnings_convexity.csv",
            "3pm_screener_put_speculation.csv",
            "3pm_screener_put_hedging.csv",
        ]
        for fname in empty_outputs:
            pd.DataFrame().to_csv(os.path.join(output_dir, fname), index=False)
        print("No usable rows found. Check that the data directory contains the expected Barchart CSV exports.")
        return ranked

    if min_premium > 0 and "total_premium" in ranked.columns:
        ranked = ranked[ranked["total_premium"] >= min_premium].copy()

    os.makedirs(output_dir, exist_ok=True)

    # Column order: keep high-signal fields first, then everything else
    first_cols = [
        "rank", "Symbol", "bias", "setup_score", "preferred_structure", "confirmation",
        "earnings_convexity_flag", "bullish_earnings_convexity_score",
        "put_flow_classification", "put_speculation_score", "put_hedging_score",
        "net_signed_premium", "total_premium", "call_premium", "put_premium",
        "ask_call_premium", "short_dte_call_ask_premium", "otm_call_ask_premium",
        "ask_put_premium", "short_dte_put_ask_premium", "otm_put_ask_premium",
        "atm_put_premium", "atm_put_bid_or_mid_premium", "flow_trades",
        "unusual_volume", "max_vol_oi", "oi_accumulation_score", "iv_expansion_score",
        "oi_unwind_score", "iv_crush_score", "high_iv_risk_score",
        "min_dte", "avg_dte", "min_unusual_dte", "iv_up_min_dte", "oi_up_min_dte",
    ]
    first_cols = [c for c in first_cols if c in ranked.columns]
    other_cols = [c for c in ranked.columns if c not in first_cols]
    ranked = ranked[first_cols + other_cols]

    # =========================
    # STORE OUTPUTS AS VARIABLES
    # =========================
    
    ranked_df = ranked.copy()
    
    bullish_df = ranked[
        ranked["bias"].eq("Bullish")
    ].head(top).copy()
    
    bearish_df = ranked[
        ranked["bias"].eq("Bearish")
    ].head(top).copy()
    
    vol_expansion_df = ranked[
        ranked["preferred_structure"].str.contains(
            "Vol expansion",
            na=False
        )
    ].head(top).copy()

    earnings_convexity_df = ranked[
        ranked["earnings_convexity_flag"].ne("No convexity alert")
    ].sort_values(
        ["bullish_earnings_convexity_score", "put_speculation_score", "total_premium"],
        ascending=False
    ).head(top).copy()

    put_speculation_df = ranked[
        ranked["put_flow_classification"].eq("Speculative bearish puts")
    ].sort_values(
        ["put_speculation_score", "total_premium"],
        ascending=False
    ).head(top).copy()

    put_hedging_df = ranked[
        ranked["put_flow_classification"].eq("Likely protective hedging")
    ].sort_values(
        ["put_hedging_score", "total_premium"],
        ascending=False
    ).head(top).copy()


    # =========================
    # SAVE CSV FILES
    # =========================
    
    ranked_df.to_csv(
        os.path.join(output_dir, "3pm_screener_ranked.csv"),
        index=False
    )
    
    bullish_df.to_csv(
        os.path.join(output_dir, "3pm_screener_bullish.csv"),
        index=False
    )
    
    bearish_df.to_csv(
        os.path.join(output_dir, "3pm_screener_bearish.csv"),
        index=False
    )
    
    vol_expansion_df.to_csv(
        os.path.join(output_dir, "3pm_screener_vol_expansion.csv"),
        index=False
    )

    earnings_convexity_df.to_csv(
        os.path.join(output_dir, "3pm_screener_earnings_convexity.csv"),
        index=False
    )

    put_speculation_df.to_csv(
        os.path.join(output_dir, "3pm_screener_put_speculation.csv"),
        index=False
    )

    put_hedging_df.to_csv(
        os.path.join(output_dir, "3pm_screener_put_hedging.csv"),
        index=False
    )


    print("\nTop candidates:")
    cols = [
        "rank", "Symbol", "bias", "setup_score", "preferred_structure",
        "confirmation", "earnings_convexity_flag", "put_flow_classification",
        "bullish_earnings_convexity_score", "put_speculation_score",
        "put_hedging_score", "net_signed_premium", "total_premium"
    ]
    cols = [c for c in cols if c in ranked.columns]
    print(ranked[cols].head(top).to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")
    globals()["ranked_df"] = ranked_df
    globals()["bullish_df"] = bullish_df
    globals()["bearish_df"] = bearish_df
    globals()["vol_expansion_df"] = vol_expansion_df
    globals()["earnings_convexity_df"] = earnings_convexity_df
    globals()["put_speculation_df"] = put_speculation_df
    globals()["put_hedging_df"] = put_hedging_df
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3PM Barchart options flow screener with IV/OI confirmation layers.")
    parser.add_argument("--data-dir", default=".", help="Folder containing Barchart CSV exports.")
    parser.add_argument("--output-dir", default=".", help="Folder to save screener output CSVs.")
    parser.add_argument("--top", type=int, default=25, help="Number of rows to print/save in side-specific files.")
    parser.add_argument("--min-premium", type=float, default=0, help="Optional minimum total premium filter.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_screener(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top=args.top,
        min_premium=args.min_premium,
    )

