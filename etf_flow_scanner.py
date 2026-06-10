"""
ETF Flow + OI + IV Scanner
--------------------------
Builds ticker-level ETF trade candidates from Barchart CSV exports:
  - options flow
  - unusual ETF options activity
  - increase/decrease change in open interest
  - increase/decrease percent change in volatility
  - highest implied volatility

Usage examples:
  python etf_flow_scanner.py --input-dir "C:/Users/ritwi/Downloads" --date 2026-05-19
  python etf_flow_scanner.py --input-dir . --out-dir ./scanner_output

Outputs:
  etf_scanner_ranked.csv
  etf_scanner_bullish.csv
  etf_scanner_bearish.csv
  etf_scanner_vol_expansion.csv
  etf_scanner_range_candidates.csv
  etf_scanner_summary.txt
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
# Helpers
# -----------------------------

FILE_PATTERNS = {
    "oi_increase": ["*etfs-increase-change-in-open-interest*.csv", "*increase-change-in-open-interest*.csv"],
    "oi_decrease": ["*etfs-decrease-change-in-open-interest*.csv", "*decrease-change-in-open-interest*.csv"],
    "iv_increase": ["*etfs-increase-percent-change-in-volatility*.csv", "*increase-percent-change-in-volatility*.csv"],
    "iv_decrease": ["*etfs-decrease-percent-change-in-volatility*.csv", "*decrease-percent-change-in-volatility*.csv"],
    "high_iv": ["*etfs-highest-implied-volatility*.csv", "*highest-implied-volatility*.csv"],
    "flow": ["*options-flow*.csv"],
    "unusual": ["*unusual-etf-options-activity*.csv", "*unusual*options*activity*.csv"],
}

ETF_PRIORITY = {
    # Broad/liquid ETFs get a small liquidity bump. Add more as needed.
    "SPY", "QQQ", "IWM", "SMH", "SOXX", "XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV",
    "TLT", "IEF", "HYG", "LQD", "GLD", "SLV", "GDX", "USO", "EEM", "IBIT", "TQQQ", "SQQQ",
}


def find_file(input_dir: str, patterns: List[str]) -> Optional[str]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(input_dir, pattern)))
    if not matches:
        return None
    # newest modified file wins if there are duplicates
    return max(matches, key=os.path.getmtime)


def read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return pd.DataFrame()


def to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "").replace("%", "")
    if s in {"", "-", "nan", "None"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if col in {"Symbol", "Type", "Side", "Code", "Time", "Exp Date", "Expires", "Bid x Size", "Ask x Size", "Moneyness", "*"}:
            continue
        sample = df[col].dropna().astype(str).head(20).tolist()
        if any(re.search(r"[$,%]|^-?\d", v.strip()) for v in sample):
            df[col] = df[col].map(to_num)
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.title().str.strip()
    return df


def weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    return float((values * weights).sum())


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0, np.nan) and not pd.isna(b) else 0.0


def zscore_clip(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - s.mean()) / std).clip(-clip, clip)


# -----------------------------
# Dataset summaries
# -----------------------------

def summarize_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    df = clean_numeric_cols(df)
    for col in ["Premium", "Size", "Volume", "Open Int", "Delta", "DTE", "Strike", "Price~"]:
        if col not in df.columns:
            df[col] = 0.0
    side = df.get("Side", pd.Series("", index=df.index)).astype(str).str.lower()
    typ = df.get("Type", pd.Series("", index=df.index)).astype(str).str.lower()
    premium = pd.to_numeric(df["Premium"], errors="coerce").fillna(0.0).abs()

    # Direction assumptions:
    # Call ask = bullish; call bid = bearish. Put ask = bearish; put bid = bullish.
    # Mid/unknown gets a weaker directional weight using option type only.
    direction = np.select(
        [
            (typ == "call") & side.str.contains("ask"),
            (typ == "call") & side.str.contains("bid"),
            (typ == "put") & side.str.contains("ask"),
            (typ == "put") & side.str.contains("bid"),
            (typ == "call"),
            (typ == "put"),
        ],
        [1.0, -1.0, -1.0, 1.0, 0.35, -0.35],
        default=0.0,
    )
    df["flow_signed_premium"] = premium * direction
    df["call_premium"] = np.where(typ == "call", premium, 0.0)
    df["put_premium"] = np.where(typ == "put", premium, 0.0)
    df["ask_premium"] = np.where(side.str.contains("ask"), premium, 0.0)
    df["bid_premium"] = np.where(side.str.contains("bid"), premium, 0.0)

    g = df.groupby("Symbol", as_index=False).agg(
        flow_trades=("Symbol", "size"),
        flow_premium=("Premium", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0).abs().sum())),
        flow_signed_premium=("flow_signed_premium", "sum"),
        flow_call_premium=("call_premium", "sum"),
        flow_put_premium=("put_premium", "sum"),
        flow_ask_premium=("ask_premium", "sum"),
        flow_bid_premium=("bid_premium", "sum"),
        flow_max_trade_premium=("Premium", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0).abs().max())),
        avg_flow_dte=("DTE", "mean"),
        avg_flow_delta=("Delta", "mean"),
        spot=("Price~", "last"),
    )
    g["flow_call_put_ratio"] = (g["flow_call_premium"] + 1) / (g["flow_put_premium"] + 1)
    g["flow_direction_score"] = g.apply(lambda r: safe_div(r["flow_signed_premium"], r["flow_premium"]), axis=1)
    return g


def summarize_unusual(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    df = clean_numeric_cols(df)
    for col in ["Volume", "Open Int", "Vol/OI", "Latest", "Delta", "DTE", "Price~", "Imp Vol"]:
        if col not in df.columns:
            df[col] = 0.0
    typ = df.get("Type", pd.Series("", index=df.index)).astype(str).str.lower()
    approx_premium = pd.to_numeric(df.get("Latest", 0), errors="coerce").fillna(0).abs() * pd.to_numeric(df["Volume"], errors="coerce").fillna(0) * 100
    dir_weight = np.where(typ == "call", 1.0, np.where(typ == "put", -1.0, 0.0))
    df["unusual_signed_notional"] = approx_premium * dir_weight
    df["unusual_call_notional"] = np.where(typ == "call", approx_premium, 0.0)
    df["unusual_put_notional"] = np.where(typ == "put", approx_premium, 0.0)
    g = df.groupby("Symbol", as_index=False).agg(
        unusual_rows=("Symbol", "size"),
        unusual_volume=("Volume", "sum"),
        avg_vol_oi=("Vol/OI", "mean"),
        max_vol_oi=("Vol/OI", "max"),
        unusual_signed_notional=("unusual_signed_notional", "sum"),
        unusual_call_notional=("unusual_call_notional", "sum"),
        unusual_put_notional=("unusual_put_notional", "sum"),
        avg_unusual_dte=("DTE", "mean"),
        spot_unusual=("Price~", "last"),
    )
    g["unusual_direction_score"] = g.apply(lambda r: safe_div(r["unusual_signed_notional"], abs(r["unusual_call_notional"]) + abs(r["unusual_put_notional"])), axis=1)
    return g


def summarize_oi(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    # kind: increase or decrease
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    df = clean_numeric_cols(df)
    for col in ["Open Int", "Volume", "OI %Chg", "Vol/OI", "Delta", "DTE", "Price~"]:
        if col not in df.columns:
            df[col] = 0.0
    typ = df.get("Type", pd.Series("", index=df.index)).astype(str).str.lower()
    oi = pd.to_numeric(df["Open Int"], errors="coerce").fillna(0)
    oi_chg = pd.to_numeric(df["OI %Chg"], errors="coerce").fillna(0)
    # Increase: calls bullish, puts bearish. Decrease: call unwind bearish, put unwind bullish.
    if kind == "increase":
        sign = np.where(typ == "call", 1.0, np.where(typ == "put", -1.0, 0.0))
        prefix = "oi_inc"
    else:
        sign = np.where(typ == "call", -1.0, np.where(typ == "put", 1.0, 0.0))
        prefix = "oi_dec"
    df[f"{prefix}_signed"] = sign * oi * np.log1p(abs(oi_chg))
    df[f"{prefix}_abs"] = oi * np.log1p(abs(oi_chg))
    g = df.groupby("Symbol", as_index=False).agg(
        **{
            f"{prefix}_rows": ("Symbol", "size"),
            f"{prefix}_open_int": ("Open Int", "sum"),
            f"{prefix}_avg_pct": ("OI %Chg", "mean"),
            f"{prefix}_max_abs_pct": ("OI %Chg", lambda x: float(pd.to_numeric(x, errors="coerce").abs().max())),
            f"{prefix}_signed": (f"{prefix}_signed", "sum"),
            f"{prefix}_abs": (f"{prefix}_abs", "sum"),
            f"{prefix}_avg_dte": ("DTE", "mean"),
        }
    )
    g[f"{prefix}_direction_score"] = g.apply(lambda r: safe_div(r[f"{prefix}_signed"], r[f"{prefix}_abs"]), axis=1)
    return g


def summarize_iv_change(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    df = clean_numeric_cols(df)
    for col in ["Volume", "IV %Chg", "IV", "Vega", "Delta", "DTE", "Price~"]:
        if col not in df.columns:
            df[col] = 0.0
    typ = df.get("Type", pd.Series("", index=df.index)).astype(str).str.lower()
    ivchg = pd.to_numeric(df["IV %Chg"], errors="coerce").fillna(0)
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    # IV increase in calls/puts is directional demand; IV decrease is vol crush / de-risking.
    dir_sign = np.where(typ == "call", 1.0, np.where(typ == "put", -1.0, 0.0))
    prefix = "iv_inc" if kind == "increase" else "iv_dec"
    df[f"{prefix}_pressure"] = dir_sign * vol * abs(ivchg)
    df[f"{prefix}_activity"] = vol * abs(ivchg)
    g = df.groupby("Symbol", as_index=False).agg(
        **{
            f"{prefix}_rows": ("Symbol", "size"),
            f"{prefix}_volume": ("Volume", "sum"),
            f"{prefix}_avg_pct": ("IV %Chg", "mean"),
            f"{prefix}_max_abs_pct": ("IV %Chg", lambda x: float(pd.to_numeric(x, errors="coerce").abs().max())),
            f"{prefix}_pressure": (f"{prefix}_pressure", "sum"),
            f"{prefix}_activity": (f"{prefix}_activity", "sum"),
            f"{prefix}_avg_iv": ("IV", "mean"),
            f"{prefix}_avg_dte": ("DTE", "mean"),
        }
    )
    g[f"{prefix}_direction_score"] = g.apply(lambda r: safe_div(r[f"{prefix}_pressure"], r[f"{prefix}_activity"]), axis=1)
    return g


def summarize_high_iv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns:
        return pd.DataFrame(columns=["Symbol"])
    df = clean_numeric_cols(df)
    for col in ["Volume", "IV", "Imp Vol", "Vega", "Delta", "DTE", "Price~"]:
        if col not in df.columns:
            df[col] = 0.0
    g = df.groupby("Symbol", as_index=False).agg(
        high_iv_rows=("Symbol", "size"),
        high_iv_volume=("Volume", "sum"),
        avg_high_iv=("IV", "mean"),
        max_high_iv=("IV", "max"),
        avg_high_iv_dte=("DTE", "mean"),
    )
    return g


# -----------------------------
# Final scoring
# -----------------------------

def merge_all(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty and "Symbol" in f.columns]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Symbol", how="outer")
    for col in out.columns:
        if col != "Symbol":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    # Direction score blends actual flow, unusual activity, OI shift, and IV demand.
    df["flow_component"] = zscore_clip(df.get("flow_signed_premium", 0)) + df.get("flow_direction_score", 0)
    df["unusual_component"] = zscore_clip(df.get("unusual_signed_notional", 0)) + df.get("unusual_direction_score", 0)
    df["oi_component"] = (
        0.8 * df.get("oi_inc_direction_score", 0)
        + 0.5 * df.get("oi_dec_direction_score", 0)
        + 0.4 * zscore_clip(df.get("oi_inc_signed", 0))
        + 0.25 * zscore_clip(df.get("oi_dec_signed", 0))
    )
    df["iv_direction_component"] = 0.7 * df.get("iv_inc_direction_score", 0) + 0.2 * df.get("iv_dec_direction_score", 0)

    df["directional_score"] = (
        2.5 * df["flow_component"]
        + 1.5 * df["unusual_component"]
        + 1.2 * df["oi_component"]
        + 0.8 * df["iv_direction_component"]
    )

    # Expansion score finds names likely to make a move or continue pricing vol.
    df["premium_activity_z"] = zscore_clip(df.get("flow_premium", 0))
    df["unusual_activity_z"] = zscore_clip(df.get("unusual_volume", 0)) + zscore_clip(df.get("avg_vol_oi", 0))
    df["oi_activity_z"] = zscore_clip(df.get("oi_inc_abs", 0)) + 0.5 * zscore_clip(df.get("oi_dec_abs", 0))
    df["iv_expansion_z"] = zscore_clip(df.get("iv_inc_activity", 0)) - 0.4 * zscore_clip(df.get("iv_dec_activity", 0))
    df["high_iv_risk_z"] = zscore_clip(df.get("avg_high_iv", 0))

    df["expansion_score"] = (
        1.7 * df["premium_activity_z"]
        + 1.4 * df["unusual_activity_z"]
        + 1.0 * df["oi_activity_z"]
        + 1.3 * df["iv_expansion_z"]
        + 0.3 * df["high_iv_risk_z"]
    )

    # Range/calendar candidate when activity is large but direction is balanced and IV is not collapsing too aggressively.
    df["balance_score"] = -abs(df["directional_score"])
    df["range_score"] = (
        1.0 * df["expansion_score"]
        + 1.2 * df["balance_score"]
        + 0.6 * zscore_clip(df.get("flow_premium", 0))
        - 0.5 * zscore_clip(df.get("iv_dec_activity", 0))
    )

    df["liquidity_bump"] = df["Symbol"].isin(ETF_PRIORITY).astype(float) * 0.5
    df["final_score"] = abs(df["directional_score"]) + 0.65 * df["expansion_score"] + df["liquidity_bump"]
    df["bias"] = np.select(
        [df["directional_score"] >= 2.0, df["directional_score"] <= -2.0],
        ["Bullish", "Bearish"],
        default="Neutral/Mixed",
    )

    def setup(row) -> str:
        bias = row["bias"]
        exp = row["expansion_score"]
        high_iv = row.get("avg_high_iv", 0)
        iv_dec = row.get("iv_dec_activity", 0)
        if bias == "Bullish" and exp >= 1.0:
            return "Bullish continuation: call diagonal or call debit spread"
        if bias == "Bearish" and exp >= 1.0:
            return "Bearish continuation: put diagonal or put debit spread"
        if bias == "Bullish":
            return "Bullish drift: bull put spread or conservative call diagonal"
        if bias == "Bearish":
            return "Bearish drift: bear call spread or conservative put diagonal"
        if row["range_score"] > 1.0:
            return "Range/vol structure: ATM calendar or iron condor around gamma levels"
        if high_iv > 80 and iv_dec > 0:
            return "High-IV risk: prefer credit spreads; avoid long premium chase"
        return "Watchlist only: mixed signals"

    df["suggested_setup"] = df.apply(setup, axis=1)
    return df.sort_values("final_score", ascending=False)


def build_scanner(input_dir: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, Optional[str]] = {k: find_file(input_dir, pats) for k, pats in FILE_PATTERNS.items()}
    print("Files used:")
    for k, p in paths.items():
        print(f"  {k:12s}: {os.path.basename(p) if p else 'NOT FOUND'}")

    data = {k: read_csv(p) for k, p in paths.items()}

    summaries = [
        summarize_flow(data["flow"]),
        summarize_unusual(data["unusual"]),
        summarize_oi(data["oi_increase"], "increase"),
        summarize_oi(data["oi_decrease"], "decrease"),
        summarize_iv_change(data["iv_increase"], "increase"),
        summarize_iv_change(data["iv_decrease"], "decrease"),
        summarize_high_iv(data["high_iv"]),
    ]
    ranked = add_scores(merge_all(summaries))

    if ranked.empty:
        raise RuntimeError("No usable ETF data found. Check input directory and file names.")

    # Friendly column order: core decision fields first, raw evidence after.
    core_cols = [
        "Symbol", "bias", "final_score", "directional_score", "expansion_score", "range_score", "suggested_setup",
        "spot", "flow_premium", "flow_signed_premium", "flow_call_put_ratio", "flow_trades",
        "unusual_volume", "avg_vol_oi", "max_vol_oi", "unusual_direction_score",
        "oi_inc_open_int", "oi_inc_direction_score", "oi_dec_open_int", "oi_dec_direction_score",
        "iv_inc_volume", "iv_inc_avg_pct", "iv_inc_direction_score", "iv_dec_volume", "iv_dec_avg_pct",
        "avg_high_iv", "max_high_iv",
    ]
    cols = [c for c in core_cols if c in ranked.columns] + [c for c in ranked.columns if c not in core_cols]
    ranked = ranked[cols]

    ranked.to_csv(os.path.join(out_dir, "etf_scanner_ranked.csv"), index=False)
    ranked[ranked["bias"] == "Bullish"].to_csv(os.path.join(out_dir, "etf_scanner_bullish.csv"), index=False)
    ranked[ranked["bias"] == "Bearish"].to_csv(os.path.join(out_dir, "etf_scanner_bearish.csv"), index=False)
    ranked.sort_values("expansion_score", ascending=False).to_csv(os.path.join(out_dir, "etf_scanner_vol_expansion.csv"), index=False)
    ranked.sort_values("range_score", ascending=False).to_csv(os.path.join(out_dir, "etf_scanner_range_candidates.csv"), index=False)

    top = ranked.head(15)[["Symbol", "bias", "final_score", "directional_score", "expansion_score", "range_score", "suggested_setup"]]
    with open(os.path.join(out_dir, "etf_scanner_summary.txt"), "w", encoding="utf-8") as f:
        f.write("ETF Scanner Summary\n")
        f.write("===================\n\n")
        f.write("Files used:\n")
        for k, p in paths.items():
            f.write(f"  {k:12s}: {os.path.basename(p) if p else 'NOT FOUND'}\n")
        f.write("\nTop ranked ETFs:\n")
        f.write(top.to_string(index=False))
        f.write("\n\nInterpretation notes:\n")
        f.write("- directional_score > 0 = bullish pressure; < 0 = bearish pressure.\n")
        f.write("- expansion_score ranks activity/IV/OI expansion; high values favor directional premium structures.\n")
        f.write("- range_score highlights mixed-direction activity that may suit calendars/condors if chart/gamma confirms.\n")
        f.write("- Always check earnings/macro events, ETF holdings, spot chart, and gamma walls before entry.\n")

    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF options flow/OI/IV scanner from Barchart CSV exports")
    parser.add_argument("--input-dir", default=".", help="Folder containing Barchart ETF CSV exports")
    parser.add_argument("--out-dir", default="scanner_output", help="Output folder for ranked CSVs")
    args = parser.parse_args()
    ranked = build_scanner(args.input_dir, args.out_dir)
    print("\nTop 15 ETFs:")
    print(ranked.head(15)[["Symbol", "bias", "final_score", "directional_score", "expansion_score", "range_score", "suggested_setup"]].to_string(index=False))
    print(f"\nSaved scanner outputs to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":

    ranked = build_scanner(".", "scanner_output")

    globals()["final_scores"] = ranked

    print(ranked.head(15))
