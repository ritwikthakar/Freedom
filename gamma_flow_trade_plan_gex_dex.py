"""
Gamma + Options Flow Trade Plan Generator
-----------------------------------------
Uses these CSV types:
  1) <ticker>-expected-move-*.csv
  2) <ticker>-options-flow-*.csv
  3) unusual-options-activity-*.csv
  4) <TICKER>_gex_history*.csv
  5) <TICKER>_dex_history*.csv
  6) net_gex_exposure_by_expiration_<TICKER>*.csv
  7) net_dex_exposure_by_expiration_<TICKER>*.csv

Output:
  - trade_plan_<TICKER>.csv
  - trade_plan_<TICKER>.txt

Important:
  This is a decision-support tool, not a prediction engine. Gamma levels are model-derived.
  Always validate output with the chart before taking a trade.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradePlan:
    ticker: str
    spot: float
    bias: str
    setup_type: str
    confidence: int
    entry_trigger: float
    stop_loss: float
    profit_target_1: float
    profit_target_2: float
    risk_per_share: float
    reward_1: float
    reward_2: float
    rr_1: float
    rr_2: float
    gamma_regime: str
    nearest_gamma_flip: float
    nearest_call_wall: float
    nearest_put_wall: float
    dex_regime: str
    front_net_gex: float
    front_net_dex: float
    nearest_dex_call_wall: float
    nearest_dex_put_wall: float
    expected_move: float
    expected_move_upper: float
    expected_move_lower: float
    flow_score: float
    unusual_score: float
    gex_history_score: float
    dex_history_score: float
    positioning_score: float
    notes: str


def clean_money_percent(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return float(x.replace('%', '').replace(',', '').strip())
    return float(x)


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
    return float(x)


def find_file(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_inputs(folder: str, ticker: str):
    t = ticker.upper()
    lower = ticker.lower()

    files = {
        "expected": find_file(os.path.join(folder, f"{lower}-expected-move-*.csv"))
                    or find_file(os.path.join(folder, f"{t}-expected-move-*.csv")),
        "flow": find_file(os.path.join(folder, f"{lower}-options-flow-*.csv"))
                or find_file(os.path.join(folder, f"{t}-options-flow-*.csv")),
        "unusual": find_file(os.path.join(folder, "unusual-options-activity-*.csv")),
        "gex_history": find_file(os.path.join(folder, f"{t}_gex_history*.csv")),
        "dex_history": find_file(os.path.join(folder, f"{t}_dex_history*.csv")),
        "gex_exp": find_file(os.path.join(folder, f"net_gex_exposure_by_expiration_{t}*.csv")),
        "dex_exp": find_file(os.path.join(folder, f"net_dex_exposure_by_expiration_{t}*.csv")),
    }

    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing required files for {ticker}: {missing}\nSearched in: {folder}")

    return {k: pd.read_csv(v) for k, v in files.items()}, files


def expected_move_snapshot(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["DTE"] = df["DTE"].apply(to_float)
    df["Price~"] = df["Price~"].apply(to_float)
    df["Expected Move"] = df["Expected Move"].apply(to_float)
    df["Upper Price"] = df["Upper Price"].apply(to_float)
    df["Lower Price"] = df["Lower Price"].apply(to_float)

    # Prefer nearest non-expired weekly/monthly expiry with DTE > 0
    row = df[df["DTE"] > 0].sort_values("DTE").iloc[0]
    return {
        "spot": float(row["Price~"]),
        "dte": float(row["DTE"]),
        "em": float(row["Expected Move"]),
        "upper": float(row["Upper Price"]),
        "lower": float(row["Lower Price"]),
        "expiration": str(row.get("Expiration Date", "")),
    }


def flow_direction_score(flow: pd.DataFrame, ticker: str) -> float:
    df = flow.copy()
    df = df[df["Symbol"].astype(str).str.upper() == ticker.upper()]
    if df.empty:
        return 0.0

    df["Premium"] = df["Premium"].apply(to_float).fillna(0)
    df["Delta"] = df["Delta"].apply(to_float).fillna(0)
    df["DTE"] = df["DTE"].apply(to_float).fillna(999)
    df["Side"] = df["Side"].astype(str).str.lower()
    df["Type"] = df["Type"].astype(str).str.lower()

    # Directional assumption:
    # ask-side call buying = bullish; ask-side put buying = bearish.
    # bid-side call selling = mildly bearish; bid-side put selling = mildly bullish.
    side_mult = np.select(
        [df["Side"].eq("ask"), df["Side"].eq("bid"), df["Side"].eq("mid")],
        [1.0, -0.7, 0.25],
        default=0.0,
    )
    type_mult = np.where(df["Type"].eq("call"), 1.0, -1.0)
    dte_weight = np.where(df["DTE"] <= 45, 1.2, np.where(df["DTE"] <= 180, 1.0, 0.6))
    raw = (df["Premium"] * side_mult * type_mult * dte_weight).sum()
    denom = max(df["Premium"].sum(), 1.0)
    return float(np.clip(100 * raw / denom, -100, 100))


def unusual_direction_score(unusual: pd.DataFrame, ticker: str) -> float:
    df = unusual.copy()
    df = df[df["Symbol"].astype(str).str.upper() == ticker.upper()]
    if df.empty:
        return 0.0

    df["Volume"] = df["Volume"].apply(to_float).fillna(0)
    df["Open Int"] = df["Open Int"].apply(to_float).replace(0, np.nan)
    df["Delta"] = df["Delta"].apply(to_float).fillna(0)
    df["DTE"] = df["DTE"].apply(to_float).fillna(999)
    df["Type"] = df["Type"].astype(str).str.lower()

    vol_oi = (df["Volume"] / df["Open Int"]).replace([np.inf, -np.inf], np.nan).fillna(1)
    vol_oi_weight = np.clip(vol_oi, 1, 10)
    dte_weight = np.where(df["DTE"] <= 45, 1.2, np.where(df["DTE"] <= 180, 1.0, 0.6))
    type_mult = np.where(df["Type"].eq("call"), 1.0, -1.0)

    raw = (df["Volume"] * type_mult * vol_oi_weight * dte_weight).sum()
    denom = max((df["Volume"] * vol_oi_weight * dte_weight).sum(), 1.0)
    return float(np.clip(100 * raw / denom, -100, 100))


def gex_snapshot(gex_exp: pd.DataFrame, spot: float) -> dict:
    df = gex_exp.copy()
    for col in ["net_gex", "call_gex", "put_gex", "call_wall", "put_wall", "gamma_flip"]:
        df[col] = df[col].apply(to_float)

    # Use nearest listed expiry as primary map.
    row = df.iloc[0]
    net_gex = float(row["net_gex"])
    flip = float(row["gamma_flip"]) if not pd.isna(row["gamma_flip"]) else np.nan
    call_wall = float(row["call_wall"])
    put_wall = float(row["put_wall"])

    if net_gex > 0 and (pd.isna(flip) or spot > flip):
        regime = "POSITIVE_GAMMA_VOL_SUPPRESSION"
    elif net_gex < 0 or (not pd.isna(flip) and spot < flip):
        regime = "NEGATIVE_GAMMA_VOL_EXPANSION"
    else:
        regime = "MIXED_GAMMA"

    return {
        "net_gex": net_gex,
        "flip": flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime": regime,
        "expiration": str(row.get("expiration", "")),
    }


def dex_snapshot(dex_exp: pd.DataFrame, spot: float) -> dict:
    """Read front-expiration DEX map.

    DEX is used as a directional pressure filter:
      - positive net DEX = call/delta exposure dominates, bullish/stabilizing
      - negative net DEX = put/delta exposure dominates, bearish/hedging pressure
    """
    df = dex_exp.copy()
    for col in ["net_dex", "call_dex", "put_dex", "call_wall", "put_wall"]:
        df[col] = df[col].apply(to_float)

    row = df.iloc[0]
    net_dex = float(row["net_dex"])
    call_wall = float(row["call_wall"])
    put_wall = float(row["put_wall"])

    if net_dex > 0:
        regime = "POSITIVE_DEX_CALL_DELTA_DOMINANT"
    elif net_dex < 0:
        regime = "NEGATIVE_DEX_PUT_DELTA_DOMINANT"
    else:
        regime = "MIXED_DEX"

    return {
        "net_dex": net_dex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime": regime,
        "expiration": str(row.get("expiration", "")),
    }


def dex_history_score(hist: pd.DataFrame) -> float:
    df = hist.copy()
    df["dex_net_oi"] = df["dex_net_oi"].apply(to_float)
    df = df.dropna(subset=["dex_net_oi"])
    if len(df) < 10:
        return 0.0
    last = df["dex_net_oi"].iloc[-1]
    prev5 = df["dex_net_oi"].iloc[-6]
    prev20 = df["dex_net_oi"].iloc[-21] if len(df) > 21 else df["dex_net_oi"].iloc[0]

    # Positive = DEX improving / call delta pressure increasing.
    # Negative = DEX deteriorating / put delta pressure increasing.
    change5 = (last - prev5) / max(abs(prev5), 1)
    change20 = (last - prev20) / max(abs(prev20), 1)
    return float(np.clip(50 * change5 + 25 * change20, -100, 100))


def positioning_score(gex: dict, dex: dict, spot: float) -> float:
    score = 0.0

    if gex["regime"].startswith("POSITIVE"):
        score += 10
    elif gex["regime"].startswith("NEGATIVE"):
        score -= 10

    if dex["regime"].startswith("POSITIVE"):
        score += 15
    elif dex["regime"].startswith("NEGATIVE"):
        score -= 15

    # Price above gamma flip is a small bullish filter; below is bearish.
    if not pd.isna(gex["flip"]):
        score += 10 if spot > gex["flip"] else -10

    return float(np.clip(score, -100, 100))


def gex_history_score(hist: pd.DataFrame) -> float:
    df = hist.copy()
    df["gex_net_oi"] = df["gex_net_oi"].apply(to_float)
    df = df.dropna(subset=["gex_net_oi"])
    if len(df) < 10:
        return 0.0
    last = df["gex_net_oi"].iloc[-1]
    prev5 = df["gex_net_oi"].iloc[-6]
    prev20 = df["gex_net_oi"].iloc[-21] if len(df) > 21 else df["gex_net_oi"].iloc[0]

    # Positive means gamma environment recently improved/stabilized; negative means weakening/destabilizing.
    change5 = (last - prev5) / max(abs(prev5), 1)
    change20 = (last - prev20) / max(abs(prev20), 1)
    return float(np.clip(50 * change5 + 25 * change20, -100, 100))


def round_level(x: float) -> float:
    if pd.isna(x) or math.isinf(x):
        return np.nan
    return round(float(x), 2)


def build_trade_plan(ticker: str, data: dict) -> TradePlan:
    ems = expected_move_snapshot(data["expected"])
    spot, em, upper, lower = ems["spot"], ems["em"], ems["upper"], ems["lower"]

    flow_score = flow_direction_score(data["flow"], ticker)
    unusual_score = unusual_direction_score(data["unusual"], ticker)
    gex_hist_score = gex_history_score(data["gex_history"])
    dex_hist_score = dex_history_score(data["dex_history"])
    gex = gex_snapshot(data["gex_exp"], spot)
    dex = dex_snapshot(data["dex_exp"], spot)
    pos_score = positioning_score(gex, dex, spot)

    # Options-only model: flow + unusual + GEX history + DEX history + front-expiration positioning.
    combined = (
        0.35 * flow_score
        + 0.20 * unusual_score
        + 0.15 * gex_hist_score
        + 0.15 * dex_hist_score
        + 0.15 * pos_score
    )

    room_to_call = (gex["call_wall"] - spot) / spot if gex["call_wall"] else 0
    room_to_put = (spot - gex["put_wall"]) / spot if gex["put_wall"] else 0

    notes = []
    notes.append(f"Nearest expected move expiry: {ems['expiration']} ({ems['dte']:.0f} DTE).")
    notes.append(f"Nearest GEX expiry: {gex['expiration']}. Gamma data is model-derived, so treat levels as zones.")
    notes.append(f"Nearest DEX expiry: {dex['expiration']}. DEX regime: {dex['regime']}.")

    if combined >= 15:
        bias = "BUY"
        if gex["regime"].startswith("POSITIVE") and room_to_call < 0.03:
            setup_type = "BUY_ONLY_ON_BREAK_ABOVE_CALL_WALL_OR_WAIT_PULLBACK"
            entry = max(gex["call_wall"] + 0.10 * em, spot + 0.15 * em)
            stop = max(spot - 0.35 * em, gex["put_wall"] if not pd.isna(gex["put_wall"]) else lower)
            pt1 = upper
            pt2 = upper + 0.50 * em
            notes.append("Bullish flow, but price is close to call wall; avoid chasing unless wall breaks/holds.")
        else:
            setup_type = "BULLISH_CONTINUATION"
            entry = spot + 0.15 * em
            stop = max(spot - 0.35 * em, lower)
            pt1 = min(upper, gex["call_wall"] if gex["call_wall"] > spot else upper)
            pt2 = upper + 0.50 * em
            notes.append("Bullish flow/positioning with enough upside room before the main resistance zone.")

    elif combined <= -15:
        bias = "SELL"
        if gex["regime"].startswith("POSITIVE") and room_to_put > 0.03:
            setup_type = "BEARISH_PULLBACK_IN_POSITIVE_GAMMA"
            entry = spot - 0.15 * em
            stop = min(spot + 0.35 * em, gex["call_wall"] if not pd.isna(gex["call_wall"]) else upper)
            pt1 = max(lower, gex["put_wall"] if gex["put_wall"] < spot else lower)
            pt2 = lower - 0.50 * em
            notes.append("Bearish signal, but positive gamma can slow the move; use confirmation below trigger.")
        else:
            setup_type = "BEARISH_EXPANSION"
            entry = spot - 0.15 * em
            stop = min(spot + 0.35 * em, upper)
            pt1 = max(lower, gex["put_wall"] if gex["put_wall"] < spot else lower)
            pt2 = lower - 0.50 * em
            notes.append("Bearish flow/positioning with expansion risk if support breaks.")
    else:
        bias = "NO_TRADE"
        setup_type = "MIXED_SIGNAL_WAIT_FOR_BREAK"
        entry = np.nan
        stop = np.nan
        pt1 = np.nan
        pt2 = np.nan
        notes.append("Signals are mixed; wait for either upper/lower expected-move break or clearer flow confirmation.")

    if bias == "BUY":
        risk = max(entry - stop, 0.01)
        reward1 = max(pt1 - entry, 0)
        reward2 = max(pt2 - entry, 0)
    elif bias == "SELL":
        risk = max(stop - entry, 0.01)
        reward1 = max(entry - pt1, 0)
        reward2 = max(entry - pt2, 0)
    else:
        risk = reward1 = reward2 = rr1 = rr2 = np.nan
        confidence = int(max(0, 100 - abs(combined)))
        return TradePlan(ticker.upper(), round_level(spot), bias, setup_type, confidence, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, np.nan, np.nan, gex["regime"], round_level(gex["flip"]),
                         round_level(gex["call_wall"]), round_level(gex["put_wall"]), dex["regime"],
                         round_level(gex["net_gex"]), round_level(dex["net_dex"]), round_level(dex["call_wall"]),
                         round_level(dex["put_wall"]), round_level(em), round_level(upper), round_level(lower),
                         round_level(flow_score), round_level(unusual_score), round_level(gex_hist_score),
                         round_level(dex_hist_score), round_level(pos_score), " ".join(notes))

    rr1 = reward1 / risk if risk else np.nan
    rr2 = reward2 / risk if risk else np.nan
    confidence = int(np.clip(50 + abs(combined) / 2 + (10 if rr1 >= 1 else -10), 1, 95))

    return TradePlan(
        ticker=ticker.upper(),
        spot=round_level(spot),
        bias=bias,
        setup_type=setup_type,
        confidence=confidence,
        entry_trigger=round_level(entry),
        stop_loss=round_level(stop),
        profit_target_1=round_level(pt1),
        profit_target_2=round_level(pt2),
        risk_per_share=round_level(risk),
        reward_1=round_level(reward1),
        reward_2=round_level(reward2),
        rr_1=round_level(rr1),
        rr_2=round_level(rr2),
        gamma_regime=gex["regime"],
        nearest_gamma_flip=round_level(gex["flip"]),
        nearest_call_wall=round_level(gex["call_wall"]),
        nearest_put_wall=round_level(gex["put_wall"]),
        dex_regime=dex["regime"],
        front_net_gex=round_level(gex["net_gex"]),
        front_net_dex=round_level(dex["net_dex"]),
        nearest_dex_call_wall=round_level(dex["call_wall"]),
        nearest_dex_put_wall=round_level(dex["put_wall"]),
        expected_move=round_level(em),
        expected_move_upper=round_level(upper),
        expected_move_lower=round_level(lower),
        flow_score=round_level(flow_score),
        unusual_score=round_level(unusual_score),
        gex_history_score=round_level(gex_hist_score),
        dex_history_score=round_level(dex_hist_score),
        positioning_score=round_level(pos_score),
        notes=" ".join(notes),
    )


def save_outputs(plan: TradePlan, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"trade_plan_{plan.ticker}.csv")
    txt_path = os.path.join(outdir, f"trade_plan_{plan.ticker}.txt")

    pd.DataFrame([asdict(plan)]).to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Trade Plan: {plan.ticker}\n")
        f.write("=" * 50 + "\n")
        for k, v in asdict(plan).items():
            f.write(f"{k}: {v}\n")

    return csv_path, txt_path


def detect_ticker(folder: str) -> str:
    """
    Auto-detect ticker from available files.
    Priority:
      1) net_gex_exposure_by_expiration_<TICKER>.csv
      2) <TICKER>_gex_history.csv
      3) <ticker>-expected-move-*.csv
    """

    # Priority 1
    files = glob.glob(os.path.join(folder, "net_gex_exposure_by_expiration_*.csv"))
    if files:
        base = os.path.basename(files[0])
        return base.replace("net_gex_exposure_by_expiration_", "").replace(".csv", "").split("(")[0].upper()

    # Priority 2
    files = glob.glob(os.path.join(folder, "*_gex_history*.csv"))
    if files:
        base = os.path.basename(files[0])
        return base.split("_gex_history")[0].upper()

    # Priority 3
    files = glob.glob(os.path.join(folder, "*-expected-move-*.csv"))
    if files:
        base = os.path.basename(files[0])
        return base.split("-expected-move-")[0].upper()

    raise FileNotFoundError("Could not auto-detect ticker from files.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate buy/sell levels from expected move, flow, unusual activity, GEX and DEX CSVs."
    )

    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker symbol (optional). If omitted, ticker will be auto-detected."
    )

    parser.add_argument(
        "--folder",
        default=".",
        help="Folder containing CSV files"
    )

    parser.add_argument(
        "--outdir",
        default=".",
        help="Output folder"
    )

    args = parser.parse_args()

    # Auto-detect ticker if not supplied
    ticker = args.ticker
    if ticker is None:
        ticker = detect_ticker(args.folder)
        print(f"Auto-detected ticker: {ticker}")

    data, files = load_inputs(args.folder, ticker)

    plan = build_trade_plan(ticker, data)

    csv_path, txt_path = save_outputs(plan, args.outdir)

    print("\nFiles used:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    print("\nTrade plan:")
    for k, v in asdict(plan).items():
        print(f"  {k}: {v}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
