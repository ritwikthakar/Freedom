"""Core logic for the Calendar Income Scanner (v2).

Pure functions only (no Streamlit) so the logic can be tested on its own.
"""
from __future__ import annotations

import io
import math
import re

import numpy as np
import pandas as pd

REQUIRED = {
    "Symbol", "Price~", "Exp Leg1", "Leg1 Strike", "Type", "Bid1",
    "Exp Leg2", "Ask2", "Net Debit", "Leg1 IV", "Leg2 IV",
    "IV Skew", "IV Rank", "IV/HV", "Net Delta", "Net Vega",
}
PCT_COLS = ["Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank"]
NUM_COLS = [
    "Price~", "Leg1 Strike", "Bid1", "Ask2", "Net Debit",
    "IV/HV", "Net Delta", "Net Vega",
]

CADENCES = {
    "Biweekly (front 7-16 DTE)": (7, 16, 21),
    "Monthly (front 21-35 DTE)": (21, 35, 28),
    "Both": (7, 35, 24),
}

_erf = np.vectorize(math.erf)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _percent(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_file(upload, source: str) -> tuple[pd.DataFrame, dict]:
    """Read a Barchart calendar CSV. Returns (frame, info)."""
    raw = upload.read() if hasattr(upload, "read") else open(upload, "rb").read()
    text = raw.decode("utf-8-sig", errors="replace")

    # Barchart appends 'Downloaded from Barchart.com as of MM-DD-YYYY ...'
    m = re.search(r"as of (\d{2})-(\d{2})-(\d{4})", text)
    as_of = pd.Timestamp(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else pd.NaT

    d = pd.read_csv(io.StringIO(text))
    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    n_before = len(d)
    d = d.dropna(how="all").copy()
    for c in PCT_COLS:
        d[c] = _percent(d[c])
    for c in NUM_COLS:
        d[c] = _numeric(d[c])
    for c in ["Exp Leg1", "Exp Leg2"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    # Drops the Barchart footer line and any other non-data row.
    good = d["Price~"].notna() & d["Exp Leg1"].notna() & d["Exp Leg2"].notna()
    dropped = int((~good).sum())
    d = d[good].copy()

    d["Symbol"] = d["Symbol"].astype("string").str.upper().str.strip()
    d["Type"] = d["Type"].astype("string").str.strip().str.title()
    d["Source"] = source
    d["As Of"] = as_of
    return d.reset_index(drop=True), {
        "as_of": as_of, "rows": len(d), "dropped": dropped, "read": n_before,
    }


# --------------------------------------------------------------------------
# Earnings / event handling
# --------------------------------------------------------------------------
def parse_earnings(text: str) -> dict:
    """Parse lines like 'AAPL,2026-10-29' or 'MSFT 10/28/2026'."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\t ]+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        dt = pd.to_datetime(parts[1].strip(), errors="coerce")
        if pd.notna(dt):
            out[parts[0].upper()] = dt.normalize()
    return out


def classify_events(x: pd.DataFrame, earnings: dict, skew_thr: float) -> pd.DataFrame:
    x = x.copy()
    x["Earnings Date"] = pd.to_datetime(x["Symbol"].astype(object).map(earnings))
    ed = x["Earnings Date"]
    has = ed.notna()

    in_front = has & (ed >= x["As Of"]) & (ed <= x["Exp Leg1"])
    after_front = has & (ed > x["Exp Leg1"]) & (ed <= x["Exp Leg2"])
    iv_ratio = x["Leg1 IV"] / x["Leg2 IV"]
    proxy = (x["IV Skew"] >= skew_thr) | (iv_ratio >= 1.35)
    suspected = ~has & proxy

    x["Event Status"] = "CLEAN"
    x.loc[after_front, "Event Status"] = "EARNINGS AFTER FRONT (back leg only)"
    x.loc[suspected, "Event Status"] = "SUSPECTED EVENT (no date)"
    x.loc[in_front, "Event Status"] = "EARNINGS IN FRONT LEG"
    x["Is Event Play"] = x["Event Status"].isin(
        ["EARNINGS IN FRONT LEG", "SUSPECTED EVENT (no date)"]
    )
    # A known date plus a big skew that earnings don't explain.
    x["Unexplained High Skew"] = has & ~in_front & (x["IV Skew"] >= skew_thr)
    return x


# --------------------------------------------------------------------------
# Option math: P&L of the calendar at front expiry
# --------------------------------------------------------------------------
def _ncdf(v):
    return 0.5 * (1.0 + _erf(np.asarray(v, dtype=float) / math.sqrt(2.0)))


def bs_price(S, K, T, r, sig, is_call):
    S = np.asarray(S, dtype=float)
    if T <= 0 or sig <= 0:
        return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    d2 = d1 - sq
    if is_call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


_Z = np.linspace(-4.0, 4.0, 321)
_W = np.exp(-0.5 * _Z ** 2)
_W = _W / _W.sum()


def pnl_model(row, rate: float, vol_ratio: float, back_iv_shift: float) -> dict:
    """Value the calendar at front expiry across a lognormal spot distribution.

    Assumptions (all visible in the sidebar):
      * long leg is repriced with Black-Scholes at Leg2 IV (+ shift), flat in spot
      * spot distribution vol = Leg1 IV * vol_ratio (realized/implied haircut)
      * zero dividends, zero commissions, debit = Ask2 - Bid1 (worst-case fill)
    """
    nan = dict.fromkeys(
        ["Peak P&L", "Peak ROI", "EV / Debit", "Prob Profit",
         "BE Low", "BE High", "BE Width (EM)", "Front EM $"], np.nan)
    S0, K = row["Price~"], row["Leg1 Strike"]
    debit = row["Net Debit"]
    Ts, Tl = row["Short DTE"] / 365.0, row["Long DTE"] / 365.0
    Tr = Tl - Ts
    s_mkt = row["Leg1 IV"] / 100.0
    s2 = row["Leg2 IV"] / 100.0 + back_iv_shift / 100.0
    if any(pd.isna(v) for v in (S0, K, debit, Ts, Tl, s_mkt, s2)) \
            or debit <= 0 or Ts <= 0 or Tr <= 0 or s_mkt <= 0:
        return nan
    s2 = max(s2, 0.01)
    is_call = str(row["Type"]).lower() == "call"

    em = S0 * s_mkt * math.sqrt(Ts)  # market-implied 1-sigma move in $
    sd = s_mkt * vol_ratio * math.sqrt(Ts)
    S = S0 * np.exp(_Z * sd - 0.5 * sd * sd)  # martingale drift

    long_val = bs_price(S, K, Tr, rate, s2, is_call)
    short_val = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    pnl = long_val - short_val - debit

    i = int(np.argmax(pnl))
    peak = float(pnl[i])
    if peak > 0:
        lo, hi = i, i
        while lo > 0 and pnl[lo - 1] > 0:
            lo -= 1
        while hi < len(pnl) - 1 and pnl[hi + 1] > 0:
            hi += 1
        be_lo, be_hi = float(S[lo]), float(S[hi])
        width = (be_hi - be_lo) / em
    else:
        be_lo = be_hi = np.nan
        width = 0.0
    return {
        "Peak P&L": peak,
        "Peak ROI": peak / debit,
        "EV / Debit": float((_W * pnl).sum() / debit),
        "Prob Profit": float((_W * (pnl > 0)).sum()),
        "BE Low": be_lo, "BE High": be_hi,
        "BE Width (EM)": width, "Front EM $": em,
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def lin(v, lo, hi):
    """0 at lo, 100 at hi, linear, clipped."""
    return ((v - lo) / (hi - lo) * 100).clip(0, 100)


def tent(v, ideal, width):
    return (100 - (v - ideal).abs() / width * 100).clip(0, 100)


def derive(d: pd.DataFrame, today_override=None) -> pd.DataFrame:
    x = d.copy()
    asof = x["As Of"].fillna(today_override if today_override is not None
                             else pd.Timestamp.now().normalize())
    x["As Of"] = asof
    x["Short DTE"] = (x["Exp Leg1"] - asof).dt.days
    x["Long DTE"] = (x["Exp Leg2"] - asof).dt.days
    x["DTE Gap"] = x["Long DTE"] - x["Short DTE"]
    x["Moneyness %"] = (x["Leg1 Strike"] - x["Price~"]) / x["Price~"] * 100
    x["Abs Delta"] = x["Net Delta"].abs()
    x["Debit % Px"] = x["Net Debit"] / x["Price~"] * 100
    x["Front EM $"] = x["Price~"] * x["Leg1 IV"] / 100 * np.sqrt(x["Short DTE"].clip(lower=0) / 365)
    x["Dist (EM)"] = (x["Leg1 Strike"] - x["Price~"]).abs() / x["Front EM $"].replace(0, np.nan)
    return x


def run_models(x: pd.DataFrame, rate: float, vol_ratio: float, back_iv_shift: float) -> pd.DataFrame:
    res = x.apply(lambda r: pd.Series(pnl_model(r, rate, vol_ratio, back_iv_shift)), axis=1)
    out = x.drop(columns=[c for c in res.columns if c in x.columns]).join(res)
    out["Peak ROI / Day %"] = out["Peak ROI"] * 100 / out["Short DTE"].replace(0, np.nan)
    return out


def score_core(x: pd.DataFrame, cadence: str, event_skew_thr: float) -> pd.DataFrame:
    """Income score for non-event calendars."""
    lo_dte, hi_dte, ideal_gap = CADENCES[cadence]
    s = pd.DataFrame(index=x.index)
    s["EV Score"] = lin(x["EV / Debit"], -0.10, 0.20)
    s["ROI Score"] = lin(x["Peak ROI"], 0.0, 1.0)
    s["Prob Score"] = lin(x["Prob Profit"], 0.30, 0.60)
    s["Skew Score"] = lin(x["IV Skew"].clip(upper=event_skew_thr), 0.0, 3.0)
    s["Center Score"] = (100 - x["Dist (EM)"] / 0.5 * 100).clip(0, 100)
    s["Delta Score"] = (100 - x["Abs Delta"] / 0.15 * 100).clip(0, 100)
    s["IV/HV Score"] = tent(x["IV/HV"], 1.05, 0.55)
    s["Gap Score"] = tent(x["DTE Gap"], ideal_gap, 14)

    x = pd.concat([x, s], axis=1)
    x["Income Score"] = (
        0.25 * x["EV Score"] + 0.20 * x["ROI Score"] + 0.15 * x["Prob Score"]
        + 0.15 * x["Skew Score"] + 0.10 * x["Center Score"]
        + 0.05 * x["Delta Score"] + 0.05 * x["IV/HV Score"] + 0.05 * x["Gap Score"]
    )

    # Soft penalties (diagnostics stay visible instead of rows vanishing)
    ivhv_pen = np.select(
        [x["IV/HV"] < 0.75, x["IV/HV"] < 0.90, x["IV/HV"] > 1.50,
         x["IV/HV"] > 1.35, x["IV/HV"] > 1.20],
        [10, 3, 12, 7, 3], default=0)
    penalty = (
        np.where(x["IV Skew"] < 0, 15, 0)
        + np.where(x["Abs Delta"] > 0.30, 20, 0)
        + np.where(x["DTE Gap"] <= 0, 50, 0)
        + np.where(x["Net Vega"] <= 0, 30, 0)
        + np.where(x["Dist (EM)"] > 0.75, 15, 0)
        + np.where(x["Event Status"].str.startswith("EARNINGS AFTER"), 5, 0)
        + ivhv_pen
    )
    x["IV/HV Penalty"] = ivhv_pen
    x["Score"] = (x["Income Score"] - penalty).clip(0, 100)

    x["Priority"] = pd.cut(
        x["Score"], bins=[-np.inf, 45, 55, 65, np.inf],
        labels=["LOW", "WATCH", "B", "A"]).astype(str)
    a_plus = (
        (x["Score"] >= 70) & (x["Dist (EM)"] <= 0.20) & (x["Abs Delta"] <= 0.10)
        & (x["EV / Debit"] > 0) & (x["Peak ROI"] >= 0.50)
        & (x["Prob Profit"] >= 0.40) & (x["DTE Gap"] >= 7)
        & (x["Event Status"] == "CLEAN")
    )
    x.loc[a_plus, "Priority"] = "A+"

    in_window = x["Short DTE"].between(lo_dte, hi_dte)
    x["In Cadence"] = in_window

    flags = pd.Series("", index=x.index)
    flags += np.where(x["Net Debit"] <= 0, "Invalid debit; ", "")
    flags += np.where(x["Net Vega"] <= 0, "Non-positive vega; ", "")
    flags += np.where(x["DTE Gap"] <= 0, "Expiry ordering; ", "")
    flags += np.where(x["IV Skew"] < 0, "Back IV richer than front; ", "")
    flags += np.where(x["Abs Delta"] > 0.30, "Too directional; ", "")
    flags += np.where(x["Dist (EM)"] > 0.75, "Strike far from spot (EM); ", "")
    flags += np.where(x["Short DTE"] < 1, "Short leg at expiry; ", "")
    flags += np.where(x["Unexplained High Skew"], "High skew, no earnings in window - other event?; ", "")
    flags += np.where(x["Event Status"].str.startswith("EARNINGS AFTER"),
                      "Earnings falls inside back-leg window; ", "")
    x["Flag"] = flags
    return x.sort_values("Score", ascending=False)


def score_earnings(x: pd.DataFrame) -> pd.DataFrame:
    """Structure-only ranking for event calendars.

    There is no earnings-move history in the CSVs, so this is a shortlist: it
    ranks cost, centering and front-IV premium. Compare the implied move with
    the stock's historical earnings moves before trading.
    """
    e = x.copy()
    e["Premium Score"] = lin(e["IV Skew"].clip(upper=30), 3.0, 25.0)
    e["Center Score"] = (100 - e["Dist (EM)"] / 0.5 * 100).clip(0, 100)
    e["Delta Score"] = (100 - e["Abs Delta"] / 0.15 * 100).clip(0, 100)
    e["Cost Score"] = (100 - e["Debit % Px"] / 4.0 * 100).clip(0, 100)
    e["Debit / EM"] = e["Net Debit"] / e["Front EM $"].replace(0, np.nan)
    e["Cost/EM Score"] = (100 - e["Debit / EM"] / 0.6 * 100).clip(0, 100)
    e["Event Score"] = (
        0.30 * e["Premium Score"] + 0.25 * e["Center Score"]
        + 0.10 * e["Delta Score"] + 0.15 * e["Cost Score"]
        + 0.20 * e["Cost/EM Score"]
    )
    e.loc[e["Net Vega"] <= 0, "Event Score"] -= 30
    e.loc[e["DTE Gap"] <= 0, "Event Score"] -= 50
    e["Event Score"] = e["Event Score"].clip(0, 100)
    return e.sort_values("Event Score", ascending=False)
