"""
Streamlit Calendar Spread Screener

Upload the two Barchart calendar CSVs:
1) Long Call Calendar screener CSV
2) Long Put Calendar screener CSV

The app normalizes the files, creates category rankings, and shows the best
calendar trades by setup type:
- Low implied volatility setup
- Consolidation setup
- Post-IV-crush / term-structure setup
- Expected-move / directional OTM setup
- Overall ranked calendar setup

Run:
    streamlit run calendar_spread_screener_app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Calendar Spread Screener",
    page_icon="📅",
    layout="wide",
)


REQUIRED_COLUMNS = [
    "Symbol", "Price~", "Exp Leg1", "Leg1 Strike", "Type", "Bid1", "Exp Leg2",
    "Ask2", "Net Debit", "Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank",
    "IV/HV", "Net Delta", "Net Vega",
]


# -----------------------------
# Helpers
# -----------------------------
def parse_percent_series(s: pd.Series) -> pd.Series:
    """Convert strings like '42.10%' to 42.10. Numeric values pass through."""
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def minmax_score(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Robust 0-100 score. NaNs become 0."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    lo, hi = x.quantile(0.05), x.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        out = pd.Series(50.0, index=s.index)
    else:
        clipped = x.clip(lo, hi)
        out = (clipped - lo) / (hi - lo) * 100
    if not higher_is_better:
        out = 100 - out
    return out.fillna(0).clip(0, 100)


def proximity_score(distance_pct: pd.Series, ideal: float = 0.0, max_good: float = 8.0) -> pd.Series:
    """Score based on closeness to ideal distance from spot."""
    d = (pd.to_numeric(distance_pct, errors="coerce") - ideal).abs()
    out = 100 * (1 - d / max_good)
    return out.fillna(0).clip(0, 100)


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)


def load_calendar_file(uploaded_file, fallback_type: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
    out["Type"] = out["Type"].astype(str).str.title().str.strip()
    if fallback_type:
        out.loc[~out["Type"].isin(["Call", "Put"]), "Type"] = fallback_type

    numeric_cols = ["Price~", "Leg1 Strike", "Bid1", "Ask2", "Net Debit", "IV/HV", "Net Delta", "Net Vega"]
    for col in numeric_cols:
        out[col] = to_numeric_series(out[col])

    for col in ["Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank"]:
        out[col] = parse_percent_series(out[col])

    out["Exp Leg1"] = pd.to_datetime(out["Exp Leg1"], errors="coerce")
    out["Exp Leg2"] = pd.to_datetime(out["Exp Leg2"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    out["DTE Short"] = (out["Exp Leg1"] - today).dt.days
    out["DTE Long"] = (out["Exp Leg2"] - today).dt.days
    out["DTE Gap"] = out["DTE Long"] - out["DTE Short"]
    out["Moneyness %"] = (out["Leg1 Strike"] - out["Price~"]) / out["Price~"] * 100
    out["Abs Moneyness %"] = out["Moneyness %"].abs()
    out["Vega/Debit"] = safe_ratio(out["Net Vega"], out["Net Debit"])
    out["Skew/Debit"] = safe_ratio(out["IV Skew"], out["Net Debit"])
    out["Short Premium/Debit"] = safe_ratio(out["Bid1"], out["Net Debit"])

    # Calendar quality guardrails
    out = out[
        (out["Net Debit"] > 0)
        & (out["Net Vega"] > 0)
        & (out["DTE Gap"] > 0)
        & out["Price~"].notna()
        & out["Leg1 Strike"].notna()
    ].copy()

    return out


@dataclass
class ScoreWeights:
    iv_skew: float = 0.20
    vega_debit: float = 0.20
    iv_rank: float = 0.15
    iv_hv: float = 0.10
    moneyness: float = 0.15
    dte_gap: float = 0.10
    debit: float = 0.10


def add_scores(df: pd.DataFrame, weights: ScoreWeights) -> pd.DataFrame:
    out = df.copy()

    # Component scores
    out["Score IV Skew"] = minmax_score(out["IV Skew"], higher_is_better=True)
    out["Score Vega/Debit"] = minmax_score(out["Vega/Debit"], higher_is_better=True)
    out["Score Low IV Rank"] = minmax_score(out["IV Rank"], higher_is_better=False)
    out["Score IV/HV Low"] = minmax_score(out["IV/HV"], higher_is_better=False)
    out["Score ATM"] = proximity_score(out["Abs Moneyness %"], ideal=0, max_good=8)
    out["Score OTM Moderate"] = proximity_score(out["Abs Moneyness %"], ideal=4, max_good=10)
    out["Score DTE Gap"] = proximity_score(out["DTE Gap"], ideal=28, max_good=35)
    out["Score Cheap Debit"] = minmax_score(out["Net Debit"], higher_is_better=False)
    out["Score Short Premium"] = minmax_score(out["Short Premium/Debit"], higher_is_better=True)
    out["Score Delta Neutral"] = proximity_score(out["Net Delta"].abs(), ideal=0, max_good=0.35)

    # 1) Low IV: cheap IV, positive vega, not too far OTM, good debit efficiency
    out["Low IV Setup Score"] = (
        0.30 * out["Score Low IV Rank"]
        + 0.20 * out["Score IV/HV Low"]
        + 0.20 * out["Score Vega/Debit"]
        + 0.15 * out["Score IV Skew"]
        + 0.10 * out["Score ATM"]
        + 0.05 * out["Score Cheap Debit"]
    )

    # 2) Consolidation: ATM, delta-neutral, front option rich enough to decay, moderate IV
    moderate_iv = 100 - (out["IV Rank"].sub(45).abs() / 45 * 100).clip(0, 100)
    out["Score Moderate IV"] = moderate_iv.fillna(0)
    out["Consolidation Setup Score"] = (
        0.30 * out["Score ATM"]
        + 0.25 * out["Score Delta Neutral"]
        + 0.20 * out["Score Short Premium"]
        + 0.15 * out["Score Moderate IV"]
        + 0.10 * out["Score DTE Gap"]
    )

    # 3) Post-IV-crush proxy: IV Rank lower/moderate, front IV > back IV, IV/HV not too hot,
    #    strong term structure edge. Since Barchart file lacks earnings date, this is a proxy.
    out["Post IV Crush Setup Score"] = (
        0.35 * out["Score IV Skew"]
        + 0.25 * out["Score Low IV Rank"]
        + 0.15 * out["Score IV/HV Low"]
        + 0.15 * out["Score Vega/Debit"]
        + 0.10 * out["Score Cheap Debit"]
    )

    # 4) Expected move / OTM directional: moderate OTM, positive vega, reasonable delta direction.
    #    Put calendars score better with negative delta; call calendars score better with positive-ish delta.
    call_delta_score = proximity_score(out["Net Delta"], ideal=0.15, max_good=0.45)
    put_delta_score = proximity_score(out["Net Delta"], ideal=-0.15, max_good=0.45)
    out["Score Directional Delta"] = np.where(out["Type"].eq("Call"), call_delta_score, put_delta_score)
    out["Expected Move Setup Score"] = (
        0.30 * out["Score OTM Moderate"]
        + 0.20 * out["Score Directional Delta"]
        + 0.20 * out["Score Vega/Debit"]
        + 0.15 * out["Score IV Skew"]
        + 0.10 * out["Score DTE Gap"]
        + 0.05 * out["Score Cheap Debit"]
    )

    # Overall ranking blends all four archetypes plus core quality.
    out["Overall Calendar Score"] = (
        0.30 * out["Consolidation Setup Score"]
        + 0.25 * out["Expected Move Setup Score"]
        + 0.25 * out["Low IV Setup Score"]
        + 0.20 * out["Post IV Crush Setup Score"]
    )

    # Best category label
    category_cols = [
        "Low IV Setup Score",
        "Consolidation Setup Score",
        "Post IV Crush Setup Score",
        "Expected Move Setup Score",
    ]
    label_map = {
        "Low IV Setup Score": "Low IV",
        "Consolidation Setup Score": "Consolidation",
        "Post IV Crush Setup Score": "Post-IV-crush proxy",
        "Expected Move Setup Score": "Expected move / OTM",
    }
    out["Best Category"] = out[category_cols].idxmax(axis=1).map(label_map)

    score_cols = [c for c in out.columns if c.endswith("Score")]
    out[score_cols] = out[score_cols].round(1)
    return out.sort_values("Overall Calendar Score", ascending=False)


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Symbol", "Type", "Price~", "Leg1 Strike", "Moneyness %",
        "Exp Leg1", "Exp Leg2", "DTE Short", "DTE Long", "Net Debit",
        "IV Rank", "IV/HV", "Leg1 IV", "Leg2 IV", "IV Skew",
        "Net Delta", "Net Vega", "Vega/Debit", "Best Category",
        "Overall Calendar Score", "Low IV Setup Score", "Consolidation Setup Score",
        "Post IV Crush Setup Score", "Expected Move Setup Score",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    for c in ["Exp Leg1", "Exp Leg2"]:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    rounded = [
        "Price~", "Leg1 Strike", "Moneyness %", "Net Debit", "IV Rank", "IV/HV",
        "Leg1 IV", "Leg2 IV", "IV Skew", "Net Delta", "Net Vega", "Vega/Debit",
        "Overall Calendar Score", "Low IV Setup Score", "Consolidation Setup Score",
        "Post IV Crush Setup Score", "Expected Move Setup Score",
    ]
    for c in rounded:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out


def get_top_unique_symbols(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(["Symbol", "Type"], as_index=False, group_keys=False)
        .head(1)
        .head(n)
    )


def build_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# UI
# -----------------------------
st.title("📅 Calendar Spread Screener")
st.caption("Ranks Barchart long call-calendar and long put-calendar CSVs into trade archetypes.")

with st.sidebar:
    st.header("Upload Barchart files")
    call_file = st.file_uploader("Long call calendar CSV", type=["csv"], key="call_file")
    put_file = st.file_uploader("Long put calendar CSV", type=["csv"], key="put_file")

    st.header("Filters")
    top_n = st.slider("Top setups per category", 3, 25, 5)
    min_score = st.slider("Minimum overall score", 0, 100, 45)
    max_abs_moneyness = st.slider("Max distance from spot (%)", 1, 30, 12)
    min_dte_gap = st.slider("Minimum DTE gap", 1, 90, 7)
    max_debit = st.number_input("Max net debit ($)", min_value=0.0, value=9999.0, step=0.5)
    unique_symbols = st.checkbox("Show one best setup per symbol/type in category tables", value=True)

    st.header("Overall score weights")
    weights = ScoreWeights(
        iv_skew=st.slider("IV skew", 0.0, 1.0, 0.20, 0.05),
        vega_debit=st.slider("Vega/debit", 0.0, 1.0, 0.20, 0.05),
        iv_rank=st.slider("Low IV rank", 0.0, 1.0, 0.15, 0.05),
        iv_hv=st.slider("Low IV/HV", 0.0, 1.0, 0.10, 0.05),
        moneyness=st.slider("Moneyness", 0.0, 1.0, 0.15, 0.05),
        dte_gap=st.slider("DTE gap", 0.0, 1.0, 0.10, 0.05),
        debit=st.slider("Cheap debit", 0.0, 1.0, 0.10, 0.05),
    )

st.info(
    "Upload both Barchart calendar files. The app uses only the columns inside those files, so the "
    "Post-IV-crush and Expected-move categories are proxies unless you later add earnings dates, GEX/DEX, or expected-move data."
)

if not call_file and not put_file:
    st.stop()

try:
    frames = []
    if call_file:
        frames.append(load_calendar_file(call_file, fallback_type="Call"))
    if put_file:
        frames.append(load_calendar_file(put_file, fallback_type="Put"))
    raw = pd.concat(frames, ignore_index=True)
except Exception as exc:
    st.error(f"Could not load files: {exc}")
    st.stop()

ranked = add_scores(raw, weights)
filtered = ranked[
    (ranked["Overall Calendar Score"] >= min_score)
    & (ranked["Abs Moneyness %"] <= max_abs_moneyness)
    & (ranked["DTE Gap"] >= min_dte_gap)
    & (ranked["Net Debit"] <= max_debit)
].copy()

# Summary cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total rows", f"{len(raw):,}")
c2.metric("After filters", f"{len(filtered):,}")
c3.metric("Calls", f"{int((filtered['Type'] == 'Call').sum()):,}")
c4.metric("Puts", f"{int((filtered['Type'] == 'Put').sum()):,}")
c5.metric("Symbols", f"{filtered['Symbol'].nunique():,}")

if filtered.empty:
    st.warning("No rows passed the filters. Loosen score, moneyness, DTE gap, or max debit.")
    st.stop()

# Overall table
st.subheader("Overall ranked calendar trades")
st.dataframe(
    format_for_display(filtered.sort_values("Overall Calendar Score", ascending=False).head(50)),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download ranked results CSV",
    data=build_csv_download(format_for_display(filtered.sort_values("Overall Calendar Score", ascending=False))),
    file_name="ranked_calendar_spread_results.csv",
    mime="text/csv",
)

# Category tabs
category_map = {
    "Low IV": "Low IV Setup Score",
    "Consolidation": "Consolidation Setup Score",
    "Post-IV-crush proxy": "Post IV Crush Setup Score",
    "Expected move / OTM": "Expected Move Setup Score",
}

tabs = st.tabs(list(category_map.keys()))
for tab, (label, score_col) in zip(tabs, category_map.items()):
    with tab:
        st.markdown(f"### Top {top_n} — {label}")
        table = filtered.copy()
        if unique_symbols:
            table = get_top_unique_symbols(table, score_col, top_n)
        else:
            table = table.sort_values(score_col, ascending=False).head(top_n)
        st.dataframe(format_for_display(table), use_container_width=True, hide_index=True)

# Visuals
st.subheader("Category distribution")
cat_counts = filtered["Best Category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]
st.bar_chart(cat_counts, x="Category", y="Count", use_container_width=True)

st.subheader("Score vs debit")
scatter_df = filtered[["Net Debit", "Overall Calendar Score", "Type", "Symbol"]].copy()
st.scatter_chart(scatter_df, x="Net Debit", y="Overall Calendar Score", color="Type", use_container_width=True)

with st.expander("Scoring logic"):
    st.markdown(
        """
**Low IV setup** favors low IV Rank, low IV/HV, high vega/debit, positive IV skew, and strikes close to spot.  
**Consolidation setup** favors ATM strikes, delta neutrality, high short-premium/debit, moderate IV Rank, and a clean DTE gap.  
**Post-IV-crush proxy** favors lower IV Rank plus front-month IV that is still richer than back-month IV. This is a proxy because earnings dates are not in the Barchart calendar files.  
**Expected-move / OTM setup** favors moderate OTM strikes, directional delta, high vega/debit, IV skew, and a clean DTE gap. This is a proxy because expected move is not in the Barchart calendar files.

Suggested next upgrade: add expected-move, earnings date, GEX/DEX, and Donchian/VWAP trend data so the category labels become more precise.
        """
    )
