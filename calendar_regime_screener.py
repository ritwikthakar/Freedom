from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calendar Regime Screener", page_icon="📅", layout="wide")

REQUIRED = {
    "Symbol", "Price~", "Exp Leg1", "Leg1 Strike", "Type", "Bid1", "Exp Leg2",
    "Ask2", "Net Debit", "Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank",
    "IV/HV", "Net Delta", "Net Vega"
}

PCT_COLS = ["Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank"]
NUM_COLS = ["Price~", "Leg1 Strike", "Bid1", "Ask2", "Net Debit", "IV/HV", "Net Delta", "Net Vega"]


def pct_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or np.isclose(lo, hi):
        out = pd.Series(50.0, index=s.index)
    else:
        out = (s - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        out = 100.0 - out
    return out.clip(0, 100)


def load_calendar_csv(file_or_path, source_label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    out = df.copy()
    # Barchart exports can contain completely blank/footer rows. Remove rows that
    # do not identify an actual option setup before calculating scores.
    out = out.dropna(how="all").copy()
    if "Symbol" in out.columns:
        out["Symbol"] = out["Symbol"].astype("string").str.strip()
        out = out[out["Symbol"].notna() & out["Symbol"].ne("")].copy()

    for c in PCT_COLS:
        out[c] = pct_to_float(out[c])
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["Exp Leg1", "Exp Leg2"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    if source_label:
        out["Source"] = source_label
    else:
        out["Source"] = out["Type"].astype(str).str.title()

    return enrich(out)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Strike Distance %"] = ((d["Leg1 Strike"] - d["Price~"]) / d["Price~"] * 100).replace([np.inf, -np.inf], np.nan)
    d["Abs Strike Distance %"] = d["Strike Distance %"].abs()
    d["Abs Delta"] = d["Net Delta"].abs()
    d["Vega / Debit"] = (d["Net Vega"] / d["Net Debit"]).replace([np.inf, -np.inf], np.nan)
    d["Delta / Debit"] = (d["Abs Delta"] / d["Net Debit"]).replace([np.inf, -np.inf], np.nan)
    d["Skew / Debit"] = (d["IV Skew"] / d["Net Debit"]).replace([np.inf, -np.inf], np.nan)

    today = pd.Timestamp.now().normalize()
    d["Short DTE"] = (d["Exp Leg1"] - today).dt.days
    d["Long DTE"] = (d["Exp Leg2"] - today).dt.days
    d["DTE Gap"] = d["Long DTE"] - d["Short DTE"]

    # Regime labels based on IV/HV. These are intentionally editable in the sidebar.
    d["Raw Regime"] = np.select(
        [d["IV/HV"] < 0.90, d["IV/HV"].between(0.90, 1.20, inclusive="both"), d["IV/HV"] > 1.20],
        ["VEGA", "THETA", "DELTA"],
        default="REVIEW"
    )
    return d


def score_dataframe(df: pd.DataFrame, vega_max: float, theta_max: float, delta_min: float, min_skew: float) -> pd.DataFrame:
    d = df.copy()

    # Cross-sectional component scores (0-100).
    skew_score = minmax(d["IV Skew"], True)
    strike_neutrality = minmax(d["Abs Strike Distance %"], False)
    delta_neutrality = minmax(d["Abs Delta"], False)
    debit_eff = minmax(d["Net Debit"], False)
    vega_eff = minmax(d["Vega / Debit"], True)
    delta_eff = minmax(d["Delta / Debit"], True)

    # Regime-fit scores: closeness to desirable zones.
    theta_ivhv = (100 - ((d["IV/HV"] - 1.05).abs() / 0.35 * 100)).clip(0, 100)
    vega_ivhv = ((vega_max - d["IV/HV"]) / max(vega_max - 0.50, 0.01) * 100).clip(0, 100)
    delta_ivhv = ((d["IV/HV"] - delta_min) / max(2.00 - delta_min, 0.01) * 100).clip(0, 100)

    ivrank_cheap = (100 - d["IV Rank"]).clip(0, 100)
    otm_directionality = minmax(d["Abs Strike Distance %"], True)

    d["Theta Score"] = (
        0.30 * skew_score +
        0.25 * theta_ivhv +
        0.20 * strike_neutrality +
        0.15 * delta_neutrality +
        0.10 * debit_eff
    )

    d["Vega Score"] = (
        0.30 * vega_ivhv +
        0.25 * vega_eff +
        0.20 * ivrank_cheap +
        0.15 * skew_score +
        0.10 * debit_eff
    )

    d["Delta Score"] = (
        0.30 * delta_ivhv +
        0.25 * delta_eff +
        0.20 * otm_directionality +
        0.15 * skew_score +
        0.10 * debit_eff
    )

    score_cols = ["Theta Score", "Vega Score", "Delta Score"]

    # Pandas 3.x/Cloud raises ValueError when idxmax() encounters a row where
    # every score is NA. This can happen with partially populated Barchart rows.
    # Keep those rows visible for diagnostics instead of crashing the whole app.
    score_matrix = d[score_cols].apply(pd.to_numeric, errors="coerce")
    has_score = score_matrix.notna().any(axis=1)
    d["Best Score"] = score_matrix.max(axis=1, skipna=True)
    d["Play Type"] = "REVIEW"
    if has_score.any():
        best_col = score_matrix.loc[has_score].fillna(-np.inf).idxmax(axis=1)
        d.loc[has_score, "Play Type"] = (
            best_col.str.replace(" Score", "", regex=False).str.upper()
        )

    # Hard guardrails / warnings.
    d["Flag"] = ""
    d.loc[~has_score, "Flag"] += "Incomplete numeric data; "
    d.loc[d["IV Skew"] < min_skew, "Flag"] += "Weak/negative term skew; "
    d.loc[d["Net Debit"] <= 0, "Flag"] += "Invalid debit; "
    d.loc[d["DTE Gap"] <= 0, "Flag"] += "Expiry ordering issue; "
    d.loc[(d["IV/HV"] > 1.60) & (d["IV Skew"] < 0), "Flag"] += "Rich IV + bad skew; "
    d.loc[(d["Play Type"] == "THETA") & (d["Abs Strike Distance %"] > 5), "Flag"] += "Too far OTM for theta; "
    d.loc[(d["Play Type"] == "DELTA") & (d["Abs Strike Distance %"] < 1), "Flag"] += "Not very directional; "

    # Override the play type when the IV/HV regime is clearly incompatible.
    d.loc[d["IV/HV"] <= vega_max, "Play Type"] = "VEGA"
    d.loc[d["IV/HV"].between(vega_max, theta_max, inclusive="right"), "Play Type"] = "THETA"
    d.loc[d["IV/HV"] >= delta_min, "Play Type"] = "DELTA"

    # Re-attach score matching final play type.
    map_score = {"THETA": "Theta Score", "VEGA": "Vega Score", "DELTA": "Delta Score"}
    def selected_score(row):
        col = map_score.get(row["Play Type"])
        if col is not None and pd.notna(row.get(col)):
            return row[col]
        return row.get("Best Score", np.nan)

    d["Regime Score"] = d.apply(selected_score, axis=1)

    # Penalize structurally poor calendars.
    penalty = np.where(d["IV Skew"] < min_skew, 12, 0) + np.where(d["DTE Gap"] < 7, 8, 0)
    d["Final Score"] = (d["Regime Score"] - penalty).clip(0, 100)

    return d.sort_values("Final Score", ascending=False)


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Symbol", "Type", "Play Type", "Final Score", "Price~", "Leg1 Strike",
        "Exp Leg1", "Exp Leg2", "Short DTE", "Long DTE", "Net Debit", "IV/HV",
        "Leg1 IV", "Leg2 IV", "IV Skew", "IV Rank", "Net Delta", "Net Vega",
        "Strike Distance %", "Vega / Debit", "Delta / Debit", "Flag"
    ]
    x = df[[c for c in cols if c in df.columns]].copy()
    for c in ["Exp Leg1", "Exp Leg2"]:
        if c in x:
            x[c] = x[c].dt.strftime("%Y-%m-%d")
    return x


st.title("📅 Calendar Spread Regime Screener")
st.caption("Classifies Barchart calendar candidates as THETA, VEGA, or DELTA plays using IV/HV, term IV skew, strike distance, debit, and net Greeks.")

with st.sidebar:
    st.header("Regime thresholds")
    vega_max = st.number_input("VEGA: IV/HV at or below", min_value=0.50, max_value=1.20, value=0.90, step=0.05)
    theta_max = st.number_input("THETA: IV/HV upper bound", min_value=0.90, max_value=1.60, value=1.20, step=0.05)
    delta_min = st.number_input("DELTA: IV/HV at or above", min_value=1.00, max_value=2.50, value=1.30, step=0.05)
    min_skew = st.number_input("Minimum preferred term IV skew (%)", min_value=-20.0, max_value=30.0, value=0.0, step=0.5)
    st.divider()
    min_score = st.slider("Minimum final score", 0, 100, 55)
    top_n = st.slider("Rows per ranking", 5, 50, 15)
    st.markdown("### 🔗 Trading Workflow")

    st.link_button(
        "⚡ Cheap Convexity Screener",
        "YOUR_CHEAP_CONVEXITY_APP_URL",
        use_container_width=True
    )
    
    st.link_button(
        "🎯 Dealer Positioning",
        "https://dealerpositioning-n2m58uzu42afb44usojrwe.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "📊 Dealer / GEX Analysis",
        "https://dealerpositioning-zenvhgc3fs3dcsd9yunvct.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🔬 Option Analysis",
        "https://freedom-fuxffx4ohuuosfojdmffxl.streamlit.app/",
        use_container_width=True
    )

st.subheader("1) Upload Barchart screeners")
c1, c2 = st.columns(2)
with c1:
    put_file = st.file_uploader("Long Put Calendar CSV", type="csv", key="put")
with c2:
    call_file = st.file_uploader("Long Call Calendar CSV", type="csv", key="call")

if not put_file and not call_file:
    st.info("Upload one or both Barchart calendar screener CSVs to begin.")
    st.stop()

frames: list[pd.DataFrame] = []
errors: list[str] = []
for f, label in [(put_file, "Put"), (call_file, "Call")]:
    if f is not None:
        try:
            frames.append(load_calendar_csv(f, label))
        except Exception as exc:
            errors.append(f"{label}: {exc}")

if errors:
    for e in errors:
        st.error(e)
if not frames:
    st.stop()

raw = pd.concat(frames, ignore_index=True)
ranked = score_dataframe(raw, vega_max, theta_max, delta_min, min_skew)

st.subheader("2) Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    symbols = st.multiselect("Symbols", sorted(ranked["Symbol"].dropna().unique()))
with f2:
    types = st.multiselect("Option type", sorted(ranked["Type"].dropna().unique()), default=sorted(ranked["Type"].dropna().unique()))
with f3:
    play_types = st.multiselect("Play type", ["THETA", "VEGA", "DELTA"], default=["THETA", "VEGA", "DELTA"])
with f4:
    max_debit = st.number_input("Maximum debit", min_value=0.0, value=float(np.nanpercentile(ranked["Net Debit"], 90)), step=0.25)

flt = ranked.copy()
if symbols:
    flt = flt[flt["Symbol"].isin(symbols)]
if types:
    flt = flt[flt["Type"].isin(types)]
if play_types:
    flt = flt[flt["Play Type"].isin(play_types)]
flt = flt[(flt["Net Debit"] <= max_debit) & (flt["Final Score"] >= min_score)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidates", f"{len(flt):,}")
m2.metric("Theta plays", int((flt["Play Type"] == "THETA").sum()))
m3.metric("Vega plays", int((flt["Play Type"] == "VEGA").sum()))
m4.metric("Delta plays", int((flt["Play Type"] == "DELTA").sum()))

st.subheader("3) Ranked setups")
tabs = st.tabs(["Overall", "🟢 Theta", "🔵 Vega", "🟠 Delta", "Diagnostics"])

with tabs[0]:
    st.dataframe(fmt_table(flt.head(top_n)), use_container_width=True, hide_index=True)
with tabs[1]:
    st.caption("Best when IV/HV is balanced, term skew is favorable, strike is near spot, and net delta is relatively neutral.")
    st.dataframe(fmt_table(flt[flt["Play Type"] == "THETA"].head(top_n)), use_container_width=True, hide_index=True)
with tabs[2]:
    st.caption("Best when IV is cheap relative to realized volatility and the calendar provides efficient positive vega exposure.")
    st.dataframe(fmt_table(flt[flt["Play Type"] == "VEGA"].head(top_n)), use_container_width=True, hide_index=True)
with tabs[3]:
    st.caption("Best when IV/HV is elevated and you intentionally want directional exposure through an OTM calendar.")
    st.dataframe(fmt_table(flt[flt["Play Type"] == "DELTA"].head(top_n)), use_container_width=True, hide_index=True)
with tabs[4]:
    st.write("Score components")
    diag_cols = ["Symbol", "Type", "Play Type", "Theta Score", "Vega Score", "Delta Score", "Final Score", "IV/HV", "IV Skew", "Abs Delta", "Vega / Debit", "Delta / Debit", "Abs Strike Distance %", "Flag"]
    st.dataframe(flt[diag_cols].head(100), use_container_width=True, hide_index=True)

st.subheader("4) Visuals")
v1, v2 = st.columns(2)
with v1:
    chart_df = flt.head(30).copy()
    if not chart_df.empty:
        chart_df["Setup"] = chart_df["Symbol"].astype(str) + " " + chart_df["Type"].astype(str) + " " + chart_df["Leg1 Strike"].astype(str)
        st.bar_chart(chart_df.set_index("Setup")[["Final Score"]])
with v2:
    if not flt.empty:
        st.scatter_chart(flt, x="IV/HV", y="IV Skew", size="Final Score", color="Play Type")

st.subheader("5) Export")
export = fmt_table(flt)
st.download_button(
    "Download ranked CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="calendar_regime_ranked.csv",
    mime="text/csv",
)

with st.expander("How the model interprets each regime"):
    st.markdown(
        """
**THETA** — IV/HV roughly balanced, positive front-vs-back IV skew, near-ATM strike, and relatively neutral delta.  
**VEGA** — IV/HV low, positive vega per dollar, preferably lower IV Rank, with acceptable term structure.  
**DELTA** — IV/HV elevated and the trade intentionally uses directional delta / OTM strike placement.  

**Important:** Barchart files do not include net theta or historical changes in IV/HV. Therefore this version uses term skew, strike proximity, debit efficiency, net delta and net vega as proxies. If you later add daily IV/HV history or net theta, the model can directly detect the stronger signal: **IV falling while HV rises = delta-transition regime**.
        """
    )
