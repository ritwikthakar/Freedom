from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Calendar Consolidation Scanner",
    page_icon="🎯",
    layout="wide",
)

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


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def percent(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def bounded_score(value, ideal, full_width):
    """100 at ideal, falling linearly to 0 at ideal +/- full_width."""
    return (100 - (value - ideal).abs() / full_width * 100).clip(0, 100)


def load_file(upload, source):
    d = pd.read_csv(upload)
    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    d = d.dropna(how="all").copy()
    d["Symbol"] = d["Symbol"].astype("string").str.upper().str.strip()
    d = d[d["Symbol"].notna() & d["Symbol"].ne("")].copy()

    for c in PCT_COLS:
        d[c] = percent(d[c])
    for c in NUM_COLS:
        d[c] = numeric(d[c])
    for c in ["Exp Leg1", "Exp Leg2"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    d["Source"] = source
    return d


def score_consolidation(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    today = pd.Timestamp.now().normalize()

    # Derived fields
    x["Short DTE"] = (x["Exp Leg1"] - today).dt.days
    x["Long DTE"] = (x["Exp Leg2"] - today).dt.days
    x["DTE Gap"] = x["Long DTE"] - x["Short DTE"]
    x["Moneyness %"] = (x["Leg1 Strike"] - x["Price~"]) / x["Price~"] * 100
    x["Abs Moneyness %"] = x["Moneyness %"].abs()
    x["Abs Delta"] = x["Net Delta"].abs()
    x["Vega / Debit"] = x["Net Vega"] / x["Net Debit"].replace(0, np.nan)
    x["Short Premium / Debit"] = x["Bid1"] / x["Net Debit"].replace(0, np.nan)

    # Absolute scores. These do not depend on the day's other rows.
    # This makes a score comparable from one scan to another.
    x["ATM Score"] = (100 - x["Abs Moneyness %"] / 4.0 * 100).clip(0, 100)
    x["Delta Neutral Score"] = (100 - x["Abs Delta"] / 0.30 * 100).clip(0, 100)

    # Front IV richer than back IV is useful for harvesting the short leg.
    # 0% skew = 40, +2% = 70, +4% or more = 100.
    x["Term Skew Score"] = (40 + x["IV Skew"] * 15).clip(0, 100)

    # Balanced IV/HV is preferred for a pure consolidation/theta calendar.
    # Peak around 1.05; fades as IV/HV becomes unusually low or high.
    x["IV/HV Balance Score"] = bounded_score(x["IV/HV"], ideal=1.05, full_width=0.55)

    # Short premium collected relative to calendar debit.
    x["Short Premium Score"] = (
        x["Short Premium / Debit"] / 0.50 * 100
    ).clip(0, 100)

    # Prefer useful separation without rewarding extremely long back legs.
    x["DTE Gap Score"] = bounded_score(x["DTE Gap"], ideal=28, full_width=28)

    # Positive vega efficiency is a secondary confirmation, not the main thesis.
    vega_eff = x["Vega / Debit"].replace([np.inf, -np.inf], np.nan)
    lo = vega_eff.quantile(0.10)
    hi = vega_eff.quantile(0.90)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(lo, hi):
        x["Vega Efficiency Score"] = 50.0
    else:
        x["Vega Efficiency Score"] = (
            (vega_eff.clip(lo, hi) - lo) / (hi - lo) * 100
        ).clip(0, 100)

    # Core consolidation score:
    # ATM + neutral delta dominate. Term structure is the next-largest input.
    x["Consolidation Score"] = (
        0.30 * x["ATM Score"]
        + 0.25 * x["Delta Neutral Score"]
        + 0.15 * x["Term Skew Score"]
        + 0.10 * x["IV/HV Balance Score"]
        + 0.10 * x["Short Premium Score"]
        + 0.05 * x["DTE Gap Score"]
        + 0.05 * x["Vega Efficiency Score"]
    )

    # Guardrails
    x["Flag"] = ""
    x.loc[x["Net Debit"] <= 0, "Flag"] += "Invalid debit; "
    x.loc[x["Net Vega"] <= 0, "Flag"] += "Non-positive vega; "
    x.loc[x["DTE Gap"] <= 0, "Flag"] += "Expiry ordering; "
    x.loc[x["IV Skew"] < 0, "Flag"] += "Back IV richer than front; "
    x.loc[x["Abs Moneyness %"] > 4, "Flag"] += "Too far from spot for consolidation; "
    x.loc[x["Abs Delta"] > 0.30, "Flag"] += "Too directional; "
    x.loc[x["Short DTE"] < 1, "Flag"] += "Short leg too close to expiry; "

    # Soft penalties rather than hard deletion, so diagnostics remain visible.
    penalty = (
        np.where(x["IV Skew"] < 0, 15, 0)
        + np.where(x["Abs Moneyness %"] > 4, 20, 0)
        + np.where(x["Abs Delta"] > 0.30, 20, 0)
        + np.where(x["DTE Gap"] <= 0, 50, 0)
        + np.where(x["Net Vega"] <= 0, 30, 0)
    )
    x["Preliminary Score"] = (x["Consolidation Score"] - penalty).clip(0, 100)

    # Research priority, not a recommendation to enter.
    x["Validation Priority"] = pd.cut(
        x["Preliminary Score"],
        bins=[-np.inf, 60, 70, 80, 88, np.inf],
        labels=["LOW", "WATCH", "B", "A", "A+"],
    ).astype(str)

    # Hybrid tag: consolidation plus useful positive-vega characteristics.
    x["Play Type"] = "CONSOLIDATION"
    hybrid = (
        (x["Preliminary Score"] >= 75)
        & (x["Vega Efficiency Score"] >= 60)
        & (x["IV Rank"] <= 55)
    )
    x.loc[hybrid, "Play Type"] = "CONSOLIDATION + VEGA"

    # Why this row ranked well
    reasons = []
    for _, r in x.iterrows():
        parts = []
        if r["Abs Moneyness %"] <= 1:
            parts.append("near ATM")
        if r["Abs Delta"] <= 0.10:
            parts.append("delta neutral")
        if r["IV Skew"] >= 2:
            parts.append("strong term skew")
        elif r["IV Skew"] >= 0:
            parts.append("positive term skew")
        if 0.90 <= r["IV/HV"] <= 1.20:
            parts.append("balanced IV/HV")
        if r["Vega Efficiency Score"] >= 70:
            parts.append("efficient vega")
        reasons.append(", ".join(parts) if parts else "limited consolidation confluence")
    x["Why"] = reasons

    return x.sort_values("Preliminary Score", ascending=False)


def display_table(d):
    cols = [
        "Symbol", "Type", "Price~", "Leg1 Strike", "Moneyness %",
        "Exp Leg1", "Exp Leg2", "Short DTE", "Long DTE", "DTE Gap",
        "Net Debit", "IV Rank", "IV/HV", "Leg1 IV", "Leg2 IV", "IV Skew",
        "Net Delta", "Net Vega", "Vega / Debit", "Short Premium / Debit",
        "Play Type", "Preliminary Score", "Validation Priority", "Why", "Flag",
    ]
    z = d[[c for c in cols if c in d.columns]].copy()
    for c in ["Exp Leg1", "Exp Leg2"]:
        z[c] = z[c].dt.strftime("%Y-%m-%d")
    num = [
        "Price~", "Leg1 Strike", "Moneyness %", "Net Debit", "IV Rank",
        "IV/HV", "Leg1 IV", "Leg2 IV", "IV Skew", "Net Delta", "Net Vega",
        "Vega / Debit", "Short Premium / Debit", "Preliminary Score",
    ]
    for c in num:
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce").round(2)
    return z


st.title("🎯 Calendar Consolidation Scanner")
st.caption(
    "Stage-1 scanner: finds near-ATM, delta-neutral calendars with favorable "
    "term structure for subsequent dealer-positioning validation."
)

with st.sidebar:
    st.header("Uploads")
    put_file = st.file_uploader("Long Put Calendar CSV", type="csv", key="put")
    call_file = st.file_uploader("Long Call Calendar CSV", type="csv", key="call")

    st.header("Filters")
    min_score = st.slider("Minimum preliminary score", 0, 100, 65)
    max_moneyness = st.slider("Maximum |moneyness| %", 0.5, 8.0, 4.0, 0.25)
    max_abs_delta = st.slider("Maximum |net delta|", 0.05, 0.50, 0.30, 0.01)
    min_skew = st.number_input("Minimum IV skew %", value=0.0, step=0.25)
    min_dte_gap = st.slider("Minimum DTE gap", 1, 60, 7)
    top_n = st.slider("Top unique symbols", 3, 25, 10)

if not put_file and not call_file:
    st.info("Upload the Barchart long-put and/or long-call calendar CSV.")
    st.stop()

frames = []
errors = []
for f, label in [(put_file, "Put"), (call_file, "Call")]:
    if f is not None:
        try:
            frames.append(load_file(f, label))
        except Exception as exc:
            errors.append(f"{label}: {exc}")

for e in errors:
    st.error(e)
if not frames:
    st.stop()

raw = pd.concat(frames, ignore_index=True)
ranked = score_consolidation(raw)

flt = ranked[
    (ranked["Preliminary Score"] >= min_score)
    & (ranked["Abs Moneyness %"] <= max_moneyness)
    & (ranked["Abs Delta"] <= max_abs_delta)
    & (ranked["IV Skew"] >= min_skew)
    & (ranked["DTE Gap"] >= min_dte_gap)
    & (ranked["Net Debit"] > 0)
    & (ranked["Net Vega"] > 0)
].copy()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Raw candidates", f"{len(raw):,}")
c2.metric("Passed filters", f"{len(flt):,}")
c3.metric("A/A+", int(flt["Validation Priority"].isin(["A", "A+"]).sum()))
c4.metric("Hybrid Vega", int(flt["Play Type"].eq("CONSOLIDATION + VEGA").sum()))
c5.metric("Symbols", flt["Symbol"].nunique())

st.subheader("Top candidates for dealer validation")
st.caption(
    "Validation Priority ranks research priority only. A/A+ means export the "
    "ticker-specific dealer files next; it is not an entry signal."
)

if flt.empty:
    st.warning("No candidates passed the current filters.")
else:
    # One best contract per symbol for the research queue.
    queue = (
        flt.sort_values("Preliminary Score", ascending=False)
        .groupby("Symbol", as_index=False, group_keys=False)
        .head(1)
        .head(top_n)
    )
    st.dataframe(display_table(queue), use_container_width=True, hide_index=True)

    st.download_button(
        "Download dealer-validation queue",
        data=display_table(queue).to_csv(index=False).encode("utf-8"),
        file_name="calendar_consolidation_validation_queue.csv",
        mime="text/csv",
    )

tabs = st.tabs(["All qualifying contracts", "Puts", "Calls", "Diagnostics"])

with tabs[0]:
    st.dataframe(display_table(flt), use_container_width=True, hide_index=True)

with tabs[1]:
    st.dataframe(
        display_table(flt[flt["Type"].str.lower().eq("put")]),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.dataframe(
        display_table(flt[flt["Type"].str.lower().eq("call")]),
        use_container_width=True,
        hide_index=True,
    )

with tabs[3]:
    diag_cols = [
        "Symbol", "Type", "Leg1 Strike", "ATM Score", "Delta Neutral Score",
        "Term Skew Score", "IV/HV Balance Score", "Short Premium Score",
        "DTE Gap Score", "Vega Efficiency Score", "Consolidation Score",
        "Preliminary Score", "Flag",
    ]
    diag = ranked[diag_cols].copy()
    score_cols = [c for c in diag.columns if "Score" in c]
    diag[score_cols] = diag[score_cols].round(1)
    st.dataframe(diag.head(150), use_container_width=True, hide_index=True)

with st.expander("Scoring model"):
    st.markdown("""
**Preliminary Consolidation Score**

- **30% ATM proximity**
- **25% delta neutrality**
- **15% front-vs-back IV term skew**
- **10% balanced IV/HV**
- **10% short-premium / debit efficiency**
- **5% DTE gap**
- **5% vega / debit efficiency**

The score intentionally does **not** use GEX, DEX, expected move, gamma flip,
call wall, put wall, or GEX-by-strike. Those belong to **Stage 2 dealer
validation** after the preliminary scanner has reduced the universe.

**A/A+ = research priority, not an entry recommendation.**
""")
