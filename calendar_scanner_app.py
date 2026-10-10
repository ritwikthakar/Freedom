from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import calendar_core as cc

st.set_page_config(page_title="Calendar Income Scanner", page_icon="🎯", layout="wide")

CORE_COLS = [
    "Symbol", "Type", "Price~", "Leg1 Strike", "Dist (EM)", "Exp Leg1", "Exp Leg2",
    "Short DTE", "DTE Gap", "Net Debit", "Debit % Px", "Peak ROI", "Peak ROI / Day %",
    "EV / Debit", "Prob Profit", "BE Low", "BE High", "BE Width (EM)",
    "IV Skew", "IV/HV", "IV Rank", "Net Delta", "Net Vega",
    "Event Status", "Earnings Date", "Score", "Priority", "Flag",
]
EVENT_COLS = [
    "Symbol", "Type", "Price~", "Leg1 Strike", "Dist (EM)", "Exp Leg1", "Exp Leg2",
    "Short DTE", "DTE Gap", "Net Debit", "Debit % Px", "Front EM $", "Debit / EM",
    "Leg1 IV", "Leg2 IV", "IV Skew", "IV/HV", "Net Delta", "Net Vega",
    "Event Status", "Earnings Date", "Event Score",
]


def show(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = d[[c for c in cols if c in d.columns]].copy()
    for c in ["Exp Leg1", "Exp Leg2", "Earnings Date"]:
        if c in z:
            z[c] = pd.to_datetime(z[c], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in z.columns:
        if pd.api.types.is_float_dtype(z[c]):
            z[c] = z[c].round(2)
    return z


def best_per_symbol(d: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    return d.sort_values(by, ascending=False).groupby("Symbol", group_keys=False).head(1).head(n)


st.title("🎯 Calendar Income Scanner")
st.caption(
    "Ranks calendars by modeled payoff at front expiry (peak return, breakeven width, "
    "probability, expected value), with earnings/event calendars tracked separately."
)

with st.sidebar:
    st.header("Uploads")
    put_file = st.file_uploader("Long Put Calendar CSV", type="csv", key="put")
    call_file = st.file_uploader("Long Call Calendar CSV", type="csv", key="call")

    st.header("Cadence")
    cadence = st.selectbox("Income cadence", list(cc.CADENCES), index=0)

    st.header("Earnings / events")
    earn_text = st.text_area(
        "Earnings dates (one per line: SYMBOL,YYYY-MM-DD)",
        height=140,
        placeholder="AAPL,2026-10-29\nJPM,2026-10-14",
    )
    event_skew = st.slider(
        "Fallback: treat IV skew >= X pts as a suspected event", 4.0, 20.0, 8.0, 0.5,
        help="Only used for symbols with no date entered. Skew is a weak proxy; "
             "dates are far more reliable.",
    )

    st.header("Filters")
    min_score = st.slider("Minimum score", 0, 100, 50)
    max_dist = st.slider("Max strike distance (front expected moves)", 0.1, 1.5, 0.75, 0.05)
    max_abs_delta = st.slider("Maximum |net delta|", 0.05, 0.50, 0.30, 0.01)
    min_skew = st.number_input("Minimum IV skew (pts)", value=0.0, step=0.25)
    min_gap = st.slider("Minimum DTE gap", 1, 60, 7)
    top_n = st.slider("Top unique symbols", 3, 25, 10)

    with st.expander("Model assumptions"):
        rate = st.number_input("Risk-free rate %", value=4.0, step=0.25) / 100
        vol_ratio = st.slider(
            "Realized / implied vol", 0.6, 1.2, 0.90, 0.05,
            help="Spot distribution vol = Leg1 IV x this. Below 1.0 assumes implied "
                 "vol overstates realized vol, which is what makes short premium pay.")
        back_shift = st.slider(
            "Back-leg IV change by front expiry (vol pts)", -10.0, 5.0, 0.0, 0.5)

if not put_file and not call_file:
    st.info("Upload the Barchart long-put and/or long-call calendar CSV.")
    st.stop()

frames, infos, errors = [], [], []
for f, label in [(put_file, "Put"), (call_file, "Call")]:
    if f is not None:
        try:
            d, info = cc.load_file(f, label)
            frames.append(d)
            infos.append((label, info))
        except Exception as exc:
            errors.append(f"{label}: {exc}")
for e in errors:
    st.error(e)
if not frames:
    st.stop()

raw = pd.concat(frames, ignore_index=True)
earnings = cc.parse_earnings(earn_text)

x = cc.derive(raw)
x = cc.classify_events(x, earnings, event_skew)
x = cc.run_models(x, rate, vol_ratio, back_shift)
ranked = cc.score_core(x, cadence, event_skew)

core = ranked[~ranked["Is Event Play"]]
core = core[
    core["In Cadence"]
    & (core["Score"] >= min_score)
    & (core["Dist (EM)"] <= max_dist)
    & (core["Abs Delta"] <= max_abs_delta)
    & (core["IV Skew"] >= min_skew)
    & (core["DTE Gap"] >= min_gap)
    & (core["Net Debit"] > 0)
    & (core["Net Vega"] > 0)
].copy()

events = cc.score_earnings(ranked[ranked["Is Event Play"]])
events = events[
    (events["Short DTE"] >= 1)
    & (events["Net Debit"] > 0)
    & (events["Net Vega"] > 0)
    & (events["Dist (EM)"] <= max_dist)
    & (events["DTE Gap"] >= min_gap)
].copy()

all_syms = sorted(raw["Symbol"].dropna().unique())
no_date = [s for s in all_syms if s not in earnings]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Contracts loaded", f"{len(raw):,}")
c2.metric("Core passed", f"{len(core):,}")
c3.metric("A / A+", int(core["Priority"].isin(["A", "A+"]).sum()))
c4.metric("Event plays", f"{len(events):,}")
c5.metric("Symbols w/o earnings date", f"{len(no_date)} / {len(all_syms)}")

if no_date:
    st.warning(
        f"{len(no_date)} of {len(all_syms)} symbols have no earnings date. For those, "
        "core rows are only screened by the IV-skew proxy, which misses events when "
        "front IV is only modestly elevated. Add dates in the sidebar before trusting "
        "A/A+ rows."
    )

tabs = st.tabs(["Core income queue", "All core contracts", "Earnings / events",
                "Diagnostics", "Scoring model"])

with tabs[0]:
    st.subheader(f"Top candidates - {cadence}")
    st.caption(
        "Ranked on modeled payoff at front expiry, not on moneyness alone. "
        "Priority is a research ranking, not an entry signal."
    )
    if core.empty:
        st.warning("No candidates passed the current filters.")
    else:
        q = best_per_symbol(core, "Score", top_n)
        st.dataframe(show(q, CORE_COLS), use_container_width=True, hide_index=True)
        st.download_button(
            "Download core queue",
            show(q, CORE_COLS).to_csv(index=False).encode("utf-8"),
            file_name="calendar_core_queue.csv", mime="text/csv",
        )

with tabs[1]:
    st.dataframe(show(core.sort_values("Score", ascending=False), CORE_COLS),
                 use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Earnings / event calendars (tracked separately)")
    st.caption(
        "Structure-only shortlist: cost, centering and front-IV premium. The CSVs have no "
        "earnings-move history and the payoff model assumes no IV crush, so it is NOT applied "
        "here. Compare the front expected move with the stock's historical earnings moves "
        "before trading. Debit / EM is cost as a fraction of the front-leg implied move."
    )
    if events.empty:
        st.info("No earnings or suspected-event calendars in the current data.")
    else:
        eq = best_per_symbol(events, "Event Score", 25)
        st.dataframe(show(eq, EVENT_COLS), use_container_width=True, hide_index=True)
        st.download_button(
            "Download event queue",
            show(eq, EVENT_COLS).to_csv(index=False).encode("utf-8"),
            file_name="calendar_event_queue.csv", mime="text/csv",
        )
        with st.expander("All event contracts"):
            st.dataframe(show(events, EVENT_COLS), use_container_width=True, hide_index=True)

with tabs[3]:
    for label, info in infos:
        st.write(
            f"**{label} file** - as of "
            f"{info['as_of'].date() if pd.notna(info['as_of']) else 'unknown (using today)'}; "
            f"{info['rows']} contracts loaded, {info['dropped']} non-data row(s) dropped."
        )
    st.write("**Symbols with no earnings date:**", ", ".join(no_date) or "none")
    dc = [
        "Symbol", "Type", "Leg1 Strike", "Short DTE", "EV Score", "ROI Score",
        "Prob Score", "Skew Score", "Center Score", "Delta Score", "IV/HV Score",
        "Gap Score", "Income Score", "IV/HV Penalty", "Score", "Event Status", "Flag",
    ]
    diag = ranked[dc].copy()
    for c in diag.columns:
        if pd.api.types.is_float_dtype(diag[c]):
            diag[c] = diag[c].round(1)
    st.dataframe(diag.head(200), use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown("""
**Payoff model.** For each calendar the app reprices the long leg at front expiry
(Black-Scholes at the back-leg IV, plus your IV-change input) across a lognormal range of
spot prices, subtracts the short leg's intrinsic value and the debit (Ask2 - Bid1, i.e.
worst-case fills). From that curve it derives peak P&L, breakeven prices and width, the
probability of profit, and expected value per dollar of debit. Dividends, commissions and
the volatility smile are ignored.

**Income score (core, non-event):**
- 25% expected value / debit
- 20% peak ROI on debit
- 15% probability of profit
- 15% term skew (capped at the event threshold; bigger skew is routed to the event tab)
- 10% strike centering, measured in front expected moves
- 5% delta neutrality, 5% IV/HV balance, 5% DTE-gap fit for the chosen cadence

Penalties apply for negative skew, strikes far from spot, high |delta|, extreme IV/HV, and
earnings that fall inside the back-leg window.

**A+** needs score >= 70, strike within 0.2 expected moves, |delta| <= 0.10, positive EV,
peak ROI >= 50%, probability >= 40%, gap >= 7 days, and no event flag.

**Events.** With a date entered, earnings on or before the short expiry puts a contract in
the event tab. Without one, skew >= the sidebar threshold (or front IV >= 1.35x back IV)
marks it as a suspected event.

All weights and thresholds are judgment calls and have not been backtested.
""")
