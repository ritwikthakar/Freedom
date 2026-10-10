from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import big_move as bm
import calendar_core as cc

st.set_page_config(page_title="Calendar Income Scanner", page_icon="🎯", layout="wide")

CORE_COLS = [
    "Symbol", "Type", "Price~", "Leg1 Strike", "Dist (EM)", "Exp Leg1", "Exp Leg2",
    "Short DTE", "DTE Gap", "Net Debit", "Debit % Px", "Peak ROI", "Peak ROI / Day %",
    "EV / Debit", "Prob Profit", "BE Low", "BE High", "BE Width (EM)",
    "IV Skew", "IV/HV", "IV Rank", "Net Delta", "Net Vega",
    "Event Status", "Earnings Date", "Earnings Timing", "Score", "Priority", "Flag",
]
EVENT_COLS = [
    "Symbol", "Type", "Price~", "Leg1 Strike", "Dist (EM)", "Exp Leg1", "Exp Leg2",
    "Short DTE", "DTE Gap", "Net Debit", "Debit % Px", "Front EM $", "Debit / EM",
    "Implied Event Move %", "Leg1 IV", "Leg2 IV", "IV Skew", "IV/HV", "Net Delta",
    "Net Vega", "Event Status", "Earnings Date", "Earnings Timing", "Event Score",
]
WATCH_COLS = [
    "Symbol", "Price", "Score", "Signals", "Catalyst", "Earnings Date", "Timing",
    "Days To Earnings", "IV Rank", "UOA Contracts", "UOA $M", "UOA Call %",
    "Flow Prints", "Flow $M", "Flow Opening %", "Flow Excluded %",
    "Term Skew", "Implied Event Move %", "Lean Label", "Notes",
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


def watch_view(w: pd.DataFrame) -> pd.DataFrame:
    """Rename and rescale the watchlist for display."""
    z = pd.DataFrame(index=w.index)
    z["Symbol"] = w["Symbol"]
    z["Price"] = w["Price"]
    z["Score"] = w["Score"]
    z["Signals"] = w["Signals"]
    z["Catalyst"] = w["Catalyst"]
    z["Earnings Date"] = pd.to_datetime(w["Earnings Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    z["Timing"] = w["Timing"]
    z["Days To Earnings"] = w["Days To Earnings"]
    z["IV Rank"] = w["IV_Rank"]
    z["UOA Contracts"] = w["UOA_Contracts"]
    z["UOA $M"] = w["UOA_Premium"] / 1e6
    z["UOA Call %"] = (w["UOA_Call"] / (w["UOA_Call"] + w["UOA_Put"]).replace(0, np.nan)) * 100
    z["Flow Prints"] = w["Flow_Prints"]
    z["Flow $M"] = w["Flow_Premium"] / 1e6
    z["Flow Opening %"] = w["Flow_Open_Share"] * 100
    z["Flow Excluded %"] = w["Flow_Excluded_Share"] * 100
    z["Term Skew"] = w["Term_Skew"]
    z["Implied Event Move %"] = w["Event_Move"]
    z["Lean Label"] = w["Lean Label"]
    z["Notes"] = w["Notes"]
    for c in z.columns:
        if pd.api.types.is_float_dtype(z[c]):
            z[c] = z[c].round(1)
    return z[[c for c in WATCH_COLS if c in z.columns]]


st.title("🎯 Calendar Income Scanner")
st.caption(
    "Ranks calendars by modeled payoff at front expiry (peak return, breakeven width, "
    "probability, expected value), with earnings/event calendars tracked separately, "
    "plus a big-move watchlist built from earnings, flow and unusual-activity files."
)

with st.sidebar:
    st.header("Calendar uploads")
    put_file = st.file_uploader("Long Put Calendar CSV", type="csv", key="put")
    call_file = st.file_uploader("Long Call Calendar CSV", type="csv", key="call")

    st.header("Market data uploads")
    earn_file = st.file_uploader("Earnings calendar CSV", type="csv", key="earn")
    flow_file = st.file_uploader("Options flow CSV", type="csv", key="flow")
    unusual_file = st.file_uploader("Unusual options activity CSV", type="csv", key="unusual")

    st.header("Cadence")
    cadence = st.selectbox("Income cadence", list(cc.CADENCES), index=0)

    st.header("Earnings / events")
    earn_text = st.text_area(
        "Manual earnings dates (override the file)",
        height=100,
        placeholder="AAPL,2026-10-29\nJPM,2026-10-14",
        help="One per line: SYMBOL,YYYY-MM-DD. Anything here wins over the uploaded earnings file.",
    )
    event_skew = st.slider(
        "Fallback: treat IV skew >= X pts as a suspected event", 4.0, 20.0, 8.0, 0.5,
        help="Only used for symbols with no date. Skew is a weak proxy; dates are far more reliable.",
    )

    st.header("Calendar filters")
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

    st.header("Big-move watch")
    bm_min_price = st.number_input("Minimum stock price $", value=5.0, step=1.0)
    bm_earn_window = st.slider("Earnings window (days)", 3, 30, 14)
    bm_min_signals = st.slider(
        "Minimum signals", 1, 4, 2,
        help="Signals = earnings in window, unusual activity, qualifying flow, front-IV inversion.")
    with st.expander("Big-move filters"):
        bm_min_voi = st.number_input("Unusual: minimum Vol/OI", value=3.0, step=0.5)
        bm_min_uvol = st.number_input("Unusual: minimum contract volume", value=500, step=100)
        bm_uoa_dte = st.slider("Unusual: DTE range", 0, 120, (1, 60))
        bm_flow_dte = st.slider("Flow: max DTE", 7, 365, 60)
        bm_flow_delta = st.slider(
            "Flow: |delta| band", 0.0, 1.0, (0.05, 0.75), 0.05,
            help="Drops deep-ITM stock-replacement prints and ultra-far-OTM lottos.")

have_cal = bool(put_file or call_file)
have_market = bool(earn_file or flow_file or unusual_file)
if not have_cal and not have_market:
    st.info("Upload the Barchart calendar CSVs for the income scanner, and/or the earnings, "
            "flow and unusual-activity files for the big-move watch.")
    st.stop()

# ---------------------------------------------------------------- loading
errors, infos = [], []
frames = []
for f, label in [(put_file, "Put"), (call_file, "Call")]:
    if f is not None:
        try:
            d, info = cc.load_file(f, label)
            frames.append(d)
            infos.append((f"{label} calendar", info))
        except Exception as exc:
            errors.append(f"{label} calendar: {exc}")

earn_df = None
if earn_file is not None:
    try:
        earn_df, info = cc.load_earnings_file(earn_file)
        infos.append(("Earnings calendar", info))
    except Exception as exc:
        errors.append(f"Earnings calendar: {exc}")

flow_df = unusual_df = None
if flow_file is not None:
    try:
        flow_df, info = bm.load_flow(flow_file)
        infos.append(("Options flow", info))
    except Exception as exc:
        errors.append(f"Options flow: {exc}")
if unusual_file is not None:
    try:
        unusual_df, info = bm.load_unusual(unusual_file)
        infos.append(("Unusual activity", info))
    except Exception as exc:
        errors.append(f"Unusual activity: {exc}")

for e in errors:
    st.error(e)

# Earnings dates: file first, manual entries override.
earnings: dict = {}
timing: dict = {}
if earn_df is not None:
    earnings.update(dict(zip(earn_df["Symbol"], earn_df["Earnings Date"])))
    timing.update(dict(zip(earn_df["Symbol"], earn_df["Timing"])))
manual = cc.parse_earnings(earn_text)
earnings.update(manual)
for s in manual:
    timing[s] = "unknown"

# ---------------------------------------------------------------- calendar pipeline
ranked = core = events = None
raw = None
no_date: list[str] = []
all_syms: list[str] = []
if frames:
    raw = pd.concat(frames, ignore_index=True)
    x = cc.derive(raw)
    x = cc.classify_events(x, earnings, event_skew, timing)
    x = cc.add_implied_event_move(x)
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
            f"{len(no_date)} of {len(all_syms)} symbols have no earnings date "
            f"({', '.join(no_date[:12])}{'...' if len(no_date) > 12 else ''}). For those, core rows "
            "are only screened by the IV-skew proxy, which misses events when front IV is only "
            "modestly elevated. Upload the earnings file or add dates in the sidebar before "
            "trusting A/A+ rows."
        )

# ---------------------------------------------------------------- big-move watchlist
asof = pd.NaT
for _, inf in infos:
    if pd.notna(inf.get("as_of")):
        asof = inf["as_of"]
        break

watch = pd.DataFrame()
if have_market or ranked is not None:
    watch = bm.build_watchlist(
        unusual_df, flow_df, earn_df, ranked, asof=asof,
        min_price=bm_min_price, min_voi=bm_min_voi, min_uvol=bm_min_uvol,
        uoa_dte=bm_uoa_dte, flow_dte_max=bm_flow_dte, flow_delta=bm_flow_delta,
        earn_window=bm_earn_window, min_signals=bm_min_signals,
    )

tabs = st.tabs(["Core income queue", "All core contracts", "Earnings / events",
                "Big move watch", "Diagnostics", "Scoring model"])

# ---- Core income queue
with tabs[0]:
    if core is None:
        st.info("Upload a long-put and/or long-call calendar CSV to see the income queue.")
    else:
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

# ---- All core
with tabs[1]:
    if core is None:
        st.info("Upload a calendar CSV first.")
    else:
        st.dataframe(show(core.sort_values("Score", ascending=False), CORE_COLS),
                     use_container_width=True, hide_index=True)

# ---- Earnings / events
with tabs[2]:
    if events is None:
        st.info("Upload a calendar CSV first.")
    else:
        st.subheader("Earnings / event calendars (tracked separately)")
        st.caption(
            "Structure-only shortlist: cost, centering and front-IV premium. The payoff model "
            "assumes no IV crush, so it is NOT applied here. 'Implied Event Move %' is the 1-sigma "
            "jump backed out of the front/back IV term structure; compare it with the stock's "
            "historical earnings moves before trading. Debit / EM is cost as a fraction of the "
            "front-leg implied move."
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

# ---- Big move watch
with tabs[3]:
    st.subheader("Big move watch")
    st.caption(
        "Ranks names by how likely a LARGE move is, not which way. In a test on TSLA and INTC "
        "history, front-IV inversion and option-volume spikes preceded bigger moves, while low "
        "IV rank and realized-vol compression did not; nothing predicted direction. 'Lean' is "
        "shown for context only and is not in the score. Unusual activity can be hedges, spreads "
        "or dividend capture."
    )
    if not have_market and ranked is None:
        st.info("Upload the earnings, options-flow and unusual-activity CSVs.")
    elif watch.empty:
        st.warning("Nothing passed the current big-move filters. Try lowering 'Minimum signals'.")
    else:
        mode = st.radio(
            "Catalyst",
            ["All", "Earnings in window", "No scheduled catalyst"],
            horizontal=True,
        )
        w = watch
        if mode == "Earnings in window":
            w = w[w["Catalyst"] != "None scheduled"]
        elif mode == "No scheduled catalyst":
            w = w[w["Catalyst"] == "None scheduled"]

        missing = []
        if unusual_df is None:
            missing.append("unusual activity")
        if flow_df is None:
            missing.append("options flow")
        if earn_df is None:
            missing.append("earnings calendar")
        if missing:
            st.info("Not uploaded: " + ", ".join(missing) + ". Scores use only the data you provided.")

        if w.empty:
            st.warning("No names in this view.")
        else:
            view = watch_view(w)
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download big-move watchlist",
                view.to_csv(index=False).encode("utf-8"),
                file_name="big_move_watchlist.csv", mime="text/csv",
            )

            st.markdown("**Drill down**")
            pick = st.selectbox("Symbol", list(w["Symbol"]))
            uo, fl = bm.symbol_detail(
                pick, unusual_df, flow_df, flow_dte_max=bm_flow_dte, flow_delta=bm_flow_delta)
            if uo is not None and len(uo):
                st.write("Unusual-activity contracts")
                ucols = ["Exp Date", "DTE", "Type", "Strike", "Moneyness", "Volume", "Open Int",
                         "Vol/OI", "IV", "Delta", "Premium"]
                uz = uo[[c for c in ucols if c in uo.columns]].copy()
                uz["Exp Date"] = uz["Exp Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(uz.head(40), use_container_width=True, hide_index=True)
            if fl is not None and len(fl):
                st.write("Large flow prints (Used In Score = single-leg, in your DTE and delta band)")
                fcols = ["Time", "Exp Date", "DTE", "Type", "Strike", "Size", "Side", "Premium",
                         "Delta", "Code", "Opening", "Used In Score"]
                fz = fl[[c for c in fcols if c in fl.columns]].copy()
                if "Exp Date" in fz:
                    fz["Exp Date"] = pd.to_datetime(fz["Exp Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                st.dataframe(fz.head(40), use_container_width=True, hide_index=True)

# ---- Diagnostics
with tabs[4]:
    for label, info in infos:
        asof_txt = info["as_of"].date() if pd.notna(info.get("as_of")) else "unknown"
        st.write(
            f"**{label}** - as of {asof_txt}; {info['rows']} rows loaded, "
            f"{info['dropped']} non-data row(s) dropped."
        )
    if ranked is not None:
        st.write("**Calendar symbols with no earnings date:**", ", ".join(no_date) or "none")
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

# ---- Scoring model
with tabs[5]:
    st.markdown("""
**Calendar payoff model.** For each calendar the app reprices the long leg at front expiry
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

**Events.** Earnings dates come from the uploaded earnings file (manual entries override).
Timing matters on the edges: an after-close report on the front expiry date lands after the
short leg expires, and a before-open report on the as-of date has already happened. With no
date, skew >= the sidebar threshold (or front IV >= 1.35x back IV) marks a suspected event.

---

**Big-move score (0-100, magnitude only):**
- 25% unusual activity: premium of contracts with high Vol/OI and volume, on a log scale
- 20% earnings catalyst: inside 7 days scores highest, then the chosen window
- 20% qualifying flow: large single-leg prints inside your DTE and |delta| band; multi-leg
  (ML/TL/MF), cancelled, deep-ITM and LEAPS prints are excluded
- 15% front-IV inversion: median IV skew of near-ATM front contracts (calendar files only)
- 10% implied event move from the term structure (calendar files only)
- 10% IV rank (earnings file)

Calendar-derived components only count when a symbol appears in your calendar files, so names
outside that universe are scored on what is available. 'Signals' counts how many independent
sources fired. 'Lean' (bullish/bearish/mixed) comes from aggressor-side flow premium, or
call/put share of unusual premium when flow is thin. It was no better than a coin flip in the
TSLA/INTC test.

All weights and thresholds are judgment calls and have not been backtested beyond the small
TSLA/INTC check.
""")
