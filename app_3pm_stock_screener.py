"""
Streamlit App 2: 3PM Stock Options Flow Screener

Upload the latest Barchart stock CSVs and generate ranked, bullish, bearish,
vol expansion, earnings convexity, put speculation, and put hedging outputs.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from three_pm_flow_screener_v3_earnings_convexity import build_screener


st.set_page_config(page_title="3PM Stock Flow Screener", layout="wide")
st.title("3PM Stock Options Flow Screener")
st.caption("Upload the latest stock Barchart CSVs and generate the 3PM scanner outputs.")

with st.sidebar:
    st.header("Settings")
    top = st.number_input("Top rows per output", min_value=5, max_value=200, value=50, step=5)
    min_premium = st.number_input("Minimum total premium filter", min_value=0.0, value=0.0, step=10000.0)
    st.link_button(
        "⚡ Convexity Screener",
        "https://convexity-bqhpmsmhg5lvruwjltk6bl.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🎯 Dealer Positioning",
        "https://dealerpositioning-n2m58uzu42afb44usojrwe.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "📊 Option Contract Analysis",
        "https://dealerpositioning-zenvhgc3fs3dcsd9yunvct.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🔬 Calendar Spread Regime Screening",
        "https://freedom-fuxffx4ohuuosfojdmffxl.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "📅 Calendar Spread Screener",
        "https://freedom-hdf89xczpjdheb6kuq2qnz.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🚀 Option Flow Screener",
        "https://freedom-4rbsvk5mg32dybsiqwxwna.streamlit.app/",
        use_container_width=True
    )

flow_file = st.file_uploader("1) options-flow-*.csv", type="csv")
unusual_file = st.file_uploader("2) unusual-stock-options-activity-*.csv", type="csv")
highest_iv_file = st.file_uploader("3) stocks-highest-implied-volatility-*.csv", type="csv")
iv_up_file = st.file_uploader("4) stocks-increase-percent-change-in-volatility-*.csv", type="csv")
iv_down_file = st.file_uploader("5) stocks-decrease-percent-change-in-volatility-*.csv", type="csv")
oi_up_file = st.file_uploader("6) stocks-increase-change-in-open-interest-*.csv", type="csv")
oi_down_file = st.file_uploader("7) stocks-decrease-change-in-open-interest-*.csv", type="csv")

uploads = {
    "options-flow-upload.csv": flow_file,
    "unusual-stock-options-activity-upload.csv": unusual_file,
    "stocks-highest-implied-volatility-upload.csv": highest_iv_file,
    "stocks-increase-percent-change-in-volatility-upload.csv": iv_up_file,
    "stocks-decrease-percent-change-in-volatility-upload.csv": iv_down_file,
    "stocks-increase-change-in-open-interest-upload.csv": oi_up_file,
    "stocks-decrease-change-in-open-interest-upload.csv": oi_down_file,
}

ready = all(v is not None for v in uploads.values())


def save_uploads(folder: str) -> None:
    for filename, file in uploads.items():
        with open(os.path.join(folder, filename), "wb") as f:
            f.write(file.getbuffer())


def show_downloads(outdir: str) -> None:
    files = [
        "3pm_screener_ranked.csv",
        "3pm_screener_bullish.csv",
        "3pm_screener_bearish.csv",
        "3pm_screener_vol_expansion.csv",
        "3pm_screener_earnings_convexity.csv",
        "3pm_screener_put_speculation.csv",
        "3pm_screener_put_hedging.csv",
    ]
    for fname in files:
        path = os.path.join(outdir, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(f"Download {fname}", f.read(), file_name=fname, mime="text/csv")


if st.button("Run 3PM stock screener", type="primary", disabled=not ready):
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)
        save_uploads(tmpdir)
        try:
            ranked = build_screener(tmpdir, outdir, top=int(top), min_premium=float(min_premium))
            st.subheader("Ranked scanner")
            st.dataframe(ranked, use_container_width=True)

            tabs = st.tabs([
                "Bullish", "Bearish", "Vol expansion", "Earnings convexity",
                "Put speculation", "Put hedging", "🔥 Research Now", "Downloads"
            ])
            output_files = [
                "3pm_screener_bullish.csv",
                "3pm_screener_bearish.csv",
                "3pm_screener_vol_expansion.csv",
                "3pm_screener_earnings_convexity.csv",
                "3pm_screener_put_speculation.csv",
                "3pm_screener_put_hedging.csv",
            ]

            for tab, fname in zip(tabs[:6], output_files):
                with tab:
                    path = os.path.join(outdir, fname)
                    if os.path.exists(path):
                        st.dataframe(pd.read_csv(path), use_container_width=True)
                    else:
                        st.warning(f"{fname} was not created.")

            with tabs[6]:
                st.subheader("🔥 Research Now")
                st.caption(
                    "High-conviction shortlist requiring independent "
                    "Flow + Unusual Activity + OI confirmation."
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    research_setup = st.number_input("Min Setup", 0.0, 100.0, 65.0, 1.0)
                with c2:
                    research_flow = st.number_input("Min Flow", 0.0, 100.0, 85.0, 1.0)
                with c3:
                    research_unusual = st.number_input("Min Unusual", 0.0, 100.0, 85.0, 1.0)
                with c4:
                    research_oi = st.number_input("Min OI Accum.", 0.0, 100.0, 65.0, 1.0)
                with c5:
                    research_iv = st.number_input("IV Expansion Flag", 0.0, 100.0, 85.0, 1.0)

                required = [
                    "setup_score", "flow_score", "unusual_score",
                    "oi_accumulation_score"
                ]
                missing = [col for col in required if col not in ranked.columns]

                if missing:
                    st.warning(
                        "Research Now cannot be calculated. Missing columns: "
                        + ", ".join(missing)
                    )
                else:
                    research = ranked.copy()

                    numeric_cols = [
                        "setup_score", "flow_score", "unusual_score",
                        "oi_accumulation_score", "iv_expansion_score",
                        "flow_direction", "unusual_direction"
                    ]
                    for col in numeric_cols:
                        if col in research.columns:
                            research[col] = pd.to_numeric(research[col], errors="coerce")

                    research = research[
                        (research["setup_score"] >= research_setup)
                        & (research["flow_score"] >= research_flow)
                        & (research["unusual_score"] >= research_unusual)
                        & (research["oi_accumulation_score"] >= research_oi)
                    ].copy()

                    if "iv_expansion_score" in research.columns:
                        research["expansion_confirmed"] = (
                            research["iv_expansion_score"] >= research_iv
                        )

                    if {"flow_direction", "unusual_direction"}.issubset(research.columns):
                        research["direction_agreement"] = (
                            (research["flow_direction"] == research["unusual_direction"])
                            & research["flow_direction"].isin([-1, 1])
                        )

                    def classify_research(row):
                        expansion = row.get("iv_expansion_score", 0) >= research_iv
                        agreement = bool(row.get("direction_agreement", False))
                        bias = str(row.get("bias", "")).lower()

                        if expansion and agreement:
                            if bias == "bullish":
                                return "🔥 Bullish Expansion"
                            if bias == "bearish":
                                return "🔥 Bearish Expansion"
                            return "🔥 Expansion"
                        if agreement:
                            if bias == "bullish":
                                return "🟢 Bullish Flow"
                            if bias == "bearish":
                                return "🔴 Bearish Flow"
                            return "Directional Flow"
                        if expansion:
                            return "⚡ Vol Expansion"
                        return "👀 Watch"

                    research["research_type"] = research.apply(classify_research, axis=1)

                    sort_cols = [
                        col for col in [
                            "setup_score", "flow_score", "unusual_score",
                            "oi_accumulation_score", "iv_expansion_score"
                        ] if col in research.columns
                    ]
                    research = research.sort_values(
                        sort_cols, ascending=[False] * len(sort_cols)
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Research Candidates", len(research))

                    if "expansion_confirmed" in research.columns:
                        m2.metric(
                            "Expansion Confirmed",
                            int(research["expansion_confirmed"].sum())
                        )
                    if "direction_agreement" in research.columns:
                        m3.metric(
                            "Directional Agreement",
                            int(research["direction_agreement"].sum())
                        )
                    if "bias" in research.columns:
                        m4.metric(
                            "Bearish",
                            int(
                                (research["bias"].astype(str).str.lower() == "bearish").sum()
                            )
                        )

                    preferred_columns = [
                        "rank", "Symbol", "bias", "research_type",
                        "setup_score", "flow_score", "unusual_score",
                        "oi_accumulation_score", "iv_expansion_score",
                        "expansion_confirmed", "direction_agreement",
                        "preferred_structure", "confirmation",
                        "total_premium", "net_signed_premium"
                    ]
                    display_columns = [
                        col for col in preferred_columns if col in research.columns
                    ]
                    remaining = [
                        col for col in research.columns if col not in display_columns
                    ]

                    st.dataframe(
                        research[display_columns + remaining],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "⬇️ Download Research Now",
                        research.to_csv(index=False).encode("utf-8"),
                        file_name="3pm_screener_research_now.csv",
                        mime="text/csv"
                    )

            with tabs[7]:
                show_downloads(outdir)
        except Exception as exc:
            st.error(f"Could not run screener: {exc}")
else:
    st.info("Upload all 7 stock scanner files, then click Run.")
