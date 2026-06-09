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

            tabs = st.tabs(["Bullish", "Bearish", "Vol expansion", "Earnings convexity", "Put speculation", "Put hedging", "Downloads"])
            output_files = [
                "3pm_screener_bullish.csv",
                "3pm_screener_bearish.csv",
                "3pm_screener_vol_expansion.csv",
                "3pm_screener_earnings_convexity.csv",
                "3pm_screener_put_speculation.csv",
                "3pm_screener_put_hedging.csv",
            ]
            for tab, fname in zip(tabs[:-1], output_files):
                with tab:
                    path = os.path.join(outdir, fname)
                    if os.path.exists(path):
                        st.dataframe(pd.read_csv(path), use_container_width=True)
                    else:
                        st.warning(f"{fname} was not created.")
            with tabs[-1]:
                show_downloads(outdir)
        except Exception as exc:
            st.error(f"Could not run screener: {exc}")
else:
    st.info("Upload all 7 stock scanner files, then click Run.")
