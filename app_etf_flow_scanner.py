"""
Streamlit App 3: ETF Flow + OI + IV Scanner

Upload the latest ETF Barchart CSVs and generate ranked, bullish, bearish,
vol expansion, and range-candidate outputs.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from etf_flow_scanner import build_scanner


st.set_page_config(page_title="ETF Flow Scanner", layout="wide")
st.title("ETF Flow + OI + IV Scanner")
st.caption("Upload the latest ETF Barchart CSVs and generate ranked ETF setup outputs.")

flow_file = st.file_uploader("1) options-flow-*.csv", type="csv")
unusual_file = st.file_uploader("2) unusual-etf-options-activity-*.csv", type="csv")
high_iv_file = st.file_uploader("3) etfs-highest-implied-volatility-*.csv", type="csv")
iv_inc_file = st.file_uploader("4) etfs-increase-percent-change-in-volatility-*.csv", type="csv")
iv_dec_file = st.file_uploader("5) etfs-decrease-percent-change-in-volatility-*.csv", type="csv")
oi_inc_file = st.file_uploader("6) etfs-increase-change-in-open-interest-*.csv", type="csv")
oi_dec_file = st.file_uploader("7) etfs-decrease-change-in-open-interest-*.csv", type="csv")

uploads = {
    "options-flow-upload.csv": flow_file,
    "unusual-etf-options-activity-upload.csv": unusual_file,
    "etfs-highest-implied-volatility-upload.csv": high_iv_file,
    "etfs-increase-percent-change-in-volatility-upload.csv": iv_inc_file,
    "etfs-decrease-percent-change-in-volatility-upload.csv": iv_dec_file,
    "etfs-increase-change-in-open-interest-upload.csv": oi_inc_file,
    "etfs-decrease-change-in-open-interest-upload.csv": oi_dec_file,
}

ready = all(v is not None for v in uploads.values())


def save_uploads(folder: str) -> None:
    for filename, file in uploads.items():
        with open(os.path.join(folder, filename), "wb") as f:
            f.write(file.getbuffer())


def show_downloads(outdir: str) -> None:
    files = [
        "etf_scanner_ranked.csv",
        "etf_scanner_bullish.csv",
        "etf_scanner_bearish.csv",
        "etf_scanner_vol_expansion.csv",
        "etf_scanner_range_candidates.csv",
        "etf_scanner_summary.txt",
    ]
    for fname in files:
        path = os.path.join(outdir, fname)
        if os.path.exists(path):
            mime = "text/csv" if fname.endswith(".csv") else "text/plain"
            with open(path, "rb") as f:
                st.download_button(f"Download {fname}", f.read(), file_name=fname, mime=mime)


if st.button("Run ETF scanner", type="primary", disabled=not ready):
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)
        save_uploads(tmpdir)
        try:
            ranked = build_scanner(tmpdir, outdir)
            st.subheader("Ranked ETF scanner")
            st.dataframe(ranked, use_container_width=True)

            tabs = st.tabs(["Bullish", "Bearish", "Vol expansion", "Range candidates", "Downloads"])
            output_files = [
                "etf_scanner_bullish.csv",
                "etf_scanner_bearish.csv",
                "etf_scanner_vol_expansion.csv",
                "etf_scanner_range_candidates.csv",
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
            import traceback

            st.error(f"Could not run ETF scanner: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())
else:
    st.info("Upload all 7 ETF scanner files, then click Run.")
