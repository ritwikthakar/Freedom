"""
Streamlit App 1: Single-Ticker Gamma + Flow + GEX/DEX Trade Plan

Upload the latest Barchart/OptionCharts CSVs, enter a ticker, and generate
a trade plan without using Spyder/local command line.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import asdict

import pandas as pd
import streamlit as st

from gamma_flow_trade_plan_gex_dex import build_trade_plan, load_inputs


st.set_page_config(page_title="Gamma + Flow Trade Plan", layout="wide")
st.title("Gamma + Flow Trade Plan")
st.caption("Upload expected move, flow, unusual activity, GEX history, DEX history, net GEX, and net DEX files.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="PLTR").upper().strip()
    st.markdown("Upload fresh CSVs from Barchart / OptionCharts.")

expected_file = st.file_uploader("1) Expected move CSV", type="csv")
flow_file = st.file_uploader("2) Options flow CSV", type="csv")
unusual_file = st.file_uploader("3) Unusual options activity CSV", type="csv")
gex_hist_file = st.file_uploader("4) GEX history CSV", type="csv")
dex_hist_file = st.file_uploader("5) DEX history CSV", type="csv")
net_gex_file = st.file_uploader("6) Net GEX by expiration CSV", type="csv")
net_dex_file = st.file_uploader("7) Net DEX by expiration CSV", type="csv")

uploaded = {
    "expected": expected_file,
    "flow": flow_file,
    "unusual": unusual_file,
    "gex_history": gex_hist_file,
    "dex_history": dex_hist_file,
    "net_gex": net_gex_file,
    "net_dex": net_dex_file,
}

all_ready = ticker and all(v is not None for v in uploaded.values())


def save_uploaded(folder: str, ticker: str, files: dict) -> None:
    """Save uploaded files with names that match the original script's glob patterns."""
    t = ticker.upper()
    lower = ticker.lower()
    name_map = {
        "expected": f"{lower}-expected-move-upload.csv",
        "flow": f"{lower}-options-flow-upload.csv",
        "unusual": "unusual-options-activity-upload.csv",
        "gex_history": f"{t}_gex_history_upload.csv",
        "dex_history": f"{t}_dex_history_upload.csv",
        "net_gex": f"net_gex_exposure_by_expiration_{t}_upload.csv",
        "net_dex": f"net_dex_exposure_by_expiration_{t}_upload.csv",
    }
    for key, file in files.items():
        out_path = os.path.join(folder, name_map[key])
        with open(out_path, "wb") as f:
            f.write(file.getbuffer())


if st.button("Generate trade plan", type="primary", disabled=not all_ready):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_uploaded(tmpdir, ticker, uploaded)
        try:
            data, files_used = load_inputs(tmpdir, ticker)
            plan = build_trade_plan(ticker, data)
            row = asdict(plan)
            df = pd.DataFrame([row])

            st.subheader(f"Trade Plan: {ticker}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bias", row.get("bias"))
            c2.metric("Confidence", row.get("confidence"))
            c3.metric("Gamma regime", row.get("gamma_regime"))
            c4.metric("DEX regime", row.get("dex_regime"))

            st.dataframe(df, use_container_width=True)

            st.subheader("Key levels")
            level_cols = [
                "spot", "entry_trigger", "stop_loss", "profit_target_1", "profit_target_2",
                "nearest_gamma_flip", "nearest_call_wall", "nearest_put_wall",
                "expected_move_upper", "expected_move_lower",
            ]
            st.dataframe(df[[c for c in level_cols if c in df.columns]], use_container_width=True)

            st.subheader("Files used")
            st.json({k: os.path.basename(v) for k, v in files_used.items()})

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            txt = "\n".join([f"{k}: {v}" for k, v in row.items()])
            st.download_button("Download CSV", csv_bytes, file_name=f"trade_plan_{ticker}.csv", mime="text/csv")
            st.download_button("Download TXT", txt.encode("utf-8"), file_name=f"trade_plan_{ticker}.txt", mime="text/plain")
        except Exception as exc:
            st.error(f"Could not generate trade plan: {exc}")
else:
    st.info("Upload all 7 files, then click Generate trade plan.")
