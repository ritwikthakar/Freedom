# Streamlit Trading Apps

This folder contains three Streamlit apps:

1. `app_gamma_trade_plan.py`  
   Single-ticker trade plan using expected move, options flow, unusual activity, GEX history, DEX history, net GEX, and net DEX.

2. `app_3pm_stock_screener.py`  
   Stock 3PM options flow scanner using Barchart stock CSV exports.

3. `app_etf_flow_scanner.py`  
   ETF flow, OI, IV scanner using Barchart ETF CSV exports.

## Local run

```bash
pip install -r requirements.txt
streamlit run app_gamma_trade_plan.py
streamlit run app_3pm_stock_screener.py
streamlit run app_etf_flow_scanner.py
```

## Streamlit Cloud setup

1. Create a GitHub repo.
2. Upload all files in this folder to the repo.
3. Go to Streamlit Cloud.
4. Create a new app from your GitHub repo.
5. Pick one app file as the entry point, for example:
   - `app_gamma_trade_plan.py`
   - `app_3pm_stock_screener.py`
   - `app_etf_flow_scanner.py`

For separate apps, create three Streamlit Cloud apps pointing to the same repo but different entry-point files.

## Daily workflow

1. Download fresh CSVs from Barchart / OptionCharts.
2. Open the Streamlit app URL.
3. Upload the fresh CSVs.
4. Click Run.
5. Download the output CSV/TXT files.
