"""Big-move watchlist built from Barchart exports.

Inputs (all optional, any combination):
  * Unusual stock options activity CSV  - contracts with volume >> open interest
  * Options flow CSV                    - large individual prints
  * Earnings calendar CSV               - date, timing, IV rank
  * Calendar-scanner frame              - front/back IV term structure

What this can and cannot do (from testing on TSLA/INTC history):
  * It ranks names by how likely a LARGE move is (magnitude).
  * Direction is shown as a low-reliability "lean" and is not part of the score.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

import calendar_core as cc

# OPRA-style trade condition codes seen in Barchart's flow export. Multi-leg
# codes (ML*, TL*, MF*) mean the print is one leg of a spread, so its side says
# little about direction. Mapping is inferred from the OPRA convention and is
# editable here.
NON_DIRECTIONAL_PREFIXES = ("ML", "TL", "MF")
CANCEL_CODES = {"CANC", "CNCL", "CNCO"}
OPEN_FLAGS = {"ToOpen", "BuyToOpen", "SellToOpen"}

WEIGHTS = {"catalyst": 20, "unusual": 25, "flow": 20, "term": 15, "move": 10, "ivrank": 10}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False).str.replace("+", "", regex=False).str.strip(),
        errors="coerce",
    )


def lin(v, lo, hi):
    return ((v - lo) / (hi - lo) * 100).clip(0, 100)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_unusual(upload) -> tuple[pd.DataFrame, dict]:
    text = cc.read_text(upload)
    d = pd.read_csv(io.StringIO(text))
    need = {"Symbol", "Price~", "Exp Date", "DTE", "Type", "Volume", "Open Int", "Vol/OI",
            "Bid", "Ask", "Latest", "Delta"}
    if need - set(d.columns):
        raise ValueError("Unusual activity file missing: " + ", ".join(sorted(need - set(d.columns))))
    n0 = len(d)
    for c in ["Price~", "DTE", "Volume", "Open Int", "Vol/OI", "Bid", "Ask", "Latest", "Delta"]:
        d[c] = _num(d[c])
    d["Exp Date"] = pd.to_datetime(d["Exp Date"], errors="coerce")
    d = d[d["Price~"].notna() & d["Exp Date"].notna() & d["Symbol"].notna()].copy()
    d["Symbol"] = d["Symbol"].astype("string").str.upper().str.strip()
    d["Type"] = d["Type"].astype("string").str.title()
    mid = (d["Bid"] + d["Ask"]) / 2
    px = np.where((d["Bid"] > 0) & (d["Ask"] > 0), mid, d["Latest"])
    d["Premium"] = d["Volume"] * px * 100.0
    # 'Imp Vol' looks like the underlying's overall IV; 'IV' the contract's.
    for c in ("Imp Vol", "IV"):
        if c in d:
            d[c] = _num(d[c])
    return d.reset_index(drop=True), {
        "as_of": cc.footer_date(text), "rows": len(d), "dropped": n0 - len(d)}


def load_flow(upload) -> tuple[pd.DataFrame, dict]:
    text = cc.read_text(upload)
    d = pd.read_csv(io.StringIO(text), keep_default_na=False, na_values=[""])
    need = {"Symbol", "Price~", "Type", "DTE", "Side", "Premium", "Size", "Delta", "Code"}
    if need - set(d.columns):
        raise ValueError("Options flow file missing: " + ", ".join(sorted(need - set(d.columns))))
    n0 = len(d)
    for c in ["Price~", "DTE", "Premium", "Size", "Delta", "Trade", "Volume", "Open Int"]:
        if c in d:
            d[c] = _num(d[c])
    d = d[d["Price~"].notna() & d["Premium"].notna() & d["Symbol"].notna()].copy()
    d["Symbol"] = d["Symbol"].astype("string").str.upper().str.strip()
    d["Type"] = d["Type"].astype("string").str.title()
    d["Side"] = d["Side"].astype(str).str.lower().str.strip()
    d["Code"] = d["Code"].astype(str).str.upper().str.strip()
    flag = d["*"].astype(str).str.strip() if "*" in d else pd.Series("", index=d.index)
    d["Opening"] = flag.isin(OPEN_FLAGS)
    return d.reset_index(drop=True), {
        "as_of": cc.footer_date(text), "rows": len(d), "dropped": n0 - len(d)}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate_unusual(u: pd.DataFrame, min_voi: float, min_vol: float,
                      dte_lo: int, dte_hi: int) -> pd.DataFrame:
    q = u[(u["Vol/OI"] >= min_voi) & (u["Volume"] >= min_vol) & u["DTE"].between(dte_lo, dte_hi)]
    if q.empty:
        return pd.DataFrame(columns=["Symbol"])
    q = q.assign(
        call_prem=np.where(q["Type"] == "Call", q["Premium"], 0.0),
        put_prem=np.where(q["Type"] == "Put", q["Premium"], 0.0),
    )
    g = q.groupby("Symbol").agg(
        UOA_Contracts=("Premium", "size"),
        UOA_Premium=("Premium", "sum"),
        UOA_Call=("call_prem", "sum"),
        UOA_Put=("put_prem", "sum"),
        UOA_Max_VolOI=("Vol/OI", "max"),
        UOA_Med_DTE=("DTE", "median"),
        Price_u=("Price~", "last"),
    ).reset_index()
    tot = (g["UOA_Call"] + g["UOA_Put"]).replace(0, np.nan)
    g["UOA_Lean"] = (g["UOA_Call"] - g["UOA_Put"]) / tot
    return g


def aggregate_flow(f: pd.DataFrame, dte_max: int, delta_lo: float, delta_hi: float) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame(columns=["Symbol"])
    multi = f["Code"].str.startswith(NON_DIRECTIONAL_PREFIXES) | f["Code"].isin(CANCEL_CODES)
    near = f["DTE"] <= dte_max
    band = f["Delta"].abs().between(delta_lo, delta_hi)
    ok = ~multi & near & band
    f = f.assign(eligible=ok, multi=multi,
                 bull=((f["Type"] == "Call") & (f["Side"] == "ask")) | ((f["Type"] == "Put") & (f["Side"] == "bid")),
                 bear=((f["Type"] == "Put") & (f["Side"] == "ask")) | ((f["Type"] == "Call") & (f["Side"] == "bid")))
    e = f[f["eligible"]]
    g = e.assign(
        bull_p=np.where(e["bull"], e["Premium"], 0.0),
        bear_p=np.where(e["bear"], e["Premium"], 0.0),
        open_p=np.where(e["Opening"], e["Premium"], 0.0),
    ).groupby("Symbol").agg(
        Flow_Prints=("Premium", "size"),
        Flow_Premium=("Premium", "sum"),
        Flow_Bull=("bull_p", "sum"),
        Flow_Bear=("bear_p", "sum"),
        Flow_Open=("open_p", "sum"),
        Flow_Largest=("Premium", "max"),
    ).reset_index()
    dirp = (g["Flow_Bull"] + g["Flow_Bear"]).replace(0, np.nan)
    g["Flow_Lean"] = (g["Flow_Bull"] - g["Flow_Bear"]) / dirp
    g["Flow_Open_Share"] = g["Flow_Open"] / g["Flow_Premium"].replace(0, np.nan)

    allp = f.groupby("Symbol")["Premium"].sum().rename("Flow_All_Premium")
    g = g.merge(allp.reset_index(), on="Symbol", how="outer")
    g["Flow_Excluded_Share"] = 1 - g["Flow_Premium"].fillna(0) / g["Flow_All_Premium"].replace(0, np.nan)
    return g


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------
def build_watchlist(unusual=None, flow=None, earnings=None, cal=None, *,
                    asof=None, min_price=5.0, min_voi=3.0, min_uvol=500,
                    uoa_dte=(1, 60), flow_dte_max=60, flow_delta=(0.05, 0.75),
                    earn_window=14, min_signals=2) -> pd.DataFrame:
    asof = pd.Timestamp(asof).normalize() if asof is not None and pd.notna(asof) else pd.Timestamp.now().normalize()

    parts = []
    u = aggregate_unusual(unusual, min_voi, min_uvol, *uoa_dte) if unusual is not None else None
    f = aggregate_flow(flow, flow_dte_max, *flow_delta) if flow is not None else None
    if u is not None and len(u):
        parts.append(u[["Symbol"]])
    if f is not None and len(f):
        parts.append(f[["Symbol"]])

    term = None
    if cal is not None and len(cal):
        c = cal[(cal["Short DTE"] <= 21) & (cal["Dist (EM)"] <= 0.5)]
        term = c.groupby("Symbol").agg(
            Term_Skew=("IV Skew", "median"),
            Event_Move=("Implied Event Move %", "median"),
        ).reset_index()
        # Calendar names join the universe only when they carry a term-structure signal.
        strong = term[term["Term_Skew"] >= 3.0][["Symbol"]]
        if len(strong):
            parts.append(strong)
    if not parts:
        return pd.DataFrame()

    w = pd.concat(parts).drop_duplicates("Symbol").reset_index(drop=True)
    if u is not None and len(u):
        w = w.merge(u, on="Symbol", how="left")
    if f is not None and len(f):
        w = w.merge(f, on="Symbol", how="left")
    if term is not None:
        w = w.merge(term, on="Symbol", how="left")
    if earnings is not None and len(earnings):
        w = w.merge(earnings.rename(columns={"IV Rank": "IV_Rank", "Price": "Price_e"}), on="Symbol", how="left")

    for col in ["UOA_Contracts", "UOA_Premium", "UOA_Call", "UOA_Put", "Flow_Prints",
                "Flow_Premium", "Flow_Bull", "Flow_Bear", "Flow_Open", "Flow_Largest"]:
        if col not in w:
            w[col] = 0.0
        w[col] = w[col].fillna(0.0)
    for col in ["Price_u", "Price_e", "IV_Rank", "Term_Skew", "Event_Move", "UOA_Lean",
                "Flow_Lean", "Flow_Open_Share", "Flow_Excluded_Share", "Barchart Exp Move %",
                "UOA_Med_DTE", "UOA_Max_VolOI"]:
        if col not in w:
            w[col] = np.nan
    if "Earnings Date" not in w:
        w["Earnings Date"] = pd.NaT
        w["Timing"] = "unknown"
    w["Price"] = w["Price_u"].fillna(w["Price_e"])
    w = w[w["Price"].fillna(0) >= min_price].copy()

    # Catalyst: earnings still ahead, within the window.
    w["Days To Earnings"] = (w["Earnings Date"] - asof).dt.days
    d = w["Days To Earnings"]
    done = (d < 0) | ((d == 0) & (w["Timing"] == "BMO"))
    live = d.notna() & ~done & (d <= earn_window)
    w["Catalyst"] = np.where(live, np.where(d <= 7, "Earnings <=7d", f"Earnings <={earn_window}d"), "None scheduled")

    s = pd.DataFrame(index=w.index)
    s["catalyst"] = np.where(live, np.where(d <= 7, 100.0, 60.0), 0.0)
    s["unusual"] = np.where(w["UOA_Premium"] > 0,
                            lin(np.log10(w["UOA_Premium"].clip(lower=1)), np.log10(2.5e5), np.log10(2.5e7)), 0.0)
    s["flow"] = np.where(w["Flow_Premium"] > 0,
                         lin(np.log10(w["Flow_Premium"].clip(lower=1)), np.log10(9e5), np.log10(5e7)), 0.0)
    s["term"] = lin(w["Term_Skew"], 2.0, 12.0)
    s["move"] = lin(w["Event_Move"], 3.0, 12.0)
    s["ivrank"] = lin(w["IV_Rank"], 20.0, 60.0)

    num = pd.Series(0.0, index=w.index)
    den = pd.Series(0.0, index=w.index)
    for k, wt in WEIGHTS.items():
        avail = s[k].notna()
        num += np.where(avail, s[k].fillna(0) * wt, 0.0)
        den += np.where(avail, wt, 0.0)
    w["Score"] = (num / den.replace(0, np.nan)).round(1)
    w["Signals"] = ((s["catalyst"] > 0).astype(int) + (s["unusual"] > 0).astype(int)
                    + (s["flow"] > 0).astype(int) + (s["term"].fillna(0) >= 25).astype(int))
    w = w[w["Signals"] >= min_signals].copy()

    # Direction is informational only.
    big_flow = (w["Flow_Bull"] + w["Flow_Bear"]) >= 2e6
    w["Lean"] = np.where(big_flow, w["Flow_Lean"], w["UOA_Lean"])
    w["Lean Label"] = np.select(
        [w["Lean"] > 0.25, w["Lean"] < -0.25, w["Lean"].notna()],
        ["Bullish lean", "Bearish lean", "Mixed"], default="n/a")

    notes = []
    for _, r in w.iterrows():
        n = []
        if pd.notna(r["Flow_Excluded_Share"]) and r["Flow_Excluded_Share"] > 0.6:
            n.append("most flow premium was spreads / LEAPS / deep ITM (excluded)")
        if pd.notna(r["Term_Skew"]) and r["Term_Skew"] >= 8 and r["Catalyst"] == "None scheduled":
            n.append("high front IV with no earnings date: other event?")
        if pd.notna(r["UOA_Med_DTE"]) and r["UOA_Med_DTE"] <= 3:
            n.append("very short-dated, could be expiry positioning")
        notes.append("; ".join(n))
    w["Notes"] = notes
    return w.sort_values(["Score", "Signals"], ascending=False).reset_index(drop=True)


def symbol_detail(sym: str, unusual=None, flow=None, flow_dte_max=60, flow_delta=(0.05, 0.75)):
    """Raw contracts and prints for one symbol, for drill-down."""
    uo = unusual[unusual["Symbol"] == sym].sort_values("Premium", ascending=False) if unusual is not None else None
    fl = None
    if flow is not None:
        fl = flow[flow["Symbol"] == sym].copy()
        multi = fl["Code"].str.startswith(NON_DIRECTIONAL_PREFIXES) | fl["Code"].isin(CANCEL_CODES)
        fl["Used In Score"] = ~multi & (fl["DTE"] <= flow_dte_max) & fl["Delta"].abs().between(*flow_delta)
        fl = fl.sort_values("Premium", ascending=False)
    return uo, fl
