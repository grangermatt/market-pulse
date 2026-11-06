#!/usr/bin/env python3
"""
Market Pulse — data engine
================================================================
Builds one daily data.json that powers every tab of the dashboard.

Pipeline (all free):
  1. Universe   -> NASDAQ stock screener API (symbol, name, marketCap,
                   country, sector, industry in one call).
                   Filter: US common stock, market cap >= $800M, no ADRs.
                   Cached to universe.json; rebuilt if older than 7 days.
  2. Prices     -> Yahoo Finance via yfinance, ~400 trading days daily OHLCV,
                   pulled in chunks.
  3. Metrics    -> per ticker: 1D/1W/1M/3M/6M/YTD/12M returns, ADR%, $Vol,
                   52-week high/low, new-high / new-low flags, 5-day spark.
  4. Grouping   -> mean performance by industry (Themes) and sector (Groups),
                   plus fixed ETF lists (S&P / EqWt / Country), new highs &
                   lows by industry, and a top-gainers tape.
  5. Output     -> data/data.json

Network libraries (requests, yfinance) are imported lazily inside the
functions that need them, so the pure compute functions can be unit-tested
without them (see selftest.py).
"""

import os
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import industry as IND

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MIN_MARKET_CAP   = 800_000_000      # $800M floor, like Ariel's universe
UNIVERSE_CACHE   = "universe.json"
UNIVERSE_MAX_AGE = 7                 # days before the universe list is rebuilt
PRICE_PERIOD     = "400d"            # enough history for 252-day windows
CHUNK            = 150               # tickers per Yahoo download call
OUT_PATH         = "data/data.json"
HOLDINGS_CACHE   = "etf_holdings.json"   # ETF top-holdings, refreshed weekly

# Manual industry corrections for names the data source misclassifies.
# (NASDAQ files solar-panel makers under "Semiconductors" — reclassify to Solar.)
INDUSTRY_OVERRIDE = {
    "FSLR": "Solar", "ENPH": "Solar", "SEDG": "Solar", "RUN": "Solar", "NOVA": "Solar",
    "ARRY": "Solar", "SHLS": "Solar", "CSIQ": "Solar", "MAXN": "Solar", "JKS": "Solar",
    "DQ": "Solar", "NXT": "Solar", "SPWR": "Solar", "SOL": "Solar", "FTCI": "Solar",
}

# Trailing-session lookbacks for each return column
TIMEFRAMES = {"d1": 1, "w1": 5, "m1": 21, "m3": 63, "m6": 126, "m12": 252}

# Fixed ETF tabs (these are just tickers; same metric math as stocks)
ETF_SP = ["IVV", "IVW", "IVE", "IJH", "IJK", "IJJ", "IJR", "IJT", "IJS",
          "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
ETF_EQWT = ["RSP", "RSPG", "RSPT", "RSPF", "RSPN", "RSPD", "RSPU", "RSPM",
            "RSPH", "RSPR", "RSPS", "RSPC"]
ETF_COUNTRY = ["EWJ", "EWY", "INDA", "MCHI", "FXI", "EWH", "EWU", "EWQ", "EWG",
               "EWZ", "EWT", "EWA", "EWC", "EWL", "EWP", "EWS", "TUR", "EWW",
               "EWI", "EWN", "EWD", "EWO", "EPHE", "THD", "VNM", "EIDO"]

# Group / thematic / industry ETFs (the "Group ETFs" tab)
ETF_GROUP = [
    "WGMI", "BITQ", "BLOK",
    "USO", "XOP", "OIH", "FCG", "UNG", "URA", "URNM", "NLR", "IXC",
    "TAN", "ICLN", "PBW", "LIT", "GRID",
    "SMH", "SOXX", "XSD", "FOTO", "DRAM", "IGV", "WCLD", "CLOU", "SKYY", "FDN",
    "AIQ", "BOTZ", "ROBO", "QTUM",
    "CIBR", "HACK", "FINX", "ARKF",
    "XBI", "IBB", "ARKG", "IHI", "PPH",
    "ITB", "XHB", "PAVE", "XME", "COPX", "GDX", "GDXJ", "SIL", "WOOD", "MOO",
    "PHO", "REMX",
    "KRE", "KBE", "IAI",
    "JETS", "IYT", "ITA", "XAR",
    "XRT", "IBUY", "ESPO", "HERO", "MSOS",
    "ARKK", "IPO", "FFTY",
    "KWEB", "CQQQ",
]

# Human-readable label for each ETF (shown in the Name / Country column)
ETF_NAMES = {
    # S&P style + sector SPDRs
    "IVV": "iShares Core S&P 500", "IVW": "S&P 500 Growth", "IVE": "S&P 500 Value",
    "IJH": "S&P MidCap 400", "IJK": "MidCap 400 Growth", "IJJ": "MidCap 400 Value",
    "IJR": "S&P SmallCap 600", "IJT": "SmallCap 600 Growth", "IJS": "SmallCap 600 Value",
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    # Equal-weight (Invesco)
    "RSP": "S&P 500 Equal Weight", "RSPG": "Energy EW", "RSPT": "Technology EW",
    "RSPF": "Financials EW", "RSPN": "Industrials EW", "RSPD": "Consumer Disc. EW",
    "RSPU": "Utilities EW", "RSPM": "Materials EW", "RSPH": "Health Care EW",
    "RSPR": "Real Estate EW", "RSPS": "Consumer Staples EW", "RSPC": "Comm. Services EW",
    # Country
    "EWJ": "Japan", "EWY": "South Korea", "INDA": "India", "MCHI": "China",
    "FXI": "China Large-Cap", "EWH": "Hong Kong", "EWU": "United Kingdom",
    "EWQ": "France", "EWG": "Germany", "EWZ": "Brazil", "EWT": "Taiwan",
    "EWA": "Australia", "EWC": "Canada", "EWL": "Switzerland", "EWP": "Spain",
    "EWS": "Singapore", "TUR": "Turkey", "EWW": "Mexico", "EWI": "Italy",
    "EWN": "Netherlands", "EWD": "Sweden", "EWO": "Austria", "EPHE": "Philippines",
    "THD": "Thailand", "VNM": "Vietnam", "EIDO": "Indonesia",
}
ETF_NAMES.update({
    "WGMI": "CoinShares Bitcoin Mining", "BITQ": "Bitwise Crypto Industry", "BLOK": "Amplify Blockchain",
    "USO": "US Oil Fund", "XOP": "SPDR Oil & Gas E&P", "OIH": "VanEck Oil Services",
    "FCG": "First Trust Natural Gas", "UNG": "US Natural Gas Fund", "URA": "Global X Uranium",
    "URNM": "Sprott Uranium Miners", "NLR": "VanEck Uranium & Nuclear", "IXC": "iShares Global Energy",
    "TAN": "Invesco Solar", "ICLN": "iShares Clean Energy", "PBW": "Invesco WilderHill Clean",
    "LIT": "Global X Lithium & Battery", "GRID": "First Trust Clean Energy Infra",
    "SMH": "VanEck Semiconductor", "SOXX": "iShares Semiconductor", "XSD": "SPDR Semiconductor",
    "FOTO": "Tuttle Photonics", "DRAM": "Roundhill Memory", "IGV": "iShares Expanded Tech-Software",
    "WCLD": "WisdomTree Cloud", "CLOU": "Global X Cloud", "SKYY": "First Trust Cloud",
    "FDN": "First Trust Dow Internet", "AIQ": "Global X Artificial Intelligence",
    "BOTZ": "Global X Robotics & AI", "ROBO": "ROBO Global Robotics", "QTUM": "Defiance Quantum",
    "CIBR": "First Trust Cybersecurity", "HACK": "Amplify Cybersecurity", "FINX": "Global X FinTech",
    "ARKF": "ARK Fintech Innovation", "XBI": "SPDR Biotech", "IBB": "iShares Biotechnology",
    "ARKG": "ARK Genomic Revolution", "IHI": "iShares US Medical Devices", "PPH": "VanEck Pharmaceutical",
    "ITB": "iShares US Home Construction", "XHB": "SPDR Homebuilders", "PAVE": "Global X US Infrastructure",
    "XME": "SPDR Metals & Mining", "COPX": "Global X Copper Miners", "GDX": "VanEck Gold Miners",
    "GDXJ": "VanEck Junior Gold Miners", "SIL": "Global X Silver Miners", "WOOD": "iShares Timber & Forestry",
    "MOO": "VanEck Agribusiness", "PHO": "Invesco Water Resources", "REMX": "VanEck Rare Earth",
    "KRE": "SPDR Regional Banking", "KBE": "SPDR Bank", "IAI": "iShares US Broker-Dealers",
    "JETS": "US Global Jets", "IYT": "iShares Transportation", "ITA": "iShares Aerospace & Defense",
    "XAR": "SPDR Aerospace & Defense", "XRT": "SPDR Retail", "IBUY": "Amplify Online Retail",
    "ESPO": "VanEck Video Gaming & eSports", "HERO": "Global X Video Games", "MSOS": "AdvisorShares US Cannabis",
    "ARKK": "ARK Innovation", "IPO": "Renaissance IPO", "FFTY": "Innovator IBD 50",
    "KWEB": "KraneShares China Internet", "CQQQ": "Invesco China Technology",
})

NASDAQ_URL = ("https://api.nasdaq.com/api/screener/stocks"
              "?tableonly=true&limit=10000&download=true")
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# ======================================================================
# PURE COMPUTE  (no network — unit tested in selftest.py)
# ======================================================================
def _pct_return(closes, n):
    """% change between the last close and the close n sessions earlier."""
    if closes is None or len(closes) <= n:
        return None
    prev = closes[-1 - n]
    if prev is None or prev == 0 or np.isnan(prev):
        return None
    return round((closes[-1] / prev - 1) * 100, 2)


def _ytd_return(closes, dates):
    """% change from the first close of the current calendar year."""
    if len(closes) == 0:
        return None
    year = dates[-1].year
    base = None
    for c, d in zip(closes, dates):
        if d.year == year:
            base = c
            break
    if base is None or base == 0 or np.isnan(base):
        return None
    return round((closes[-1] / base - 1) * 100, 2)


def adr_pct(highs, lows, n=20):
    """Qullamaggie ADR%: 100 * (mean(high/low over last n days) - 1)."""
    h = np.asarray(highs[-n:], dtype=float)
    l = np.asarray(lows[-n:], dtype=float)
    if len(h) == 0:
        return None
    ratio = h / np.where(l == 0, np.nan, l)
    m = np.nanmean(ratio)
    if np.isnan(m):
        return None
    return round((m - 1) * 100, 2)


def ticker_metrics(df):
    """
    Compute every metric for one ticker.
    df: DataFrame indexed by date (ascending) with Open/High/Low/Close/Volume.
    Returns a dict, or None if there isn't enough data.
    """
    if df is None or len(df) < 2:
        return None
    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return None

    closes = df["Close"].to_numpy(dtype=float)
    highs  = df["High"].to_numpy(dtype=float)
    lows   = df["Low"].to_numpy(dtype=float)
    vols   = df["Volume"].to_numpy(dtype=float)
    dates  = list(df.index)

    out = {tf: _pct_return(closes, n) for tf, n in TIMEFRAMES.items()}
    out["ytd"] = _ytd_return(closes, dates)
    out["price"] = round(float(closes[-1]), 2)
    out["dollar_vol"] = float(closes[-1] * vols[-1])
    out["adr"] = adr_pct(highs, lows, 20)

    look = min(252, len(highs))
    hi52 = float(np.nanmax(highs[-look:]))
    lo52 = float(np.nanmin(lows[-look:]))
    out["hi52"] = round(hi52, 2)
    out["lo52"] = round(lo52, 2)
    out["pct_from_hi"] = round((closes[-1] / hi52 - 1) * 100, 2) if hi52 else None
    out["pct_from_lo"] = round((closes[-1] / lo52 - 1) * 100, 2) if lo52 else None

    # A genuine new 52-week high/low MADE TODAY (needs >=100 sessions of history)
    enough = look >= 100
    out["new_high"] = bool(enough and highs[-1] >= np.nanmax(highs[-look:]))
    out["new_low"]  = bool(enough and lows[-1]  <= np.nanmin(lows[-look:]))

    # 5-session spark as % from the start of the window
    if len(closes) >= 6 and closes[-6] != 0:
        out["spark"] = [round(x, 2) for x in (closes[-6:] / closes[-6] - 1) * 100]
    else:
        out["spark"] = []
    return out


def aggregate(metrics_by_ticker, key_of, min_members=3):
    """
    Group tickers and average each timeframe.
    metrics_by_ticker: {symbol: metrics_dict}
    key_of: function(symbol) -> group name (industry or sector)
    """
    groups = {}
    for sym, m in metrics_by_ticker.items():
        if m is None:
            continue
        k = key_of(sym)
        if not k:
            continue
        groups.setdefault(k, []).append((sym, m))

    rows = []
    cols = list(TIMEFRAMES.keys()) + ["ytd"]
    for name, members in groups.items():
        if len(members) < min_members:
            continue
        row = {"name": name, "count": len(members)}
        for tf in cols:
            vals = [m[tf] for _, m in members if m.get(tf) is not None]
            # median, not mean: one name can't spike a small industry's reading
            row[tf] = round(float(np.median(vals)), 2) if vals else None
        row["new_highs"] = sum(1 for _, m in members if m.get("new_high"))
        row["new_lows"]  = sum(1 for _, m in members if m.get("new_low"))
        # members sorted by 1-month perf, strongest first
        row["tickers"] = [s for s, _ in sorted(
            members, key=lambda t: (t[1].get("m1") is None, -(t[1].get("m1") or 0)))]
        rows.append(row)
    rows.sort(key=lambda r: (r.get("m1") is None, -(r.get("m1") or 0)))
    return rows


# ======================================================================
# NETWORK  (lazy imports; runs inside the GitHub Action)
# ======================================================================
def build_universe():
    """Fetch and filter the universe from the NASDAQ screener."""
    import requests
    r = requests.get(NASDAQ_URL, headers=NASDAQ_HEADERS, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]

    uni = {}
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym or any(ch in sym for ch in (".", "^", "/", "~", " ")):
            continue  # drop warrants / units / preferreds / odd classes
        # market cap -> float
        raw = (row.get("marketCap") or "").replace("$", "").replace(",", "").strip()
        try:
            mcap = float(raw)
        except ValueError:
            continue
        if mcap < MIN_MARKET_CAP:
            continue
        # No country filter: include ADRs / foreign-domiciled names >= $800M,
        # matching Ariel's "any stock over $800M market cap, that's all."
        ind = (row.get("industry") or "").strip() or "Unknown"
        uni[sym] = {
            "name": row.get("name", sym),
            "sector": (row.get("sector") or "").strip() or "Unknown",
            "industry": INDUSTRY_OVERRIDE.get(sym, ind),
            "marketcap": mcap,
        }
    if len(uni) < 500:
        raise RuntimeError(f"Universe suspiciously small ({len(uni)}); aborting.")
    return uni


def load_universe():
    """Always rebuild fresh (one quick screener call) so filters/overrides apply immediately."""
    print("Building universe from NASDAQ screener...")
    uni = build_universe()
    print(f"Universe: {len(uni)} names.")
    return uni


def _one_ticker_frame(data, sym, multi):
    """Extract a single ticker's OHLCV frame from a yf.download result."""
    try:
        df = data[sym] if multi else data
    except (KeyError, TypeError):
        return None
    if df is None or df.empty:
        return None
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_prices(symbols):
    """Download OHLCV for a list of symbols, chunked, from Yahoo."""
    import yfinance as yf
    frames = {}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        print(f"  prices {i+1}-{i+len(chunk)} / {len(symbols)}")
        try:
            data = yf.download(chunk, period=PRICE_PERIOD, interval="1d",
                               auto_adjust=False, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            print(f"    chunk failed, retrying once: {e}")
            time.sleep(5)
            data = yf.download(chunk, period=PRICE_PERIOD, interval="1d",
                               auto_adjust=False, group_by="ticker",
                               threads=True, progress=False)
        multi = len(chunk) > 1
        for sym in chunk:
            f = _one_ticker_frame(data, sym, multi)
            if f is not None and len(f) >= 2:
                frames[sym] = f
        time.sleep(1)  # be polite to Yahoo
    return frames


# ======================================================================
# ASSEMBLY
# ======================================================================
def fetch_holdings(etf_syms):
    """Top holdings per ETF via yfinance funds_data (graceful if unavailable)."""
    import yfinance as yf
    out = {}
    for s in etf_syms:
        try:
            td = yf.Ticker(s).funds_data.top_holdings
            rows = []
            for sym, r in td.head(10).iterrows():
                try:
                    w = float(r.get("Holding Percent"))
                except Exception:
                    w = 0.0
                if w <= 1.5:          # yfinance gives a fraction -> percent
                    w *= 100
                try:
                    nm = str(r.get("Name") or "")
                except Exception:
                    nm = ""
                rows.append({"sym": str(sym), "name": nm, "w": round(w, 2)})
            if rows:
                out[s] = rows
        except Exception:
            pass
        time.sleep(0.25)
    return out


def load_holdings(etf_syms):
    if os.path.exists(HOLDINGS_CACHE):
        if (time.time() - os.path.getmtime(HOLDINGS_CACHE)) / 86400 < 7:
            try:
                with open(HOLDINGS_CACHE) as f:
                    return json.load(f)
            except Exception:
                pass
    print(f"Fetching ETF holdings for {len(etf_syms)} ETFs...")
    h = fetch_holdings(etf_syms)
    try:
        with open(HOLDINGS_CACHE, "w") as f:
            json.dump(h, f)
    except Exception:
        pass
    print(f"  holdings found for {len(h)} ETFs")
    return h


def etf_rows(etf_syms, metrics):
    rows = []
    for s in etf_syms:
        m = metrics.get(s)
        if not m:
            continue
        rows.append({"sym": s, "name": ETF_NAMES.get(s, s),
                     **{k: m.get(k) for k in
                        list(TIMEFRAMES) + ["ytd", "price", "adr", "dollar_vol",
                                            "pct_from_hi", "new_high", "new_low"]}})
    rows.sort(key=lambda r: (r.get("m1") is None, -(r.get("m1") or 0)))
    return rows


def build_payload(universe, metrics):
    industry_of = lambda s: universe.get(s, {}).get("industry")
    sector_of   = lambda s: universe.get(s, {}).get("sector")

    # Stock-only metrics (exclude ETFs from the group aggregates)
    stock_metrics = {s: m for s, m in metrics.items() if s in universe and m}

    themes = aggregate(stock_metrics, industry_of)
    groups = aggregate(stock_metrics, sector_of)

    # New highs / lows by industry
    nhl = []
    for row in themes:
        if row["new_highs"] or row["new_lows"]:
            nh = [s for s in row["tickers"] if stock_metrics[s].get("new_high")]
            nl = [s for s in row["tickers"] if stock_metrics[s].get("new_low")]
            nhl.append({"industry": row["name"],
                        "new_highs": len(nh), "new_lows": len(nl),
                        "nh_tickers": nh, "nl_tickers": nl})
    nhl.sort(key=lambda r: -(r["new_highs"] - r["new_lows"]))
    totals = {"new_highs": sum(r["new_highs"] for r in nhl),
              "new_lows":  sum(r["new_lows"]  for r in nhl)}

    # Per-ticker payload (slim)
    tickers = {}
    for s, m in stock_metrics.items():
        u = universe[s]
        tickers[s] = {"name": u["name"], "sector": u["sector"],
                      "industry": u["industry"], "mcap": u["marketcap"], **m}

    gainers = sorted(
        [{"sym": s, "pct": m["d1"]} for s, m in stock_metrics.items()
         if m.get("d1") is not None],
        key=lambda x: -x["pct"])[:12]

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universe_size": len(stock_metrics),
        "params": {"min_market_cap": MIN_MARKET_CAP, "includes_adr": True,
                   "country": "all"},
        "tabs": {
            "themes": themes,
            "groups": groups,
            "sp": etf_rows(ETF_SP, metrics),
            "eqwt": etf_rows(ETF_EQWT, metrics),
            "country": etf_rows(ETF_COUNTRY, metrics),
            "group": etf_rows(ETF_GROUP, metrics),
        },
        "new_high_low": {"by_industry": nhl, "totals": totals},
        "top_gainers": gainers,
        "tickers": tickers,
    }


def main():
    universe = load_universe()

    # Reclassify sector/industry to the Morningstar scheme (= Finviz / TC2000),
    # keeping NASDAQ's values as the fallback for anything Yahoo can't classify.
    fb = {s: {"sector": universe[s]["sector"], "industry": universe[s]["industry"]}
          for s in universe}
    ind_map = IND.enrich(sorted(universe.keys()), fallback=fb, time_budget_s=1500)
    for s in universe:
        universe[s]["sector"] = ind_map[s]["sector"]
        universe[s]["industry"] = ind_map[s]["industry"]

    syms = sorted(universe.keys())
    all_syms = syms + ETF_SP + ETF_EQWT + ETF_COUNTRY + ETF_GROUP

    print(f"Fetching prices for {len(all_syms)} symbols...")
    frames = fetch_prices(all_syms)
    print(f"Got price history for {len(frames)} symbols.")

    metrics = {s: ticker_metrics(f) for s, f in frames.items()}
    metrics = {s: m for s, m in metrics.items() if m}

    payload = build_payload(universe, metrics)
    payload["etf_holdings"] = load_holdings(sorted(set(ETF_SP + ETF_EQWT + ETF_COUNTRY + ETF_GROUP)))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {OUT_PATH}: {payload['universe_size']} stocks, "
          f"{len(payload['tabs']['themes'])} industries, "
          f"{payload['new_high_low']['totals']['new_highs']} new highs / "
          f"{payload['new_high_low']['totals']['new_lows']} new lows.")


if __name__ == "__main__":
    main()
