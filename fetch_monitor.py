#!/usr/bin/env python3
"""
Market Monitor — data engine  (Stockbee-style breadth)
======================================================================
Separate from the dashboard. Runs on its own LIQUID universe:
  common stock + ADR, price >= $3, 1-day dollar-volume >= $20M  (~2,230 names)

Computes, for each of the last ~250 trading days, across that universe:
  - Up 4% / Down 4% today          (>= +/-4%, vol >= 100k, vol > prior day)
  - 5-day and 10-day up/down ratio
  - Up/Down 25% in a quarter        (vs 65-day low/high)
  - Up/Down 25% / 50% in a month    (20-day return)
  - Up/Down 13% in 34 days          (vs 34-day low/high)
  - 10x ATR Extended                (close >= 50-MA + 10 * ATR14)
  - % above 50-day SMA
  - universe count
  - S&P 500 close (^GSPC)
plus today's name lists (for the 4%-up/4%-down drill-downs) and the
up-4%/down-4% distribution (so the 300 "big day" line can be checked
against real data).

Output: data/monitor.json

Network libs (requests, yfinance) are imported lazily so the pure
compute function can be unit-tested without them (see selftest_monitor.py).
"""
import os, json, time, datetime as dt
import numpy as np
import pandas as pd

MIN_PRICE        = 3.0
MIN_DOLLAR_VOL   = 20_000_000       # 1-day $-volume floor
UNIVERSE_CACHE   = "monitor_universe.json"
UNIVERSE_MAX_AGE = 7
PRICE_PERIOD     = "400d"
CHUNK            = 150
OUT_DAYS         = 335              # rows recomputed per run (~1.3yr seed); history accumulates beyond this
OUT_PATH         = "data/monitor.json"
BIG_DAY          = 300             # dark-green threshold for 4% up/down

NASDAQ_URL = ("https://api.nasdaq.com/api/screener/stocks"
              "?tableonly=true&limit=10000&download=true")
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# ======================================================================
# PURE COMPUTE  (no network)
# ======================================================================
def _cnt(mask):            # count True across tickers, per day
    return mask.sum(axis=1)

def compute_monitor(close, high, low, volume, spx=None, out_days=OUT_DAYS, big_day=BIG_DAY):
    """
    close/high/low/volume: DataFrames indexed by date (asc), columns = tickers.
    spx: Series of S&P 500 close indexed by date (optional).
    Returns the monitor.json payload dict.
    """
    ret1  = close.pct_change() * 100
    vprev = volume.shift(1)

    up4 = (ret1 >= 4)  & (volume >= 100_000) & (volume > vprev)
    dn4 = (ret1 <= -4) & (volume >= 100_000) & (volume > vprev)

    minc65, maxc65 = close.rolling(65).min(), close.rolling(65).max()
    minc34, maxc34 = close.rolling(34).min(), close.rolling(34).max()
    avgc20 = close.rolling(20).mean()
    avgv20 = volume.rolling(20).mean()
    liq    = (avgc20 * avgv20) >= 250_000
    c20    = close.shift(20)
    ret20  = (close - c20) / c20 * 100

    up25q = (((close - minc65) / (minc65 + 0.01)) * 100 >= 25) & liq
    dn25q = (((close - maxc65) / (maxc65 + 0.01)) * 100 <= -25) & liq
    up13  = (((close - minc34) / (minc34 + 0.01)) * 100 >= 13) & liq
    dn13  = (((close - maxc34) / (maxc34 + 0.01)) * 100 <= -13) & liq
    up25m = (c20 >= 5) & liq & (ret20 >= 25)
    dn25m = (c20 >= 5) & liq & (ret20 <= -25)
    up50m = (c20 >= 5) & liq & (ret20 >= 50)
    dn50m = (c20 >= 5) & liq & (ret20 <= -50)

    # True Range -> ATR14 ; extended = close >= 50-MA + 10*ATR14
    pc = close.shift(1)
    trv = np.maximum.reduce([(high - low).values,
                             (high - pc).abs().values,
                             (low - pc).abs().values])
    tr  = pd.DataFrame(trv, index=close.index, columns=close.columns)
    atr14 = tr.rolling(14).mean()
    sma50 = close.rolling(50).mean()
    ext   = (close - sma50) >= (10 * atr14)
    above50 = close > sma50

    # per-day count series
    up4c, dn4c = _cnt(up4), _cnt(dn4)
    series = {
        "up4": up4c, "dn4": dn4c,
        "up25q": _cnt(up25q), "dn25q": _cnt(dn25q),
        "up25m": _cnt(up25m), "dn25m": _cnt(dn25m),
        "up50m": _cnt(up50m), "dn50m": _cnt(dn50m),
        "up13": _cnt(up13),   "dn13": _cnt(dn13),
        "atr10x": _cnt(ext),
    }
    # ratios (guard divide-by-zero)
    r5  = up4c.rolling(5).sum()  / dn4c.rolling(5).sum().replace(0, np.nan)
    r10 = up4c.rolling(10).sum() / dn4c.rolling(10).sum().replace(0, np.nan)

    uni = close.notna().sum(axis=1)
    have50 = (close.notna() & sma50.notna())
    pct50  = (above50 & sma50.notna()).sum(axis=1) / have50.sum(axis=1).replace(0, np.nan) * 100

    spx_al = spx.reindex(close.index) if spx is not None else None

    def val(s, d, nd=2):
        v = s.get(d)
        return None if v is None or pd.isna(v) else round(float(v), nd)

    idx = list(close.index[-out_days:])
    rows = []
    for d in reversed(idx):
        row = {"date": str(pd.Timestamp(d).date()),
               "universe": int(uni[d]),
               "r5": val(r5, d), "r10": val(r10, d),
               "atr10x": int(series["atr10x"][d]),
               "pct50": val(pct50, d, 1),
               "spx": (val(spx_al, d) if spx_al is not None else None)}
        for k in ("up4","dn4","up25q","dn25q","up25m","dn25m","up50m","dn50m","up13","dn13"):
            row[k] = int(series[k][d])
        rows.append(row)

    # today's name lists (latest day) for the 4% drill-downs
    d0 = idx[-1]
    today = {
        "up4": sorted([c for c in close.columns if bool(up4.loc[d0, c])]),
        "dn4": sorted([c for c in close.columns if bool(dn4.loc[d0, c])]),
    }

    def dist(s):
        w = s.iloc[-out_days:].dropna()
        return {"min": int(w.min()), "median": int(w.median()),
                "p90": int(w.quantile(.90)), "p95": int(w.quantile(.95)),
                "max": int(w.max())} if len(w) else {}

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universe_size": int(uni.iloc[-1]),
        "big_day": big_day,
        "params": {"min_price": MIN_PRICE, "min_dollar_vol": MIN_DOLLAR_VOL,
                   "includes_adr": True},
        "rows": rows,
        "today_members": today,
        "distribution": {"up4": dist(up4c), "dn4": dist(dn4c)},
    }

# ======================================================================
# NETWORK
# ======================================================================
def build_universe():
    import requests
    r = requests.get(NASDAQ_URL, headers=NASDAQ_HEADERS, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    uni = []
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym or any(ch in sym for ch in (".", "^", "/", "~", " ")):
            continue  # drop warrants / units / preferreds
        try:
            price = float((row.get("lastsale") or "").replace("$", "").replace(",", ""))
            vol   = float(str(row.get("volume") or "0").replace(",", ""))
        except ValueError:
            continue
        if price < MIN_PRICE or price * vol < MIN_DOLLAR_VOL:
            continue            # ADRs kept (no country filter)
        uni.append(sym)
    uni = sorted(set(uni))
    if len(uni) < 500:
        raise RuntimeError(f"Monitor universe too small ({len(uni)})")
    return uni

def load_universe():
    if os.path.exists(UNIVERSE_CACHE):
        if (time.time() - os.path.getmtime(UNIVERSE_CACHE)) / 86400 < UNIVERSE_MAX_AGE:
            with open(UNIVERSE_CACHE) as f:
                return json.load(f)
    uni = build_universe()
    with open(UNIVERSE_CACHE, "w") as f:
        json.dump(uni, f)
    print(f"Monitor universe rebuilt: {len(uni)} names")
    return uni

def fetch_prices(symbols):
    import yfinance as yf
    frames = {}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i+CHUNK]
        print(f"  prices {i+1}-{i+len(chunk)} / {len(symbols)}")
        for attempt in (1, 2):
            try:
                data = yf.download(chunk, period=PRICE_PERIOD, interval="1d",
                                   auto_adjust=False, group_by="ticker",
                                   threads=True, progress=False)
                break
            except Exception as e:
                if attempt == 2: raise
                print(f"    retry: {e}"); time.sleep(5)
        multi = len(chunk) > 1
        for s in chunk:
            try:
                df = data[s] if multi else data
            except (KeyError, TypeError):
                continue
            if df is None or df.empty: continue
            frames[s] = df[["Open","High","Low","Close","Volume"]].copy()
        time.sleep(1)
    return frames

def panel(frames, field):
    return pd.DataFrame({s: f[field] for s, f in frames.items()}).sort_index()

def main():
    syms = load_universe()
    print(f"Fetching prices for {len(syms)} names + ^GSPC ...")
    frames = fetch_prices(syms)
    print(f"Got {len(frames)} price histories")

    import yfinance as yf
    spx_df = yf.download("^GSPC", period=PRICE_PERIOD, interval="1d",
                         auto_adjust=False, progress=False)
    spx = spx_df["Close"]
    if isinstance(spx, pd.DataFrame):
        spx = spx.iloc[:, 0]

    close  = panel(frames, "Close")
    high   = panel(frames, "High")
    low    = panel(frames, "Low")
    volume = panel(frames, "Volume")

    payload = compute_monitor(close, high, low, volume, spx)

    # Accumulate history: keep every past day already on file, refresh the
    # recomputed window, so the record grows forward on the live universe.
    existing = {}
    if os.path.exists(OUT_PATH):
        try:
            old = json.load(open(OUT_PATH))
            for r in old.get("rows", []):
                existing[r["date"]] = r
        except Exception:
            pass
    for r in payload["rows"]:
        existing[r["date"]] = r
    payload["rows"] = sorted(existing.values(), key=lambda r: r["date"], reverse=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    latest = payload["rows"][0]
    print(f"Wrote {OUT_PATH}: {payload['universe_size']} names | "
          f"latest up4={latest['up4']} dn4={latest['dn4']} "
          f"r5={latest['r5']} atr10x={latest['atr10x']} pct>50dma={latest['pct50']}")
    print(f"up-4% distribution: {payload['distribution']['up4']}")

if __name__ == "__main__":
    main()
