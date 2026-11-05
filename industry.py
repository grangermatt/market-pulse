#!/usr/bin/env python3
"""
Shared industry classifier (Morningstar scheme, via Yahoo) for the dashboard
and the monitor, so both tabs match Finviz / TC2000.

- Caches results in industry_map.json  (sym -> {"sector","industry"}).
  Industries rarely change, so the cache is effectively permanent; the
  GitHub Action commits it, so later runs reuse it and stay fast.
- Only fetches names not already cached, within a time budget, saving the
  cache incrementally so progress persists even if a run is cut short.
- Any name Yahoo can't classify falls back to the caller's value (NASDAQ),
  so grouping always still computes.

The network fetch is isolated in _yahoo_fetch and can be injected (fetcher=)
for testing without yfinance or a network.
"""
import os, json, time

IND_MAP_PATH = "industry_map.json"

# Belt-and-suspenders fixes for the rare names Yahoo mislabels the way NASDAQ does.
INDUSTRY_OVERRIDE = {
    "FSLR": "Solar", "ENPH": "Solar", "SEDG": "Solar", "RUN": "Solar", "NOVA": "Solar",
    "ARRY": "Solar", "SHLS": "Solar", "CSIQ": "Solar", "MAXN": "Solar", "JKS": "Solar",
    "DQ": "Solar", "NXT": "Solar", "SPWR": "Solar", "SOL": "Solar", "FTCI": "Solar",
}


def load_map(path=IND_MAP_PATH):
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def save_map(m, path=IND_MAP_PATH):
    try:
        with open(path, "w") as f:
            json.dump(m, f)
    except Exception:
        pass


def _yahoo_fetch(sym):
    """Return {'sector','industry'} for one symbol from Yahoo, or None."""
    import yfinance as yf
    try:
        info = yf.Ticker(sym).info
    except Exception:
        return None
    ind = (info.get("industry") or "").strip()
    sec = (info.get("sector") or "").strip()
    if not ind:
        return None
    return {"sector": sec or "Unknown", "industry": ind}


def enrich(symbols, fallback=None, time_budget_s=1500, path=IND_MAP_PATH, fetcher=None):
    """
    Ensure every symbol has a Morningstar sector/industry, fetching the
    missing ones (up to a time budget). Returns {sym: {"sector","industry"}}.

    fallback: {sym: {"sector","industry"}} used for names not yet fetched
              (typically the NASDAQ values the caller already has).
    fetcher:  injectable one-symbol fetch for testing; defaults to Yahoo.
    """
    if fetcher is None:
        fetcher = _yahoo_fetch
    fallback = fallback or {}

    m = load_map(path)
    missing = [s for s in symbols if s not in m]
    start = time.time()
    fetched = 0
    for s in missing:
        if time.time() - start > time_budget_s:
            print(f"  industry fetch hit time budget; {len(missing)-fetched} left for next run")
            break
        res = fetcher(s)
        if res:
            m[s] = res
            fetched += 1
            if fetched % 50 == 0:
                save_map(m, path)
        time.sleep(0.15)
    save_map(m, path)

    out = {}
    for s in symbols:
        base = m.get(s) or fallback.get(s) or {"sector": "Unknown", "industry": "Unknown"}
        ind = INDUSTRY_OVERRIDE.get(s, base.get("industry", "Unknown"))
        out[s] = {"sector": base.get("sector", "Unknown"), "industry": ind}
    print(f"  industry: {len([s for s in symbols if s in m])}/{len(symbols)} Morningstar-classified "
          f"({fetched} newly fetched this run)")
    return out
