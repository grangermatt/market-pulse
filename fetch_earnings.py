#!/usr/bin/env python3
"""
Earnings + economic calendar engine  ->  data/earnings.json
======================================================================
Scope: S&P 500 constituents.
Source: Financial Modeling Prep (free API key in env FMP_API_KEY) for
        earnings estimates/actuals, S&P list, and the economic calendar;
        Yahoo (yfinance) for each reporter's day move + volume.

Produces three sections:
  econ[]     - upcoming + recent US macro events (date, event, impact,
               estimate, previous, actual)
  calendar[] - upcoming S&P reporters (date, ticker, time, EPS est, Rev est)
  results[]  - recent S&P reporters with EPS act/est, Rev act/est,
               EPS & Rev YoY, forward consensus (next-Q EPS/Rev est),
               day % move, volume

Every network call is wrapped: if the free tier doesn't serve an endpoint,
that section comes back empty rather than crashing the run. Pure transforms
are separated from I/O so they can be unit-tested (see selftest_earnings.py).
"""
import os, json, time, datetime as dt

FMP_KEY        = os.environ.get("FMP_API_KEY", "").strip()
BASE           = "https://financialmodelingprep.com/api/v3"
OUT            = "data/earnings.json"
SP500_CACHE    = "sp500.json"
LOOKBACK_DAYS  = 5
LOOKAHEAD_DAYS = 14

# ======================================================================
# PURE TRANSFORMS  (no network — unit tested)
# ======================================================================
def _yoy(cur, prev):
    try:
        if cur is None or prev in (None, 0):
            return None
        return round((cur - prev) / abs(prev) * 100, 1)
    except Exception:
        return None


def pick_econ(rows, keep=("High", "Medium")):
    """US macro events at the chosen importance levels, sorted by date."""
    out = []
    for r in rows or []:
        country = (r.get("country") or "").upper()
        if country not in ("US", "USA", "UNITED STATES"):
            continue
        imp = (r.get("impact") or r.get("importance") or "").title()
        if keep and imp not in keep:
            continue
        out.append({"date": r.get("date"), "event": r.get("event"),
                    "impact": imp, "actual": r.get("actual"),
                    "estimate": r.get("estimate"), "previous": r.get("previous")})
    out.sort(key=lambda x: x["date"] or "")
    return out


def build_calendar(cal_rows, sp500):
    """Upcoming S&P reporters with consensus estimates."""
    out = []
    for r in cal_rows or []:
        s = r.get("symbol")
        if s not in sp500:
            continue
        out.append({"date": r.get("date"), "sym": s, "name": sp500.get(s, s),
                    "time": r.get("time") or "", "eps_est": r.get("epsEstimated"),
                    "rev_est": r.get("revenueEstimated")})
    out.sort(key=lambda x: (x["date"] or "", x["sym"]))
    return out


def forward_consensus(future_rows, sp500):
    """Each S&P symbol's NEXT scheduled estimate = forward consensus."""
    fwd = {}
    for r in sorted(future_rows or [], key=lambda x: x.get("date") or ""):
        s = r.get("symbol")
        if s in sp500 and s not in fwd:
            fwd[s] = {"eps_est": r.get("epsEstimated"), "rev_est": r.get("revenueEstimated")}
    return fwd


def build_results(recent_rows, sp500, income_by_sym, quotes, fwd_by_sym):
    """Recent S&P reporters, enriched with YoY growth, forward consensus,
    market cap, and day move — Earnings-Whispers-style."""
    out = []
    for r in recent_rows or []:
        s = r.get("symbol")
        if s not in sp500:
            continue
        if r.get("eps") is None and r.get("revenue") is None:
            continue  # scheduled but not reported yet
        inc = income_by_sym.get(s) or []
        eps_yoy = rev_yoy = None
        if len(inc) >= 5:
            eps_yoy = _yoy(inc[0].get("eps"), inc[4].get("eps"))
            rev_yoy = _yoy(inc[0].get("revenue"), inc[4].get("revenue"))
        q = quotes.get(s) or {}
        fwd = fwd_by_sym.get(s) or {}
        out.append({
            "date": r.get("date"), "sym": s, "name": sp500.get(s, s),
            "time": r.get("time") or "",
            "mktcap": q.get("mktcap"),
            "eps": r.get("eps"), "eps_est": r.get("epsEstimated"), "eps_yoy": eps_yoy,
            "rev": r.get("revenue"), "rev_est": r.get("revenueEstimated"), "rev_yoy": rev_yoy,
            "fwd_eps_est": fwd.get("eps_est"), "fwd_rev_est": fwd.get("rev_est"),
            "pct": q.get("pct"), "vol": q.get("vol"),
        })
    out.sort(key=lambda x: (x["date"] or "", x["sym"]), reverse=True)
    return out


# ======================================================================
# NETWORK  (lazy; graceful on any failure)
# ======================================================================
def _get(path, params=None):
    import requests
    if not FMP_KEY:
        raise RuntimeError("FMP_API_KEY not set — add it as a GitHub secret")
    p = dict(params or {}); p["apikey"] = FMP_KEY
    try:
        r = requests.get(f"{BASE}/{path}", params=p, timeout=40)
        if r.status_code != 200:
            print(f"  FMP /{path} -> HTTP {r.status_code} (free-tier limit?)")
            return None
        return r.json()
    except Exception as e:
        print(f"  FMP /{path} error: {e}")
        return None


def sp500_map():
    data = _get("sp500_constituent")
    if isinstance(data, list) and data:
        m = {d["symbol"]: d.get("name", d["symbol"]) for d in data if d.get("symbol")}
        try:
            json.dump(m, open(SP500_CACHE, "w"))
        except Exception:
            pass
        return m
    if os.path.exists(SP500_CACHE):   # fall back to last good list
        try:
            return json.load(open(SP500_CACHE))
        except Exception:
            pass
    return {}


def earning_calendar(frm, to):
    data = _get("earning_calendar", {"from": frm, "to": to})
    return data if isinstance(data, list) else []


def economic_calendar(frm, to):
    data = _get("economic_calendar", {"from": frm, "to": to})
    return data if isinstance(data, list) else []


def income_quarters(sym, limit=6):
    data = _get(f"income-statement/{sym}", {"period": "quarter", "limit": limit})
    return data if isinstance(data, list) else []


def fmp_quotes(syms):
    """Market cap + day % change + volume in one batched quote call per 50."""
    out = {}
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        data = _get("quote/" + ",".join(chunk))
        for q in data or []:
            s = q.get("symbol")
            if s:
                out[s] = {"mktcap": q.get("marketCap"),
                          "pct": q.get("changesPercentage"),
                          "vol": q.get("volume")}
        time.sleep(0.1)
    return out


def main():
    if not FMP_KEY:
        print("FMP_API_KEY not set — writing empty earnings.json. "
              "Add the secret and re-run to populate.")
        payload = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "scope": "S&P 500", "econ": [], "calendar": [], "results": [],
                   "note": "no_api_key"}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(payload, open(OUT, "w"), separators=(",", ":"))
        return

    today = dt.date.today()
    frm_r = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    to_f  = (today + dt.timedelta(days=LOOKAHEAD_DAYS)).isoformat()
    tomorrow = (today + dt.timedelta(days=1)).isoformat()

    sp = sp500_map()
    print(f"S&P constituents: {len(sp)}")

    recent = earning_calendar(frm_r, today.isoformat())
    future = earning_calendar(tomorrow, to_f)
    fwd = forward_consensus(future, sp)

    recent_sp = [r for r in recent if r.get("symbol") in sp
                 and (r.get("eps") is not None or r.get("revenue") is not None)]
    result_syms = sorted({r["symbol"] for r in recent_sp})
    print(f"Recent S&P reporters: {len(result_syms)}")

    income_by_sym = {}
    for s in result_syms:
        income_by_sym[s] = income_quarters(s)
        time.sleep(0.1)
    quotes = fmp_quotes(result_syms)

    econ = pick_econ((economic_calendar(today.isoformat(), to_f) or [])
                     + (economic_calendar(frm_r, today.isoformat()) or []))

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "S&P 500",
        "econ": econ,
        "calendar": build_calendar(future, sp),
        "results": build_results(recent, sp, income_by_sym, quotes, fwd),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), separators=(",", ":"))
    print(f"Wrote {OUT}: {len(payload['calendar'])} upcoming, "
          f"{len(payload['results'])} results, {len(econ)} econ events")


if __name__ == "__main__":
    main()
