"""
pull_prices.py -- fetch daily price history for MSTR and BTC-USD.

BTC:  Hyperliquid candleSnapshot endpoint (BTC-PERP, daily close price).
      This is the perpetual futures close, not a traditional spot close.
      The perp and spot track very closely for BTC; any basis divergence
      is small relative to weekly return magnitudes.
MSTR: yfinance, ticker "MSTR", adjusted daily close.

Output: data/prices_raw.csv  (long format: date, ticker, close, source)

Timing note: BTC close = UTC midnight; MSTR close ~21:00 UTC. Within a
weekly series the offset is negligible.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

# Start a few weeks before first ATM issuance row (Nov 2024) so the first
# weekly return calc has a prior close to diff against.
START_DATE = datetime(2024, 10, 14, tzinfo=timezone.utc)   # ~4 weeks buffer
END_DATE   = datetime(2026, 6, 24, tzinfo=timezone.utc)    # today


# ---------------------------------------------------------------------------
# Hyperliquid BTC-PERP daily closes
# ---------------------------------------------------------------------------

HL_URL = "https://api.hyperliquid.xyz/info"
HL_HEADERS = {"Content-Type": "application/json"}


def _hl_post(payload: dict) -> list:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(HL_URL, data=body, headers=HL_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def pull_btc_hl(start: datetime, end: datetime) -> pd.DataFrame:
    """Return daily BTC-PERP closes from Hyperliquid."""
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    try:
        rows = _hl_post(payload)
    except Exception as e:
        sys.exit(f"[ERROR] Hyperliquid request failed: {e}")

    if not rows:
        sys.exit("[ERROR] Hyperliquid returned 0 candles for BTC -- cannot continue.")

    records = []
    for r in rows:
        # t = open-time ms; T = close-time ms. Use open-time as the date label
        # (UTC midnight for 1d candles) so dates align with calendar days.
        dt = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc).date()
        records.append({"date": dt, "ticker": "BTC", "close": float(r["c"]), "source": "hyperliquid"})

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# yfinance MSTR daily closes
# ---------------------------------------------------------------------------

def pull_mstr_yf(start: datetime, end: datetime) -> pd.DataFrame:
    """Return adjusted daily MSTR closes from yfinance."""
    # yfinance end is exclusive, so add 1 day to include end_date itself.
    end_inc = end + timedelta(days=1)
    ticker = yf.Ticker("MSTR")
    hist = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end_inc.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )

    if hist.empty:
        sys.exit("[ERROR] yfinance returned no data for MSTR -- cannot continue.")

    df = hist[["Close"]].copy()
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None).date
    df = df.reset_index()
    df.columns = ["date", "close"]
    df["ticker"] = "MSTR"
    df["source"] = "yfinance"
    df = df[["date", "ticker", "close", "source"]].sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def _check_gaps(df: pd.DataFrame, ticker: str, is_equity: bool):
    """Print any unexpected gaps. Equity: flag missing weekdays. BTC: flag missing calendar days."""
    dates = sorted(df["date"].tolist())
    if not dates:
        return
    d = dates[0]
    end = dates[-1]
    date_set = set(dates)
    gaps = []
    while d <= end:
        if is_equity:
            # Only flag missing Mon-Fri
            if d.weekday() < 5 and d not in date_set:
                gaps.append(d)
        else:
            if d not in date_set:
                gaps.append(d)
        d += timedelta(days=1)

    if gaps:
        # Group consecutive gaps for readability
        print(f"  [GAPS] {ticker}: {len(gaps)} missing {'weekdays' if is_equity else 'days'} "
              f"(first 5: {gaps[:5]})", file=sys.stderr)
    else:
        print(f"  [OK] {ticker}: no unexpected gaps.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "prices_raw.csv"

    print(f"Pulling BTC-PERP daily closes from Hyperliquid ({START_DATE.date()} to {END_DATE.date()})...")
    btc_df = pull_btc_hl(START_DATE, END_DATE)

    print(f"Pulling MSTR daily closes from yfinance ({START_DATE.date()} to {END_DATE.date()})...")
    mstr_df = pull_mstr_yf(START_DATE, END_DATE)

    # Validate both non-empty before writing anything
    if btc_df.empty:
        sys.exit("[ERROR] BTC dataframe is empty after pull.")
    if mstr_df.empty:
        sys.exit("[ERROR] MSTR dataframe is empty after pull.")

    combined = pd.concat([btc_df, mstr_df], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} rows to {out_path}\n")

    # Summary per ticker
    for ticker, grp in combined.groupby("ticker"):
        is_eq = (ticker == "MSTR")
        print(f"{ticker}: {len(grp)} rows  |  {grp['date'].min()} to {grp['date'].max()}  "
              f"|  source={grp['source'].iloc[0]}")
        sys.stdout.flush()
        _check_gaps(grp, ticker, is_equity=is_eq)
        sys.stderr.flush()


if __name__ == "__main__":
    main()
