"""
build_dataset.py -- join ATM issuance and price data into a weekly dataset.

Week definition: each ATM filing's own window_start_raw / window_end_raw,
NOT standard ISO weeks. Filings disclose irregular windows (mostly 7 days,
sometimes 2-3 around quarter/year-end), so returns are computed over the
actual disclosed period.

Return window alignment
-----------------------
The disclosed window is inclusive of its first day: a "June 1 to June 7"
filing covers selling that happened ON June 1. The return that spans that
selling therefore runs from the close BEFORE the window opens to the close
at the window's end:

    start price = last trading close STRICTLY BEFORE window_start
    end price   = last trading close ON OR BEFORE window_end

For the typical Monday-to-Sunday window that is prior-Friday close to
Friday close, a full 5 trading days. Using the window_start close instead
would silently drop Monday, the first day of issuance, from the measured
return. It also leaves a Friday-to-Monday hole between consecutive weeks,
which breaks the lead-lag analysis downstream.

Endpoint matching
-----------------
MSTR and BTC returns are computed over the SAME two calendar dates (the MSTR
trading days above), not over the raw window boundaries. BTC trades on
weekends and MSTR does not; pairing a Sunday BTC close against a Friday MSTR
close injects two days of unhedged BTC move into every observation and
suppresses the BTC-beta fit. The actual dates used are written to the output
as px_start_date / px_end_date.

Returns: (close_end - close_start) / close_start  (arithmetic, not log)

Output: data/weekly.csv
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
ATM_PATH  = DATA_DIR / "atm_issuance_raw.csv"
PRICE_PATH = DATA_DIR / "prices_raw.csv"
OUT_PATH   = DATA_DIR / "weekly.csv"

# Windows shorter or longer than this are flagged in output.
WINDOW_MIN_DAYS = 5
WINDOW_MAX_DAYS = 10

# How far back to walk when looking for a trading day.
MAX_LOOKBACK_DAYS = 7


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------

def _build_lookup(prices: pd.DataFrame, ticker: str) -> dict:
    """Return {date -> close} dict for a single ticker."""
    sub = prices[prices["ticker"] == ticker].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.date
    return dict(zip(sub["date"], sub["close"]))


def _trading_day_on_or_before(px_map: dict, target: date) -> date:
    """Latest date in px_map that is <= target."""
    d = target
    for _ in range(MAX_LOOKBACK_DAYS + 1):
        if d in px_map:
            return d
        d -= timedelta(days=1)
    raise KeyError(f"No trading day within {MAX_LOOKBACK_DAYS} days on or before {target}")


def _trading_day_before(px_map: dict, target: date) -> date:
    """Latest date in px_map that is strictly < target."""
    return _trading_day_on_or_before(px_map, target - timedelta(days=1))


def _is_routine_weekend(window_boundary: date, used: date) -> bool:
    """True when the only reason we moved back was Sat/Sun, not a holiday."""
    return window_boundary.weekday() in (5, 6) and (window_boundary - used).days <= 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Load inputs ---------------------------------------------------------
    atm = pd.read_csv(ATM_PATH)
    prices = pd.read_csv(PRICE_PATH)

    if atm.empty:
        sys.exit("[ERROR] atm_issuance_raw.csv is empty.")
    if prices.empty:
        sys.exit("[ERROR] prices_raw.csv is empty.")

    mstr_px = _build_lookup(prices, "MSTR")
    btc_px  = _build_lookup(prices, "BTC")

    # -- Process each filing row ---------------------------------------------
    records = []
    n_subst = 0       # MSTR holiday substitutions (weekends are routine, not flagged)
    n_short_long = 0  # windows outside [WINDOW_MIN_DAYS, WINDOW_MAX_DAYS]
    n_skipped = 0

    for idx, row in atm.iterrows():
        try:
            ws = pd.to_datetime(row["window_start_raw"]).date()
            we = pd.to_datetime(row["window_end_raw"]).date()
        except Exception as e:
            print(f"[SKIP row {idx}] Date parse error: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        window_days = (we - ws).days

        # Endpoint dates: the close before the window opens, and the close at
        # the window's end. Both are MSTR trading days.
        try:
            px_start_date = _trading_day_before(mstr_px, ws)
            px_end_date   = _trading_day_on_or_before(mstr_px, we)
        except KeyError as e:
            print(f"[SKIP row {idx}] MSTR price missing: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        if px_end_date <= px_start_date:
            print(f"[SKIP row {idx}] {ws}--{we}: degenerate window, endpoints "
                  f"resolve to {px_start_date} and {px_end_date}", file=sys.stderr)
            n_skipped += 1
            continue

        # BTC at the SAME dates, so both legs span an identical interval.
        if px_start_date not in btc_px or px_end_date not in btc_px:
            print(f"[SKIP row {idx}] BTC price missing for "
                  f"{px_start_date} or {px_end_date}", file=sys.stderr)
            n_skipped += 1
            continue

        mstr_return = (mstr_px[px_end_date] - mstr_px[px_start_date]) / mstr_px[px_start_date]
        btc_return  = (btc_px[px_end_date]  - btc_px[px_start_date])  / btc_px[px_start_date]

        # Trading days spanned, exclusive of the start close. Normally 5.
        trading_days = sum(
            1 for d in mstr_px
            if px_start_date < d <= px_end_date
        )

        # Flag only genuine holiday substitutions at the window end. Walking a
        # Sunday window_end back to Friday is how these filings always look.
        subst_flag = (px_end_date != we) and not _is_routine_weekend(we, px_end_date)
        if subst_flag:
            n_subst += 1
            print(f"  [SUBST] {ws}--{we}: window_end {we} resolved to "
                  f"{px_end_date} (market holiday)", file=sys.stderr)

        nonstandard_flag = window_days < WINDOW_MIN_DAYS or window_days > WINDOW_MAX_DAYS
        if nonstandard_flag:
            n_short_long += 1

        records.append({
            "window_start":      ws,
            "window_end":        we,
            "window_days":       window_days,
            "px_start_date":     px_start_date,
            "px_end_date":       px_end_date,
            "trading_days":      trading_days,
            "mstr_return":       round(mstr_return, 6),
            "btc_return":        round(btc_return, 6),
            "shares_sold":       int(row["shares_sold"]),
            "net_proceeds_usd":  float(row["net_proceeds_usd"]),
            "format":            row["format"],
            "price_subst_flag":  subst_flag,
            "nonstandard_window": nonstandard_flag,
        })

    if not records:
        sys.exit("[ERROR] No rows built -- check date parsing and price coverage.")

    out = pd.DataFrame(records).sort_values("window_start").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    # -- Summary -------------------------------------------------------------
    sys.stdout.flush(); sys.stderr.flush()
    print(f"\nWrote {len(out)} rows to {OUT_PATH}")
    print(f"Date range:  {out['window_start'].min()}  to  {out['window_end'].max()}")
    print(f"Skipped:     {n_skipped} rows (price data missing)")
    print(f"Holiday subst: {n_subst} rows (window_end fell on a market holiday)")
    print(f"Non-std windows (<{WINDOW_MIN_DAYS}d or >{WINDOW_MAX_DAYS}d): {n_short_long}")

    # Contiguity check: with correct alignment, one week's end close is the
    # next week's start close. Breaks here mean a missing filing.
    std = out[~out["nonstandard_window"]].reset_index(drop=True)
    contiguous = sum(
        std["px_end_date"].iloc[i] == std["px_start_date"].iloc[i + 1]
        for i in range(len(std) - 1)
    )
    print(f"Contiguous week pairs: {contiguous}/{max(len(std) - 1, 0)} "
          f"(breaks correspond to missing filings)")

    td = out[~out["nonstandard_window"]]["trading_days"]
    print(f"Trading days per standard window: "
          f"median={td.median():.0f}  min={td.min()}  max={td.max()}")

    if n_short_long:
        print("\nNon-standard windows:")
        ns = out[out["nonstandard_window"]]
        for _, r in ns.iterrows():
            print(f"  {r['window_start']} to {r['window_end']}  ({r['window_days']}d)  "
                  f"net_proceeds=${r['net_proceeds_usd']/1e6:.1f}M")


if __name__ == "__main__":
    main()
