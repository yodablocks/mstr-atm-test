"""
build_dataset.py -- join ATM issuance and price data into a weekly dataset.

Week definition: each ATM filing's own window_start_raw / window_end_raw,
NOT standard ISO weeks. Filings disclose irregular windows (mostly 7 days,
sometimes 2-3 around quarter/year-end), so returns are computed over the
actual disclosed period.

Returns: (close_end - close_start) / close_start  (arithmetic, not log)

MSTR: equity, closed on weekends/holidays. If window_start or window_end
      falls on a non-trading day, use the nearest prior trading day's close.
      Flag these substitutions.
BTC:  trades every day. Exact date lookups -- no fallback needed.

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


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------

def _build_lookup(prices: pd.DataFrame, ticker: str) -> dict:
    """Return {date -> close} dict for a single ticker."""
    sub = prices[prices["ticker"] == ticker].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.date
    return dict(zip(sub["date"], sub["close"]))


def _nearest_prior(px_map: dict, target: date, max_lookback: int = 5) -> tuple[float, bool]:
    """
    Return (close, holiday_substituted).

    Sunday/Saturday -> Friday is routine for these filings (window ends are
    disclosed through Sunday). That case is NOT flagged (substituted=False).
    Only flag when the nearest prior trading day is further back than the
    weekend itself -- i.e., the Friday before was also a market holiday.
    """
    if target in px_map:
        return px_map[target], False

    # Walk back to find the actual trading day used
    d = target - timedelta(days=1)
    for _ in range(max_lookback):
        if d in px_map:
            # Routine weekend: target was Sat/Sun and we landed on Friday
            if target.weekday() in (5, 6) and (target - d).days <= 2:
                return px_map[d], False
            # Otherwise: a weekday or a longer-than-2-day gap -- genuine holiday
            return px_map[d], True
        d -= timedelta(days=1)
    raise KeyError(f"No price within {max_lookback} days before {target}")


def _exact(px_map: dict, target: date) -> float:
    """Return exact close; raise KeyError with informative message if missing."""
    if target not in px_map:
        raise KeyError(f"No price for {target}")
    return px_map[target]


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
    n_subst = 0       # MSTR weekend/holiday substitutions
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

        # MSTR: nearest-prior fallback on non-trading days
        try:
            mstr_start, sub_s = _nearest_prior(mstr_px, ws)
            mstr_end,   sub_e = _nearest_prior(mstr_px, we)
        except KeyError as e:
            print(f"[SKIP row {idx}] MSTR price missing: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        # BTC: exact lookups only
        try:
            btc_start = _exact(btc_px, ws)
            btc_end   = _exact(btc_px, we)
        except KeyError as e:
            print(f"[SKIP row {idx}] BTC price missing: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        mstr_return = (mstr_end - mstr_start) / mstr_start
        btc_return  = (btc_end  - btc_start)  / btc_start

        subst_flag = sub_s or sub_e
        if subst_flag:
            n_subst += 1
            print(f"  [SUBST] {ws}--{we}: MSTR date substituted "
                  f"(start_ok={not sub_s}, end_ok={not sub_e})", file=sys.stderr)

        nonstandard_flag = window_days < WINDOW_MIN_DAYS or window_days > WINDOW_MAX_DAYS
        if nonstandard_flag:
            n_short_long += 1

        records.append({
            "window_start":      ws,
            "window_end":        we,
            "window_days":       window_days,
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
    print(f"Price subst: {n_subst} rows (MSTR weekend/holiday boundary)")
    print(f"Non-std windows (<{WINDOW_MIN_DAYS}d or >{WINDOW_MAX_DAYS}d): {n_short_long}")

    if n_short_long:
        print("\nNon-standard windows:")
        ns = out[out["nonstandard_window"]]
        for _, r in ns.iterrows():
            print(f"  {r['window_start']} to {r['window_end']}  ({r['window_days']}d)  "
                  f"net_proceeds=${r['net_proceeds_usd']/1e6:.1f}M")


if __name__ == "__main__":
    main()
