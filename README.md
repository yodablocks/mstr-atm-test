# MSTR ATM Abnormal Return Replication

Tests whether weekly ATM common stock issuance by Strategy Inc. (MSTR) shows
up as a detectable abnormal return, after stripping out BTC-beta exposure.

This replicates and extends a public analysis of the same question (OLS
R^2 ≈ 0.49, described as "dead"), re-testing with methods more robust to
the issuance distribution's lumpiness: rank correlation, MAD-based outlier
handling, and lead-lag checks.

## Dependencies

```
pandas
numpy
scipy
statsmodels
yfinance
```

---

## Run order

```bash
python pull_filings.py   # needs EDGAR_USER_AGENT env var with a real email
python pull_prices.py
python build_dataset.py
python analysis.py
```

EDGAR requires a `User-Agent` header with a real contact email. Set `EDGAR_USER_AGENT="Name contact@email.com"` before running `pull_filings.py` or edit the default in the script.

`pull_prices.py` hits Hyperliquid and Yahoo Finance directly -- requires live network access.

---

## Known limitations

- ATM window dates are not always calendar-week aligned. We use the filing's own disclosed window rather than snapping to ISO weeks.
- Two 2025 weeks are missing from the issuance dataset (filings exist but proceeds weren't parsed).
- BTC price is perpetual futures close, not a traditional spot exchange close.
- Coverage thins out before Nov 2024; the weekly 8-K disclosure cadence is a relatively recent practice for MSTR.
