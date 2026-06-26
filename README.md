# MSTR ATM Abnormal Return Replication

Tests whether weekly ATM common stock issuance by Strategy Inc. (MSTR) shows
up as a detectable abnormal return, after stripping out BTC-beta exposure.

A public analysis of this same question, using a single-pass OLS regression,
reported R^2 ~ 0.49 and called the relationship "dead." This replicates that
test and extends it with methods more robust to the issuance distribution's
lumpiness: rank correlation, MAD-based outlier handling, and lead-lag checks.

## Dependencies

pandas
numpy
scipy
statsmodels
yfinance

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

---

## Conclusion

A public analysis of this same question, using a single-pass OLS regression,
reported R^2 ~ 0.49 and called the relationship "dead." That conclusion turns
out to be directionally right but methodologically lucky. OLS R^2 is not a
robust statistic on a distribution this lumpy: in this dataset, the five
largest issuance weeks account for the majority of total proceeds. A test
built for a roughly normal distribution will swing on outliers in a sample
shaped like this one, and a result built that way is not something you can
trust to hold on a different sample window.

Re-tested here with three approaches built for this kind of distribution:

- OLS R^2: 0.020 (all weeks), 0.042 (non-zero issuance weeks only)
- Spearman rank correlation: rho = -0.081
- MAD-based outlier handling: result collapses toward zero once the largest
  issuance weeks are downweighted
- Lead-lag check across adjacent weeks: no signal

Every method points the same direction: no detectable relationship between
ATM issuance size and abnormal return, before or after stripping BTC-beta.
"Dead" was the right call. This is the version of that call that survives
a change in test, sample window, or outlier handling -- which the original
single-test result does not.
