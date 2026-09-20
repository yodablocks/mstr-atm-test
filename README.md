# MSTR ATM Abnormal Return Replication

Tests whether weekly ATM common stock issuance by Strategy Inc. (MSTR) shows
up as a detectable abnormal return, after stripping out BTC-beta exposure.

A public analysis of this question, using a single-pass OLS regression,
reported R^2 ~ 0.49 and called the relationship "dead." This attempts to
reproduce that result and then asks a question the original did not: given
89 weeks of data and the noise in MSTR's abnormal returns, what size of
effect could this test have detected at all?

Sample: 2024-11-11 to 2026-09-13, 89 weekly ATM disclosures.

---

## Headline result

The original statistic could not be reproduced, and the test is too weak to
support the conclusion drawn from it.

```
Joint regression, mstr_return ~ btc_return + issuance, n = 89 weeks

  issuance slope   -0.700% abnormal return per $1B
  95% CI           [-2.855%, +1.455%] per $1B
  p                0.520

  minimum detectable effect at 80% power:  3.04% per $1B
```

The interval is asymmetric, so its two ends answer different questions:

- A negative price impact **worse than -2.85% per $1B** is ruled out. Selling
  pressure predicts a negative sign, so this is the bound that matters for
  the question being asked.
- A positive effect **larger than +1.45% per $1B** is ruled out.
- Anything inside that interval is untestable here. A true effect below
  3.04% per $1B fails to reject zero most of the time at n = 89.

A 2% abnormal return per $1B issued is not a small effect, and this sample
cannot distinguish it from nothing.

### The point estimate is not stable

Three passes over substantially the same question:

```
  77 weeks, return window misaligned by one day   -1.01% per $1B
  77 weeks, window corrected                      +0.37% per $1B
  89 weeks, window corrected                      -0.70% per $1B
```

Every one of these sits comfortably inside every other one's confidence
interval. The sign of the headline estimate flips on a one-day alignment
choice and flips back on a twelve-week sample extension. That is what an
underpowered test looks like from the inside, and it is the strongest
argument in this repo against reading any single point estimate as a result.

### More data made the test weaker

Extending the sample from 77 to 89 weeks did not improve power:

```
                      77 weeks    89 weeks
  residual SD           5.18%       6.03%
  minimum detectable    2.70%       3.04%
```

The twelve added weeks are noisier than the average of the preceding
seventy-seven, and residual volatility enters the minimum detectable effect
faster than sample size reduces it. Collecting more weeks at this rate will
not settle the question on any useful timescale. That is an argument for
changing the design, not for waiting.

So "no detectable relationship" is correct. "No relationship" is not
supported. A null result here is the expected outcome across a wide range of
real, economically meaningful impacts, not only under zero impact.

---

## On the 0.49 figure

Our replication of the second-stage regression gives R^2 = 0.0035, not 0.49.
The gap is not explained by missing weeks: 7 of 96 calendar weeks in the span
are absent, and the relationship stays flat with or without them.

Worth stating plainly: an R^2 of 0.49 would be a *strong* relationship, not a
dead one. Roughly half the variance in abnormal returns explained by issuance
alone would be a remarkable finding. So the 0.49 is very unlikely to be the
R^2 of this regression. Two candidates were checked:

```
Pearson |r|, all weeks                      0.0588
directional hit rate, non-zero weeks        0.450   (coin flip = 0.50)
```

Neither lands on 0.49 either. Without the original's code or data the
statistic cannot be identified, so this repo does not claim to have
replicated it. It reports its own numbers and says where they came from.

---

## Method notes

**Return window.** The disclosed ATM window is inclusive of its first day. A
"June 1 to June 7" filing covers selling on June 1, so the return runs from
the close *before* the window opens to the close at its end. For a typical
Monday-to-Sunday window that is prior-Friday close to Friday close, five
trading days. Measuring from the Monday close instead drops the first day of
issuance from every observation and leaves a Friday-to-Monday hole between
consecutive weeks. This matters: the earlier version of this analysis did
exactly that, and its headline slope had the opposite sign.

**Endpoint matching.** MSTR and BTC returns span the same two calendar dates.
BTC trades on weekends and MSTR does not, so pairing a Sunday BTC close with
a Friday MSTR close injects two days of unhedged BTC move into every row.
Matching endpoints raises the BTC-beta R^2 to 0.642.

**Joint, not two-stage.** The issuance coefficient comes from one regression,
`mstr_return ~ btc_return + issuance`, rather than from regressing the BTC
residual on issuance. Issuance correlates with BTC returns at +0.52 in this
sample, since Strategy sells into strength, so the two-stage version leaves
the BTC component inside the regressor and attenuates the slope by 28%. This
also handles the fact that BTC-beta is estimated on a sample that includes
the event weeks, which a textbook event study would avoid by using a clean
estimation window. There is no clean window here: Strategy issued in 60 of
89 weeks.

**Robustness by downweighting, not deletion.** Issuance is heavily
right-skewed with a large mass at zero, so the concern about outlier
sensitivity is real. Deleting the large weeks is not the fix. A MAD rule at
k = 3 flags 17 of 89 weeks; the 72 survivors hold 32.5% of total proceeds.
Re-running on them and finding nothing is circular, because 67% of the
dollars under test are gone before the regression runs. That figure is
reported as a diagnostic and is not used as evidence. Robustness instead
comes from Huber M-estimation, a log1p specification, and p95 winsorization,
all retaining every row:

```
  Huber M-estimator      -0.318% per $1B   p = 0.665
  log1p(issuance $B)     -1.551% per unit  p = 0.390
  winsorized at p95      -1.273% per $1B   p = 0.311
```

All three agree with the joint estimate at approximately zero. Same
conclusion, reached without discarding the data.

**Rank correlation.** 29 of 89 weeks have zero issuance, which enters
Spearman as a large block of ties. The non-zero-only version is the one to
read: rho = -0.164, p = 0.212, n = 60.

**Lead-lag.** Contemporaneous, one week back, one week forward:

```
  lag  -1   rho = +0.099   p = 0.375   n = 82
  lag   0   rho = -0.082   p = 0.442   n = 89
  lag  +1   rho = +0.007   p = 0.949   n = 82
```

Nothing significant, and nothing close. Pairs that straddle a missing week
are excluded; there are 7 such weeks, and a prior version of this code
treated single-week gaps as contiguous.

**Multiple comparisons.** This script runs 20 issuance-related tests on one
dataset, so roughly one spurious p < 0.05 is expected by chance. The
Bonferroni threshold is p < 0.0025. No issuance test clears that, and none
clears an unadjusted 0.05 either. The BTC-beta regression in section a is
excluded from the count: it is the hedge specification rather than a test of
the issuance question, and it is significant at any threshold.

---

## The identification problem

Even with more data, the contemporaneous regression cannot identify price
impact. Strategy chooses when to issue, and it issues more after strong
weeks and when the mNAV premium is wide. Issuance is therefore endogenous to
returns, and the lag -1 result is consistent with exactly that. A design that
could actually answer the question needs an instrument for issuance, or
intraday data around the selling itself rather than weekly aggregates.

This repo does not have that. It establishes an upper bound on the weekly
effect and nothing more.

---

## Dependencies

See `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Run order

```bash
export EDGAR_USER_AGENT="Your Name your@email.com"   # required, no default
python pull_filings.py
python pull_prices.py
python build_dataset.py
python analysis.py
```

`data/weekly.csv` is committed, so `analysis.py` runs on a fresh clone
without any network access. The first two scripts only need re-running to
extend the sample.

EDGAR requires a `User-Agent` header with a real contact email and will
throttle or block requests without one. `pull_filings.py` exits rather than
sending a placeholder.

`pull_prices.py` hits Hyperliquid and Yahoo Finance directly and needs live
network access.

---

## Known limitations

- **Power, as above.** Effects below 3.04% per $1B in magnitude are
  undetectable here, and extending the sample has so far made this worse
  rather than better.
  This is the binding constraint on everything the repo can conclude.
- **Endogeneity, as above.** Issuance is not randomly assigned.
- ATM window dates are not always calendar-week aligned. The filing's own
  disclosed window is used rather than snapping to ISO weeks. Three
  1-to-2-day year-end stubs are dropped, leaving 77 of 80 filings.
- 7 of the 96 calendar weeks in the span are missing from the issuance
  dataset, across 6 breaks. `analysis.py` lists them by date. They are
  dropped rather than interpolated, and lead-lag pairs across them are
  excluded.
- Issuance is in raw dollars, not scaled by market cap or dollar volume.
  MSTR's market cap moves substantially across this window, so $1B is not a
  constant shock. Scaling would be the next improvement.
- BTC price is Hyperliquid perpetual futures close, not a spot exchange
  close.
- Coverage thins out before Nov 2024. The weekly 8-K disclosure cadence is a
  relatively recent practice for MSTR.
- Eleven windows end on a market holiday and resolve back to the prior
  trading day. Flagged as `price_subst_flag` in `data/weekly.csv`.
- Issuance figures are parsed out of 8-K exhibit tables whose layout changes
  without notice. `build_dataset.py` refuses to write the panel if any week's
  implied share price (net proceeds / shares sold) falls outside 0.70x to
  1.40x the MSTR close over the same window. This exists because a footnote
  marker appearing in the Net Proceeds cell in August 2026 filings shifted
  the positional column read to "Available for Issuance" and produced five
  weeks of $20B proceeds against real figures near $290M. The ratio is
  written to `weekly.csv` as `implied_px_ratio`; it currently spans 0.919 to
  1.153 with a median of 1.017.

---

## Conclusion

Every specification points the same direction: no detectable relationship
between ATM issuance size and abnormal return, before or after stripping
BTC-beta, contemporaneously or at a one-week lag, with or without robust
handling of the skew.

That is a weaker claim than "dead," and it is the one the data support. The
sample rules out a weekly price impact worse than -2.85% per $1B. Below that
threshold this test has nothing to say, and no amount of additional
robustness checking on the same 89 weeks will change it, because the limit is
statistical power rather than method choice. Neither will waiting: adding
twelve weeks moved the minimum detectable effect the wrong way.

If the question is worth settling, the next step is not another test on this
dataset. It is intraday data around the selling windows, or a design that
addresses the fact that Strategy chooses when to sell.
