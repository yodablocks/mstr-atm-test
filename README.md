# MSTR ATM Abnormal Return Replication

Tests whether weekly ATM common stock issuance by Strategy Inc. (MSTR) shows
up as a detectable abnormal return, after stripping out BTC-beta exposure.

A public analysis of this question, using a single-pass OLS regression,
reported R^2 ~ 0.49 and called the relationship "dead." This attempts to
reproduce that result and then asks a question the original did not: given
77 weeks of data and the noise in MSTR's abnormal returns, what size of
effect could this test have detected at all?

---

## Headline result

The original statistic could not be reproduced, and the test is too weak to
support the conclusion drawn from it.

```
Joint regression, mstr_return ~ btc_return + issuance, n = 77 weeks

  issuance slope   +0.365% abnormal return per $1B
  95% CI           [-1.556%, +2.286%] per $1B
  p                0.706

  minimum detectable effect at 80% power:  2.70% per $1B
```

The confidence interval contains price impacts up to about 2.3% per $1B in
either direction. That is not a small number. The honest statement is:

- The data **rule out** a contemporaneous price impact larger than roughly
  2.3% per $1B issued.
- The data **say nothing** about any effect smaller than that, because the
  design cannot detect one. A true effect below 2.70% per $1B fails to reject
  zero most of the time at n = 77.

So "no detectable relationship" is correct. "No relationship" is not
supported. A null result here is the expected outcome across a wide range of
real, economically meaningful impacts, not only under zero impact.

---

## On the 0.49 figure

Our replication of the second-stage regression gives R^2 = 0.0014, not 0.49.
The gap is not explained by missing weeks: the Apr-Oct 2025 period that was
initially unparsed is now covered, and the relationship stays flat.

Worth stating plainly: an R^2 of 0.49 would be a *strong* relationship, not a
dead one. Roughly half the variance in abnormal returns explained by issuance
alone would be a remarkable finding. So the 0.49 is very unlikely to be the
R^2 of this regression. Two candidates were checked:

```
Pearson |r|, all weeks                      0.0377
directional hit rate, non-zero weeks        0.451   (coin flip = 0.50)
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
Matching endpoints raises the BTC-beta R^2 to 0.686.

**Joint, not two-stage.** The issuance coefficient comes from one regression,
`mstr_return ~ btc_return + issuance`, rather than from regressing the BTC
residual on issuance. Issuance correlates with BTC returns at +0.52 in this
sample, since Strategy sells into strength, so the two-stage version leaves
the BTC component inside the regressor and attenuates the slope by 27%. This
also handles the fact that BTC-beta is estimated on a sample that includes
the event weeks, which a textbook event study would avoid by using a clean
estimation window. There is no clean window here: Strategy issued in 51 of
77 weeks.

**Robustness by downweighting, not deletion.** Issuance is heavily
right-skewed with a large mass at zero, so the concern about outlier
sensitivity is real. Deleting the large weeks is not the fix. A MAD rule at
k = 3 flags 22 of 77 weeks; the 55 survivors hold 16.3% of total proceeds.
Re-running on them and finding nothing is circular, because 84% of the
dollars under test are gone before the regression runs. That figure is
reported as a diagnostic and is not used as evidence. Robustness instead
comes from Huber M-estimation, a log1p specification, and p95 winsorization,
all retaining every row:

```
  Huber M-estimator      +0.092% per $1B   p = 0.894
  log1p(issuance $B)     +0.322% per unit  p = 0.845
  winsorized at p95      +0.007% per $1B   p = 0.995
```

All three agree with the joint estimate at approximately zero. Same
conclusion, reached without discarding the data.

**Rank correlation.** 26 of 77 weeks have zero issuance, which enters
Spearman as a large block of ties. The non-zero-only version is the one to
read: rho = -0.106, p = 0.459, n = 51.

**Lead-lag.** Contemporaneous, one week back, one week forward:

```
  lag  -1   rho = +0.178   p = 0.127   n = 75
  lag   0   rho = -0.007   p = 0.952   n = 77
  lag  +1   rho = -0.088   p = 0.453   n = 75
```

Nothing significant. The mildly suggestive one is lag -1, and its sign points
the wrong way for price impact: strong abnormal returns *precede* larger
issuance. That is the endogeneity channel, not an effect of selling.

**Multiple comparisons.** This script runs 21 tests on one dataset, so
roughly one spurious p < 0.05 is expected by chance. The Bonferroni threshold
is p < 0.0023. Nothing above clears that, and nothing above clears an
unadjusted 0.05 either.

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

- **Power, as above.** Effects below 2.70% per $1B are undetectable here.
  This is the binding constraint on everything the repo can conclude.
- **Endogeneity, as above.** Issuance is not randomly assigned.
- ATM window dates are not always calendar-week aligned. The filing's own
  disclosed window is used rather than snapping to ISO weeks. Three
  1-to-2-day year-end stubs are dropped, leaving 77 of 80 filings.
- Two 2025 weeks are missing from the issuance dataset: May 19-25 and
  Oct 6-12. Filings were not located.
- Issuance is in raw dollars, not scaled by market cap or dollar volume.
  MSTR's market cap moves substantially across this window, so $1B is not a
  constant shock. Scaling would be the next improvement.
- BTC price is Hyperliquid perpetual futures close, not a spot exchange
  close.
- Coverage thins out before Nov 2024. The weekly 8-K disclosure cadence is a
  relatively recent practice for MSTR.
- Nine windows end on a market holiday and resolve back to the prior trading
  day. Flagged as `price_subst_flag` in `data/weekly.csv`.

---

## Conclusion

Every specification points the same direction: no detectable relationship
between ATM issuance size and abnormal return, before or after stripping
BTC-beta, contemporaneously or at a one-week lag, with or without robust
handling of the skew.

That is a weaker claim than "dead," and it is the one the data support. The
sample rules out a weekly price impact above roughly 2.3% per $1B. Below
that, this test has nothing to say, and no amount of additional robustness
checks on the same 77 weeks will change that, because the limit is
statistical power rather than method choice.

If the question is worth settling, the next step is not another test on this
dataset. It is intraday data around the selling windows, or a design that
addresses the fact that Strategy chooses when to sell.
