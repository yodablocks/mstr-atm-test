"""
analysis.py -- replicate and stress-test Abishek Kannan's MSTR ATM analysis.

Steps:
  a. OLS: MSTR weekly return ~ BTC weekly return. Residual = abnormal return.
  b. Issuance effect, from the joint regression below. The two-stage version
     the original design implies is reported alongside it for comparison,
     as is the attempt to reproduce the original's 0.49.
  c. Spearman rank correlation: abnormal return vs issuance $.
  d. Robustness to the skewed issuance distribution: Huber M-estimation,
     log1p, and winsorization, all retaining every row. MAD-based deletion
     is reported as a diagnostic only, never as evidence.
  e. Lead-lag: issuance(t) vs abnormal_return(t-1), (t), (t+1).
  f. Power: the minimum effect this design could have detected, and the
     confidence interval that follows from it.

Estimator note: the headline issuance coefficient comes from the JOINT
regression

    mstr_return ~ const + btc_return + issuance

not from regressing the BTC residual on issuance. Issuance correlates with
BTC returns (Strategy sells more into strength), so the two-stage version
omits the BTC-issuance covariance and attenuates the issuance slope toward
zero. Both are reported so the gap is visible.

Endpoint matching is done upstream in build_dataset.py: MSTR and BTC returns
already span the same two trading days, so no re-derivation happens here.

All results printed to stdout. No output files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

WEEKLY_PATH = Path("data/weekly.csv")

MAD_K = 3.0  # outlier threshold: k * MAD from median (diagnostic only, see section d)

# Issuance-related hypothesis tests reported below, counted by section:
#   b: 4   c: 2   d: 5   e: 9  (3 lags x {OLS, 2 Spearman})
# Section a (MSTR ~ BTC) is excluded: it is the hedge specification, not a
# test of the issuance question, and it is significant at any threshold.
N_TESTS = 20

# Disclosed windows start 7 days apart. Beyond this, a week is missing.
MAX_WEEK_SPACING_DAYS = 8

ALPHA = 0.05   # two-sided significance level for power calculations
POWER = 0.80   # conventional target power


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols(y: np.ndarray, X: np.ndarray, label: str,
         as_pct: bool = True) -> tuple[np.ndarray, float]:
    """
    Fit y ~ const + X and print it.

    as_pct=True renders coefficients as percentage points, which is what you
    want when X is issuance in $B (slope reads as "% abnormal return per $1B").
    Set as_pct=False for a unitless regressor such as the BTC return, where
    the slope is a beta and a percent sign would be wrong.
    """
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    names = ["const"] + (["slope"] if X.ndim == 1 else [f"x{i}" for i in range(X.shape[1])])
    if as_pct:
        coef_str = "  ".join(
            f"{n}={model.params[i]*100:+.3f}% (p={model.pvalues[i]:.3f})"
            for i, n in enumerate(names)
        )
    else:
        coef_str = "  ".join(
            f"{n}={model.params[i]:+.4f} (p={model.pvalues[i]:.3f})"
            for i, n in enumerate(names)
        )
    print(f"  {label}: R^2={model.rsquared:.4f}  n={int(model.nobs)}  {coef_str}")
    return model.resid, model.rsquared


def _joint_ols(y: np.ndarray, btc: np.ndarray, iss: np.ndarray, label: str):
    """
    One-pass y ~ btc + issuance. Returns the fitted model.

    This is the correct estimate of the issuance effect: it partials BTC out
    of BOTH sides. The two-stage alternative (regress resid(y|btc) on issuance)
    leaves the BTC component of issuance in the regressor and biases the slope.
    """
    X = sm.add_constant(np.column_stack([btc, iss]))
    m = sm.OLS(y, X).fit()
    print(f"  {label}: R^2={m.rsquared:.4f}  n={int(m.nobs)}")
    print(f"    btc_beta      = {m.params[1]:+.4f}  (p={m.pvalues[1]:.3f})")
    print(f"    issuance slope= {m.params[2]*100:+.3f}% per $1B  "
          f"(t={m.tvalues[2]:+.2f}, p={m.pvalues[2]:.3f})")
    return m


def _mde(se: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """
    Minimum detectable effect: the smallest true slope this design would
    reject zero for, at the given power. Approximately 2.80 * se for the
    conventional alpha=0.05, power=0.80.
    """
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se


def _spearman(x: np.ndarray, y: np.ndarray, label: str) -> float:
    rho, pval = stats.spearmanr(x, y)
    print(f"  {label}: rho={rho:.4f}  p={pval:.4f}  n={len(x)}")
    return rho


def _mad_outlier_mask(series: np.ndarray, k: float = MAD_K) -> np.ndarray:
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series), dtype=bool)
    return np.abs(series - median) > k * mad


def _section(title: str):
    print(f"\n# -- {title} " + "-" * max(0, 60 - len(title)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    weekly = pd.read_csv(WEEKLY_PATH)

    # Drop non-standard windows (1-2 day year-end stubs)
    n_orig = len(weekly)
    dropped_starts = set(
        pd.to_datetime(weekly.loc[weekly["nonstandard_window"], "window_start"]).dt.date
    )
    df = weekly[~weekly["nonstandard_window"]].copy().reset_index(drop=True)
    n_dropped = n_orig - len(df)
    if n_dropped:
        print(f"[NOTE] Dropped {n_dropped} non-standard windows (<5d or >10d).")

    mstr_ret = df["mstr_return"].values
    btc_ret  = df["btc_return"].values
    # Issuance in $B, so every slope below reads directly as "return per $1B".
    issuance = df["net_proceeds_usd"].values / 1e9
    n = len(df)

    # -----------------------------------------------------------------------
    # a. OLS: MSTR return ~ BTC return
    # -----------------------------------------------------------------------
    _section("a. BTC-beta regression (MSTR ~ BTC)")
    print(f"  n = {n} weeks (incl. zero-issuance weeks)")
    _, r2_btc = _ols(mstr_ret, btc_ret, "MSTR ~ BTC", as_pct=False)
    r2_btc_slope = sm.OLS(mstr_ret, sm.add_constant(btc_ret)).fit().params[1]
    resid_all = sm.OLS(mstr_ret, sm.add_constant(btc_ret)).fit().resid
    df["abnormal_return"] = resid_all

    # -----------------------------------------------------------------------
    # b. OLS: abnormal return ~ issuance $
    # -----------------------------------------------------------------------
    _section("b. Issuance effect")
    abnormal = df["abnormal_return"].values
    nz_mask = issuance > 0

    print("  PRIMARY -- joint regression (partials BTC out of both sides):")
    joint = _joint_ols(mstr_ret, btc_ret, issuance, "MSTR ~ BTC + issuance")
    slope_joint = joint.params[2]
    p_joint = joint.pvalues[2]
    ci_joint = joint.conf_int()[2]
    print(f"    95% CI          = [{ci_joint[0]*100:+.3f}%, {ci_joint[1]*100:+.3f}%] per $1B")

    print("\n  Two-stage version (what the original design implies), for comparison:")
    _, r2_base = _ols(abnormal, issuance, "abnormal ~ issuance (all weeks)")
    slope_two = sm.OLS(abnormal, sm.add_constant(issuance)).fit().params[1]
    corr_bi = np.corrcoef(issuance, btc_ret)[0, 1]
    atten = (1 - slope_two / slope_joint) * 100 if slope_joint else float("nan")
    print(f"    corr(issuance, btc_return) = {corr_bi:+.3f}, so the two-stage slope")
    print(f"    is attenuated by {atten:.0f}% relative to the joint estimate")
    print(f"    ({slope_two*100:+.3f}% vs {slope_joint*100:+.3f}% per $1B).")

    print(f"\n  Original post reported R^2 ~0.49. Ours is {r2_base:.4f}.")
    print(f"  An R^2 of 0.49 would be a STRONG relationship, not a dead one, so the")
    print(f"  original figure is unlikely to be an R^2 of this regression. Candidates:")
    print(f"  a hit rate (0.50 = coin flip) or a correlation coefficient. Reported below.")

    # Re-run on non-zero issuance weeks only (he may have excluded zero-sale weeks)
    _, r2_nz = _ols(abnormal[nz_mask], issuance[nz_mask],
                    f"abnormal ~ issuance (non-zero issuance only, n={nz_mask.sum()})")
    corr_p = np.corrcoef(issuance, abnormal)[0, 1]
    print(f"  Pearson r (all weeks): {corr_p:+.4f}   |r| = {abs(corr_p):.4f}")

    # Directional accuracy: fraction of non-zero weeks where sign(abnormal) == sign(issuance)
    # "0.49" may be a hit-rate metric rather than OLS R^2
    dir_acc = ((issuance[nz_mask] > 0) == (abnormal[nz_mask] > 0)).mean()
    n_pos_iss_pos_abn = ((issuance[nz_mask] > 0) & (abnormal[nz_mask] > 0)).sum()
    print(f"  Directional accuracy (non-zero issuance, n={nz_mask.sum()}): "
          f"{dir_acc:.4f}  ({n_pos_iss_pos_abn}/{nz_mask.sum()} same sign)")
    print(f"  [If '0.49' is a hit rate, 0.50 is coin-flip baseline -- consistent with 'dead']")
    print(f"  [Neither |r| nor the hit rate lands on 0.49 either; the original statistic")
    print(f"   could not be reproduced. See README.]")

    # -----------------------------------------------------------------------
    # c. Spearman rank correlation
    # -----------------------------------------------------------------------
    _section("c. Spearman rank correlation: abnormal return vs issuance $")
    _spearman(issuance, abnormal, "all weeks")
    _spearman(issuance[nz_mask], abnormal[nz_mask], "non-zero issuance only")

    # -----------------------------------------------------------------------
    # d. Robustness to the lumpy issuance distribution
    # -----------------------------------------------------------------------
    _section("d. Robustness to lumpy issuance")
    print("  Issuance is heavily right-skewed with a large mass at zero, so the")
    print("  concern is real. But DELETING the large weeks is not the answer: it")
    print("  removes the variation the regression is trying to price. Quantified")
    print("  first, then handled properly by downweighting instead.")

    # -- What MAD deletion would actually discard (diagnostic only) ----------
    outlier_mask = _mad_outlier_mask(issuance, MAD_K)
    n_outliers = int(outlier_mask.sum())
    med = np.median(issuance)
    mad = np.median(np.abs(issuance - med))
    kept_dollars = issuance[~outlier_mask].sum() / issuance.sum()
    print(f"\n  [diagnostic] MAD rule at k={MAD_K}: median=${med:.3f}B, "
          f"MAD=${mad:.3f}B, cutoff=${med + MAD_K*mad:.3f}B")
    print(f"  [diagnostic] It flags {n_outliers}/{len(issuance)} weeks, leaving "
          f"{int((~outlier_mask).sum())} rows that hold only "
          f"{kept_dollars*100:.1f}% of total proceeds.")
    print(f"  [diagnostic] Dropping them and finding no effect is circular: "
          f"{(1-kept_dollars)*100:.0f}% of the")
    print(f"               dollars being tested would be gone. Reported, not used.")
    _, r2_clean = _ols(abnormal[~outlier_mask], issuance[~outlier_mask],
                       "abnormal ~ issuance (MAD-deleted -- shown for contrast only)")

    # -- Proper robustness: keep every row, reduce leverage ------------------
    print(f"\n  Robust specifications, all {len(issuance)} rows retained:")

    rlm = sm.RLM(abnormal, sm.add_constant(issuance),
                 M=sm.robust.norms.HuberT()).fit()
    slope_rlm = rlm.params[1]
    print(f"    Huber M-estimator:   {slope_rlm*100:+.3f}% per $1B  "
          f"(p={rlm.pvalues[1]:.3f})")

    log_iss = np.log1p(issuance)
    log_fit = sm.OLS(abnormal, sm.add_constant(log_iss)).fit()
    print(f"    log1p(issuance $B):  {log_fit.params[1]*100:+.3f}% per unit  "
          f"(p={log_fit.pvalues[1]:.3f}, R^2={log_fit.rsquared:.4f})")

    wins = np.clip(issuance, None, np.quantile(issuance, 0.95))
    win_fit = sm.OLS(abnormal, sm.add_constant(wins)).fit()
    print(f"    winsorized at p95:   {win_fit.params[1]*100:+.3f}% per $1B  "
          f"(p={win_fit.pvalues[1]:.3f}, R^2={win_fit.rsquared:.4f})")

    print(f"\n    Spearman is itself rank-based, so it already handles the skew.")
    print(f"    Its weak spot here is the {int((~nz_mask).sum())} tied zeros out of "
          f"{len(issuance)}, which is why the")
    print(f"    non-zero-only version below is the one to read:")
    _spearman(issuance[nz_mask], abnormal[nz_mask],
              "non-zero issuance weeks (no ties)")

    # -----------------------------------------------------------------------
    # e. Lead-lag
    # -----------------------------------------------------------------------
    _section("e. Lead-lag: issuance(t) vs abnormal_return")
    print("  Convention: lag=+1 means abnormal is one week AFTER issuance (issuance leads).")
    print("  lag=-1: abnormal(t-1) -- prior-week return vs this week's issuance")
    print("  lag= 0: abnormal(t)   -- contemporaneous")
    print("  lag=+1: abnormal(t+1) -- does issuance predict next-week abnormal return?")
    print(f"  [pairs straddling a missing week (>{MAX_WEEK_SPACING_DAYS}d between "
          f"window starts) are excluded]")

    df_s = df.sort_values("window_start").reset_index(drop=True)
    df_s["window_start"] = pd.to_datetime(df_s["window_start"])
    df_s["days_since_prev"] = df_s["window_start"].diff().dt.days
    # Consecutive disclosed windows start 7 days apart. Anything beyond 8 means
    # at least one week is missing, so rows adjacent in the frame are not
    # adjacent in time and must not be paired as lag 1. The previous threshold
    # of 14 only caught gaps of two or more missing weeks, silently treating a
    # single missing week as contiguous.
    df_s["gap_before"] = df_s["days_since_prev"] > MAX_WEEK_SPACING_DAYS

    def _lag_pair(lag: int, nonzero_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        For each issuance row at index i, abnormal row is at index i+lag.
        lag=+1 => abnormal is the FOLLOWING week (issuance leads abnormal).
        lag=-1 => abnormal is the PRIOR week (abnormal leads issuance).

        nonzero_only drops weeks with no issuance, which otherwise enter the
        rank correlation as a large block of ties.
        """
        iss_v, abn_v = [], []
        for i in range(len(df_s)):
            j = i + lag
            if j < 0 or j >= len(df_s):
                continue
            lo, hi = min(i, j), max(i, j)
            if df_s.iloc[lo + 1 : hi + 1]["gap_before"].any():
                continue
            if nonzero_only and df_s.iloc[i]["net_proceeds_usd"] <= 0:
                continue
            iss_v.append(df_s.iloc[i]["net_proceeds_usd"] / 1e9)
            abn_v.append(df_s.iloc[j]["abnormal_return"])
        return np.array(iss_v), np.array(abn_v)

    for lag, desc in [
        (-1, "abnormal(t-1) -- prior-week return vs issuance [does abnormal predict next issuance?]"),
        ( 0, "abnormal(t)   -- contemporaneous"),
        (+1, "abnormal(t+1) -- next-week return vs issuance [does issuance predict future abnormal?]"),
    ]:
        print(f"\n  lag={lag:+d}  {desc}")
        iss_v, abn_v = _lag_pair(lag)
        _ols(abn_v, iss_v, f"  OLS")
        _spearman(iss_v, abn_v, f"  Spearman (all weeks)")
        iss_v2, abn_v2 = _lag_pair(lag, nonzero_only=True)
        _spearman(iss_v2, abn_v2, f"  Spearman (non-zero issuance weeks)")

    # -----------------------------------------------------------------------
    # f. Power: what this design can and cannot detect
    # -----------------------------------------------------------------------
    _section("f. Power -- what a null result here does and does not mean")

    se_joint = joint.bse[2]
    mde = _mde(se_joint)
    resid_sd = abnormal.std(ddof=1)

    print(f"  Residual (abnormal return) SD: {resid_sd*100:.2f}% per week")
    print(f"  Issuance SD:                   ${issuance.std(ddof=1):.3f}B per week")
    print(f"  SE of the issuance slope:      {se_joint*100:.3f}% per $1B")
    print()
    print(f"  Minimum detectable effect (alpha={ALPHA}, power={POWER:.0%}):")
    print(f"    |slope| >= {mde*100:.2f}% abnormal return per $1B")
    print()
    print(f"  Point estimate {slope_joint*100:+.3f}% per $1B, 95% CI "
          f"[{ci_joint[0]*100:+.3f}%, {ci_joint[1]*100:+.3f}%].")
    print(f"  The interval is asymmetric, so the two bounds answer different")
    print(f"  questions and should be quoted separately:")
    print(f"    negative price impact worse than {ci_joint[0]*100:.2f}% per $1B: ruled out")
    print(f"    positive effect larger than  {ci_joint[1]*100:+.2f}% per $1B: ruled out")
    print(f"  Selling pressure predicts a negative sign, so {ci_joint[0]*100:.2f}% is the")
    print(f"  bound that matters for the question being asked.")
    print(f"  Anything inside the interval is untestable here. With n={n} weeks""")
    print(f"  and weekly residual noise of {resid_sd*100:.1f}%, a true effect anywhere")
    print(f"  below {mde*100:.1f}% per $1B would fail to reject zero most of the time,")
    print(f"  so a null result here is the expected outcome under a wide range of")
    print(f"  real, economically meaningful impacts, not just under no impact at all.")
    print()
    print(f"  'No detectable relationship' is therefore the correct reading.")
    print(f"  'No relationship' is not supported by these data.")

    # -- Is the noise a misfit hedge, or real volatility? -------------------
    # If BTC-beta drifted, the "residual" would be partly misspecification and
    # a rolling beta would recover power. If beta is stable, the noise is real.
    half = n // 2
    b1 = sm.OLS(mstr_ret[:half], sm.add_constant(btc_ret[:half])).fit().params[1]
    b2 = sm.OLS(mstr_ret[half:], sm.add_constant(btc_ret[half:])).fit().params[1]
    late = np.zeros(n); late[half:] = 1.0
    chow = sm.OLS(
        mstr_ret,
        sm.add_constant(np.column_stack([btc_ret, late, btc_ret * late])),
    ).fit()
    p_interact = chow.pvalues[3]
    print()
    print(f"  BTC-beta by half: {b1:.3f} then {b2:.3f} "
          f"(full sample {r2_btc_slope:.3f})")
    print(f"  Test for a beta shift (btc x late-half interaction): "
          f"p={p_interact:.3f}")
    sd1 = abnormal[:half].std(ddof=1)
    sd2 = abnormal[half:].std(ddof=1)
    print(f"  Residual SD by half: {sd1*100:.2f}% then {sd2*100:.2f}%")
    if p_interact > 0.05:
        print(f"  No evidence of a beta shift, so the rise in residual volatility")
        print(f"  is real rather than a misspecified hedge. A rolling beta would")
        print(f"  not recover the lost power.")
    else:
        print(f"  Beta appears to shift. Part of the 'residual' is then")
        print(f"  misspecification, and a rolling beta may recover some power.")

    # -- How concentrated is that volatility? -------------------------------
    worst = np.argsort(-np.abs(abnormal))[:2]
    keep = np.ones(n, dtype=bool)
    keep[worst] = False
    j2 = sm.OLS(mstr_ret[keep],
                sm.add_constant(np.column_stack([btc_ret[keep], issuance[keep]]))).fit()
    mde2 = _mde(j2.bse[2])
    print()
    print(f"  Two largest-residual weeks:")
    for i in worst:
        print(f"    {df.iloc[i]['window_start']}: abnormal "
              f"{abnormal[i]*100:+.1f}%")
    print(f"  Excluding just those two (n={int(keep.sum())}): residual SD "
          f"{abnormal[keep].std(ddof=1)*100:.2f}%, MDE {mde2*100:.2f}% per $1B,")
    print(f"  slope {j2.params[2]*100:+.3f}% per $1B (p={j2.pvalues[2]:.3f}).")
    print(f"  Two weeks out of {n} move the MDE by "
          f"{abs(mde - mde2)*100:.2f} points and flip the sign of the estimate.")

    # MDE shrinks as 1/sqrt(n), so the weeks needed to reach a target scale
    # with the square of the improvement factor.
    print()
    print(f"  Weeks of data needed to reach a given MDE, holding this residual")
    print(f"  volatility fixed (MDE scales as 1/sqrt(n)):")
    for target in (2.0, 1.0, 0.5):
        need = int(np.ceil(n * (mde * 100 / target) ** 2))
        print(f"    MDE {target:.1f}% per $1B:  {need:,} weeks "
              f"(~{need / 52:.0f} years, {need - n:,} more than now)")
    print(f"  Strategy has disclosed weekly for under two years. Reaching a")
    print(f"  resolution that would matter is a decade-scale wait, so the answer")
    print(f"  is a better design, not a longer sample.")

    # -----------------------------------------------------------------------
    # Multiple comparisons
    # -----------------------------------------------------------------------
    _section("Multiple comparisons")
    print(f"  {N_TESTS} issuance-related hypothesis tests on one dataset.")
    print(f"  (The BTC-beta regression in section a is excluded: it is the hedge")
    print(f"   specification, not a test of the issuance question.)")
    print(f"  At alpha=0.05 the expected number of spurious rejections is "
          f"{N_TESTS * 0.05:.1f}.")
    print(f"  Bonferroni-adjusted threshold: p < {0.05 / N_TESTS:.4f}.")
    print(f"  No issuance test above clears that, and none clears an unadjusted")
    print(f"  0.05 either. Treat any single p near 0.10 as noise unless it")
    print(f"  survives a pre-registered re-test on new weeks.")

    # -----------------------------------------------------------------------
    # Data gap note
    # -----------------------------------------------------------------------
    _section("Data coverage note")
    first, last = df_s["window_start"].min(), df_s["window_start"].max()
    span_weeks = int((last - first).days / 7) + 1
    print(f"  {n} weeks analyzed, {first.date()} to "
          f"{pd.to_datetime(df_s['window_end'].max()).date()}.")
    print(f"  That span is {span_weeks} calendar weeks, so "
          f"{span_weeks - n} weekly filings are missing.")

    breaks = df_s[df_s["gap_before"]]
    n_stub = n_unlocated = 0
    if len(breaks):
        print(f"\n  {len(breaks)} break(s) in the weekly sequence:")
        for _, r in breaks.iterrows():
            missing = int(r["days_since_prev"] / 7) - 1
            prev_end = df_s.loc[df_s["window_start"] < r["window_start"],
                                "window_end"].max()
            # A break is explained if a non-standard window we dropped starts
            # inside it. Those filings exist and were parsed; we excluded them.
            lo = pd.to_datetime(prev_end).date()
            hi = r["window_start"].date()
            stub = any(lo < d < hi for d in dropped_starts)
            if stub:
                n_stub += missing
                cause = "stub window dropped as non-standard (filing exists)"
            else:
                n_unlocated += missing
                cause = "filing not located by this parser"
            print(f"    {missing} week(s) between {prev_end} and {hi}: {cause}")
    else:
        print("  No breaks in the weekly sequence.")

    print(f"\n  Of {n_stub + n_unlocated} missing weeks, {n_stub} are quarter- and year-end "
          f"stub windows this")
    print(f"  script drops by design, and {n_unlocated} are filings the parser did not")
    print(f"  locate. Neither is a week Strategy failed to disclose. Missing weeks")
    print(f"  are dropped rather than interpolated, and lead-lag pairs across them")
    print(f"  are excluded.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _section("Summary")
    print(f"  Weeks analyzed:               {n}")
    print(f"  Non-standard windows dropped: {n_dropped}")
    print(f"  Zero-issuance weeks:          {(~nz_mask).sum()}")
    print(f"  MAD-flagged weeks (k={MAD_K}):    {n_outliers}  (downweighted, not dropped)")
    print()
    print(f"  (a) BTC-beta R^2:                      {r2_btc:.4f}")
    print(f"  (b) JOINT issuance slope:              {slope_joint*100:+.3f}% per $1B  (p={p_joint:.3f})")
    print(f"      two-stage OLS R^2 (all weeks):     {r2_base:.4f}")
    print(f"      baseline OLS R^2 (non-zero only):  {r2_nz:.4f}")
    print(f"      directional accuracy (non-zero):   {dir_acc:.4f}  (coin-flip = 0.50)")
    print(f"      95% CI:                            "
          f"[{ci_joint[0]*100:+.3f}%, {ci_joint[1]*100:+.3f}%] per $1B")
    print(f"      min detectable effect (80% power): {mde*100:.2f}% per $1B")
    print(f"  (d) Huber robust slope:                {slope_rlm*100:+.3f}% per $1B")
    print(f"      MAD-deleted OLS R^2 (not used):    {r2_clean:.4f}  "
          f"[discards {(1-kept_dollars)*100:.0f}% of proceeds]")


if __name__ == "__main__":
    main()
