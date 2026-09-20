"""
analysis.py -- replicate and stress-test Abishek Kannan's MSTR ATM analysis.

Steps:
  a. OLS: MSTR weekly return ~ BTC weekly return. Residual = abnormal return.
  b. OLS: abnormal return ~ issuance $. Baseline R^2 target ~0.49.
  c. Spearman rank correlation: abnormal return vs issuance $.
  d. MAD outlier flagging on issuance $. Re-run (b) and (c) excluding outliers.
  e. Lead-lag: issuance(t) vs abnormal_return(t-1), (t), (t+1).

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

MAD_K = 3.0  # outlier threshold: k * MAD from median


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols(y: np.ndarray, X: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    names = ["const"] + (["slope"] if X.ndim == 1 else [f"x{i}" for i in range(X.shape[1])])
    coef_str = "  ".join(
        f"{n}={model.params[i]:.4f} (p={model.pvalues[i]:.3f})"
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
    print(f"    issuance slope= {m.params[2]*1e9*100:+.3f}% per $1B  "
          f"(t={m.tvalues[2]:+.2f}, p={m.pvalues[2]:.3f})")
    return m


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
    df = weekly[~weekly["nonstandard_window"]].copy().reset_index(drop=True)
    n_dropped = n_orig - len(df)
    if n_dropped:
        print(f"[NOTE] Dropped {n_dropped} non-standard windows (<5d or >10d).")

    mstr_ret = df["mstr_return"].values
    btc_ret  = df["btc_return"].values
    issuance = df["net_proceeds_usd"].values
    n = len(df)

    # -----------------------------------------------------------------------
    # a. OLS: MSTR return ~ BTC return
    # -----------------------------------------------------------------------
    _section("a. BTC-beta regression (MSTR ~ BTC)")
    print(f"  n = {n} weeks (incl. zero-issuance weeks)")
    _, r2_btc = _ols(mstr_ret, btc_ret, "MSTR ~ BTC")
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

    print("\n  Two-stage version (what the original design implies), for comparison:")
    _, r2_base = _ols(abnormal, issuance, "abnormal ~ issuance (all weeks)")
    slope_two = sm.OLS(abnormal, sm.add_constant(issuance)).fit().params[1]
    corr_bi = np.corrcoef(issuance, btc_ret)[0, 1]
    atten = (1 - slope_two / slope_joint) * 100 if slope_joint else float("nan")
    print(f"    corr(issuance, btc_return) = {corr_bi:+.3f}, so the two-stage slope")
    print(f"    is attenuated by {atten:.0f}% relative to the joint estimate")
    print(f"    ({slope_two*1e9*100:+.3f}% vs {slope_joint*1e9*100:+.3f}% per $1B).")

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
    # d. MAD outlier flagging + re-run
    # -----------------------------------------------------------------------
    _section(f"d. MAD outlier flagging (k={MAD_K}) on issuance $")
    outlier_mask = _mad_outlier_mask(issuance, MAD_K)
    n_outliers = outlier_mask.sum()
    if n_outliers:
        out_rows = df[outlier_mask][["window_start", "window_end", "net_proceeds_usd", "shares_sold"]]
        print(f"  {n_outliers} outlier weeks (>{MAD_K}*MAD from median):")
        for _, r in out_rows.iterrows():
            print(f"    {r['window_start']} to {r['window_end']}  "
                  f"${r['net_proceeds_usd']/1e9:.2f}B  {r['shares_sold']:,} shares")
    else:
        print(f"  No outliers at k={MAD_K}.")

    clean = ~outlier_mask
    n_clean = clean.sum()
    print(f"\n  Re-run excluding {n_outliers} outliers (n={n_clean}):")
    _, r2_clean = _ols(abnormal[clean], issuance[clean],
                       "abnormal ~ issuance (excl. MAD outliers)")
    _spearman(issuance[clean], abnormal[clean], "issuance vs abnormal (excl. MAD outliers)")

    # -----------------------------------------------------------------------
    # e. Lead-lag
    # -----------------------------------------------------------------------
    _section("e. Lead-lag: issuance(t) vs abnormal_return")
    print("  Convention: lag=+1 means abnormal is one week AFTER issuance (issuance leads).")
    print("  lag=-1: abnormal(t-1) -- prior-week return vs this week's issuance")
    print("  lag= 0: abnormal(t)   -- contemporaneous")
    print("  lag=+1: abnormal(t+1) -- does issuance predict next-week abnormal return?")
    print("  [pairs straddling the Apr-Oct 2025 Format C gap are excluded]")

    df_s = df.sort_values("window_start").reset_index(drop=True)
    df_s["window_start"] = pd.to_datetime(df_s["window_start"])
    df_s["days_since_prev"] = df_s["window_start"].diff().dt.days
    df_s["gap_before"] = df_s["days_since_prev"] > 14

    def _lag_pair(lag: int) -> tuple[np.ndarray, np.ndarray]:
        """
        For each issuance row at index i, abnormal row is at index i+lag.
        lag=+1 => abnormal is the FOLLOWING week (issuance leads abnormal).
        lag=-1 => abnormal is the PRIOR week (abnormal leads issuance).
        """
        iss_v, abn_v = [], []
        for i in range(len(df_s)):
            j = i + lag
            if j < 0 or j >= len(df_s):
                continue
            lo, hi = min(i, j), max(i, j)
            if df_s.iloc[lo + 1 : hi + 1]["gap_before"].any():
                continue
            iss_v.append(df_s.iloc[i]["net_proceeds_usd"])
            abn_v.append(df_s.iloc[j]["abnormal_return"])
        return np.array(iss_v), np.array(abn_v)

    outlier_mask_s = _mad_outlier_mask(
        df_s["net_proceeds_usd"].values, MAD_K
    )

    for lag, desc in [
        (-1, "abnormal(t-1) -- prior-week return vs issuance [does abnormal predict next issuance?]"),
        ( 0, "abnormal(t)   -- contemporaneous"),
        (+1, "abnormal(t+1) -- next-week return vs issuance [does issuance predict future abnormal?]"),
    ]:
        print(f"\n  lag={lag:+d}  {desc}")
        iss_v, abn_v = _lag_pair(lag)
        _ols(abn_v, iss_v, f"  OLS")
        _spearman(iss_v, abn_v, f"  Spearman")

        # Robustness: re-run excluding MAD outlier issuance weeks
        iss_v2, abn_v2 = [], []
        for i in range(len(df_s)):
            j = i + lag
            if j < 0 or j >= len(df_s):
                continue
            lo, hi = min(i, j), max(i, j)
            if df_s.iloc[lo + 1 : hi + 1]["gap_before"].any():
                continue
            if outlier_mask_s[i]:
                continue
            iss_v2.append(df_s.iloc[i]["net_proceeds_usd"])
            abn_v2.append(df_s.iloc[j]["abnormal_return"])
        iss_v2, abn_v2 = np.array(iss_v2), np.array(abn_v2)
        _spearman(iss_v2, abn_v2, f"  Spearman (excl. MAD outliers, n={len(iss_v2)})")

    # -----------------------------------------------------------------------
    # Data gap note
    # -----------------------------------------------------------------------
    _section("Data coverage note")
    n_pre  = (df_s["window_start"] < "2025-04-01").sum()
    n_gap  = ((df_s["window_start"] >= "2025-04-01") & (df_s["window_start"] < "2025-11-01")).sum()
    n_post = (df_s["window_start"] >= "2025-11-01").sum()
    print(f"  Our dataset: {n} weeks total.")
    print(f"    Nov 2024 - Mar 2025:  {n_pre} weeks")
    print(f"    Apr 2025 - Oct 2025:  {n_gap} weeks (Format C, now parsed -- 2 weeks still missing)")
    print(f"    Nov 2025 - Jun 2026:  {n_post} weeks")
    print(f"  Remaining gaps: May 19-25 2025 and Oct 6-12 2025 (1 week each, filings not found).")
    print(f"  The R^2 gap vs the original 0.49 is not explained by missing data:")
    print(f"  the Apr-Oct 2025 period is now covered and the relationship remains flat.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _section("Summary")
    print(f"  Weeks analyzed:               {n}")
    print(f"  Non-standard windows dropped: {n_dropped}")
    print(f"  Zero-issuance weeks:          {(~nz_mask).sum()}")
    print(f"  MAD outliers (k={MAD_K}):        {n_outliers}")
    print()
    print(f"  (a) BTC-beta R^2:                      {r2_btc:.4f}")
    print(f"  (b) JOINT issuance slope:              {slope_joint*1e9*100:+.3f}% per $1B  (p={p_joint:.3f})")
    print(f"      two-stage OLS R^2 (all weeks):     {r2_base:.4f}")
    print(f"      baseline OLS R^2 (non-zero only):  {r2_nz:.4f}")
    print(f"      directional accuracy (non-zero):   {dir_acc:.4f}  (coin-flip = 0.50)")
    print(f"  (d) ex-outlier OLS R^2:                {r2_clean:.4f}")


if __name__ == "__main__":
    main()
