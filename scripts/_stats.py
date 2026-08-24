"""
_stats.py — shared estimation helpers for the CAR regressions in RQ2 and RQ3.

This module is deliberately NOT one of the numbered research-question scripts.
It exists so that the RQ2 average-effect regression and the RQ3 bridge
regressions cannot drift apart: both clusters, both absorption checks and the
identification guard live here once.

Contents
    stars              significance markers
    fit_cluster        OLS with one-way or two-way cluster-robust SEs
    absorption         how much of a regressor the fixed effects absorb
    check_identified   raises NotIdentified rather than returning a
                       pseudo-inverse artefact
    n_bad_se           count of NaN SEs (two-way clustering is not PSD in small
                       samples, and we report that rather than hide it)
    mde                minimum detectable effect at 80% power
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import ABSORBED_R2, POWER_MULT


class NotIdentified(RuntimeError):
    """The term of interest is collinear with the rest of the design. We stop and
    report rather than printing whatever the pseudo-inverse happens to return."""


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "(n/a)"
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(n.s.)"


def fit_cluster(df: pd.DataFrame, formula: str, groups: list[str],
                use_t: bool = True):
    """OLS with cluster-robust standard errors.

    groups = ['permno']            cluster on the bank
             ['doc_id']            cluster on the event
             ['permno', 'doc_id']  two-way (Cameron-Gelbach-Miller)

    use_t=True reads inference off t with G-1 degrees of freedom. With 11 event
    clusters the normal reference is far too generous, so this is the default.

    The two-way estimator subtracts an intersection term and is NOT guaranteed
    positive semi-definite: in this sample some fixed-effect dummies come out
    with a negative variance and therefore a NaN SE. Callers should report
    n_bad_se() alongside, and should treat the EVENT-clustered SE as the headline
    inference.
    """
    g = df[groups[0]].values if len(groups) == 1 else df[groups].values
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": g}, use_t=use_t)


def n_bad_se(m) -> int:
    return int(np.sum(~np.isfinite(np.asarray(m.bse, dtype=float))))


def absorption(df: pd.DataFrame, target: str, rhs: str) -> dict:
    """Regress `target` on `rhs` (normally the fixed effects) and report how much
    is left. R^2 = 1 to machine precision means the term is perfectly absorbed
    and cannot be estimated alongside rhs."""
    m = smf.ols(f"{target} ~ {rhs}", data=df).fit()
    r = np.asarray(m.resid, dtype=float)
    return {"r2": float(m.rsquared), "max_abs": float(np.abs(r).max()),
            "sd": float(r.std(ddof=0)), "absorbed": float(m.rsquared) >= ABSORBED_R2}


def check_identified(df: pd.DataFrame, term: str, others: str) -> dict:
    """Verify the term of interest retains variation once every other regressor
    in the spec is partialled out. Raises NotIdentified if it does not."""
    a = absorption(df, term, others)
    if a["absorbed"]:
        raise NotIdentified(
            f"{term} has R^2 = {a['r2']:.12f} on the remaining regressors "
            f"({others}); residual SD {a['sd']:.3e}. Not identified — not reported.")
    return a


def mde(se: float, sd_regressor: float = 1.0, car_sd: float | None = None) -> dict:
    """Minimum detectable effect at two-sided 5% and 80% power."""
    raw = POWER_MULT * se
    per_sd = raw * sd_regressor
    out = {"raw": raw, "per_sd": per_sd, "per_sd_pp": per_sd * 100}
    if car_sd:
        out["pct_of_car_sd"] = 100 * per_sd / car_sd
    return out


def coef_row(lab: str, m, term: str, width: int = 30) -> str:
    """One formatted line: coefficient, SE, t, p, 95% CI, stars. Unrounded."""
    ci = m.conf_int()
    return ("    {:<{w}s} {:>14.8f} {:>13.8f} {:>8.3f} {:>10.6f}  "
            "[{:+.8f}, {:+.8f}]  {}").format(
        lab, m.params[term], m.bse[term], m.tvalues[term], m.pvalues[term],
        ci.loc[term, 0], ci.loc[term, 1], stars(m.pvalues[term]), w=width)


def coef_header(width: int = 30) -> list[str]:
    return ["    {:<{w}s} {:>14s} {:>13s} {:>8s} {:>10s}  {:<28s} {}".format(
        "spec", "coef", "SE", "t", "p", "95% CI", "sig", w=width),
        "    " + "-" * (width + 82)]


def wrap(text: str, width: int = 74, indent: str = "  ") -> list[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return out
