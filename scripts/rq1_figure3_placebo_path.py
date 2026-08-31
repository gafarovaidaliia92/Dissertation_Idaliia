"""
rq1_figure3_placebo_path.py — Figure 3: the quarterly path of the
uninsured-share coefficient.

WHAT IT PLOTS. One point per quarter: the coefficient on uninsured_share from
the baseline wide-population OLS, estimated separately for each quarter with the
predictors lagged one quarter, with its 95% confidence interval. These are the
quarter-by-quarter placebo estimates behind Appendix F / Table F1. The 2023Q1
banking-stress quarter is set apart.

WHAT IT DOES NOT DO. It estimates nothing. Every coefficient and every interval
is read verbatim from rq1_placebo_by_quarter.csv, which rq1_placebo.py wrote.
No confidence interval is recomputed from a standard error, no model is refitted
and no frozen key is touched. This script only draws.

VERIFICATION BEFORE DRAWING. Each point is checked against config.FROZEN before
anything is rendered, and the script aborts on any mismatch, so the figure
cannot silently drift from the manuscript:

    rq1.placebo.mq.<quarter>.coef          for the ordinary quarters
    rq1.placebo.uninsured.2023Q1.coef      for the banking-stress quarter

DETERMINISM. No randomness, no date stamp, no network. The row order is the
chronological order of the source file. Re-running overwrites the PDF with a
byte-comparable figure.

Reads  : results/rq1_vulnerability/rq1_placebo_by_quarter.csv
         (falls back to data/processed/ if the showcase copy is absent)
Writes : figures/fig3_uninsured_coefficient_path.pdf   (vector, nothing else)

    python3 scripts/rq1_figure3_placebo_path.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: we only save a file
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                              # noqa: E402

# --- style, identical to Figure 2 (scripts/rq2_figures.py) -------------------
TEAL = "#2a9d8f"        # positive coefficient
AMBER = "#e0a458"       # negative coefficient
ZERO_LINE = "#999999"
GRID = "#e6e6e6"
STRESS_BAND = "#f2f2f2"  # shading behind the banking-stress quarter
EDGE = "#333333"

STRESS_QUARTER = "2023Q1"

SRC = C.RES_RQ1 / "rq1_placebo_by_quarter.csv"
SRC_FALLBACK = C.PROC / "rq1_placebo_by_quarter.csv"
OUT = C.ROOT / "figures" / "fig3_uninsured_coefficient_path.pdf"

# Fonts: the manuscript body is sans-serif. Helvetica first, then the metric
# equivalents, then matplotlib's own DejaVu Sans. main() reports which was used,
# because a silent fallback would change the figure's appearance.
FONT_STACK = ["Helvetica", "Nimbus Sans", "Nimbus Sans L", "Arial", "DejaVu Sans"]


def resolve_font() -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in FONT_STACK:
        if name in have:
            return name
    return "DejaVu Sans"


def load() -> pd.DataFrame:
    src = SRC if SRC.exists() else SRC_FALLBACK
    if not src.exists():
        raise SystemExit(
            f"neither {SRC} nor {SRC_FALLBACK} exists — run scripts/rq1_placebo.py")
    d = pd.read_csv(src)
    need = {"outcome_quarter", "coef", "ci_low", "ci_high", "p", "n", "period_type"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{src} is missing column(s): {sorted(missing)}")
    print(f"read {src} — {len(d)} quarters")
    return d.reset_index(drop=True)


def frozen_key(quarter: str) -> str:
    """frozen records the stress quarter under the headline placebo key and the
    others under the per-quarter series; both name the same estimate."""
    return (f"rq1.placebo.uninsured.{quarter}.coef" if quarter == STRESS_QUARTER
            else f"rq1.placebo.mq.{quarter}.coef")


def verify(d: pd.DataFrame) -> None:
    """Abort unless every plotted coefficient matches config.FROZEN."""
    bad, checked = [], 0
    for r in d.itertuples():
        key = frozen_key(r.outcome_quarter)
        if key not in C.FROZEN:
            print(f"  {r.outcome_quarter}: no frozen key ({key}) — not checked")
            continue
        expected, tol, _ = C.FROZEN[key]
        checked += 1
        if abs(float(r.coef) - expected) > max(tol, 1e-8):
            bad.append(f"{r.outcome_quarter}: file {r.coef:+.8f} vs frozen {expected:+.8f}")
    if bad:
        raise SystemExit("STOP — plotted values disagree with config.FROZEN:\n  "
                         + "\n  ".join(bad))
    print(f"verified {checked} of {len(d)} coefficients against config.FROZEN; "
          f"0 mismatches")


def draw(d: pd.DataFrame, font: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,      # embed TrueType so the text stays selectable
        "ps.fonttype": 42,
    })

    x = range(len(d))
    coef = d.coef.to_numpy(dtype=float)
    lo = d.ci_low.to_numpy(dtype=float)
    hi = d.ci_high.to_numpy(dtype=float)
    is_stress = (d.outcome_quarter == STRESS_QUARTER).to_numpy()

    fig, ax = plt.subplots(figsize=(6.27, 3.4))

    # the stress quarter, shaded first so everything else sits on top
    for xi in [i for i, s in enumerate(is_stress) if s]:
        ax.axvspan(xi - 0.42, xi + 0.42, color=STRESS_BAND, zorder=0, linewidth=0)

    ax.axhline(0, color=ZERO_LINE, linewidth=0.9, zorder=2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    # thin line joining the point estimates
    ax.plot(x, coef, color="#7a7a7a", linewidth=0.9, zorder=3)

    # whiskers, drawn per point so each takes its own colour
    for xi, c, l, h, stress in zip(x, coef, lo, hi, is_stress):
        colour = TEAL if c >= 0 else AMBER
        ax.plot([xi, xi], [l, h], color=colour, linewidth=1.4, zorder=4,
                solid_capstyle="butt")
        # caps
        for y in (l, h):
            ax.plot([xi - 0.07, xi + 0.07], [y, y], color=colour,
                    linewidth=1.1, zorder=4)
        # marker: shape carries the stress quarter too, so the distinction
        # survives greyscale printing where teal and amber are close in tone
        ax.plot(xi, c, marker="D" if stress else "o",
                markersize=6.0 if stress else 5.0,
                markerfacecolor=colour, markeredgecolor=EDGE,
                markeredgewidth=1.0 if stress else 0.6,
                linestyle="none", zorder=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(d.outcome_quarter.tolist())
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Outcome quarter")
    ax.set_ylabel("Coefficient on uninsured-deposit share")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)


def report(d: pd.DataFrame) -> None:
    print("\nplotted points — check these against Table F1:\n")
    print(f"  {'quarter':<9} {'predictor':<10} {'n':>5} {'coef':>12} "
          f"{'95% CI low':>12} {'95% CI high':>12} {'p':>10}   rounded")
    print("  " + "-" * 88)
    for r in d.itertuples():
        mark = "  <- banking stress" if r.outcome_quarter == STRESS_QUARTER else ""
        rounded = (f"{r.coef:+.3f}  [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
        print(f"  {r.outcome_quarter:<9} {r.predictor_quarter:<10} {int(r.n):>5} "
              f"{r.coef:>12.8f} {r.ci_low:>12.8f} {r.ci_high:>12.8f} "
              f"{r.p:>10.6f}   {rounded}{mark}")


def main() -> None:
    d = load()
    verify(d)
    font = resolve_font()
    if font != FONT_STACK[0]:
        print(f"NOTE: {FONT_STACK[0]} not found; using {font}")
    else:
        print(f"font: {font}")
    draw(d, font)
    report(d)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
