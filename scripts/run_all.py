"""
run_all.py — master entry point. Regenerates the whole project in dependency
order and then proves the numbers did not move.

    python3 scripts/run_all.py --check-results  # NO data needed: check results/
    python3 scripts/run_all.py              # idempotent: reuses frozen artefacts
    python3 scripts/run_all.py --force      # re-estimate everything (~25 min)
    python3 scripts/run_all.py --verify     # skip the pipeline, reconcile only

--check-results is the entry point for a fresh clone. It recomputes nothing and
touches no data/ path; it reads the published headline numbers from
results/shared_inputs/ and reconciles them against config.FROZEN. --verify and
--force both need data/processed/ and data/raw/, which are not in the repository.

ORDER (each stage consumes the previous one's outputs):

    RQ1   rq1_build_panel -> rq1_model -> rq1_scores -> rq1_robustness
          -> rq1_descriptives
    RQ2   rq2_signal -> rq2_validation -> rq2_car -> rq2_reaction -> rq2_avg_effect
    RQ3   rq3_link -> rq3_interaction -> rq3_measures -> rq3_custody_check
          collect_results

IDEMPOTENCE. Every stage skips itself when its outputs already exist, because
the empirical results are FINAL and re-running an estimator is a chance to lose
them, not to improve them. --force overrides that, stage by stage.

TWO STAGES ARE NEVER RUN AUTOMATICALLY, even with --force:
  * rq2_signal's classification step calls an LLM. The sentence labels are
    frozen and every RQ2/RQ3 number depends on them.
  * rq2_car.py is the CAR construction and is treated as read-only.
Both must be invoked deliberately, on their own.

RECONCILIATION. After the pipeline, every headline coefficient is recomputed and
compared against config.FROZEN, which records the values produced before the
refactor. The table is printed and written to data/processed/reconciliation.txt.
A mismatch means the refactor broke something; it is not a new finding.
"""

from __future__ import annotations

import json
import sys
import traceback

import config as C

FORCE = "--force" in sys.argv
VERIFY_ONLY = "--verify" in sys.argv
CHECK_RESULTS = "--check-results" in sys.argv


def check_published() -> int:
    """Reconcile the PUBLISHED results against config.FROZEN, recomputing nothing.

    This is the one entry point that works in a fresh clone: it reads
    results/shared_inputs/headline_numbers.json, which the pipeline wrote when
    those results were produced, and compares it key by key against FROZEN.
    data/ need not exist at all. Nothing is written.
    """
    src = C.RES_SHARED / "headline_numbers.json"
    if not src.exists():
        raise SystemExit(
            f"[run_all] {src} not found.\n"
            "          --check-results reads the published results snapshot; run\n"
            "          the pipeline (needs data/) or restore results/ from git.")
    got = {k: v for k, v in json.loads(src.read_text()).items() if v is not None}
    text, n_bad = reconcile(got)
    print("\n" + text)
    print(f"read {src} — {len(got)} published values, nothing recomputed")
    return n_bad


def stage(name: str, fn, *, optional: bool = False):
    print("\n" + "=" * 78)
    print(f"  {name}")
    print("=" * 78)
    try:
        return fn()
    except Exception as e:                                        # noqa: BLE001
        if optional:
            print(f"[run_all] {name} FAILED (optional): {type(e).__name__}: {e}")
            return None
        traceback.print_exc()
        raise SystemExit(f"[run_all] {name} failed — stopping.")


# --------------------------------------------------------------------------- #
#  pipeline
# --------------------------------------------------------------------------- #
def run_pipeline() -> dict:
    """Returns the recomputed headline numbers keyed exactly as config.FROZEN."""
    got: dict = {}

    import rq1_build_panel
    import rq1_model
    import rq1_robustness
    import rq1_scores
    stage("RQ1 1/5 — build the panels", rq1_build_panel.main)
    stage("RQ1 2/5 — H1a and H1b", rq1_model.main)
    stage("RQ1 3/5 — leakage-free vulnerability scores", rq1_scores.main)
    stage("RQ1 4/5 — failed-bank robustness", rq1_robustness.main)

    import rq1_descriptives
    got.update(stage("RQ1 5/5 — Tables 1 and 2 (descriptives)",
                     rq1_descriptives.main))

    print("\n[run_all] rq2_signal and rq2_car are NOT run automatically:")
    print("          the signal calls an LLM and the CAR construction is frozen.")

    import rq2_avg_effect
    import rq2_reaction
    got.update(stage("RQ2 1/2 — did the market react at all?", rq2_reaction.main))
    got.update(stage("RQ2 2/2 — average effect of communication type",
                     rq2_avg_effect.main))

    import rq3_interaction
    import rq3_link
    import rq3_measures
    got.update(stage("RQ3 1/3 — the level link (delta1)", rq3_link.main))
    got.update(stage("RQ3 2/3 — two-way FE interaction (b)", rq3_interaction.main))
    got.update(stage("RQ3 3/4 — measures, level-vs-reaction, artefacts",
                     rq3_measures.main))

    # Appendix C. Read-only re-estimation under the custody exclusions; it
    # contributes no FROZEN key, it just has to exist as a file rather than as
    # terminal output someone has to remember to capture.
    import rq3_custody_check
    stage("RQ3 4/4 — custody robustness (Appendix C)", rq3_custody_check.main)

    import rq2_robustness
    got.update(stage("ROBUSTNESS — Section 3.7 windows, S_norm, threshold",
                     rq2_robustness.main))

    # Inference add-ons. Neither re-specifies anything: the first tabulates the
    # nested-tuning metrics behind Table 5 (parsed, not re-estimated), the second
    # recomputes the p-value of every event-clustered coefficient by a wild
    # cluster bootstrap. It runs AFTER rq2_robustness because it bootstraps that
    # stage's variants too, and it reproduces every published CRV1 fit to 1e-8
    # before it will bootstrap anything.
    import rq1_tuned_table
    import rq_wildboot
    got.update(stage("INFERENCE 1/3 — Table 5 tuned metrics",
                     rq1_tuned_table.main))
    got.update(stage("INFERENCE 2/3 — wild cluster bootstrap p-values",
                     lambda: rq_wildboot.main(with_variants=True)))

    # The pre-crisis comparison period. Builds its own panels straight from the
    # Call Report archives, so it needs data/raw/ but nothing from the RQ2/RQ3
    # stages. It refuses to run unless its parameterised panel builder first
    # reproduces panel_2022Q4_wide.csv and Table 3 column 2 exactly.
    import rq1_placebo
    got.update(stage("INFERENCE 3/3 — pre-crisis comparison period (2022Q2)",
                     rq1_placebo.main))

    import rq2_figures
    stage("FIGURE 2 — Safeguard score by event", rq2_figures.main, optional=True)

    got.update(read_rq1_headlines())
    return got


def read_rq1_headlines() -> dict:
    """RQ1's headline numbers are parsed back out of its reports rather than
    re-estimated: re-running the 30-repeat CV and the nested tuning just to
    reconcile would cost 20 minutes and risk the very drift we are checking for."""
    out: dict = {}
    if C.RQ1_H1B_ROBUST.exists():
        txt = C.RQ1_H1B_ROBUST.read_text()
        wide, narrow = txt.split("NARROW, N=277")[0], txt.split("NARROW, N=277")[-1]
        for tag, blob in (("wide", wide), ("narrow", narrow)):
            wins = [l for l in blob.splitlines() if "RF beats OLS in" in l]
            for i, key in enumerate(("full", "pruned")):
                if i < len(wins):
                    out[f"rq1.repeated.{tag}.{key}.rf_wins"] = float(
                        wins[i].split("in")[1].split("of")[0])
        for line in wide.splitlines():
            s = line.split()
            # "full (10 features)  <model>  <rmse>  <r2>" — exactly 6 tokens, so
            # the "OLS - RF gap" line cannot be mistaken for a model row
            if len(s) == 6 and s[0] == "full" and s[2] == "features)":
                if s[3] == "OLS":
                    out["rq1.wide.oos.OLS.rmse"] = float(s[4])
                    out["rq1.wide.oos.OLS.r2"] = float(s[5])
                elif s[3] == "RandomForest":
                    out["rq1.wide.oos.RF.rmse"] = float(s[4])
                    out["rq1.wide.oos.RF.r2"] = float(s[5])
            if len(s) == 5 and s[0] == "uninsured_share" and s[4] == "KEPT":
                out["rq1.wide.ols.uninsured_share.coef"] = float(s[1])
                out["rq1.wide.ols.uninsured_share.p"] = float(s[2])
        for line in narrow.splitlines():
            s = line.split()
            # repeated-CV summary row: model, mean, sd, min, max, meanR2
            if len(s) == 6 and s[0] in ("OLS", "RandomForest"):
                key = "OLS" if s[0] == "OLS" else "RF"
                out.setdefault(f"rq1.repeated.narrow.full.{key}.mean_r2", float(s[5]))
            if len(s) == 6 and s[0] == "full" and s[2] == "features)" and s[3] == "OLS":
                out.setdefault("rq1.narrow.oos.OLS.rmse", float(s[4]))
    if C.RQ1_FAILED_ROBUST.exists():
        txt = C.RQ1_FAILED_ROBUST.read_text()
        wide, narrow = txt.split("NARROW, N=277")[0], txt.split("NARROW, N=277")[-1]
        for tag, blob in (("wide", wide), ("narrow", narrow)):
            block = blob.split("uninsured_share:")[1].split("unrealised_losses:")[0]
            for line in block.splitlines():
                s = line.split()
                # columns: ... N coef p beta_std sig  -> coef=s[-4], p=s[-3]
                if "(a) baseline" in line and len(s) >= 5:
                    out[f"rq1.failed.{tag}.baseline.coef"] = float(s[-4])
                if "(b) failed banks excluded" in line and len(s) >= 5:
                    out[f"rq1.failed.{tag}.excluded.coef"] = float(s[-4])
                    out[f"rq1.failed.{tag}.excluded.p"] = float(s[-3])
    if C.RQ1_COMPARISON.exists():
        for line in C.RQ1_COMPARISON.read_text().splitlines():
            if "transferred to" in line and "R2=" in line:
                out["rq1.transfer.r2"] = float(line.split("R2=")[1].split()[0])
    return out


# --------------------------------------------------------------------------- #
#  reconciliation
# --------------------------------------------------------------------------- #
def reconcile(got: dict) -> tuple[str, int]:
    rows, n_bad, n_missing = [], 0, 0
    width = max(len(k) for k in C.FROZEN)
    rows.append("=" * 110)
    src_label = ("PUBLISHED results/ snapshot" if CHECK_RESULTS
                 else "refactored pipeline")
    rows.append(f"RECONCILIATION — {src_label} against the pre-refactor values")
    rows.append("=" * 110)
    rows.append("")
    if CHECK_RESULTS:
        rows.append("  Every headline number is read from the published results/ snapshot")
        rows.append("  and compared against config.FROZEN. NOTHING IS RECOMPUTED here — this")
        rows.append("  checks that the published results are the ones the study reports.")
    else:
        rows.append("  Every headline number is recomputed by the new code and compared")
        rows.append("  against config.FROZEN. The results are final; a mismatch here is a")
        rows.append("  bug in the refactor, not a finding.")
    rows.append("")
    rows.append("  {:<{w}s} {:>16s} {:>16s} {:>12s}  {}".format(
        "key", "frozen", "published" if CHECK_RESULTS else "recomputed",
        "abs diff", "status", w=width))
    rows.append("  " + "-" * (width + 66))
    for key, (frozen, tol, desc) in C.FROZEN.items():
        if key not in got:
            rows.append("  {:<{w}s} {:>16.8f} {:>16s} {:>12s}  NOT RECOMPUTED".format(
                key, frozen, "-", "-", w=width))
            n_missing += 1
            continue
        val = float(got[key])
        diff = abs(val - frozen)
        ok = diff <= tol
        n_bad += (not ok)
        rows.append("  {:<{w}s} {:>16.8f} {:>16.8f} {:>12.2e}  {}".format(
            key, frozen, val, diff, "match" if ok else "*** MISMATCH ***", w=width))
    rows.append("")
    rows.append(f"  checked {len(C.FROZEN) - n_missing} of {len(C.FROZEN)} keys; "
                f"{n_bad} mismatch(es); {n_missing} not present in this run")
    rows.append("")
    if "rq3.custody_drop.p_event_headline" in got:
        rows.append("  INTENDED CHANGE, not a mismatch: the headline standard error for")
        rows.append("  the RQ3 interaction is now the EVENT-clustered one, because the")
        rows.append("  two-way covariance is not PSD at this sample size. The archived")
        rows.append("  custody-drop p was two-way clustered and still reconciles above;")
        rows.append(f"  the new headline figure is "
                    f"{got['rq3.custody_drop.p_event_headline']:.8f} "
                    f"(event-clustered).")
        rows.append("")
    if n_bad:
        rows.append("  STOP. At least one number moved. Do not use these outputs until")
        rows.append("  the cause is found.")
    else:
        if CHECK_RESULTS:
            rows.append("  All published numbers match the frozen values.")
        else:
            rows.append("  All recomputed numbers match the frozen values. The refactor is")
            rows.append("  behaviour-preserving.")
    rows.append("")
    return "\n".join(rows), n_bad


def main() -> None:
    if CHECK_RESULTS:
        raise SystemExit(1 if check_published() else 0)

    got = {} if VERIFY_ONLY else run_pipeline()
    if VERIFY_ONLY:
        import rq1_tuned_table
        import rq2_avg_effect
        import rq2_reaction
        import rq3_interaction
        import rq3_link
        import rq3_measures
        import rq_wildboot
        # --variants is NOT passed here, for the same reason rq2_robustness is
        # not in this list: --verify recomputes the baseline headline numbers,
        # and the Section 3.7 variants need a CAR rebuild off data/raw/wrds. The
        # variant bootstrap keys therefore report "NOT RECOMPUTED" under
        # --verify, exactly as the variant CRV1 keys already do.
        for fn in (rq2_reaction.main, rq2_avg_effect.main, rq3_link.main,
                   rq3_interaction.main, rq3_measures.main,
                   rq1_tuned_table.main, rq_wildboot.main):
            got.update(fn())
        got.update(read_rq1_headlines())

    C.HEADLINE_JSON.write_text(json.dumps(
        {k: (float(v) if v == v else None) for k, v in sorted(got.items())}, indent=2))
    text, n_bad = reconcile(got)
    C.RECONCILIATION.write_text(text + "\n")
    print("\n" + text)
    print(f"wrote {C.RECONCILIATION}")
    print(f"wrote {C.HEADLINE_JSON}")

    if not VERIFY_ONLY:
        import collect_results
        stage("SHOWCASE — rebuild results/", collect_results.main, optional=True)

    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
