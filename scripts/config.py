"""
config.py — every path, seed, feature list and threshold in the project, in one
place. No analysis script may hard-code a path or a magic number.

The project answers three research questions, and the module layout follows them:

  RQ1  Which balance-sheet characteristics predict the 2023Q1 deposit outflow, and
       do tree models beat OLS out of sample?
         rq1_build_panel.py -> rq1_model.py -> rq1_scores.py -> rq1_robustness.py

  RQ2  Did the equity market react to Federal Reserve CBDC communications, and does
       the AVERAGE reaction depend on the type of communication?
         rq2_signal.py -> rq2_validation.py -> rq2_car.py -> rq2_reaction.py
                                                          -> rq2_avg_effect.py

  RQ3  Do the two vulnerability proxies meet: does a bank the RQ1 model calls
       vulnerable react differently to CBDC communications?
         rq3_link.py -> rq3_interaction.py -> rq3_measures.py

FROZEN NUMBERS. The empirical results are final. FROZEN below records the headline
figures produced before the refactor; run_all.py recomputes them and prints a
reconciliation table. A mismatch is a bug in the refactor, not a new finding.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"

WRDS = RAW / "wrds"
FACTORS = RAW / "factors"
FDIC_DIR = RAW / "fdic"
FED_DIR = RAW / "fed"
NIC_DIR = RAW / "nic"
MDRM_DIR = RAW / "mdrm"

# results showcase: one directory per research question
RES_RQ1 = RESULTS / "rq1_vulnerability"
RES_RQ2 = RESULTS / "rq2_communications"
RES_RQ3 = RESULTS / "rq3_bridge"
RES_SHARED = RESULTS / "shared_inputs"
RES_ARCHIVE = RESULTS / "_archive"

# ---- frozen inputs ----------------------------------------------------------
# data/frozen/ holds the four artefacts NO script can rebuild: the LLM sentence
# labels, the hand-coded validation template, the listed-bank sample and the
# RSSD<->permno crosswalk. They live outside data/processed/ because that folder
# is disposable output, and they are the one part of data/ kept under version
# control (see the !data/frozen/ exception in .gitignore).
FROZEN_DIR = DATA / "frozen"
FROZEN_SUPERSEDED = FROZEN_DIR / "superseded"   # snapshots behind results/_archive/

# ---- inputs -----------------------------------------------------------------
SAMPLE_BANKS = FROZEN_DIR / "sample_banks.csv"
CROSSWALK = FROZEN_DIR / "crosswalk_rssd_permno.csv"
COMMS_XLSX = FED_DIR / "Fed_Communications_Block3.xlsx"   # the 11-document corpus
DOC1_FULLTEXT = FED_DIR / "fulltext_01_money_and_payments.txt"
FAILURES_CSV = FDIC_DIR / "fdic_failures.csv"
CODEBOOK = ROOT / "docs" / "rq2_codebook.md"

# ---- RQ1 --------------------------------------------------------------------
PANEL_NARROW = PROC / "panel_2022Q4.csv"
PANEL_WIDE = PROC / "panel_2022Q4_wide.csv"
PANEL_WIDE_ALL = PROC / "panel_2022Q4_wide_allcharters.csv"
RQ1_RESULTS = PROC / "rq1_results.txt"                 # narrow, canonical
RQ1_RESULTS_NARROW = PROC / "rq1_results_narrow.txt"
RQ1_RESULTS_WIDE = PROC / "rq1_results_wide.txt"
RQ1_COMPARISON = PROC / "comparison_narrow_vs_wide.txt"
RQ1_H1B_ROBUST = PROC / "rq1_h1b_robustness.txt"
RQ1_FAILED_ROBUST = PROC / "rq1_failed_bank_robustness.txt"
RQ1_DESCRIPTIVES = PROC / "rq1_descriptives.txt"     # Tables 1 and 2
SCORES_NARROW = PROC / "vulnerability_scores_narrow.csv"
SCORES_WIDE = PROC / "vulnerability_scores_wide.csv"   # leakage-free, RQ3 input
SCORES_OLS = PROC / "vulnerability_scores_ols.csv"     # leakage-free, OLS variant

# ---- RQ2 --------------------------------------------------------------------
RQ2_SENTENCES = PROC / "rq2_sentences.csv"
RQ2_LABELS = FROZEN_DIR / "rq2_sentence_labels.csv"        # FROZEN: LLM output
RQ2_VALIDATION_TEMPLATE = FROZEN_DIR / "rq2_validation_template.csv"  # FROZEN: hand-coded
RQ2_SCORES = PROC / "rq2_safeguard_scores.csv"         # 11 events, the S variable
RQ2_RUN_META = PROC / "rq2_run_metadata.json"
RQ2_VALIDATION = PROC / "rq2_validation_results.txt"
RQ2_CAR = PROC / "rq2_car.csv"                         # produced by rq2_car.py
RQ2_CAR_AUG = PROC / "rq2_car_augmented.csv"
RQ2_CAR_DROPLOG = PROC / "rq2_car_droplog.csv"
RQ2_CAR_SANITY = PROC / "rq2_car_sanity.txt"
RQ2_REACTION = PROC / "rq2_reaction.txt"               # RQ2 authoritative file
RQ2_AVG_EFFECT = PROC / "rq2_avg_effect.txt"
RQ2_ROBUSTNESS = PROC / "rq2_robustness.txt"   # Section 3.7 checks
RQ2_FIG_SIGNAL = PROC / "rq2_safeguard_by_event.png"   # Figure 2
RQ2_CORPUS_TABLE = PROC / "rq2_corpus_table.txt"       # Appendix B

# ---- RQ3 --------------------------------------------------------------------
RQ3_LINK = PROC / "rq3_link.txt"
RQ3_INTERACTION = PROC / "rq3_interaction.txt"
RQ3_MEASURES = PROC / "rq3_measures.txt"
RQ3_BRIDGE_CSV = PROC / "rq3_bridge.csv"
RQ3_CUSTODY = PROC / "rq3_custody_check.txt"   # Appendix C source

# ---- inference add-ons (supervisor's final review) --------------------------
# These ADD to the reported inference; they re-specify nothing. rq_wildboot.py
# recomputes the p-value of every event-clustered baseline coefficient by a wild
# cluster bootstrap (11 clusters is too few for CRV1) and reports the MDE for the
# uninsured x Safeguard interaction. rq1_tuned_table.py reports the nested-tuning
# metrics behind Table 5, which the pipeline computed but never tabulated.
RQ_WILDBOOT = PROC / "rq_wildboot.txt"
RQ1_TUNED_CSV = PROC / "rq1_tuned_metrics.csv"     # Table 5, machine-readable
RQ1_TUNED_TEX = PROC / "rq1_tuned_metrics.tex"     # Table 5, booktabs fragment

# ---- pre-crisis comparison period (2022Q2) ----------------------------------
# rq1_placebo.py re-estimates the RQ1 specification on a non-banking-stress
# period to establish whether the uninsured-deposit relationship is specific to
# the 2023 stress or a persistent structural correlate. Never "calm period".
RQ1_PLACEBO = PROC / "rq1_placebo.txt"
RQ1_PLACEBO_MD = PROC / "rq1_placebo.md"
RQ1_PLACEBO_TEX = PROC / "rq1_placebo_table.tex"
RQ1_PLACEBO_CSV = PROC / "rq1_placebo_table.csv"
RQ1_PLACEBO_MQ_CSV = PROC / "rq1_placebo_by_quarter.csv"
RQ1_DEPGROWTH_CSV = PROC / "rq1_deposit_growth_by_quarter.csv"

# ---- reconciliation ---------------------------------------------------------
HEADLINE_JSON = PROC / "headline_numbers.json"
RECONCILIATION = PROC / "reconciliation.txt"

# --------------------------------------------------------------------------- #
#  Global seed and estimation settings
# --------------------------------------------------------------------------- #
SEED = 42
N_SPLITS = 5
N_REPEATS = 30          # repeated CV: 30 independent shuffles
INNER_SPLITS = 3        # inner folds of the nested tuning
SIG_LEVEL = 0.05

# two-sided 5%, 80% power: MDE = 2.80 x SE
POWER_MULT = 2.80
# CAR standard deviation used to scale MDEs into "% of a CAR SD"
CAR_SD_REF = 0.048

# A regressor whose R^2 on the fixed effects reaches this is treated as perfectly
# collinear with them and is NOT estimated. Printing a pseudo-inverse split of a
# collinear pair is an artefact, not an estimate.
ABSORBED_R2 = 1.0 - 1e-9

MIN_BANKS_PER_EVENT = 30

# --------------------------------------------------------------------------- #
#  Variables
# --------------------------------------------------------------------------- #
OUTCOME = "dep_growth"

FEATURES = [
    "uninsured_share", "unrealised_losses", "deposit_reliance", "liquidity",
    "capital", "size", "ROA", "NPL_ratio", "equity_ratio", "int_inc_ratio",
]

# bank controls in every CAR regression, read at the characteristic quarter of
# the event (so they vary within bank across events)
CONTROLS = ["size", "ROA", "NPL_ratio", "equity_ratio"]

EXCLUDE_IDRSSD = [119528]   # Farmers National Bank of Emlenton — merger exit, null y

# The three banks that failed in 2023, censored at dep_growth = -1.0 in the
# baseline panel. rq1_robustness.py re-runs RQ1 under four treatments of them.
FAILED_IDRSSD = {802866: "SILICON VALLEY BANK",
                 2942690: "SIGNATURE BANK",
                 4114567: "FIRST REPUBLIC BANK"}

# --------------------------------------------------------------------------- #
#  Vulnerability measures (RQ3). ALL oriented HIGHER = MORE VULNERABLE.
# --------------------------------------------------------------------------- #
MEASURES = [
    dict(
        key="uninsured_share",
        name="uninsured_share",
        static=False,
        orient=("uninsured deposits / total deposits, read from the last Call Report "
                "quarter ending strictly before t0. Raw fraction on [0, 1]; already "
                "oriented higher = more vulnerable. NOT rescaled. Bank-EVENT level: "
                "the 11 events draw on 8 distinct characteristic quarters, so it is "
                "NOT absorbed by bank fixed effects."),
    ),
    dict(
        key="score_rf",
        name="RF score (leakage-free)",
        static=True,
        orient=("= -1 x predicted 2023Q1 deposit growth from the RandomForest fitted "
                "on the 688 non-listed wide-population banks and applied to the listed "
                "banks. Higher = more vulnerable, because a predicted OUTFLOW maps to a "
                "positive score. NOT rescaled. STATIC 2022Q4 profile: one value per "
                "bank, so it IS absorbed by bank fixed effects."),
    ),
    dict(
        key="score_ols",
        name="OLS score (same split)",
        static=True,
        orient=("identical leakage-free protocol to the RF score — same features, same "
                "688/277 train-apply split — with LinearRegression in place of "
                "RandomForest. Higher = more vulnerable. STATIC, absorbed by bank FE."),
    ),
]
MEASURE_KEYS = [m["key"] for m in MEASURES]

# --------------------------------------------------------------------------- #
#  Tuning grids (searched on inner folds only — never on scoring data)
# --------------------------------------------------------------------------- #
RF_PARAMS = dict(n_estimators=500, max_depth=4, min_samples_leaf=5,
                 max_features=0.6, random_state=SEED, n_jobs=-1)
GB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=2,
                 min_samples_leaf=8, subsample=0.8, random_state=SEED)

RF_GRID = {
    "n_estimators": [200],
    "max_depth": [3, 4, 6, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [0.4, 0.6, 1.0],
}
GB_GRID = {
    "n_estimators": [200, 500],
    "learning_rate": [0.03, 0.10],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 10],
}

# --------------------------------------------------------------------------- #
#  FROZEN headline numbers — the pre-refactor values, for reconciliation.
#  key -> (value, tolerance, human description)
#  Tolerances are the precision at which the source file reported the number.
#
#  RE-FROZEN 2026-08-17, 12 of the 46 keys. Call Report coverage now starts at
#  2019Q3 instead of 2019Q4, so doc_id 8 (Brainard, 2019-10-16) draws its bank
#  characteristics from a quarter that actually precedes the event instead of
#  the one two months after it. That removed the last char_lookahead=True rows
#  (258 of 2956 bank-events). Only estimates that take balance-sheet controls
#  on the right-hand side moved: gamma1 in the controlled specification, all
#  three delta1 measures, and b for uninsured_share (the one bank-event
#  measure). CAR, alpha and beta are unchanged to 1e-15, so the 34 remaining
#  keys — every RQ1 number, the uncontrolled gamma1 specifications, b for both
#  model scores, and the custody-drop check — still reconcile against their
#  original values. Keys re-frozen here are tagged "(re-frozen 2019Q3)".
# --------------------------------------------------------------------------- #
FROZEN: dict[str, tuple[float, float, str]] = {
    # ---- RQ1 ----
    "rq1.wide.ols.uninsured_share.coef": (
        -0.07713467, 1e-8, "WIDE OLS coefficient on uninsured_share"),
    "rq1.wide.ols.uninsured_share.p": (
        0.00006744, 1e-8, "WIDE OLS p-value on uninsured_share"),
    "rq1.wide.oos.OLS.rmse": (0.09195066, 1e-8, "WIDE OOS RMSE, OLS (seed 42)"),
    "rq1.wide.oos.RF.rmse": (0.09129361, 1e-8, "WIDE OOS RMSE, RandomForest (seed 42)"),
    "rq1.wide.oos.OLS.r2": (0.02497642, 1e-8, "WIDE OOS R2, OLS (seed 42)"),
    "rq1.wide.oos.RF.r2": (0.03886107, 1e-8, "WIDE OOS R2, RandomForest (seed 42)"),
    "rq1.narrow.oos.OLS.rmse": (0.13194880, 1e-8, "NARROW OOS RMSE, OLS (seed 42)"),
    "rq1.repeated.wide.full.rf_wins": (
        26, 0, "WIDE full features: repeats out of 30 where RF beats OLS"),
    "rq1.repeated.wide.pruned.rf_wins": (
        9, 0, "WIDE pruned features: repeats out of 30 where RF beats OLS"),
    "rq1.repeated.narrow.full.rf_wins": (
        23, 0, "NARROW full features: repeats out of 30 where RF beats OLS"),
    "rq1.repeated.narrow.pruned.rf_wins": (
        17, 0, "NARROW pruned features: repeats out of 30 where RF beats OLS"),
    "rq1.repeated.narrow.full.OLS.mean_r2": (
        0.01427315, 1e-8, "NARROW mean OOS R2 over 30 repeats, OLS (positive!)"),
    "rq1.repeated.narrow.full.RF.mean_r2": (
        0.03352837, 1e-8, "NARROW mean OOS R2 over 30 repeats, RF (positive!)"),
    "rq1.transfer.r2": (0.0302, 1e-4, "RF score transfer R2 on the 277 listed banks"),
    "rq1.failed.wide.baseline.coef": (
        -0.07713467, 1e-8, "RUN4 WIDE uninsured_share, baseline censored -1.0"),
    "rq1.failed.wide.excluded.coef": (
        -0.03336049, 1e-8, "RUN4 WIDE uninsured_share, three failures excluded"),
    "rq1.failed.wide.excluded.p": (
        0.03134800, 1e-7, "RUN4 WIDE p, three failures excluded (still significant)"),
    "rq1.failed.narrow.excluded.coef": (
        -0.05945248, 1e-8, "RUN4 NARROW uninsured_share, three failures excluded"),
    "rq1.failed.narrow.excluded.p": (
        0.10458407, 1e-7, "RUN4 NARROW p, three failures excluded (NOT significant)"),

    # ---- Tables 1 and 2 (rq1_descriptives.py) ----
    "rq1.desc.N": (277, 0, "listed banks in Tables 1 and 2"),
    "rq1.desc.uninsured_share.mean": (
        0.39796361, 1e-8, "Table 1 mean, uninsured_share"),
    "rq1.desc.uninsured_share.sd": (
        0.14440797, 1e-8, "Table 1 sd, uninsured_share"),
    "rq1.desc.uninsured_share.min": (
        0.03228398, 1e-8, "Table 1 min, uninsured_share"),
    "rq1.desc.uninsured_share.max": (
        0.86517654, 1e-8, "Table 1 max, uninsured_share"),
    "rq1.desc.size.mean": (
        15.95225801, 1e-8, "Table 1 mean, size"),
    "rq1.desc.size.sd": (
        1.59677055, 1e-8, "Table 1 sd, size"),
    "rq1.desc.size.min": (
        12.85169049, 1e-8, "Table 1 min, size"),
    "rq1.desc.size.max": (
        21.88702334, 1e-8, "Table 1 max, size"),
    "rq1.desc.ROA.mean": (
        0.01227905, 1e-8, "Table 1 mean, ROA"),
    "rq1.desc.ROA.sd": (
        0.00388855, 1e-8, "Table 1 sd, ROA"),
    "rq1.desc.ROA.min": (
        -0.00190058, 1e-8, "Table 1 min, ROA"),
    "rq1.desc.ROA.max": (
        0.04227701, 1e-8, "Table 1 max, ROA"),
    "rq1.desc.NPL_ratio.mean": (
        0.00460212, 1e-8, "Table 1 mean, NPL_ratio"),
    "rq1.desc.NPL_ratio.sd": (
        0.00446492, 1e-8, "Table 1 sd, NPL_ratio"),
    "rq1.desc.NPL_ratio.min": (
        0.00000000, 1e-8, "Table 1 min, NPL_ratio"),
    "rq1.desc.NPL_ratio.max": (
        0.02529396, 1e-8, "Table 1 max, NPL_ratio"),
    "rq1.desc.equity_ratio.mean": (
        0.09917691, 1e-8, "Table 1 mean, equity_ratio"),
    "rq1.desc.equity_ratio.sd": (
        0.02343209, 1e-8, "Table 1 sd, equity_ratio"),
    "rq1.desc.equity_ratio.min": (
        0.04188154, 1e-8, "Table 1 min, equity_ratio"),
    "rq1.desc.equity_ratio.max": (
        0.17729094, 1e-8, "Table 1 max, equity_ratio"),
    "rq1.desc.deposit_reliance.mean": (
        0.92354411, 1e-8, "Table 1 mean, deposit_reliance"),
    "rq1.desc.deposit_reliance.sd": (
        0.05138319, 1e-8, "Table 1 sd, deposit_reliance"),
    "rq1.desc.deposit_reliance.min": (
        0.68858759, 1e-8, "Table 1 min, deposit_reliance"),
    "rq1.desc.deposit_reliance.max": (
        0.99632617, 1e-8, "Table 1 max, deposit_reliance"),
    "rq1.corr.uninsured_share.size": (
        0.35056473, 1e-8, "Table 2 corr(uninsured_share, size)"),
    "rq1.corr.uninsured_share.ROA": (
        -0.09333631, 1e-8, "Table 2 corr(uninsured_share, ROA)"),
    "rq1.corr.uninsured_share.NPL_ratio": (
        -0.09899051, 1e-8, "Table 2 corr(uninsured_share, NPL_ratio)"),
    "rq1.corr.uninsured_share.equity_ratio": (
        0.01268865, 1e-8, "Table 2 corr(uninsured_share, equity_ratio)"),
    "rq1.corr.uninsured_share.deposit_reliance": (
        -0.11197202, 1e-8, "Table 2 corr(uninsured_share, deposit_reliance)"),
    "rq1.corr.size.ROA": (
        0.10984765, 1e-8, "Table 2 corr(size, ROA)"),
    "rq1.corr.size.NPL_ratio": (
        0.14993497, 1e-8, "Table 2 corr(size, NPL_ratio)"),
    "rq1.corr.size.equity_ratio": (
        0.06061249, 1e-8, "Table 2 corr(size, equity_ratio)"),
    "rq1.corr.size.deposit_reliance": (
        -0.31792586, 1e-8, "Table 2 corr(size, deposit_reliance)"),
    "rq1.corr.ROA.NPL_ratio": (
        0.01100415, 1e-8, "Table 2 corr(ROA, NPL_ratio)"),
    "rq1.corr.ROA.equity_ratio": (
        0.18961143, 1e-8, "Table 2 corr(ROA, equity_ratio)"),
    "rq1.corr.ROA.deposit_reliance": (
        0.02402644, 1e-8, "Table 2 corr(ROA, deposit_reliance)"),
    "rq1.corr.NPL_ratio.equity_ratio": (
        0.06551437, 1e-8, "Table 2 corr(NPL_ratio, equity_ratio)"),
    "rq1.corr.NPL_ratio.deposit_reliance": (
        -0.08633964, 1e-8, "Table 2 corr(NPL_ratio, deposit_reliance)"),
    "rq1.corr.equity_ratio.deposit_reliance": (
        -0.02464201, 1e-8, "Table 2 corr(equity_ratio, deposit_reliance)"),

    # ---- RQ2 ----
    "rq2.unconditional.n_significant": (
        0, 0, "events of 11 with a cross-correlation-robust mean-CAR reaction"),
    "rq2.gamma1.event_level.coef": (
        0.005629, 1e-6, "gamma1, 11 event-mean CARs on 11 S values"),
    "rq2.gamma1.event_level.se": (0.018874, 1e-6, "gamma1 SE, event level"),
    "rq2.gamma1.event_level.p": (0.7723, 1e-4, "gamma1 p, event level"),
    "rq2.gamma1.pooled_ctrl.coef": (
        0.006454, 1e-6, "gamma1, pooled with controls, event-clustered (re-frozen 2019Q3)"),
    "rq2.gamma1.pooled_ctrl.se": (
        0.017129, 1e-6, "gamma1 SE, pooled with controls (re-frozen 2019Q3)"),
    "rq2.gamma1.pooled_ctrl.p": (
        0.7142, 1e-4, "gamma1 p, pooled with controls (re-frozen 2019Q3)"),
    "rq2.corr.pearson": (0.0989, 1e-4, "Pearson corr(event-mean CAR, S)"),
    "rq2.corr.spearman": (0.0137, 1e-4, "Spearman corr(event-mean CAR, S)"),

    # ---- RQ3 ----
    "rq3.delta1.uninsured.coef": (
        -0.01877493, 1e-8,
        "delta1 on uninsured_share, event FE (SIGNIFICANT) (re-frozen 2019Q3)"),
    "rq3.delta1.uninsured.p": (
        0.000321, 1e-6, "delta1 p on uninsured_share (re-frozen 2019Q3)"),
    "rq3.delta1.score_rf.coef": (
        0.06011100, 1e-8, "delta1 on the RF score (re-frozen 2019Q3)"),
    "rq3.delta1.score_rf.p": (
        0.332233, 1e-6, "delta1 p on the RF score (re-frozen 2019Q3)"),
    "rq3.delta1.score_ols.coef": (
        0.01232232, 1e-8, "delta1 on the OLS score (re-frozen 2019Q3)"),
    "rq3.delta1.score_ols.p": (
        0.782140, 1e-6, "delta1 p on the OLS score (re-frozen 2019Q3)"),
    "rq3.b.uninsured_share.coef": (
        -0.00049680, 1e-8,
        "b on S x uninsured_share, event+bank FE (re-frozen 2019Q3)"),
    "rq3.b.uninsured_share.se_twoway": (
        0.02093823, 1e-8, "b SE, two-way clustered (re-frozen 2019Q3)"),
    "rq3.b.uninsured_share.p_event": (
        0.9819, 1e-4, "b p, event-clustered (headline) (re-frozen 2019Q3)"),
    "rq3.b.score_rf.coef": (-0.38507629, 1e-8, "b on S x RF score, event+bank FE"),
    "rq3.b.score_rf.se_twoway": (0.14229917, 1e-8, "b SE, two-way clustered"),
    "rq3.b.score_rf.p_event": (0.0115, 1e-4, "b p, event-clustered (headline)"),
    "rq3.b.score_ols.coef": (-0.26387427, 1e-8, "b on S x OLS score, event+bank FE"),
    "rq3.corr.rf_uninsured.spearman": (
        0.05754306, 1e-8, "Spearman corr(RF score, uninsured_share) across banks"),
    "rq3.corr.rf_ols.spearman": (
        0.68145928, 1e-8, "Spearman corr(RF score, OLS score) across banks"),
    "rq3.placebo.corr_S_time": (
        0.60333196, 1e-8, "Pearson corr(S, calendar rank) across the 11 events"),
    "rq3.custody_drop.b": (
        -0.13939961, 1e-8, "b on S x RF score with trust/custody banks dropped"),
    "rq3.custody_drop.p": (0.063319, 1e-6, "its p (falls below 5%)"),

    # ---- 3.7 robustness, standardised (per one SD of the signal) ----
    # S runs on [-0.67, +1.0] and S_norm on [-0.06, +0.09], so raw
    # interaction coefficients are ~11x apart for the same effect. These
    # make the two directly comparable.
    "rq2.robust.snorm.beta_per_sd.score_ols": (
        -0.13031909, 1e-8, "3.7 robustness: snorm.beta per sd.score ols"),
    "rq2.robust.snorm.beta_per_sd.score_rf": (
        -0.18087696, 1e-8, "3.7 robustness: snorm.beta per sd.score rf"),
    "rq2.robust.snorm.beta_per_sd.uninsured_share": (
        0.00041651, 1e-8, "3.7 robustness: snorm.beta per sd.uninsured share"),
    "rq2.robust.snorm.gamma1_per_sd": (
        0.00353544, 1e-8, "3.7 robustness: snorm.gamma1 per sd"),
    "rq2.robust.snorm.signal_sd": (
        0.04011294, 1e-8, "3.7 robustness: snorm.signal sd"),
    "rq2.robust.thresh9.beta_per_sd.score_ols": (
        -0.14043134, 1e-8, "3.7 robustness: thresh9.beta per sd.score ols"),
    "rq2.robust.thresh9.beta_per_sd.score_rf": (
        -0.21249193, 1e-8, "3.7 robustness: thresh9.beta per sd.score rf"),
    "rq2.robust.thresh9.beta_per_sd.uninsured_share": (
        -0.00445453, 1e-8, "3.7 robustness: thresh9.beta per sd.uninsured share"),
    "rq2.robust.thresh9.gamma1_per_sd": (
        -0.00250787, 1e-8, "3.7 robustness: thresh9.gamma1 per sd"),
    "rq2.robust.thresh9.signal_sd": (
        0.29812720, 1e-8, "3.7 robustness: thresh9.signal sd"),
    "rq2.robust.win_base.beta_per_sd.score_ols": (
        -0.13046948, 1e-8, "3.7 robustness: win base.beta per sd.score ols"),
    "rq2.robust.win_base.beta_per_sd.score_rf": (
        -0.19039637, 1e-8, "3.7 robustness: win base.beta per sd.score rf"),
    "rq2.robust.win_base.beta_per_sd.uninsured_share": (
        -0.00024564, 1e-8, "3.7 robustness: win base.beta per sd.uninsured share"),
    "rq2.robust.win_base.gamma1_per_sd": (
        0.00319124, 1e-8, "3.7 robustness: win base.gamma1 per sd"),
    "rq2.robust.win_base.signal_sd": (
        0.49443805, 1e-8, "3.7 robustness: win base.signal sd"),
    "rq2.robust.win_m1p3.beta_per_sd.score_ols": (
        -0.11463564, 1e-8, "3.7 robustness: win m1p3.beta per sd.score ols"),
    "rq2.robust.win_m1p3.beta_per_sd.score_rf": (
        -0.15268676, 1e-8, "3.7 robustness: win m1p3.beta per sd.score rf"),
    "rq2.robust.win_m1p3.beta_per_sd.uninsured_share": (
        -0.00608672, 1e-8, "3.7 robustness: win m1p3.beta per sd.uninsured share"),
    "rq2.robust.win_m1p3.gamma1_per_sd": (
        -0.00070010, 1e-8, "3.7 robustness: win m1p3.gamma1 per sd"),
    "rq2.robust.win_m1p3.signal_sd": (
        0.49443805, 1e-8, "3.7 robustness: win m1p3.signal sd"),
    "rq2.robust.win_m5p5.beta_per_sd.score_ols": (
        -0.15080379, 1e-8, "3.7 robustness: win m5p5.beta per sd.score ols"),
    "rq2.robust.win_m5p5.beta_per_sd.score_rf": (
        -0.22751634, 1e-8, "3.7 robustness: win m5p5.beta per sd.score rf"),
    "rq2.robust.win_m5p5.beta_per_sd.uninsured_share": (
        -0.00656866, 1e-8, "3.7 robustness: win m5p5.beta per sd.uninsured share"),
    "rq2.robust.win_m5p5.gamma1_per_sd": (
        0.00461274, 1e-8, "3.7 robustness: win m5p5.gamma1 per sd"),
    "rq2.robust.win_m5p5.signal_sd": (
        0.49443805, 1e-8, "3.7 robustness: win m5p5.signal sd"),

    # ---- Section 3.7 robustness (rq2_robustness.py) ----
    # Alternative event windows, the normalised signal and the design-count
    # threshold. The win_base rows duplicate the headline specification on
    # purpose: they are the control that proves the variant machinery
    # reproduces the published numbers before any variant is believed.
    "rq2.robust.snorm.beta.score_ols.coef": (
        -3.24880456, 1e-8, "3.7 robustness: snorm.beta.score ols.coef"),
    "rq2.robust.snorm.beta.score_ols.p": (
        0.01526694, 1e-8, "3.7 robustness: snorm.beta.score ols.p"),
    "rq2.robust.snorm.beta.score_rf.coef": (
        -4.50919251, 1e-8, "3.7 robustness: snorm.beta.score rf.coef"),
    "rq2.robust.snorm.beta.score_rf.p": (
        0.04205960, 1e-8, "3.7 robustness: snorm.beta.score rf.p"),
    "rq2.robust.snorm.beta.uninsured_share.coef": (
        0.01038347, 1e-8, "3.7 robustness: snorm.beta.uninsured share.coef"),
    "rq2.robust.snorm.beta.uninsured_share.p": (
        0.97035077, 1e-8, "3.7 robustness: snorm.beta.uninsured share.p"),
    "rq2.robust.snorm.beta_excustody.score_ols.coef": (
        -1.79037101, 1e-8, "3.7 robustness: snorm.beta excustody.score ols.coef"),
    "rq2.robust.snorm.beta_excustody.score_ols.p": (
        0.03015094, 1e-8, "3.7 robustness: snorm.beta excustody.score ols.p"),
    "rq2.robust.snorm.beta_excustody.score_rf.coef": (
        -1.96050025, 1e-8, "3.7 robustness: snorm.beta excustody.score rf.coef"),
    "rq2.robust.snorm.beta_excustody.score_rf.p": (
        0.08751632, 1e-8, "3.7 robustness: snorm.beta excustody.score rf.p"),
    "rq2.robust.snorm.beta_excustody.uninsured_share.coef": (
        0.05725998, 1e-8, "3.7 robustness: snorm.beta excustody.uninsured share.coef"),
    "rq2.robust.snorm.beta_excustody.uninsured_share.p": (
        0.83666714, 1e-8, "3.7 robustness: snorm.beta excustody.uninsured share.p"),
    "rq2.robust.snorm.delta1.score_ols.coef": (
        0.01232232, 1e-8, "3.7 robustness: snorm.delta1.score ols.coef"),
    "rq2.robust.snorm.delta1.score_ols.p": (
        0.78213952, 1e-8, "3.7 robustness: snorm.delta1.score ols.p"),
    "rq2.robust.snorm.delta1.score_rf.coef": (
        0.06011100, 1e-8, "3.7 robustness: snorm.delta1.score rf.coef"),
    "rq2.robust.snorm.delta1.score_rf.p": (
        0.33223292, 1e-8, "3.7 robustness: snorm.delta1.score rf.p"),
    "rq2.robust.snorm.delta1.uninsured_share.coef": (
        -0.01877493, 1e-8, "3.7 robustness: snorm.delta1.uninsured share.coef"),
    "rq2.robust.snorm.delta1.uninsured_share.p": (
        0.00032066, 1e-8, "3.7 robustness: snorm.delta1.uninsured share.p"),
    "rq2.robust.snorm.gamma1.coef": (
        0.08813715, 1e-8, "3.7 robustness: snorm.gamma1.coef"),
    "rq2.robust.snorm.gamma1.p": (
        0.71072530, 1e-8, "3.7 robustness: snorm.gamma1.p"),
    "rq2.robust.snorm.gamma1.se": (
        0.23095404, 1e-8, "3.7 robustness: snorm.gamma1.se"),
    "rq2.robust.thresh9.beta.score_ols.coef": (
        -0.47104505, 1e-8, "3.7 robustness: thresh9.beta.score ols.coef"),
    "rq2.robust.thresh9.beta.score_ols.p": (
        0.03853969, 1e-8, "3.7 robustness: thresh9.beta.score ols.p"),
    "rq2.robust.thresh9.beta.score_rf.coef": (
        -0.71275593, 1e-8, "3.7 robustness: thresh9.beta.score rf.coef"),
    "rq2.robust.thresh9.beta.score_rf.p": (
        0.04204450, 1e-8, "3.7 robustness: thresh9.beta.score rf.p"),
    "rq2.robust.thresh9.beta.uninsured_share.coef": (
        -0.01494172, 1e-8, "3.7 robustness: thresh9.beta.uninsured share.coef"),
    "rq2.robust.thresh9.beta.uninsured_share.p": (
        0.76210470, 1e-8, "3.7 robustness: thresh9.beta.uninsured share.p"),
    "rq2.robust.thresh9.beta_excustody.score_ols.coef": (
        -0.29867817, 1e-8, "3.7 robustness: thresh9.beta excustody.score ols.coef"),
    "rq2.robust.thresh9.beta_excustody.score_ols.p": (
        0.04765666, 1e-8, "3.7 robustness: thresh9.beta excustody.score ols.p"),
    "rq2.robust.thresh9.beta_excustody.score_rf.coef": (
        -0.45042440, 1e-8, "3.7 robustness: thresh9.beta excustody.score rf.coef"),
    "rq2.robust.thresh9.beta_excustody.score_rf.p": (
        0.05798860, 1e-8, "3.7 robustness: thresh9.beta excustody.score rf.p"),
    "rq2.robust.thresh9.beta_excustody.uninsured_share.coef": (
        -0.00866735, 1e-8, "3.7 robustness: thresh9.beta excustody.uninsured share.coef"),
    "rq2.robust.thresh9.beta_excustody.uninsured_share.p": (
        0.86108420, 1e-8, "3.7 robustness: thresh9.beta excustody.uninsured share.p"),
    "rq2.robust.thresh9.bw_significant": (
        0.00000000, 0, "3.7 robustness: thresh9.bw significant"),
    "rq2.robust.thresh9.delta1.score_ols.coef": (
        0.02915844, 1e-8, "3.7 robustness: thresh9.delta1.score ols.coef"),
    "rq2.robust.thresh9.delta1.score_ols.p": (
        0.56042668, 1e-8, "3.7 robustness: thresh9.delta1.score ols.p"),
    "rq2.robust.thresh9.delta1.score_rf.coef": (
        0.07180388, 1e-8, "3.7 robustness: thresh9.delta1.score rf.coef"),
    "rq2.robust.thresh9.delta1.score_rf.p": (
        0.21294273, 1e-8, "3.7 robustness: thresh9.delta1.score rf.p"),
    "rq2.robust.thresh9.delta1.uninsured_share.coef": (
        -0.01406690, 1e-8, "3.7 robustness: thresh9.delta1.uninsured share.coef"),
    "rq2.robust.thresh9.delta1.uninsured_share.p": (
        0.01691491, 1e-8, "3.7 robustness: thresh9.delta1.uninsured share.p"),
    "rq2.robust.thresh9.gamma1.coef": (
        -0.00841209, 1e-8, "3.7 robustness: thresh9.gamma1.coef"),
    "rq2.robust.thresh9.gamma1.p": (
        0.84895022, 1e-8, "3.7 robustness: thresh9.gamma1.p"),
    "rq2.robust.thresh9.gamma1.se": (
        0.04230458, 1e-8, "3.7 robustness: thresh9.gamma1.se"),
    "rq2.robust.win_base.beta.score_ols.coef": (
        -0.26387427, 1e-8, "3.7 robustness: win base.beta.score ols.coef"),
    "rq2.robust.win_base.beta.score_ols.p": (
        0.00285943, 1e-8, "3.7 robustness: win base.beta.score ols.p"),
    "rq2.robust.win_base.beta.score_rf.coef": (
        -0.38507629, 1e-8, "3.7 robustness: win base.beta.score rf.coef"),
    "rq2.robust.win_base.beta.score_rf.p": (
        0.01148834, 1e-8, "3.7 robustness: win base.beta.score rf.p"),
    "rq2.robust.win_base.beta.uninsured_share.coef": (
        -0.00049680, 1e-8, "3.7 robustness: win base.beta.uninsured share.coef"),
    "rq2.robust.win_base.beta.uninsured_share.p": (
        0.98188477, 1e-8, "3.7 robustness: win base.beta.uninsured share.p"),
    "rq2.robust.win_base.beta_excustody.score_ols.coef": (
        -0.12160487, 1e-8, "3.7 robustness: win base.beta excustody.score ols.coef"),
    "rq2.robust.win_base.beta_excustody.score_ols.p": (
        0.11229032, 1e-8, "3.7 robustness: win base.beta excustody.score ols.p"),
    "rq2.robust.win_base.beta_excustody.score_rf.coef": (
        -0.13316361, 1e-8, "3.7 robustness: win base.beta excustody.score rf.coef"),
    "rq2.robust.win_base.beta_excustody.score_rf.p": (
        0.11626604, 1e-8, "3.7 robustness: win base.beta excustody.score rf.p"),
    "rq2.robust.win_base.beta_excustody.uninsured_share.coef": (
        0.00372397, 1e-8, "3.7 robustness: win base.beta excustody.uninsured share.coef"),
    "rq2.robust.win_base.beta_excustody.uninsured_share.p": (
        0.86682439, 1e-8, "3.7 robustness: win base.beta excustody.uninsured share.p"),
    "rq2.robust.win_base.bw_significant": (
        0.00000000, 0, "3.7 robustness: win base.bw significant"),
    "rq2.robust.win_base.delta1.score_ols.coef": (
        0.01232232, 1e-8, "3.7 robustness: win base.delta1.score ols.coef"),
    "rq2.robust.win_base.delta1.score_ols.p": (
        0.78213952, 1e-8, "3.7 robustness: win base.delta1.score ols.p"),
    "rq2.robust.win_base.delta1.score_rf.coef": (
        0.06011100, 1e-8, "3.7 robustness: win base.delta1.score rf.coef"),
    "rq2.robust.win_base.delta1.score_rf.p": (
        0.33223292, 1e-8, "3.7 robustness: win base.delta1.score rf.p"),
    "rq2.robust.win_base.delta1.uninsured_share.coef": (
        -0.01877493, 1e-8, "3.7 robustness: win base.delta1.uninsured share.coef"),
    "rq2.robust.win_base.delta1.uninsured_share.p": (
        0.00032066, 1e-8, "3.7 robustness: win base.delta1.uninsured share.p"),
    "rq2.robust.win_base.gamma1.coef": (
        0.00645428, 1e-8, "3.7 robustness: win base.gamma1.coef"),
    "rq2.robust.win_base.gamma1.p": (
        0.71418685, 1e-8, "3.7 robustness: win base.gamma1.p"),
    "rq2.robust.win_base.gamma1.se": (
        0.01712870, 1e-8, "3.7 robustness: win base.gamma1.se"),
    "rq2.robust.win_m1p3.beta.score_ols.coef": (
        -0.23185036, 1e-8, "3.7 robustness: win m1p3.beta.score ols.coef"),
    "rq2.robust.win_m1p3.beta.score_ols.p": (
        0.00051934, 1e-8, "3.7 robustness: win m1p3.beta.score ols.p"),
    "rq2.robust.win_m1p3.beta.score_rf.coef": (
        -0.30880869, 1e-8, "3.7 robustness: win m1p3.beta.score rf.coef"),
    "rq2.robust.win_m1p3.beta.score_rf.p": (
        0.00129787, 1e-8, "3.7 robustness: win m1p3.beta.score rf.p"),
    "rq2.robust.win_m1p3.beta.uninsured_share.coef": (
        -0.01231038, 1e-8, "3.7 robustness: win m1p3.beta.uninsured share.coef"),
    "rq2.robust.win_m1p3.beta.uninsured_share.p": (
        0.24163661, 1e-8, "3.7 robustness: win m1p3.beta.uninsured share.p"),
    "rq2.robust.win_m1p3.beta_excustody.score_ols.coef": (
        -0.11193718, 1e-8, "3.7 robustness: win m1p3.beta excustody.score ols.coef"),
    "rq2.robust.win_m1p3.beta_excustody.score_ols.p": (
        0.17085228, 1e-8, "3.7 robustness: win m1p3.beta excustody.score ols.p"),
    "rq2.robust.win_m1p3.beta_excustody.score_rf.coef": (
        -0.09194120, 1e-8, "3.7 robustness: win m1p3.beta excustody.score rf.coef"),
    "rq2.robust.win_m1p3.beta_excustody.score_rf.p": (
        0.35978762, 1e-8, "3.7 robustness: win m1p3.beta excustody.score rf.p"),
    "rq2.robust.win_m1p3.beta_excustody.uninsured_share.coef": (
        -0.00832452, 1e-8, "3.7 robustness: win m1p3.beta excustody.uninsured share.coef"),
    "rq2.robust.win_m1p3.beta_excustody.uninsured_share.p": (
        0.40547894, 1e-8, "3.7 robustness: win m1p3.beta excustody.uninsured share.p"),
    "rq2.robust.win_m1p3.bw_significant": (
        0.00000000, 0, "3.7 robustness: win m1p3.bw significant"),
    "rq2.robust.win_m1p3.delta1.score_ols.coef": (
        0.01788361, 1e-8, "3.7 robustness: win m1p3.delta1.score ols.coef"),
    "rq2.robust.win_m1p3.delta1.score_ols.p": (
        0.56711705, 1e-8, "3.7 robustness: win m1p3.delta1.score ols.p"),
    "rq2.robust.win_m1p3.delta1.score_rf.coef": (
        0.01333148, 1e-8, "3.7 robustness: win m1p3.delta1.score rf.coef"),
    "rq2.robust.win_m1p3.delta1.score_rf.p": (
        0.75866998, 1e-8, "3.7 robustness: win m1p3.delta1.score rf.p"),
    "rq2.robust.win_m1p3.delta1.uninsured_share.coef": (
        -0.00853550, 1e-8, "3.7 robustness: win m1p3.delta1.uninsured share.coef"),
    "rq2.robust.win_m1p3.delta1.uninsured_share.p": (
        0.03629983, 1e-8, "3.7 robustness: win m1p3.delta1.uninsured share.p"),
    "rq2.robust.win_m1p3.gamma1.coef": (
        -0.00141596, 1e-8, "3.7 robustness: win m1p3.gamma1.coef"),
    "rq2.robust.win_m1p3.gamma1.p": (
        0.88970547, 1e-8, "3.7 robustness: win m1p3.gamma1.p"),
    "rq2.robust.win_m1p3.gamma1.se": (
        0.00995380, 1e-8, "3.7 robustness: win m1p3.gamma1.se"),
    "rq2.robust.win_m5p5.beta.score_ols.coef": (
        -0.30500037, 1e-8, "3.7 robustness: win m5p5.beta.score ols.coef"),
    "rq2.robust.win_m5p5.beta.score_ols.p": (
        0.06222672, 1e-8, "3.7 robustness: win m5p5.beta.score ols.p"),
    "rq2.robust.win_m5p5.beta.score_rf.coef": (
        -0.46015136, 1e-8, "3.7 robustness: win m5p5.beta.score rf.coef"),
    "rq2.robust.win_m5p5.beta.score_rf.p": (
        0.05043008, 1e-8, "3.7 robustness: win m5p5.beta.score rf.p"),
    "rq2.robust.win_m5p5.beta.uninsured_share.coef": (
        -0.01328510, 1e-8, "3.7 robustness: win m5p5.beta.uninsured share.coef"),
    "rq2.robust.win_m5p5.beta.uninsured_share.p": (
        0.59112222, 1e-8, "3.7 robustness: win m5p5.beta.uninsured share.p"),
    "rq2.robust.win_m5p5.beta_excustody.score_ols.coef": (
        -0.13272511, 1e-8, "3.7 robustness: win m5p5.beta excustody.score ols.coef"),
    "rq2.robust.win_m5p5.beta_excustody.score_ols.p": (
        0.22455093, 1e-8, "3.7 robustness: win m5p5.beta excustody.score ols.p"),
    "rq2.robust.win_m5p5.beta_excustody.score_rf.coef": (
        -0.15957310, 1e-8, "3.7 robustness: win m5p5.beta excustody.score rf.coef"),
    "rq2.robust.win_m5p5.beta_excustody.score_rf.p": (
        0.17653026, 1e-8, "3.7 robustness: win m5p5.beta excustody.score rf.p"),
    "rq2.robust.win_m5p5.beta_excustody.uninsured_share.coef": (
        -0.00792083, 1e-8, "3.7 robustness: win m5p5.beta excustody.uninsured share.coef"),
    "rq2.robust.win_m5p5.beta_excustody.uninsured_share.p": (
        0.74548098, 1e-8, "3.7 robustness: win m5p5.beta excustody.uninsured share.p"),
    "rq2.robust.win_m5p5.bw_significant": (
        1.00000000, 0, "3.7 robustness: win m5p5.bw significant"),
    "rq2.robust.win_m5p5.delta1.score_ols.coef": (
        -0.00429454, 1e-8, "3.7 robustness: win m5p5.delta1.score ols.coef"),
    "rq2.robust.win_m5p5.delta1.score_ols.p": (
        0.93203169, 1e-8, "3.7 robustness: win m5p5.delta1.score ols.p"),
    "rq2.robust.win_m5p5.delta1.score_rf.coef": (
        0.07300125, 1e-8, "3.7 robustness: win m5p5.delta1.score rf.coef"),
    "rq2.robust.win_m5p5.delta1.score_rf.p": (
        0.27772735, 1e-8, "3.7 robustness: win m5p5.delta1.score rf.p"),
    "rq2.robust.win_m5p5.delta1.uninsured_share.coef": (
        -0.01884552, 1e-8, "3.7 robustness: win m5p5.delta1.uninsured share.coef"),
    "rq2.robust.win_m5p5.delta1.uninsured_share.p": (
        0.00312915, 1e-8, "3.7 robustness: win m5p5.delta1.uninsured share.p"),
    "rq2.robust.win_m5p5.gamma1.coef": (
        0.00932925, 1e-8, "3.7 robustness: win m5p5.gamma1.coef"),
    "rq2.robust.win_m5p5.gamma1.p": (
        0.57029546, 1e-8, "3.7 robustness: win m5p5.gamma1.p"),
    "rq2.robust.win_m5p5.gamma1.se": (
        0.01589635, 1e-8, "3.7 robustness: win m5p5.gamma1.se"),

    # ======================================================================= #
    #  ADDED 2026-08-25 — the supervisor's final-review inference checks.
    #
    #  These keys ADD to the record; not one existing key above is touched. The
    #  additions re-test and report what was already estimated — they do not
    #  re-specify anything:
    #
    #    * rq_wildboot.py recomputes the p-value of every baseline coefficient
    #      whose SEs are clustered on the EVENT. With 11 clusters CRV1 is
    #      unreliable, so the p-values are redone by a wild cluster bootstrap
    #      (Rademacher, null imposed, bootstrap type "11"). Every point estimate
    #      and standard error is unchanged, and the script refuses to run unless
    #      it first reproduces the published CRV1 fit to 1e-8.
    #    * rq1_tuned_table.py tabulates the nested-tuning metrics behind Table 5,
    #      parsed out of rq1_h1b_robustness.txt rather than re-estimated.
    #
    #  Table 8's LINK column (delta1) is clustered on the BANK, 276 clusters, and
    #  is deliberately NOT bootstrapped.
    # ======================================================================= #

    # ---- Task 1 — WILD CLUSTER BOOTSTRAP p-VALUES (rq_wildboot.py). 11 event clusters,
    #        so 2^11 = 2048 sign patterns are FULLY ENUMERATED: each p is an exact
    #        k/2048, deterministic, and reconciles at tolerance 0.
    "rq2.wildboot.gamma1.augmented.p": (
        0.43359375000, 0, "Table 7 gamma1, augmented CAR: wild cluster bootstrap p"),
    "rq2.wildboot.gamma1.controls.p": (
        0.66894531250, 0, "Table 7 gamma1, with controls: wild cluster bootstrap p"),
    "rq2.wildboot.gamma1.nocontrols.p": (
        0.70410156250, 0, "Table 7 gamma1, no controls: wild cluster bootstrap p"),
    "rq3.wildboot.b.score_ols.p": (
        0.04687500000, 0, "Table 8 b on S x OLS score: bootstrap p (CRV1 was 0.00285943)"),
    "rq3.wildboot.b.score_rf.p": (
        0.06835937500, 0, "Table 8 b on S x RF score: bootstrap p (CRV1 was 0.01148834)"),
    "rq3.wildboot.b.uninsured_share.p": (
        0.98535156250, 0, "Table 8 b on S x uninsured_share: bootstrap p"),
    "rq3.wildboot.excustody.b.score_ols.p": (
        0.18750000000, 0, "Appendix C b on S x OLS score, ex-custody: bootstrap p"),
    "rq3.wildboot.excustody.b.score_rf.p": (
        0.26269531250, 0, "Appendix C b on S x RF score, ex-custody: bootstrap p"),
    "rq3.wildboot.excustody.b.uninsured_share.p": (
        0.85839843750, 0, "Appendix C b on S x uninsured_share, ex-custody: bootstrap p"),

    # ---- Task 1 (cont.) — the same treatment for the Section 3.7 robustness variants.
    #        The threshold variant has 7 events, so its p is k/128.
    "rq2.robust.wildboot.snorm.beta.score_ols.p": (
        0.04492187500, 0, "3.7 robustness, bootstrap p: snorm beta score_ols"),
    "rq2.robust.wildboot.snorm.beta.score_rf.p": (
        0.04980468750, 0, "3.7 robustness, bootstrap p: snorm beta score_rf"),
    "rq2.robust.wildboot.snorm.beta.uninsured_share.p": (
        0.97851562500, 0, "3.7 robustness, bootstrap p: snorm beta uninsured_share"),
    "rq2.robust.wildboot.snorm.gamma1.p": (
        0.66796875000, 0, "3.7 robustness, bootstrap p: snorm gamma1"),
    "rq2.robust.wildboot.thresh9.beta.score_ols.p": (
        0.03125000000, 0, "3.7 robustness, bootstrap p: thresh9 beta score_ols"),
    "rq2.robust.wildboot.thresh9.beta.score_rf.p": (
        0.07812500000, 0, "3.7 robustness, bootstrap p: thresh9 beta score_rf"),
    "rq2.robust.wildboot.thresh9.beta.uninsured_share.p": (
        0.57812500000, 0, "3.7 robustness, bootstrap p: thresh9 beta uninsured_share"),
    "rq2.robust.wildboot.thresh9.gamma1.p": (
        0.71875000000, 0, "3.7 robustness, bootstrap p: thresh9 gamma1"),
    "rq2.robust.wildboot.win_base.beta.score_ols.p": (
        0.04687500000, 0, "3.7 robustness, bootstrap p: win_base beta score_ols"),
    "rq2.robust.wildboot.win_base.beta.score_rf.p": (
        0.06835937500, 0, "3.7 robustness, bootstrap p: win_base beta score_rf"),
    "rq2.robust.wildboot.win_base.beta.uninsured_share.p": (
        0.98535156250, 0, "3.7 robustness, bootstrap p: win_base beta uninsured_share"),
    "rq2.robust.wildboot.win_base.gamma1.p": (
        0.66894531250, 0, "3.7 robustness, bootstrap p: win_base gamma1"),
    "rq2.robust.wildboot.win_m1p3.beta.score_ols.p": (
        0.05664062500, 0, "3.7 robustness, bootstrap p: win_m1p3 beta score_ols"),
    "rq2.robust.wildboot.win_m1p3.beta.score_rf.p": (
        0.08300781250, 0, "3.7 robustness, bootstrap p: win_m1p3 beta score_rf"),
    "rq2.robust.wildboot.win_m1p3.beta.uninsured_share.p": (
        0.32226562500, 0, "3.7 robustness, bootstrap p: win_m1p3 beta uninsured_share"),
    "rq2.robust.wildboot.win_m1p3.gamma1.p": (
        0.88867187500, 0, "3.7 robustness, bootstrap p: win_m1p3 gamma1"),
    "rq2.robust.wildboot.win_m5p5.beta.score_ols.p": (
        0.16894531250, 0, "3.7 robustness, bootstrap p: win_m5p5 beta score_ols"),
    "rq2.robust.wildboot.win_m5p5.beta.score_rf.p": (
        0.14746093750, 0, "3.7 robustness, bootstrap p: win_m5p5 beta score_rf"),
    "rq2.robust.wildboot.win_m5p5.beta.uninsured_share.p": (
        0.81933593750, 0, "3.7 robustness, bootstrap p: win_m5p5 beta uninsured_share"),
    "rq2.robust.wildboot.win_m5p5.gamma1.p": (
        0.58984375000, 0, "3.7 robustness, bootstrap p: win_m5p5 gamma1"),

    # ---- Task 2 — TABLE 5, the nested-tuning metrics (rq1_tuned_table.py). Parsed from
    #        rq1_h1b_robustness.txt section 3.4; nothing re-estimated.
    "rq1.tuned.listed.full.OLS.r2": (
        -0.02209562, 1e-8, "Table 5 nested tuning: listed, all 10 features, OLS OOS R2"),
    "rq1.tuned.listed.full.OLS.rmse": (
        0.13194880, 1e-8, "Table 5 nested tuning: listed, all 10 features, OLS OOS RMSE"),
    "rq1.tuned.listed.full.RF.r2": (
        -0.13476043, 1e-8, "Table 5 nested tuning: listed, all 10 features, RF OOS R2"),
    "rq1.tuned.listed.full.RF.rmse": (
        0.13903104, 1e-8, "Table 5 nested tuning: listed, all 10 features, RF OOS RMSE"),
    "rq1.tuned.listed.sig.OLS.r2": (
        0.02253105, 1e-8, "Table 5 nested tuning: listed, OLS-significant features, OLS OOS R2"),
    "rq1.tuned.listed.sig.OLS.rmse": (
        0.12903609, 1e-8, "Table 5 nested tuning: listed, OLS-significant features, OLS OOS RMSE"),
    "rq1.tuned.listed.sig.RF.r2": (
        -0.03104933, 1e-8, "Table 5 nested tuning: listed, OLS-significant features, RF OOS R2"),
    "rq1.tuned.listed.sig.RF.rmse": (
        0.13252549, 1e-8, "Table 5 nested tuning: listed, OLS-significant features, RF OOS RMSE"),
    "rq1.tuned.wide.full.OLS.r2": (
        0.02497642, 1e-8, "Table 5 nested tuning: wide, all 10 features, OLS OOS R2"),
    "rq1.tuned.wide.full.OLS.rmse": (
        0.09195066, 1e-8, "Table 5 nested tuning: wide, all 10 features, OLS OOS RMSE"),
    "rq1.tuned.wide.full.RF.r2": (
        0.00250510, 1e-8, "Table 5 nested tuning: wide, all 10 features, RF OOS R2"),
    "rq1.tuned.wide.full.RF.rmse": (
        0.09300421, 1e-8, "Table 5 nested tuning: wide, all 10 features, RF OOS RMSE"),
    "rq1.tuned.wide.sig.OLS.r2": (
        0.03395703, 1e-8, "Table 5 nested tuning: wide, OLS-significant features, OLS OOS R2"),
    "rq1.tuned.wide.sig.OLS.rmse": (
        0.09152622, 1e-8, "Table 5 nested tuning: wide, OLS-significant features, OLS OOS RMSE"),
    "rq1.tuned.wide.sig.RF.r2": (
        0.03052772, 1e-8, "Table 5 nested tuning: wide, OLS-significant features, RF OOS R2"),
    "rq1.tuned.wide.sig.RF.rmse": (
        0.09168852, 1e-8, "Table 5 nested tuning: wide, OLS-significant features, RF OOS RMSE"),

    # ---- Task 4 — MDE for the uninsured x Safeguard interaction, on the project's
    #        standard convention: 2.80 x the analytic event-clustered SE.
    "rq3.mde.b.uninsured_share.pct_of_car_sd": (
        25.86825066, 1e-8, "MDE for b on S x uninsured_share, % of a CAR SD"),
    "rq3.mde.b.uninsured_share.per_sd_pp": (
        1.24298773, 1e-8, "MDE for b on S x uninsured_share, pp of CAR per 1 SD"),
    "rq3.mde.b.uninsured_share.raw": (
        0.05975250, 1e-8, "MDE for b on S x uninsured_share, raw (analytic event-clustered SE)"),

    # ======================================================================= #
    #  ADDED 2026-08-26 — the pre-crisis comparison period (2022Q2).
    #
    #  rq1_placebo.py re-estimates the RQ1 specification on a
    #  NON-BANKING-STRESS period to establish whether the uninsured-deposit
    #  relationship in Table 3 is specific to the 2023 stress or a persistent
    #  structural correlate. Nothing existing is re-specified: the comparison
    #  period applies the SAME variables, screens and specification with the
    #  dates shifted, and its panel builder is proven output-identical to the
    #  frozen one (rebuilding 2022Q4 -> 2023Q1 reproduces panel_2022Q4_wide.csv
    #  and Table 3 column 2 exactly) before any of these numbers are computed.
    #
    #  Terminology: "pre-crisis comparison period" / "non-banking-stress
    #  comparison period". Never "calm period" — 2022Q2 is in fact the first
    #  quarter of the system-wide deposit contraction (see
    #  rq1_deposit_growth_by_quarter.csv).
    # ======================================================================= #

    # ---- STEP 0/L6 — period-specific regressions and samples ----
    "rq1.placebo.banks.2022Q2_only": (
        41, 0, "banks in the comparison period only"),
    "rq1.placebo.banks.2023Q1_only": (
        34, 0, "banks in the stress period only"),
    "rq1.placebo.banks.both": (
        919, 0, "banks appearing in BOTH periods"),
    "rq1.placebo.banks.unique": (
        994, 0, "unique banks in the pooled model"),
    "rq1.placebo.n.2022Q2": (
        960, 0, "N, pre-crisis comparison period (2022Q2), all screens at 2022Q1"),
    "rq1.placebo.n.2023Q1": (
        953, 0, "N, banking-stress period (2023Q1) = the frozen 953"),
    "rq1.placebo.uninsured.2022Q2.coef": (
        -0.00897881, 1e-8, "uninsured_share coef, comparison period OLS"),
    "rq1.placebo.uninsured.2022Q2.p": (
        0.50840653, 1e-8, "uninsured_share p, comparison period OLS"),
    "rq1.placebo.uninsured.2022Q2.t": (
        -0.66157141, 1e-8, "uninsured_share t, comparison period OLS"),
    "rq1.placebo.uninsured.2023Q1.coef": (
        -0.07713467, 1e-8, "uninsured_share coef, stress period (= Table 3 col 2)"),
    "rq1.placebo.uninsured.2023Q1.p": (
        0.00006744, 1e-8, "uninsured_share p, stress period (= Table 3 col 2)"),
    "rq1.placebo.uninsured.2023Q1.t": (
        -4.00307542, 1e-8, "uninsured_share t, stress period (= Table 3 col 2)"),

    # ---- L5 HEADLINE — the pooled two-period interaction, clustered by bank ----
    "rq1.placebo.pooled.interaction.coef": (
        -0.06671359, 1e-8, "HEADLINE: uninsured_share x 1[2023Q1] coef"),
    "rq1.placebo.pooled.interaction.p": (
        0.09527093, 1e-8, "HEADLINE: interaction p (bank-clustered)"),
    "rq1.placebo.pooled.interaction.se": (
        0.03995266, 1e-8, "HEADLINE: interaction SE, clustered by bank"),
    "rq1.placebo.pooled.interaction.t": (
        -1.66981605, 1e-8, "HEADLINE: interaction t"),
    "rq1.placebo.pooled.n_banks": (
        994, 0, "pooled unique banks (= clusters)"),
    "rq1.placebo.pooled.n_obs": (
        1913, 0, "pooled bank-period observations"),
    "rq1.placebo.pooled.stress.coef": (
        -0.01077912, 1e-8, "pooled model: 1[2023Q1] dummy coef"),
    "rq1.placebo.pooled.stress.p": (
        0.55013391, 1e-8, "pooled model: 1[2023Q1] dummy p"),
    "rq1.placebo.pooled.uninsured_main.coef": (
        -0.00592065, 1e-8, "pooled model: uninsured_share (pre-crisis slope)"),
    "rq1.placebo.pooled.uninsured_main.p": (
        0.77798748, 1e-8, "pooled model: uninsured_share p"),

    # ---- STEP 3 — the uninsured_share coefficient across ordinary quarters ----
    "rq1.placebo.mq.2021Q2.coef": (
        0.06489867, 1e-8, "ordinary-quarter series: uninsured_share coef, outcome 2021Q2"),
    "rq1.placebo.mq.2021Q2.p": (
        0.00000015, 1e-8, "ordinary-quarter series: uninsured_share p, outcome 2021Q2"),
    "rq1.placebo.mq.2021Q3.coef": (
        0.05592256, 1e-8, "ordinary-quarter series: uninsured_share coef, outcome 2021Q3"),
    "rq1.placebo.mq.2021Q3.p": (
        0.00000563, 1e-8, "ordinary-quarter series: uninsured_share p, outcome 2021Q3"),
    "rq1.placebo.mq.2021Q4.coef": (
        0.07643888, 1e-8, "ordinary-quarter series: uninsured_share coef, outcome 2021Q4"),
    "rq1.placebo.mq.2021Q4.p": (
        0.07685345, 1e-8, "ordinary-quarter series: uninsured_share p, outcome 2021Q4"),
    "rq1.placebo.mq.2022Q1.coef": (
        -0.04980422, 1e-8, "ordinary-quarter series: uninsured_share coef, outcome 2022Q1"),
    "rq1.placebo.mq.2022Q1.p": (
        0.03623411, 1e-8, "ordinary-quarter series: uninsured_share p, outcome 2022Q1"),
    "rq1.placebo.mq.2022Q2.coef": (
        -0.00897881, 1e-8, "ordinary-quarter series: uninsured_share coef, outcome 2022Q2"),
    "rq1.placebo.mq.2022Q2.p": (
        0.50840653, 1e-8, "ordinary-quarter series: uninsured_share p, outcome 2022Q2"),
}
