"""rq2_validation.py: inter-coder reliability for the sentence labels, in two
stages with hand coding in between.

    python scripts/rq2_validation.py template
        writes data/frozen/rq2_validation_template.csv, a stratified sample of
        about 150 sentences across all documents, over-sampling those likely to
        concern design. The columns human_mentions_design and
        human_design_stance are left blank for the coder, and the model's own
        labels are not in the file, so the coding is blind.

    python scripts/rq2_validation.py score
        run after the template is filled. Merges the hand codes with the model
        labels from rq2_sentence_labels.csv and writes percentage agreement,
        Cohen's kappa on mentions_design, weighted kappa and Krippendorff's
        alpha on design_stance, the confusion matrices, and a document-level
        check against the preliminary hand codes in the corpus spreadsheet.

Stratification uses a keyword heuristic rather than the model's labels, so the
sample is drawn independently of what is being judged.

Writes    data/processed/rq2_validation_results.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data") / "processed"
SENTENCES = PROC / "rq2_sentences.csv"
LABELS = PROC.parent / "frozen" / "rq2_sentence_labels.csv"
TEMPLATE = PROC.parent / "frozen" / "rq2_validation_template.csv"
RESULTS = PROC / "rq2_validation_results.txt"
XLSX = Path("data/raw/fed") / "Fed_Communications_Block3.xlsx"

SAMPLE_N = 150
SEED = 42

# Transparent design-likelihood heuristic for stratified over-sampling.
# not used for labelling — only to decide which sentences to over-sample so the
# ~150 has enough design sentences to estimate stance kappa.
DESIGN_KEYWORDS = [
    "non-interest", "interest-bearing", "interest bearing", "remunerat",
    "holding cap", "holding limit", "limit the amount", "limit on", "cap on",
    "hold or transfer", "quantity limit", "intermediated", "two-tier", "two tier",
    "unintermediated", "direct to", "disintermediat", "design feature",
    "design choice", "design of", "eligibility", "access to", "offline",
    "privacy", "convert", "convertib", "safeguard", "tiered",
]


def design_likely(sentence: str) -> bool:
    s = sentence.lower()
    return any(k in s for k in DESIGN_KEYWORDS)


# --------------------------------------------------------------------------- #
#  STAGE 1 — build the blind validation template
# --------------------------------------------------------------------------- #
def build_template() -> None:
    if not SENTENCES.exists():
        sys.exit(f"STOP: {SENTENCES} missing — run scripts/rq2_sentences.py first.")
    s = pd.read_csv(SENTENCES)
    s["design_likely"] = s.sentence.map(design_likely)
    rng = np.random.RandomState(SEED)

    # Over-sample likely-design sentences: target ~55% design-likely in the 150,
    # allocated across documents proportionally so every doc is represented.
    n_design = min(int(SAMPLE_N * 0.55), int(s.design_likely.sum()))
    n_other = SAMPLE_N - n_design

    def stratified(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        # proportional-per-doc with at least 1 where possible
        picks = []
        docs = pool.doc_id.unique()
        base = {d: max(1, round(n * (pool.doc_id == d).sum() / len(pool)))
                for d in docs}
        for d, k in base.items():
            g = pool[pool.doc_id == d]
            picks.append(g.sample(min(k, len(g)), random_state=rng))
        out = pd.concat(picks)
        if len(out) > n:
            out = out.sample(n, random_state=rng)
        return out

    design = stratified(s[s.design_likely], n_design)
    other = stratified(s[~s.design_likely].drop(index=design.index, errors="ignore"),
                       n_other)
    sample = (pd.concat([design, other])
              .drop_duplicates("sent_id")
              .sort_values(["doc_id", "sent_index"]))

    tmpl = pd.DataFrame({
        "doc_id": sample.doc_id.astype(int),
        "sent_id": sample.sent_id,
        "sentence": sample.sentence,
        "human_mentions_design": "",   # analyst fills: true / false
        "human_design_stance": "",     # analyst fills: -1 / 0 / 1
    })
    PROC.mkdir(parents=True, exist_ok=True)
    tmpl.to_csv(TEMPLATE, index=False)
    print(f"wrote {TEMPLATE}: {len(tmpl)} sentences "
          f"({int(sample.design_likely.sum())} design-likely, "
          f"{len(tmpl) - int(sample.design_likely.sum())} other) "
          f"across {tmpl.doc_id.nunique()} docs")
    print("  Fill human_mentions_design (true/false) and human_design_stance "
          "(-1/0/1), then run: python scripts/rq2_validation.py score")


# --------------------------------------------------------------------------- #
#  STAGE 2 — kappa / alpha / confusion
# --------------------------------------------------------------------------- #
def _to_bool(x):
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def cohen_kappa(a: list, b: list) -> tuple[float, float]:
    """Cohen's kappa for two coders. Returns (kappa, pct_agreement)."""
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    row, col = m.sum(1) / n, m.sum(0) / n
    pe = float((row * col).sum())
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return kappa, po


def weighted_kappa(a: list, b: list, cats=(-1, 0, 1)) -> float:
    """Linear-weighted kappa for the ordinal stance labels."""
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    W = np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    row, col = O.sum(1), O.sum(0)
    E = np.outer(row, col) / n
    denom = (W * E).sum()
    return 1 - (W * O).sum() / denom if denom else float("nan")


def krippendorff_alpha_ordinal(a: list, b: list, cats=(-1, 0, 1)) -> float:
    """Krippendorff's alpha, ordinal metric, two coders, no missing."""
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    coincidence = np.zeros((k, k))
    for x, y in zip(a, b):
        i, j = idx[x], idx[y]
        coincidence[i, j] += 1
        coincidence[j, i] += 1
    n_total = coincidence.sum()
    if n_total == 0:
        return float("nan")
    nc = coincidence.sum(0)

    def ordinal_delta(c, k_):
        lo, hi = sorted((c, k_))
        s = sum(nc[g] for g in range(lo, hi + 1)) - (nc[c] + nc[k_]) / 2
        return s ** 2

    Do = sum(coincidence[c, k_] * ordinal_delta(c, k_)
             for c in range(k) for k_ in range(k))
    De = 0.0
    for c in range(k):
        for k_ in range(k):
            De += nc[c] * nc[k_] * ordinal_delta(c, k_)
    De = De / (n_total - 1)
    return 1 - (Do / De) if De else float("nan")


def confusion(a: list, b: list, cats) -> str:
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), int)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    hdr = "human\\claude  " + "  ".join(f"{c!s:>6}" for c in cats)
    lines = [hdr]
    for c in cats:
        lines.append(f"{c!s:>10}    " +
                     "  ".join(f"{m[idx[c], idx[cc]]:>6}" for cc in cats))
    return "\n".join(lines)


def score() -> None:
    if not TEMPLATE.exists():
        sys.exit(f"STOP: {TEMPLATE} missing — run the 'template' stage and fill it.")
    if not LABELS.exists():
        sys.exit(f"STOP: {LABELS} missing — run scripts/rq2_classify.py first.")

    human = pd.read_csv(TEMPLATE, dtype=str)
    filled = human[(human.human_mentions_design.astype(str).str.strip() != "")
                   & (human.human_mentions_design.notna())]
    if filled.empty:
        sys.exit("STOP: the template has no filled human codes yet.")

    claude = pd.read_csv(LABELS)
    m = filled.merge(claude[["sent_id", "mentions_design", "design_stance",
                             "flagged"]], on="sent_id", how="left")
    m = m[m.flagged != True]  # noqa: E712 — drop unparsed Claude rows

    m["h_md"] = m.human_mentions_design.map(_to_bool)
    m["c_md"] = m.mentions_design.map(_to_bool)
    m = m[m.h_md.notna() & m.c_md.notna()].copy()
    m["h_st"] = pd.to_numeric(m.human_design_stance, errors="coerce").astype("Int64")
    m["c_st"] = pd.to_numeric(m.design_stance, errors="coerce").astype("Int64")

    lines = []
    add = lines.append
    add("=" * 70)
    add("RQ2 VALIDATION — inter-coder reliability (human vs Claude)")
    add("=" * 70)
    add(f"template: {TEMPLATE.name}   labels: {LABELS.name}")
    add(f"filled & merged sentences: {len(m)}")
    add("")

    # mentions_design — Cohen's kappa
    k_md, po_md = cohen_kappa(list(m.h_md), list(m.c_md))
    add("--- mentions_design (binary) ---")
    add(f"  % agreement     : {po_md:.3f}")
    add(f"  Cohen's kappa   : {k_md:.3f}   ({kappa_label(k_md)})")
    add("  confusion (rows=human, cols=Claude):")
    add(confusion(list(m.h_md), list(m.c_md), [False, True]))
    add("")

    # design_stance — only where BOTH coders said design present (stance is defined)
    both = m[(m.h_md) & (m.c_md)].copy()
    both = both[both.h_st.notna() & both.c_st.notna()]
    add("--- design_stance (-1/0/1, where both coded mentions_design=true) ---")
    add(f"  n = {len(both)}")
    if len(both) >= 2:
        h = [int(x) for x in both.h_st]
        c = [int(x) for x in both.c_st]
        _, po_st = cohen_kappa(h, c)
        add(f"  % agreement            : {po_st:.3f}")
        add(f"  linear-weighted kappa  : {weighted_kappa(h, c):.3f}")
        add(f"  Krippendorff alpha (ord): {krippendorff_alpha_ordinal(h, c):.3f}")
        add("  confusion (rows=human, cols=Claude):")
        add(confusion(h, c, [-1, 0, 1]))
    else:
        add("  too few jointly-design sentences to estimate stance kappa.")
    add("")

    # weak-agreement warning
    if k_md < 0.6:
        add("!! mentions_design kappa < 0.60 — agreement is weak. Refine the "
            "codebook/prompt and re-run classification before trusting S.")
    add("")

    # free doc-level secondary check vs preliminary hand codes
    add("--- secondary check: sentence-aggregated vs preliminary doc hand-codes ---")
    try:
        pre = pd.read_excel(XLSX, sheet_name="communications")[["id", "mentions_design"]]
        pre = pre.rename(columns={"id": "doc_id"})
        pre["hand_any_design"] = pre.mentions_design.astype(str).str.strip().str.lower().eq("yes")
        agg = (claude[claude.flagged != True]  # noqa: E712
               .groupby("doc_id").mentions_design.apply(
                   lambda x: bool(pd.to_numeric(
                       x.map(_to_bool).astype("boolean")).fillna(0).sum() > 0)))
        cmp = pre.merge(agg.rename("claude_any_design"), on="doc_id", how="inner")
        match = int((cmp.hand_any_design == cmp.claude_any_design).sum())
        add(f"  docs compared: {len(cmp)}   doc-level agreement: "
            f"{match}/{len(cmp)}")
        for _, r in cmp.iterrows():
            flag = "" if r.hand_any_design == r.claude_any_design else "  <-- differs"
            add(f"    doc {int(r.doc_id):>2}: hand={r.hand_any_design!s:>5}  "
                f"claude={r.claude_any_design!s:>5}{flag}")
    except Exception as e:  # noqa: BLE001
        add(f"  (skipped: {e})")

    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS}")


def kappa_label(k: float) -> str:
    if k < 0: return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "template":
        build_template()
    elif stage == "score":
        score()
    else:
        sys.exit("usage: python scripts/rq2_validation.py [template|score]")


if __name__ == "__main__":
    main()
