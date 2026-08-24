"""
rq2_score.py — RQ2 step 3: aggregate sentence labels to one Safeguard score S per
document, ready to feed the event study.

Reads : data/frozen/rq2_sentence_labels.csv, data/raw/fed/Fed_Communications_Block3.xlsx
Writes: data/processed/rq2_safeguard_scores.csv

Per document:
    S      = (#stance==+1  -  #stance==-1) / (#mentions_design==true)
    S_norm = (#stance==+1  -  #stance==-1) / (#sentences)              [robustness]

Higher S = more protective ("Safeguard") design language. A document with zero
design sentences gets S = 0 and design_zero_flag = True (S is undefined there; 0
is a convention, not a measurement — the flag ensures no one reads S blind).

Flagged/unparsed sentences are excluded from the counts and reported separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROC = Path("data") / "processed"
LABELS = PROC.parent / "frozen" / "rq2_sentence_labels.csv"
XLSX = Path("data/raw/fed") / "Fed_Communications_Block3.xlsx"
OUT = PROC / "rq2_safeguard_scores.csv"


def main() -> None:
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} missing — run scripts/rq2_classify.py first.")
    df = pd.read_csv(LABELS)

    meta = (pd.read_excel(XLSX, sheet_name="communications")[["id", "date", "speaker"]]
            .rename(columns={"id": "doc_id"}))
    meta["doc_id"] = meta["doc_id"].astype(int)
    meta["date"] = meta["date"].astype(str).str[:10]

    rows = []
    for doc_id, g in df.groupby("doc_id"):
        total = len(g)
        flagged = int(g.flagged.sum()) if "flagged" in g else 0
        ok = g[g.flagged != True] if "flagged" in g else g  # noqa: E712
        design = ok[ok.mentions_design == True]  # noqa: E712
        n_design = len(design)
        n_pos = int((design.design_stance == 1).sum())
        n_neg = int((design.design_stance == -1).sum())
        n_neu = int((design.design_stance == 0).sum())
        zero = n_design == 0
        S = 0.0 if zero else (n_pos - n_neg) / n_design
        S_norm = (n_pos - n_neg) / total if total else 0.0
        rows.append({
            "doc_id": int(doc_id),
            "n_sentences": total, "n_flagged": flagged,
            "n_design": n_design, "n_protective": n_pos,
            "n_expansive": n_neg, "n_neutral_design": n_neu,
            "S": round(S, 4), "S_norm": round(S_norm, 4),
            "design_zero_flag": zero,
        })

    scores = (pd.DataFrame(rows)
              .merge(meta, on="doc_id", how="left")
              .sort_values("date"))
    cols = ["doc_id", "date", "speaker", "n_sentences", "n_flagged", "n_design",
            "n_protective", "n_expansive", "n_neutral_design", "S", "S_norm",
            "design_zero_flag"]
    scores = scores[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT, index=False)

    print(f"wrote {OUT} ({len(scores)} documents)\n")
    print(scores.to_string(index=False))
    valid = scores[~scores.design_zero_flag]
    if len(valid):
        print("\nmost PROTECTIVE (high S):")
        print(valid.sort_values("S", ascending=False)
              .head(3)[["doc_id", "date", "speaker", "S", "n_design"]].to_string(index=False))
        print("\nmost EXPANSIVE (low S):")
        print(valid.sort_values("S")
              .head(3)[["doc_id", "date", "speaker", "S", "n_design"]].to_string(index=False))
    zeros = scores[scores.design_zero_flag]
    if len(zeros):
        print(f"\ndesign_zero_flag=True (S=0 by convention): "
              f"docs {list(zeros.doc_id)}")


if __name__ == "__main__":
    main()
