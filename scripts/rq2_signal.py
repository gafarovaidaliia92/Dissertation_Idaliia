"""
rq2_signal.py — RQ2, step 1: build the Safeguard signal S from the 11 Federal
Reserve CBDC communications.

    S = (protective - expansive) / n_design_sentences

computed per document, so S is an EVENT-level variable with 11 values (10
distinct — two documents both score 1.00). Higher S = the communication puts more
weight on protective design (holding limits, non-remuneration, intermediation).

Pipeline, in order:
    _rq2_sentences.py  segment the 11 documents into sentences (1338 of them)
    _rq2_classify.py   label each sentence protective / expansive / neutral-design
                       with an LLM, against rq2_codebook.md
    _rq2_score.py      aggregate to the 11 document scores

DO NOT RE-RUN CASUALLY. The classification step calls an LLM. Even at
temperature 0 a re-run can differ, and every downstream number in RQ2 and RQ3 is
conditioned on the labels in rq2_sentence_labels.csv. Those labels are FROZEN.
This script refuses to overwrite them unless --force is passed, and
rq2_run_metadata.json records the model, temperature and codebook hash that
produced them.

Inter-coder reliability against a human coder is in rq2_validation.py:
Cohen's kappa 0.839 on whether a sentence is about design, weighted kappa 0.847
on the stance. That is the evidence the labels are trustworthy.

MEASUREMENT CAVEAT to carry into the write-up: design-sentence density ranges
from 2 to 65 per document, and BOTH documents scoring S = +1.00 rest on 3 and 4
sentences. The extremes of S are its noisiest points.

Outputs   data/processed/rq2_sentences.csv
          data/frozen/rq2_sentence_labels.csv   FROZEN
          data/processed/rq2_safeguard_scores.csv  the 11-row S table
          data/processed/rq2_run_metadata.json
"""

from __future__ import annotations

import sys

import config as C


def main() -> None:
    force = "--force" in sys.argv
    if C.RQ2_LABELS.exists() and not force:
        print(f"[rq2_signal] FROZEN labels present: {C.RQ2_LABELS.name}")
        print("[rq2_signal] the classification step calls an LLM and is NOT re-run.")
        print("[rq2_signal] every RQ2/RQ3 number depends on these labels.")
        print("[rq2_signal] pass --force only if you intend to regenerate the signal.")
        if C.RQ2_SCORES.exists():
            print(f"[rq2_signal] S table present: {C.RQ2_SCORES.name}")
            return
        import _rq2_score
        _rq2_score.main()
        return

    import _rq2_classify
    import _rq2_score
    import _rq2_sentences
    _rq2_sentences.main()
    _rq2_classify.main()
    _rq2_score.main()


if __name__ == "__main__":
    main()
