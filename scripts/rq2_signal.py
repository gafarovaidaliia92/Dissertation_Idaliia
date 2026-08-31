"""rq2_signal.py: builds the Safeguard signal S from the eleven Federal Reserve
communications.

    S = (protective - expansive) / n_design_sentences

computed per document, so S is an event-level variable taking eleven values, ten
of them distinct because two documents both score 1.00. A higher value means the
communication puts more weight on protective design: holding limits,
non-remuneration, intermediation.

Pipeline:
    _rq2_sentences.py  segments the eleven documents into 1338 sentences
    _rq2_classify.py   labels each sentence against docs/rq2_codebook.md
    _rq2_score.py      aggregates to the eleven document scores

The classification step calls a language model, and every RQ2 and RQ3 number is
conditioned on the resulting labels, so those labels are frozen in
data/frozen/rq2_sentence_labels.csv. This script refuses to overwrite them
without --force, and rq2_run_metadata.json records the model, the temperature
and the SHA-256 of the codebook that produced them. Reliability against the
author's hand coding is reported by rq2_validation.py: Cohen's kappa 0.839 on
whether a sentence concerns design, weighted kappa 0.847 on its stance.

Design-sentence density ranges from 2 to 65 per document, and both documents
scoring S = +1.00 rest on 3 and 4 sentences, so the extremes of S are its
noisiest points. rq2_robustness.py reports the threshold and normalised variants
that address this.

Outputs   data/processed/rq2_sentences.csv
          data/frozen/rq2_sentence_labels.csv
          data/processed/rq2_safeguard_scores.csv
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
