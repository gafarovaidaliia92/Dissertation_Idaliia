# data/frozen/ — the inputs no script can rebuild

Everything else under `data/` is disposable: `data/raw/` is downloaded and
`data/processed/` is written by the pipeline, and both are gitignored. The files
here are different. Nothing in `scripts/` produces them, so if they are lost the
project cannot be reconstructed from this repository. They are therefore the one
part of `data/` kept under version control, via a targeted exception in
`.gitignore`.

## The four inputs

| File | What it is | Why it cannot be rebuilt |
|---|---|---|
| `rq2_sentence_labels.csv` | the LLM's design/stance label for each of the 1,338 sentences | produced by an LLM run (`_rq2_classify.py`, `claude-sonnet-4-5` at temperature 0). Re-running costs an API call and is not bit-reproducible once the model version moves. Every RQ2 and RQ3 number is conditioned on these labels |
| `rq2_validation_template.csv` | 149 sentences hand-coded by the author, blind to the model's labels | human coding. It is the basis of the Cohen's kappa 0.839 and weighted kappa 0.847 in Section 4.3 |
| `sample_banks.csv` | the 278 listed banks: bank_IDRSSD, holder_RSSD, permno, name, failed | assembled outside the repository |
| `crosswalk_rssd_permno.csv` | RSSD ↔ permno, 342 bank RSSDs mapping to 278 permnos | built by hand from the FFIEC NIC relationship files (bank → holding company → permco → permno). That step does not exist in the pipeline |

`rq2_signal.py` refuses to overwrite the labels without `--force`, and
`_rq2_classify.py` stops rather than fabricating them if credentials are missing.

## superseded/

Snapshots of earlier pipeline runs. **The pipeline does not regenerate them** —
the scripts that produced them are gone (see `results/_archive/README.md`) — but
`collect_results.py` still reads them: `ARCHIVE` maps each one into
`results/_archive/` under a descriptive name, which is how a number quoted in an
earlier draft can still be traced.

All eleven sources of `results/_archive/` live here. The RQ1 group, before the
supervisor's robustness runs:

| Snapshot | Becomes, in `results/_archive/` |
|---|---|
| `comparison_narrow_vs_wide_prev.txt` | `rq1_comparison_before_robustness.txt` |
| `rq1_results_wide_prev.txt` | `rq1_results_wide_before_robustness.txt` |
| `rq1_results_narrow_prev.txt` | `rq1_results_narrow_before_robustness.txt` |
| `vulnerability_scores_wide_prev.csv` | `rq1_scores_wide_before_robustness.csv` |

The RQ2/RQ3 group, spanning the re-split that moved the Safeguard × vulnerability
interaction out of H2 and into RQ3 (oldest first):

| Snapshot | Becomes, in `results/_archive/` |
|---|---|
| `rq2_event_regression_interaction.txt` | `rq2_h2_v1_interaction_as_h2.txt` |
| `rq2_event_regression_prev.txt` | `rq2_h2_v2_car_on_S.txt` |
| `rq2_event_regression.txt` | `rq2_h2_v3_combined_before_resplit.txt` |
| `rq2_measures_compare.txt` | `rq3_measures_v1_inside_rq2.txt` |
| `rq2_car_sanity_interaction.txt` | `rq2_car_sanity_v1.txt` |
| `rq2_car_sanity_prev.txt` | `rq2_car_sanity_v2.txt` |
| `rq2_enrichment_results_interaction.txt` | `rq2_enrichment_v1.txt` |

They were moved here from `data/processed/` because that directory is gitignored
and is wiped by any clean rebuild, which would have left `results/_archive/`
permanently un-rebuildable. Verified by destruction: deleting `results/_archive/`
outright and re-running `collect_results.py` restores all twelve files from this
directory alone.

## Do not edit anything in this directory

These files are inputs, not outputs. Changing one changes published results
without any stage of the pipeline recording that it happened.

> **Not in the public repository.** `crosswalk_rssd_permno.csv` held the
> RSSD-to-CRSP-`permno` mapping. CRSP identifiers are part of a licensed
> product, so the file is omitted and the `permno` column has been removed
> from every CSV here. Banks are still identified by `bank_IDRSSD`. See
> [`DATA.md`](../../DATA.md) for how a WRDS subscriber rebuilds it.
