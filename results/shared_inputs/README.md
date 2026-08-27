# Shared inputs and verification

| File | What it is |
|---|---|
| `sample_banks.csv` | 278 listed banks: bank_IDRSSD, holder_RSSD, permno, name, failed |
| `crosswalk_rssd_permno.csv` | RSSD <-> permno crosswalk |
| `headline_numbers.json` | every headline coefficient the pipeline recomputes |
| `reconciliation.txt` | **proof the refactor did not move any number** |

**Reproducibility note.** Neither `sample_banks.csv` nor the crosswalk is built
by a script in `scripts/` — they were assembled outside the repo (via NIC:
bank -> holder -> permco -> permno, N=278). If the sample ever has to be rebuilt,
that step does not exist in the pipeline. Both therefore live in `data/frozen/`,
which is under version control, rather than in the disposable `data/processed/`.

RQ1 models 277 banks: IDRSSD 119528 (Farmers National Bank of Emlenton, permno
93131) is excluded as a merger exit with a null outcome.
