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

> **Not in the public repository.** `crosswalk_rssd_permno.csv` held the
> RSSD-to-CRSP-`permno` mapping. CRSP identifiers are part of a licensed
> product, so the file is omitted and the `permno` column has been removed
> from every CSV here. Banks are still identified by `bank_IDRSSD`. See
> [`DATA.md`](../../DATA.md) for how a WRDS subscriber rebuilds it.
