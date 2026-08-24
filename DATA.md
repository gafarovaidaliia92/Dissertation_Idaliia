# Data: sources, what ships here, and what does not

The **code** in this repository is under the MIT licence (see `LICENSE`). The
**data** is not uniformly licensable, so this file states, source by source,
what is included and on what terms.

The short version: everything derived from public regulatory filings is here;
nothing derived from the licensed CRSP database is, beyond the estimated
coefficients that constitute the study's results.

## What is in this repository

| Path | What it is | Terms |
|---|---|---|
| `scripts/` | the whole pipeline | MIT (`LICENSE`) |
| `results/` | the published outputs: reports, figures, and the bank-event panels | derived estimates — see the CRSP note below |
| `data/raw/fed/` | the eleven-document Federal Reserve corpus (xlsx), the *Money and Payments* report (PDF) and the extracted body text of document 1 | works of the US Government, public domain |
| `data/frozen/rq2_sentence_labels.csv` | the LLM sentence labels behind the Safeguard signal | produced by this project |
| `data/frozen/rq2_validation_template.csv` | ~150 sentences hand-coded by the author for the reliability check | produced by this project |
| `data/frozen/sample_banks.csv` | the 278 listed banks, by FFIEC RSSD | FFIEC identifiers, public |
| `data/frozen/superseded/` | superseded snapshots kept as an audit trail | as above |

## What is NOT in this repository, and why

**CRSP daily stock data (`data/raw/wrds/`).** Daily returns, the value-weighted
index and delisting returns come from CRSP through WRDS under a subscription
that does not permit redistribution. None of it is here. Reproducing the event
study requires your own WRDS access.

**CRSP identifiers (`permno`, `permco`).** These are CRSP's own security
identifiers and are part of the licensed product, so they have been removed from
every CSV published here, and `crosswalk_rssd_permno.csv` — whose only content
was the RSSD-to-permno mapping — is omitted entirely. Banks remain fully
identified by `bank_IDRSSD` and `name`, which are public FFIEC fields, so every
table and panel here can still be read, joined and checked.

*If you have WRDS access* and want to re-run `scripts/rq2_car.py`, you need to
restore a `permno` column on `data/frozen/sample_banks.csv`. Build it by
matching each bank's RSSD to its holding company in the FFIEC NIC relationship
files and then to a CRSP `permco`/`permno` in the CRSP `stocknames` table; that
is exactly how the omitted crosswalk was originally constructed.

**The bulk public inputs (`data/raw/` other than `fed/`).** Sixteen quarterly
Call Report archives, the MDRM dictionary, the NIC relationship and attribute
files, the FDIC extracts and the cached factor downloads come to roughly 280 MB.
All of it is free to download, so it is left out for size rather than for
licensing. `README.md` lists each source and where to get it.

**The dissertation text.** Not included.

## About the results that *are* published

`results/rq2_communications/rq2_car.csv` and the panels derived from it carry
`CAR`, `alpha` and `beta`. These are **estimated statistics** — the output of a
market-model regression — not CRSP data. Publishing estimates computed from
licensed inputs is the normal practice for a replication package, and no raw or
near-raw CRSP series appears in any file here. The Call Report characteristics
alongside them (`uninsured_share`, `size`, `ROA`, `NPL_ratio`, `equity_ratio`,
`deposit_reliance`) are computed from public FFIEC filings.

## Verifying without any data at all

```bash
python3 scripts/run_all.py --check-results
```

reads the published numbers from `results/shared_inputs/` and reconciles all 220
of them against `scripts/config.py`'s frozen values. It needs no `data/`
directory and no subscription.

## Source list

| Source | Used for | Access |
|---|---|---|
| FFIEC Call Reports (CDR Bulk Data) | all bank characteristics, deposit growth | public |
| FFIEC MDRM dictionary | mapping reporting codes to variables | public |
| FFIEC NIC | RSSD-to-holding-company structure | public |
| FDIC BankFind | charter classification, failed-bank list | public |
| Federal Reserve Board | the eleven-document CBDC corpus | public domain |
| Kenneth R. French Data Library | daily factors, 49 industry portfolios | free for research |
| FRED (DGS2) | two-year Treasury yield | public |
| **CRSP via WRDS** | **daily returns, market index, delisting returns** | **subscription; not redistributed here** |
