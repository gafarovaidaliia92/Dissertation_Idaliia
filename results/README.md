# Results

Assembled by `scripts/collect_results.py` — 2026-08-24.

`data/processed/` is the pipeline's working directory; this folder is the
shop window. Rebuild after any re-run:

```bash
python3 scripts/collect_results.py
```

## Sections

| Folder | Question | Verdict |
|---|---|---|
| `rq1_vulnerability/` | H1a/H1b — balance-sheet model of outflow | H1a partial, H1b **not robust** |
| `rq2_communications/` | H2 — did the market react, on average? | **no reaction**, gamma1 null |
| `rq3_bridge/` | H3 — do the two proxies meet? | **mechanism not supported** |
| `shared_inputs/` | sample, crosswalk, reconciliation | 278 banks; 220/220 numbers reconcile |
| `_archive/` | superseded versions | incl. the interaction-as-H2 era |

## The through-line

The market did not react to CBDC communications at all (0 of 11 events,
with volatility and turnover suppressed on 7 of 11). With no average
reaction there is nothing for the communication type to modulate (RQ2
gamma1 null) and nothing for a cross-section of vulnerability to explain
(RQ3 mechanism unsupported). The three nulls are one finding, not three.

Two caveats that must travel with that sentence:

- the nulls are **not equally tight**. RQ3's coefficients are bounded at
  ~5-9.5% of a CAR SD; RQ2's gamma1 only at ~49%. A modest average effect
  is not excluded.
- `uninsured_share` **does** have a real level effect on CAR
  (-0.01877493, p = 0.00032066). It is not a reaction to CBDC content —
  its interaction with the signal is null — but it is not nothing.
