# Archive — superseded versions

Nothing here is current. It is kept so any number quoted in an earlier draft can
still be traced. Names are chronological.

## The one conceptual change

The Safeguard x vulnerability interaction used to be presented as the **H2** test
inside RQ2. That was a mis-filing: H2 is about the AVERAGE effect of the
communication type, while the interaction asks whether VULNERABLE banks react
differently — which is H3. It now lives in `../rq3_bridge/`.

| File | What it was |
|---|---|
| `rq2_h2_v1_interaction_as_h2.txt` | v1 — the interaction presented as the H2 headline |
| `rq2_h2_v2_car_on_S.txt` | v2 — after H2 was realigned to CAR ~ S, interaction demoted |
| `rq2_h2_v3_combined_before_resplit.txt` | v3 — RQ2 and the interaction in one file, before the split |
| `rq3_measures_v1_inside_rq2.txt` | the three-measure comparison while it still sat in RQ2 |
| `rq2_car_sanity_v1.txt`, `rq2_car_sanity_v2.txt` | earlier per-event CAR sanity tables |
| `rq2_enrichment_v1.txt` | the enrichment report before the H2 rework |
| `rq1_comparison_before_robustness.txt` | main RQ1 document before the robustness runs |
| `rq1_results_{wide,narrow}_before_robustness.txt` | per-population reports, same vintage |
| `rq1_scores_wide_before_robustness.csv` | the RQ3 input score, same vintage |

Two superseded scripts that also lived here were not retained
and were never under version control. Their outputs are the `.txt` files above,
and their logic survives in the current `rq2_avg_effect.py`,
`rq3_interaction.py` and `rq3_measures.py`.

Every current number is reconciled against these in
`../shared_inputs/reconciliation.txt`.
