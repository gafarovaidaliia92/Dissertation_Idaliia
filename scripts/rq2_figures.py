"""rq2_figures.py: Figure 2, the Safeguard score by event.

Draws the eleven document scores in calendar order with the least-squares trend
over event order, the same time measure as the first row of Appendix C, Table
C2. Nothing is estimated here: S is read as published and the only computation
is that trend line.

Also writes the Appendix B corpus table, in which every column is computed from
the pipeline: the number of sentences per document, the number the classifier
marks as describing a concrete design feature, their split into protective and
expansive, and the resulting score.

Reads     data/processed/rq2_safeguard_scores.csv
Writes    data/processed/rq2_safeguard_by_event.png
          data/processed/rq2_corpus_table.txt
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

import config as C                         # noqa: E402

TEAL = "#2a9d8f"      # protective
AMBER = "#e0a458"     # expansive
TREND = "#3d3d3d"


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(C.RQ2_SCORES)
    ev["date"] = pd.to_datetime(ev["date"])
    return ev.sort_values("date").reset_index(drop=True)


def draw(ev: pd.DataFrame, out_path) -> None:
    x = np.arange(len(ev))                 # event order, evenly spaced
    s = ev["S"].to_numpy(dtype=float)
    colours = [TEAL if v >= 0 else AMBER for v in s]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x, s, color=colours, width=0.62, zorder=3)

    # linear trend over event order, as the caption specifies
    slope, intercept = np.polyfit(x, s, 1)
    ax.plot(x, slope * x + intercept, linestyle="--", linewidth=1.6,
            color=TREND, zorder=4,
            label=f"linear trend (slope {slope:+.3f} per event)")

    ax.axhline(0, color="#999999", linewidth=0.9, zorder=2)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel("Safeguard score $S$")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in ev["date"]],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.7, len(ev) - 0.3)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # value labels, placed clear of the bar in the direction it points
    for xi, v in zip(x, s):
        ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=8,
                    color="#444444")

    ax.set_title("Safeguard score $S$ for the eleven Federal Reserve "
                 "CBDC communications, 2019–2023", fontsize=11, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def corpus_table(ev: pd.DataFrame) -> str:
    """Appendix B, rebuilt from the pipeline.

    The published version carries hand-assigned Design and Valence columns that
    the signal contradicts: four communications marked "No design" in fact carry
    2, 16, 14 and 4 design sentences, and the Quarles speech marked "No" has more
    design content than five of those marked "Yes". Counts replace judgements."""
    meta = pd.read_excel(C.COMMS_XLSX, sheet_name="communications")
    meta = meta.rename(columns={"id": "doc_id"})[["doc_id", "title", "type"]]
    d = ev.merge(meta, on="doc_id", how="left").sort_values("date")

    L = ["=" * 118,
         "APPENDIX B — THE ELEVEN FEDERAL RESERVE COMMUNICATIONS",
         "=" * 118, "",
         "  Every column below is computed from the pipeline. n_design is the",
         "  number of sentences the classifier marks as describing a concrete CBDC",
         "  design feature; n_protective and n_expansive are their stances; and",
         "  S = (n_protective - n_expansive) / n_design is the Safeguard score.",
         "  All eleven communications carry design sentences, so all eleven",
         "  contribute to S.", "",
         "  {:<12s} {:<26s} {:>7s} {:>9s} {:>7s} {:>7s} {:>9s}  {}".format(
             "date", "speaker", "n_sent", "n_design", "n_prot", "n_exp", "S", "title"),
         "  " + "-" * 116]
    for r in d.itertuples():
        L.append("  {:<12s} {:<26s} {:>7d} {:>9d} {:>7d} {:>7d} {:>+9.4f}  {}".format(
            r.date.strftime("%Y-%m-%d"), str(r.speaker)[:26], int(r.n_sentences),
            int(r.n_design), int(r.n_protective), int(r.n_expansive), r.S,
            str(r.title)[:60]))
    L += ["  " + "-" * 116,
          "  {:<12s} {:<26s} {:>7d} {:>9d} {:>7d} {:>7d}".format(
              "TOTAL", "", int(d.n_sentences.sum()), int(d.n_design.sum()),
              int(d.n_protective.sum()), int(d.n_expansive.sum())),
          "",
          f"  S ranges {d.S.min():+.4f} to {d.S.max():+.4f}; "
          f"{int((d.S > 0).sum())} score above zero, {int((d.S < 0).sum())} below, "
          f"{int((d.S == 0).sum())} exactly zero.",
          f"  Design sentences per document range {int(d.n_design.min())} to "
          f"{int(d.n_design.max())}.",
          "", "=" * 118, ""]
    return "\n".join(L)


def main() -> None:
    ev = load_events()
    if len(ev) != 11:
        raise SystemExit(f"expected 11 events, found {len(ev)}")
    draw(ev, C.RQ2_FIG_SIGNAL)
    pos = int((ev.S > 0).sum())
    neg = int((ev.S < 0).sum())
    zero = int((ev.S == 0).sum())
    print(f"[rq2_figures] {len(ev)} events: {pos} protective, {neg} expansive, "
          f"{zero} exactly zero")
    print(f"[rq2_figures] S range {ev.S.min():+.4f} .. {ev.S.max():+.4f}")
    print(f"wrote {C.RQ2_FIG_SIGNAL}")
    table = corpus_table(ev)
    C.RQ2_CORPUS_TABLE.write_text(table)
    print(table)
    print(f"wrote {C.RQ2_CORPUS_TABLE}")


if __name__ == "__main__":
    main()
