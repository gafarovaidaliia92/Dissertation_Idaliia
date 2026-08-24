"""
rq2_sentences.py — RQ2 step 1: split each of the 11 Fed CBDC communications into
sentences with a real tokenizer, robust to "U.S.", "e.g.", "Mr.", decimals and
numbered lists.

Reads   : data/raw/fed/Fed_Communications_Block3.xlsx (sheet "communications")
          data/raw/fed/fulltext_01_money_and_payments.txt (doc id=1 body;
          its full_text exceeds Excel's ~32,767-char cell limit and is stored
          separately)
Writes  : data/processed/rq2_sentences.csv
          columns: sent_id, doc_id, sent_index, date, speaker, sentence

Stable sent_id = f"{doc_id}-{sent_index}" (sent_index is 0-based within a document).

NB: this is the TEXT half of RQ2 only (sentence prep). No API calls here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    import nltk
    from nltk.tokenize import PunktTokenizer
except ImportError as e:  # pragma: no cover
    raise SystemExit("nltk is required: pip install nltk") from e

XLSX = Path("data/raw/fed") / "Fed_Communications_Block3.xlsx"
SHEET = "communications"
# Doc id=1's body lives in a separate .txt because it exceeds Excel's cell limit.
# The stub in the spreadsheet names this file. If it is absent, doc 1 is skipped
# and reported (never silently emitted from the 464-char stub).
DOC1_TXT = Path("data/raw/fed") / "fulltext_01_money_and_payments.txt"
OUT = Path("data") / "processed" / "rq2_sentences.csv"

# Abbreviations that must NOT end a sentence (punkt learns some, but the corpus is
# small — seed the obvious government/finance ones so a lone doc doesn't over-split).
EXTRA_ABBREV = {
    "u.s", "e.g", "i.e", "etc", "vs", "cf", "mr", "mrs", "ms", "dr", "prof",
    "gov", "sen", "rep", "st", "no", "fig", "al", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.k", "u.s.c",
    "p.m", "a.m", "inc", "corp", "co", "ltd", "approx", "sec", "art", "para",
}


def build_tokenizer() -> PunktTokenizer:
    """Pretrained English Punkt tokenizer, seeded with extra finance/government
    abbreviations so a lone document does not over-split on 'e.g.', 'Inc.', etc."""
    try:
        tok = PunktTokenizer("english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        tok = PunktTokenizer("english")
    for a in EXTRA_ABBREV:
        tok._params.abbrev_types.add(a)
    return tok


def normalize(text: str) -> str:
    """Whitespace/typographic cleanup before splitting. Protects decimals and
    numbered-list markers from being read as sentence ends."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", "—").replace("–", "-")
                .replace(" ", " "))
    # join hard-wrapped lines inside a paragraph (single newline -> space),
    # keep paragraph breaks (blank line) as a hard boundary marker
    text = re.sub(r"\n\s*\n", "\n \n", text)   # paragraph sep marker
    text = re.sub(r"(?<!\n)\n(?! )", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_doc(tok: PunktTokenizer, text: str) -> list[str]:
    """Split one document into clean sentences."""
    sents: list[str] = []
    for para in normalize(text).split(" "):
        para = para.strip()
        if not para:
            continue
        for s in tok.tokenize(para):
            s = s.strip()
            # drop empties and pure enumerators ("1.", "a)") that survived
            if s and not re.fullmatch(r"[\(\[]?\w{1,3}[\.\)\]]", s):
                sents.append(s)
    return sents


def load_docs() -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name=SHEET)
    df = df[["id", "date", "speaker", "full_text"]].rename(columns={"id": "doc_id"})
    df["doc_id"] = df["doc_id"].astype(int)

    # doc 1: substitute the external full text if available
    if DOC1_TXT.exists():
        body = DOC1_TXT.read_text(encoding="utf-8")
        df.loc[df.doc_id == 1, "full_text"] = body
        print(f"[doc 1] loaded external body: {len(body):,} chars from {DOC1_TXT}")
    else:
        print(f"[doc 1] WARNING: {DOC1_TXT} not found — doc 1 will be SKIPPED "
              f"(its spreadsheet cell is only a stub, not the full text).")
    return df


def main() -> None:
    tok = build_tokenizer()
    docs = load_docs()

    rows = []
    skipped = []
    for _, d in docs.sort_values("doc_id").iterrows():
        text = d["full_text"]
        # skip doc 1 if only the stub is present
        if d.doc_id == 1 and not DOC1_TXT.exists():
            skipped.append(1)
            continue
        if not isinstance(text, str) or len(text.strip()) < 50:
            skipped.append(int(d.doc_id))
            continue
        sents = split_doc(tok, text)
        for i, s in enumerate(sents):
            rows.append({"sent_id": f"{d.doc_id}-{i}", "doc_id": int(d.doc_id),
                         "sent_index": i, "date": str(d["date"])[:10],
                         "speaker": d["speaker"], "sentence": s})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}: {len(out)} sentences across "
          f"{out.doc_id.nunique()} documents")
    print(out.groupby(["doc_id", "speaker"]).size().rename("n_sentences").to_string())
    if skipped:
        print(f"\nSKIPPED docs (no usable full text): {sorted(set(skipped))}")
        print("  -> supply data/raw/fed/fulltext_01_money_and_payments.txt to add doc 1.")


if __name__ == "__main__":
    main()
