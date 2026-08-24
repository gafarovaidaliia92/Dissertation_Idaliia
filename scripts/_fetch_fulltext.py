"""
_fetch_fulltext.py — rebuild data/raw/fed/fulltext_01_money_and_payments.txt, the body
text of doc id=1 in Fed_Communications_Block3.xlsx (the Board's January 2022
discussion paper "Money and Payments: The U.S. Dollar in the Age of Digital
Transformation").

Doc 1's body is held in a .txt rather than in the spreadsheet because it exceeds
Excel's ~32,767-char cell limit; the xlsx cell is a 464-char stub.

Reads   : data/raw/fed/Fed_Communications_Block3.xlsx (row id=1 -> landing page)
Fetches : the PDF linked from that landing page -> data/raw/fed/
Writes  : data/raw/fed/fulltext_01_money_and_payments.txt

Output format: one paragraph per line, paragraphs separated by a blank line.
That is exactly what rq2_sentences.normalize() expects — it maps a blank line to
the U+2029 paragraph separator it splits on, and joins any single newline inside
a paragraph. No hard-wrapped lines are emitted, so nothing is mis-joined.

WHAT IS KEPT (the unit of analysis for RQ2 is a body sentence):
  * body prose, 10 pt, printed pages 1-32 (PDF pages 5-36)
  * bulleted list items, each as its own paragraph
  * section headings          -- only with --headings   (default: dropped)
  * footnote prose, 8 pt      -- only with --footnotes   (default: dropped)

WHAT IS DROPPED, and why:
  * cover + "The Federal Reserve System is the central bank..." (PDF pp. 1-2)
    and the Contents page (p. 3) -- front matter, not authored prose
  * running headers, 9 pt at top of page (page number + document/section title)
  * inline superscript footnote markers, 5-7 pt -- otherwise they weld onto the
    body text as "another bank.4Accordingly"
  * the 7 pt note under the payment-chain figure on printed p. 29
  * References (PDF pp. 37-39) -- citation strings and URLs, which would emit
    hundreds of non-sentence "sentences" into the RQ2 census
  * the back cover (PDF p. 40)

Extraction notes:
  * x_tolerance=2 (not pdfplumber's default 3): at 3, italic-to-roman runs lose
    their space and yield "Central bank moneyis a liability" / "monetary
    policyto promote". 2 fixes those without over-splitting any word (verified:
    identical stray-singleton count vs. the default).
  * Paragraph breaks are geometric: intra-paragraph leading is 16 pt, so a
    vertical gap >= 18 pt starts a new paragraph. Bullet glyphs also start one.
  * End-of-line hyphens are resolved against a vocabulary built from the
    document itself: "pub-/lic's" -> "public's" (joined form attested), while
    "cross-/border" -> "cross-border" (hyphenated form attested). Unattested
    either way defaults to joining, since typographic hyphenation dominates.

Usage:
    python scripts/_fetch_fulltext.py                   # body prose only
    python scripts/_fetch_fulltext.py --footnotes        # + footnote prose
    python scripts/_fetch_fulltext.py --headings         # + section headings
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import urllib.request
from pathlib import Path

try:
    import pdfplumber
except ImportError as e:  # pragma: no cover
    raise SystemExit("pdfplumber is required: pip install pdfplumber") from e

import pandas as pd

XLSX = Path("data/raw/fed") / "Fed_Communications_Block3.xlsx"
SHEET = "communications"
PDF_DIR = Path("data") / "raw" / "fed"
PDF_PATH = PDF_DIR / "money-and-payments-20220120.pdf"
OUT = Path("data/raw/fed") / "fulltext_01_money_and_payments.txt"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# --- page ranges (1-indexed, inclusive) ------------------------------------ #
BODY_FIRST, BODY_LAST = 5, 36     # printed pp. 1-32: Exec Summary .. Appendix B
# pp. 1-4 front matter, 37-39 References, 40 back cover

# --- font-size bands (pt), measured from the file -------------------------- #
HEADER_SIZE = 9.0                 # running header only
FOOTNOTE_LO, FOOTNOTE_HI = 7.5, 8.5
BODY_MIN = 9.5                    # 10 body, 12/14/23.9 headings
HEADING_MIN = 11.5                # 12, 14, 23.9 -> headings

HEADER_TOP_MAX = 55              # running header sits at top ~39.5

PARA_GAP = 18.0                  # intra-paragraph leading is 16 pt
BULLETS = "•■◆–—"

# Body prose is left-aligned at x0 = 90, bullet text at 101-110. The Appendix C
# payment-chain diagram sets its labels at 10 pt too -- same size as body -- but
# at x0 >= 118 ("Sender", "Front end", "Payment infrastructure", stray "1 2 3").
# Every body-band line beyond 112 in this document is diagram text, so this gate
# keeps those labels out of the sentence census.
BODY_X0_MAX = 112.0

# Footnote text is set at x0 = 100 on every one of the 18 pages that has any.
# Figure/diagram labels are also 8 pt but start at x0 >= 109 (printed pp. 29-30
# figures), so this gate separates the two exactly.
FOOT_X0_LO, FOOT_X0_HI = 97.0, 103.0
# Each footnote opens with its number as a 5 pt superscript; 8 pt inline refs in
# the body are 6.5 pt. So 5 pt char tops mark where a new note begins.
FOOT_MARKER_SIZE = 5.0
FOOT_MARKER_TOL = 4.0


# --------------------------------------------------------------------------- #
#  fetch
# --------------------------------------------------------------------------- #
def landing_url() -> str:
    df = pd.read_excel(XLSX, sheet_name=SHEET)
    row = df.loc[df["id"] == 1]
    if row.empty:
        sys.exit(f"STOP: no row with id=1 in {XLSX}")
    return str(row.iloc[0]["url"]).strip()


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def ensure_pdf() -> Path:
    """Resolve the PDF from the landing page recorded in the spreadsheet."""
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 500_000:
        print(f"[pdf] cached: {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
        return PDF_PATH

    land = landing_url()
    print(f"[pdf] landing page (xlsx id=1): {land}")
    html = get(land).decode("utf-8", "replace")
    m = re.search(r'href="([^"]*money-and-payments[^"]*\.pdf)"', html)
    if not m:
        sys.exit("STOP: no money-and-payments*.pdf link found on the landing page.")
    pdf_url = urllib.request.urljoin(land, m.group(1))
    print(f"[pdf] downloading: {pdf_url}")

    blob = get(pdf_url)
    if not blob.startswith(b"%PDF"):
        sys.exit(f"STOP: {pdf_url} did not return a PDF.")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.write_bytes(blob)
    print(f"[pdf] saved {PDF_PATH} ({len(blob):,} bytes)")
    return PDF_PATH


# --------------------------------------------------------------------------- #
#  layout -> lines -> paragraphs
# --------------------------------------------------------------------------- #
def page_lines(page, keep) -> list[tuple[float, float, str, float]]:
    """Group a page's kept chars into visual lines.

    Returns (top, x0, text, max_size) per line, in reading order. Text is
    rebuilt from pdfplumber words at x_tolerance=2 so italic/roman runs keep
    their spaces.
    """
    chars = [c for c in page.chars if keep(c)]
    if not chars:
        return []
    kept = {(round(c["x0"], 1), round(c["top"], 1)) for c in chars}
    words = [w for w in page.extract_words(x_tolerance=2, y_tolerance=2,
                                           extra_attrs=["size"])
             if (round(w["x0"], 1), round(w["top"], 1)) in kept
             or any(abs(w["top"] - t) < 1.5 and abs(w["x0"] - x) < 1.5
                    for x, t in kept)]

    rows: dict[float, list] = collections.defaultdict(list)
    for w in words:
        rows[round(w["top"] / 3) * 3].append(w)

    out = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w["x0"])
        out.append((min(w["top"] for w in ws),
                    min(w["x0"] for w in ws),
                    " ".join(w["text"] for w in ws),
                    max(w["size"] for w in ws)))
    return out


def to_paragraphs(lines: list[tuple[float, float, str, float]]) -> list[list[str]]:
    """Merge lines into paragraphs on vertical gap / bullet glyphs.

    Returns a list of paragraphs, each a list of its raw lines (lines are kept
    separate so dehyphenate() can still see end-of-line hyphens).
    """
    paras: list[list[str]] = []
    prev_top = None
    for top, x0, text, _size in lines:
        text = text.strip()
        if not text:
            continue
        starts_bullet = text[0] in BULLETS
        gap_break = prev_top is not None and (top - prev_top) >= PARA_GAP
        if starts_bullet:
            text = text[1:].strip()
        if not paras or gap_break or starts_bullet:
            paras.append([text])
        else:
            paras[-1].append(text)
        prev_top = top
    return paras


# A paragraph that runs over a page break has no vertical gap to detect, so
# only start a new paragraph across pages when the previous page actually ended
# a sentence. Breaking mid-sentence would hand punkt two fragments; an extra
# break between two complete sentences costs nothing.
TERMINAL = re.compile(r"[.!?:;”\"’')\]]$")


def footnote_paragraphs(page) -> list[list[str]]:
    """One paragraph per footnote, using 5 pt leader positions as note starts."""
    lines = [l for l in page_lines(page, lambda c: (FOOTNOTE_LO <= c["size"] < FOOTNOTE_HI
                                                    and c["top"] > HEADER_TOP_MAX))
             if FOOT_X0_LO <= l[1] <= FOOT_X0_HI]
    marks = [c["top"] for c in page.chars
             if abs(c["size"] - FOOT_MARKER_SIZE) < 0.3]

    notes: list[list[str]] = []
    for top, _x0, text, _size in lines:
        starts = any(-1.0 <= (top - m) <= FOOT_MARKER_TOL for m in marks)
        if not notes or starts:
            notes.append([text])
        else:
            notes[-1].append(text)
    return notes


def stitch_pages(pages: list[list[list[str]]]) -> list[list[str]]:
    """Concatenate per-page paragraphs, rejoining ones split by a page break."""
    out: list[list[str]] = []
    for paras in pages:
        if not paras:
            continue
        if out and not TERMINAL.search(out[-1][-1].strip()):
            out[-1].extend(paras[0])       # continuation of the same paragraph
            out.extend(paras[1:])
        else:
            out.extend(paras)
    return out


# --------------------------------------------------------------------------- #
#  de-hyphenation
# --------------------------------------------------------------------------- #
WORD_RE = re.compile(r"[A-Za-z][A-Za-z’'-]*")


def build_vocab(paras: list[list[str]]) -> set[str]:
    """Words attested in the document, from lines NOT ending in a hyphen."""
    vocab: set[str] = set()
    for lines in paras:
        for line in lines:
            line = re.sub(r"[A-Za-z’'-]+-$", "", line)   # drop the split word
            vocab.update(w.lower() for w in WORD_RE.findall(line))
    return vocab


def dehyphenate(lines: list[str], vocab: set[str]) -> str:
    """Join a paragraph's lines, resolving end-of-line hyphens against vocab."""
    buf = lines[0]
    for nxt in lines[1:]:
        m = re.search(r"([A-Za-z][A-Za-z’']*)-$", buf)
        n = re.match(r"([A-Za-z][A-Za-z’']*)", nxt)
        if m and n:
            joined, hyph = (m.group(1) + n.group(1)).lower(), \
                           (m.group(1) + "-" + n.group(1)).lower()
            if joined in vocab:
                buf = buf[:-1] + nxt            # drop hyphen: pub-lic -> public
                continue
            if hyph in vocab:
                buf = buf + nxt                 # keep hyphen: cross-border
                continue
            buf = buf[:-1] + nxt                # default: typographic hyphen
            continue
        buf = buf + " " + nxt
    return re.sub(r"\s+", " ", buf).strip()


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def extract(pdf_path: Path, want_headings: bool, want_footnotes: bool) -> list[str]:
    body_pages: list[list[list[str]]] = []
    foot_pages: list[list[list[str]]] = []
    n_head_dropped = 0
    n_fig_dropped = 0

    def is_body(c):
        return (c["size"] >= BODY_MIN and c["top"] > HEADER_TOP_MAX
                and abs(c["size"] - HEADER_SIZE) > 0.2)

    with pdfplumber.open(pdf_path) as pdf:
        for pi in range(BODY_FIRST - 1, BODY_LAST):
            page = pdf.pages[pi]

            lines = page_lines(page, is_body)
            before = len(lines)
            lines = [l for l in lines if l[1] <= BODY_X0_MAX]
            n_fig_dropped += before - len(lines)
            if not want_headings:
                before = len(lines)
                lines = [l for l in lines if l[3] < HEADING_MIN]
                n_head_dropped += before - len(lines)
            body_pages.append(to_paragraphs(lines))

            if want_footnotes:
                foot_pages.append(footnote_paragraphs(page))

    body = stitch_pages(body_pages)
    foot = stitch_pages(foot_pages)

    vocab = build_vocab(body + foot)
    paras = [dehyphenate(p, vocab) for p in body + foot]
    paras = [p for p in paras if len(p) > 1]
    print(f"[extract] dropped {n_fig_dropped} figure-label lines (x0 > "
          f"{BODY_X0_MAX:.0f}, Appendix C diagram)")
    if not want_headings:
        print(f"[extract] dropped {n_head_dropped} heading lines "
              f"(pass --headings to keep them)")
    print(f"[extract] body: {len(body)} paragraphs from "
          f"{len(body_pages)} pages (page-break splits rejoined)")
    if want_footnotes:
        print(f"[extract] footnotes: {len(foot)} paragraphs")
    return paras


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headings", action="store_true",
                    help="keep section headings as their own paragraphs")
    ap.add_argument("--footnotes", action="store_true",
                    help="append footnote prose after the body")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    pdf_path = ensure_pdf()
    paras = extract(pdf_path, args.headings, args.footnotes)

    text = "\n\n".join(paras) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    words = len(WORD_RE.findall(text))
    print(f"\nwrote {args.out}")
    print(f"  {len(paras)} paragraphs, {len(text):,} chars, ~{words:,} words")
    print(f"  (xlsx stub was 464 chars; Excel cell limit is 32,767)")


if __name__ == "__main__":
    main()
