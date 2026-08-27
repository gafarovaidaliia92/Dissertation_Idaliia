"""
make_public.py — rebuild the PUBLIC mirror of this repository.

The public repository is not a clone. It is a filtered copy: the dissertation,
the audit notes and every CRSP identifier are stripped out of it. Copying the
tree by hand would eventually publish one of them, so the filtering lives here
and is re-run on every update.

    python3 scripts/make_public.py            # rebuild ../Dissertation_CBDC_public
    python3 scripts/make_public.py --dry-run  # list what would change, touch nothing

What it does, in order:
  1. mirrors the allowed paths, deleting anything that no longer exists upstream
  2. removes `permno` / `permco` from every CSV (CRSP identifiers are licensed)
  3. omits crosswalk_rssd_permno.csv, whose only content was that mapping
  4. re-applies the public README edits (licence line, docs/ block, CRSP row)
  5. leaves LICENSE, DATA.md and .git alone — they exist only in the mirror

It never touches git. Review the diff, then commit and push from the mirror.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
DST = SRC.parent / "Dissertation_CBDC_public"
DRY = "--dry-run" in sys.argv

# Whole directories mirrored wholesale.
MIRROR_DIRS = ["scripts", "results", "data/raw/fed", "data/frozen"]

# Individual files copied from the root.
MIRROR_FILES = ["README.md", "requirements.txt", "requirements.lock.txt", ".gitignore"]

# docs/ is filtered rather than mirrored: only the methodological documents go
# public. Anything reviewing the dissertation text, and the Russian working
# notes, stay private. Add a filename here to publish it.
DOCS_PUBLIC = ["RESULTS.md", "rq2_codebook.md", "rq1_pipeline_variants.md"]

# Never copied, wherever they appear.
EXCLUDE_NAMES = {"crosswalk_rssd_permno.csv"}
EXCLUDE_DIRS = {"__pycache__", "dissertation"}

# Created by the mirror, not by this repository — never delete these.
KEEP_IN_DST = {"LICENSE", "DATA.md", ".git"}

DROP_COLUMNS = {"permno", "permco"}


def _skip(p: Path) -> bool:
    return p.name in EXCLUDE_NAMES or any(d in p.parts for d in EXCLUDE_DIRS)


def mirror() -> tuple[list[str], list[str]]:
    """Copy the allowed tree; return (written, deleted) as relative paths."""
    wanted: dict[Path, Path] = {}
    for d in MIRROR_DIRS:
        for src in (SRC / d).rglob("*"):
            if src.is_file() and not _skip(src):
                wanted[src.relative_to(SRC)] = src
    for f in MIRROR_FILES:
        if (SRC / f).exists():
            wanted[Path(f)] = SRC / f
    for name in DOCS_PUBLIC:
        if (SRC / "docs" / name).exists():
            wanted[Path("docs") / name] = SRC / "docs" / name

    written = []
    for rel, src in sorted(wanted.items()):
        dst = DST / rel
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        written.append(str(rel))
        if not DRY:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # anything in the mirror that upstream no longer has
    deleted = []
    for dst in sorted(DST.rglob("*")):
        if not dst.is_file():
            continue
        rel = dst.relative_to(DST)
        if rel.parts[0] in KEEP_IN_DST or rel.parts[0] == ".git":
            continue
        if rel not in wanted:
            deleted.append(str(rel))
            if not DRY:
                dst.unlink()
    return written, deleted


def strip_identifiers() -> list[str]:
    """Remove CRSP identifier columns from every CSV in the mirror."""
    out = []
    for p in sorted(DST.rglob("*.csv")):
        rows = list(csv.reader(p.open(newline="")))
        if not rows:
            continue
        hdr = rows[0]
        idx = [i for i, c in enumerate(hdr) if c.strip().lower() in DROP_COLUMNS]
        if not idx:
            continue
        keep = [i for i in range(len(hdr)) if i not in idx]
        out.append(f"{p.relative_to(DST)} (-{','.join(hdr[i] for i in idx)})")
        if not DRY:
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                for row in rows:
                    w.writerow([row[i] for i in keep] if len(row) == len(hdr) else row)
    return out


def patch_readme() -> bool:
    """Re-apply the public-only README edits to the copied README."""
    p = DST / "README.md"
    if not p.exists():
        return False
    t = orig = p.read_text()

    licence_line = ("Code: MIT (`LICENSE`). Data sources and what is and is not "
                    "redistributed here: **[`DATA.md`](DATA.md)**.")
    if licence_line not in t:
        lines = t.splitlines()
        lines.insert(2, licence_line + "\n")
        t = "\n".join(lines)

    # the docs/ listing: only what actually ships
    start = t.find("docs/\n")
    if start != -1:
        end = t.find("```", start)
        if end != -1:
            t = t[:start] + (
                "docs/\n"
                "  RESULTS.md                 the authoritative results write-up, RQ1 -> RQ2 -> RQ3\n"
                "  rq2_codebook.md            the coding scheme behind the Safeguard signal\n"
                "  rq1_pipeline_variants.md   narrow vs wide population, charter filter\n"
            ) + t[end:]

    # drop any paragraph about the dissertation file, which is not published
    i = t.find("`docs/dissertation/Dissertation_current.docx`")
    if i != -1:
        j = t.find("\n\n", i)
        t = t[:i] + (t[j + 2:] if j != -1 else "")

    t = t.replace("| `wrds/{dsf,dsi,dsedelist}.parquet` | ",
                  "| `wrds/{dsf,dsi,dsedelist}.parquet` (**not in this repo**) | ")

    if t != orig and not DRY:
        p.write_text(t)
    return t != orig


def main() -> None:
    if not DST.exists():
        raise SystemExit(f"{DST} does not exist — build it once before updating")
    written, deleted = mirror()
    stripped = strip_identifiers()
    patched = patch_readme()

    tag = "[dry-run] would " if DRY else ""
    print(f"{tag}update {len(written)} file(s), delete {len(deleted)}")
    for f in written[:40]:
        print(f"   + {f}")
    if len(written) > 40:
        print(f"   ... and {len(written) - 40} more")
    for f in deleted:
        print(f"   - {f}")
    print(f"{tag}strip CRSP identifiers from {len(stripped)} CSV(s)")
    for f in stripped[:10]:
        print(f"   ~ {f}")
    if len(stripped) > 10:
        print(f"   ... and {len(stripped) - 10} more")
    print(f"README patched: {patched}")

    leaks = [str(p.relative_to(DST)) for p in DST.rglob("*.csv")
             if "permno" in p.read_text(errors="ignore")[:4000].split("\n")[0].lower()]
    print("CRSP identifiers remaining in CSV headers:", leaks or "none")
    print(f"\nmirror: {DST}")
    if not DRY:
        print("next:  cd", DST, "&& git add -A && git commit && git push")


if __name__ == "__main__":
    main()
