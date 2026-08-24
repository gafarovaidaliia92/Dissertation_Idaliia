"""
fetch_fdic_class.py — one-off fetch of FDIC institution classification for every
2022Q4 Call Report filer, cached to data/raw/fdic/institutions_class.csv.

Source: FDIC BankFind Suite API (https://api.fdic.gov/banks/institutions), no
auth. We keep BKCLASS (charter class) and SPECGRP/SPECGRPN (FDIC's own
specialization grouping, derived from the balance sheet), which is what
distinguishes a credit-card bank or a servicer from an ordinary commercial bank.

The API returns INACTIVE institutions too (ACTIVE=0), so the three failed banks
(SVB / Signature / First Republic) are covered — unlike the NIC ATTRIBUTES file,
which only lists live entities.

Fail loud: if the API can't be reached or returns fewer certs than requested,
this raises. The downstream pipeline must NOT silently fall back to a name list.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
ZIP_2022Q4 = RAW / "call_12312022.zip"
OUT = RAW / "fdic" / "institutions_class.csv"

API = "https://api.fdic.gov/banks/institutions"
FIELDS = "CERT,NAME,BKCLASS,SPECGRP,SPECGRPN,ACTIVE"
BATCH = 40
RETRIES = 3


def all_certs(zip_path: Path) -> list[int]:
    zf = zipfile.ZipFile(zip_path)
    por = sorted(n for n in zf.namelist() if "Bulk POR" in n)[0]
    df = pd.read_csv(zf.open(por), sep="\t", dtype=str, encoding="latin-1")
    df.columns = [c.strip().strip('"') for c in df.columns]
    cert_col = next(c for c in df.columns if "Certificate" in c)
    certs = pd.to_numeric(df[cert_col], errors="coerce").dropna().astype(int)
    return sorted(set(certs.tolist()))


def fetch_batch(chunk: list[int]) -> list[dict]:
    q = "CERT:(" + " OR ".join(str(c) for c in chunk) + ")"
    url = API + "?" + urllib.parse.urlencode(
        {"filters": q, "fields": FIELDS, "limit": 1000})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                d = json.load(r)
            return [x["data"] for x in d["data"]]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"FDIC API failed for batch starting {chunk[0]}: {last}")


def main() -> None:
    certs = all_certs(ZIP_2022Q4)
    print(f"[fetch] {len(certs)} certs to look up from {ZIP_2022Q4.name}")
    rows: list[dict] = []
    for i in range(0, len(certs), BATCH):
        rows.extend(fetch_batch(certs[i:i + BATCH]))
        if i % (BATCH * 20) == 0:
            print(f"        {i + BATCH:>5}/{len(certs)} ...")
        time.sleep(0.25)

    out = pd.DataFrame(rows).drop_duplicates("CERT")
    got = set(out.CERT.astype(int))
    missing = sorted(set(certs) - got)
    print(f"[fetch] received {len(out)} unique certs; missing {len(missing)}")
    if len(out) < 0.98 * len(certs):
        raise RuntimeError(
            f"FDIC returned only {len(out)}/{len(certs)} certs — refusing to "
            f"write a partial snapshot. Missing e.g. {missing[:10]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["CERT", "NAME", "BKCLASS", "SPECGRP", "SPECGRPN", "ACTIVE"]].to_csv(
        OUT, index=False)
    print(f"[fetch] wrote {OUT} ({len(out)} rows)")
    print("        BKCLASS:", dict(out.BKCLASS.value_counts()))
    print("        SPECGRP:", dict(out.SPECGRP.value_counts().sort_index()))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nSTOP: {e}", file=sys.stderr)
        sys.exit(1)
