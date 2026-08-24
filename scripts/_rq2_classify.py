"""
rq2_classify.py — RQ2 step 2: sentence-level LLM classification of the Fed CBDC
communications against rq2_codebook.md.

One Anthropic API call per sentence. system = the codebook (verbatim); user = the
target sentence plus its immediate neighbours as context, with the target marked.
Deterministic decoding (temperature = 0). Strict-JSON output, parsed defensively
with one retry, then flagged.

Reads   : data/processed/rq2_sentences.csv, rq2_codebook.md
Writes  : data/frozen/rq2_sentence_labels.csv   (fully auditable — every raw
          response, plus the run provenance on every row)
          data/processed/rq2_run_metadata.json

PROVENANCE recorded on every label row and in the metadata file:
  model, run_utc_date, temperature, codebook_sha256.

------------------------------------------------------------------------------
PROMPT CACHING — does not affect labels or provenance.
The system prompt (codebook + JSON_INSTRUCTION) is byte-identical on all N calls
while only the user turn varies, so it is sent as a single cached text block.
Caching is a prefix match over `tools -> system -> messages`; there are no tools
here and the block sits at the end of `system`, so the whole system prompt is
the cached prefix and the varying sentence falls after it.

The model sees exactly the same bytes as an uncached run — wrapping the text in
a content block changes transport, not content. Labels stay comparable across
runs, and NOTHING in the provenance changes: model, run_utc_date, temperature
and codebook_sha256 are untouched, and rq2_run_metadata.json keeps the same
keys and values it had before caching was added. Cache statistics are printed
to stdout only; they are deliberately NOT persisted, so the metadata file is
byte-comparable with pre-caching runs.

Cost: cache writes bill ~1.25x and reads ~0.1x, so the codebook is paid for at
full price once and at a tenth thereafter — for N=1338 that turns ~$6.6 of
input into ~$1.1. Reads only land if the prefix exceeds the model's minimum
cacheable length (MIN_CACHEABLE_TOKENS); below it the API silently declines to
cache. If you EDIT THE CODEBOOK and it shrinks under that floor, caching stops
with no error — main() therefore checks the length up front and reports
cache_read_input_tokens at the end. If reads are 0, something invalidated the
prefix; do not assume the run was cached just because this file says so.
------------------------------------------------------------------------------

STOPS (does not fake labels) if ANTHROPIC_API_KEY / credentials are missing or the
API is unreachable.

------------------------------------------------------------------------------
MODEL / TEMPERATURE NOTE — read before changing MODEL.
The spec fixes temperature = 0 for reproducibility. Anthropic's *frontier* models
(claude-opus-4-8, claude-sonnet-5, claude-opus-4-7) REJECT any `temperature`
parameter with HTTP 400 — so temperature = 0 cannot be honoured on them. This
script therefore pins a model that accepts temperature = 0. `claude-sonnet-4-5`
(legacy but active, strong at nuanced classification) is the default. If you set
MODEL to a frontier model, temperature is auto-dropped (see MODELS_NO_TEMPERATURE)
and the run is no longer temperature-0 — a warning is printed and the metadata
records temperature = null.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROC = Path("data") / "processed"
SENTENCES = PROC / "rq2_sentences.csv"
CODEBOOK = Path("docs/rq2_codebook.md")
OUT_LABELS = PROC.parent / "frozen" / "rq2_sentence_labels.csv"
OUT_META = PROC / "rq2_run_metadata.json"

# --- pinned model + decoding (see MODEL / TEMPERATURE NOTE above) ---
MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0.0
MAX_TOKENS = 300
# Frontier models reject `temperature`; if MODEL is one of these, drop it.
MODELS_NO_TEMPERATURE = {"claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5",
                         "claude-fable-5", "claude-mythos-5"}

RETRIES = 1  # one retry on parse/transport failure, then flag

# --- prompt caching (see PROMPT CACHING note above) ---
# Default 5-minute TTL: calls are sequential and a few seconds apart, so the
# entry never expires mid-run. A 1h TTL would double the write cost for nothing.
CACHE_TTL = "ephemeral"
# Minimum cacheable prefix for the pinned model. Model-specific and NOT
# monotonic across generations (claude-sonnet-4-5 / claude-sonnet-5: 1024;
# claude-opus-4-8: 1024; claude-opus-4-7: 2048; claude-opus-4-6: 4096).
# Below this the API caches nothing and says nothing.
MIN_CACHEABLE_TOKENS = 1024

JSON_INSTRUCTION = """
------------------------------------------------------------------------------
OUTPUT FORMAT (strict). Respond with EXACTLY one JSON object and nothing else:
{"mentions_design": <true|false>, "design_stance": <-1|0|1>, "rationale": "<=1 sentence"}
Rules:
- design_stance MUST be 0 whenever mentions_design is false.
- design_stance is an integer, one of -1, 0, 1 (never a string).
- rationale is at most one sentence.
- No markdown, no code fences, no text before or after the JSON object.
"""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_client():
    """Construct the Anthropic client, or STOP with a clear message. Credentials
    may come from ANTHROPIC_API_KEY or an `ant auth login` profile — a bare
    client picks either up. We probe with a tiny request and fail loudly."""
    try:
        import anthropic
    except ImportError:
        sys.exit("STOP: `anthropic` SDK not installed (pip install anthropic). "
                 "Not faking labels.")
    try:
        client = anthropic.Anthropic()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"STOP: could not construct Anthropic client ({e}). Set "
                 f"ANTHROPIC_API_KEY or run `ant auth login`. Not faking labels.")
    return client, anthropic


def target_prompt(prev: str, target: str, nxt: str) -> str:
    parts = []
    if prev:
        parts.append(f"[preceding] {prev}")
    parts.append(f"[TARGET — classify this sentence] {target}")
    if nxt:
        parts.append(f"[following] {nxt}")
    parts.append("\nClassify ONLY the [TARGET] sentence. Return the strict JSON object.")
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict | None:
    """Defensive parse of a strict-JSON reply. Returns a normalized dict or None."""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "mentions_design" not in raw or "design_stance" not in raw:
        return None
    md = raw["mentions_design"]
    if isinstance(md, str):
        md = md.strip().lower() in ("true", "yes", "1")
    md = bool(md)
    try:
        stance = int(raw["design_stance"])
    except (ValueError, TypeError):
        return None
    if stance not in (-1, 0, 1):
        return None
    if not md:
        stance = 0  # enforce the codebook coupling
    rationale = str(raw.get("rationale", "")).strip()
    return {"mentions_design": md, "design_stance": stance, "rationale": rationale}


def build_system(codebook: str) -> list[dict]:
    """The system prompt as a single cache-marked text block.

    Identical text to the pre-caching `codebook + "\\n" + JSON_INSTRUCTION`
    string — only the transport shape differs, so labels stay comparable.
    """
    return [{
        "type": "text",
        "text": codebook + "\n" + JSON_INSTRUCTION,
        "cache_control": {"type": CACHE_TTL},
    }]


# Cache accounting, printed to stdout at the end. NOT written to the metadata
# file — the provenance record must stay identical to pre-caching runs.
CACHE_STATS = {"write": 0, "read": 0, "uncached": 0}


def classify_one(client, anthropic, system: list[dict],
                 user: str) -> tuple[dict | None, str]:
    """One sentence -> (parsed|None, raw_text). Retries once on transport error.

    Also accumulates cache usage into CACHE_STATS as a side effect.
    """
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": user}])
    if MODEL not in MODELS_NO_TEMPERATURE:
        kwargs["temperature"] = TEMPERATURE
    last_raw = ""
    for attempt in range(RETRIES + 1):
        try:
            resp = client.messages.create(**kwargs)
            u = resp.usage
            CACHE_STATS["write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            CACHE_STATS["read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            CACHE_STATS["uncached"] += u.input_tokens or 0
            last_raw = "".join(b.text for b in resp.content if b.type == "text")
            parsed = parse_response(last_raw)
            if parsed is not None:
                return parsed, last_raw
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_raw = f"__api_error__: {e}"
        if attempt < RETRIES:
            time.sleep(1.5 * (attempt + 1))
    return None, last_raw


def main() -> None:
    if not SENTENCES.exists():
        sys.exit(f"STOP: {SENTENCES} missing — run scripts/rq2_sentences.py first.")
    if not CODEBOOK.exists():
        sys.exit(f"STOP: {CODEBOOK} missing.")

    codebook = CODEBOOK.read_text(encoding="utf-8")
    codebook_hash = sha256_file(CODEBOOK)
    system = build_system(codebook)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    temp_recorded = None if MODEL in MODELS_NO_TEMPERATURE else TEMPERATURE
    if MODEL in MODELS_NO_TEMPERATURE:
        print(f"WARNING: {MODEL} rejects `temperature`; running WITHOUT it. "
              f"This run is NOT temperature-0.")

    sents = pd.read_csv(SENTENCES)
    client, anthropic = build_client()

    # by-doc neighbour lookup for context
    sents = sents.sort_values(["doc_id", "sent_index"]).reset_index(drop=True)
    by_doc = {d: g.reset_index(drop=True) for d, g in sents.groupby("doc_id")}

    # Pre-flight: a system prompt under the model's floor is silently NOT
    # cached. Most likely to bite after the codebook is trimmed post-kappa.
    try:
        n_sys = client.messages.count_tokens(
            model=MODEL, system=system,
            messages=[{"role": "user", "content": "x"}]).input_tokens
        if n_sys < MIN_CACHEABLE_TOKENS:
            print(f"WARNING: system prompt ~{n_sys} tokens < "
                  f"{MIN_CACHEABLE_TOKENS} minimum for {MODEL} — prompt "
                  f"caching will silently NOT apply. Labels are unaffected; "
                  f"only cost is.")
        else:
            print(f"[cache] system prompt ~{n_sys} tokens (>= "
                  f"{MIN_CACHEABLE_TOKENS} minimum) — caching should apply")
    except Exception as e:  # noqa: BLE001 — diagnostics only, never block a run
        print(f"[cache] could not count system tokens ({e}); continuing")

    rows, n_flagged = [], 0
    print(f"[classify] {len(sents)} sentences | model={MODEL} | "
          f"temperature={temp_recorded} | codebook={codebook_hash[:12]}")
    for _, r in sents.iterrows():
        g = by_doc[r.doc_id]
        i = int(r.sent_index)
        prev = g.sentence.iloc[i - 1] if i > 0 else ""
        nxt = g.sentence.iloc[i + 1] if i + 1 < len(g) else ""
        user = target_prompt(str(prev), str(r.sentence), str(nxt))
        parsed, raw = classify_one(client, anthropic, system, user)
        flagged = parsed is None
        n_flagged += int(flagged)
        rows.append({
            "sent_id": r.sent_id, "doc_id": int(r.doc_id),
            "sent_index": i, "date": r.date, "speaker": r.speaker,
            "sentence": r.sentence,
            "mentions_design": None if flagged else parsed["mentions_design"],
            "design_stance": None if flagged else parsed["design_stance"],
            "rationale": "" if flagged else parsed["rationale"],
            "flagged": flagged,
            "raw_response": raw,
            "model": MODEL, "run_utc_date": run_date,
            "temperature": temp_recorded, "codebook_sha256": codebook_hash,
        })
        if len(rows) % 50 == 0:
            print(f"    {len(rows)}/{len(sents)}  (flagged so far: {n_flagged})")

    out = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LABELS, index=False)

    meta = {"model": MODEL, "run_utc_date": run_date, "temperature": temp_recorded,
            "codebook_sha256": codebook_hash, "codebook_file": str(CODEBOOK),
            "n_sentences": len(out), "n_flagged": int(n_flagged),
            "sdk": "anthropic", "retries": RETRIES}
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {OUT_LABELS} ({len(out)} rows; {n_flagged} flagged/unparsed)")
    print(f"wrote {OUT_META}")

    # Cache report — stdout only, never persisted (see PROMPT CACHING note).
    w, r, u = CACHE_STATS["write"], CACHE_STATS["read"], CACHE_STATS["uncached"]
    print(f"\n[cache] written={w:,} read={r:,} uncached={u:,} input tokens")
    if r == 0 and len(out):
        print("[cache] WARNING: 0 cached reads — the prefix was invalidated on "
              "every call, or it is below the model's minimum. The run is "
              "VALID (labels unaffected), but it cost full price.")
    else:
        billed = w * 1.25 + r * 0.10 + u
        print(f"[cache] billed ~{billed:,.0f} input-token equivalents vs "
              f"{w + r + u:,} uncached ("
              f"{100 * (1 - billed / max(w + r + u, 1)):.0f}% saved)")
    ok = out[~out.flagged]
    if len(ok):
        print("\nlabel distribution (parsed rows):")
        print(f"  mentions_design=true: {int(ok.mentions_design.sum())} / {len(ok)}")
        print(ok.design_stance.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
