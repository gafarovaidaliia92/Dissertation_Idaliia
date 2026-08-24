# RQ2 "Safeguard" text-signal codebook

This codebook is the classification rubric for the sentence-level LLM coding of the
11 Fed CBDC communications. It is passed **verbatim** as the system prompt to the
model, and a SHA-256 hash of this file is recorded with every label so the exact
rubric behind each classification is auditable.

The coder assigns **two independent labels** to each sentence, judged **only** from
the sentence itself in the context of its immediate neighbours (the preceding and
following sentence are supplied as context; the target sentence is marked).

---

## Label 1 — `mentions_design` (boolean)

**Does the sentence discuss a concrete CBDC *design* feature or parameter?**

`true` if the sentence refers to any specific design choice, such as:

- **holding / balance caps** — a limit on how much CBDC an end user may hold or transfer;
- **interest vs non-interest-bearing** — whether the CBDC pays remuneration;
- **intermediated / two-tier vs direct model** — whether private-sector intermediaries
  (banks, PSPs) stand between the central bank and the end user, or the central bank
  issues directly to the public;
- **access / eligibility** — who may hold or use the CBDC (broad public vs restricted);
- **remuneration** — the rate or rule for any interest paid;
- **offline** capability;
- **privacy / data** design of the instrument.

`false` if the sentence is about CBDC **in general**, its **motivation**, its **macro
or financial-stability effects**, adoption, international competition, or the case for
or against issuance — **without** naming a concrete design feature. A sentence can be
*about* CBDC and still be `false`: the test is whether a specific design feature or
parameter is present.

---

## Label 2 — `design_stance` (−1 | 0 | +1)

Coded **only when `mentions_design = true`**. When `mentions_design = false`, set
`design_stance = 0`.

The axis is **the threat the described design poses to bank deposits** — NOT whether
the speaker likes CBDC.

- **+1 PROTECTIVE** (the "Safeguard" pole): the feature **limits** CBDC's threat to
  bank deposits — e.g. holding/balance caps, non-interest-bearing, the intermediated
  / two-tier model, restricted access. These make CBDC a *weaker* substitute for
  commercial bank deposits.

- **−1 EXPANSIVE**: the feature makes CBDC a **stronger** deposit substitute — e.g.
  interest-bearing, direct / unintermediated issuance, unlimited holdings, broad
  access as a deposit alternative, frictionless convertibility of deposits into CBDC.

- **0 NEUTRAL / OFF-AXIS**: a design feature is mentioned but the sentence is
  descriptive or neutral, lists design dimensions without endorsing a limiting or
  expanding choice, or the feature is off the deposit-substitution axis (e.g. a
  purely technical or privacy point with no deposit-substitution implication).

### Independence rule

**Judge `design_stance` independently of the speaker's overall favourability toward
CBDC.** A speaker who is sceptical of CBDC in general can still *describe* a
protective feature (that sentence is +1); an enthusiastic speaker can describe an
expansive feature (that sentence is −1). Code the feature in the sentence, not the
speaker's thesis.

---

## Worked boundary examples

Drawn from the preliminary hand-coded `design_quote` column.

**Example A — motivation vs feature (the `mentions_design` boundary).**
Doc 11 (Brainard): *"Banks play a critical role in credit intermediation and monetary
policy transmission, as well as in payments."*
→ `mentions_design = false`. This is the macro role of banks — the motivation for
caring — with **no** design feature named. (The *following* sentence in that passage,
*"the design of any CBDC would need to include safeguards to protect against
disintermediation of banks"*, **is** `true`, +1.)

**Example B — clear PROTECTIVE (+1).**
Docs 3/4 (Brainard): *"offering a non-interest bearing CBDC ... and limiting the amount
of CBDC an end user could hold or transfer."*
→ `mentions_design = true`, `design_stance = +1`. Two named features (non-interest-
bearing, holding/transfer cap) that both *limit* deposit substitution.

**Example C — PROTECTIVE feature from a sceptical speaker (independence rule).**
Doc 6 (Bowman — overall sceptical of CBDC, preliminary doc stance −1):
*"an intermediated CBDC, with private-sector service providers, could be designed in a
way that maintains financial institution involvement and minimizes disruptions to the
financial system."*
→ `mentions_design = true`, `design_stance = +1`. The **intermediated / two-tier
model** is a protective feature, even though the speaker is not pro-CBDC. Do not let
the speaker's scepticism pull this to −1 or 0.

**Example D — EXPANSIVE / frictionless convertibility (−1).**
Doc 9 (Brainard): *"the ability to convert commercial bank deposits into CBDC with a
simple swipe."*
→ `mentions_design = true`, `design_stance = −1`. Frictionless convertibility makes
CBDC a *stronger* deposit substitute (this is the run-risk channel). If a sentence
instead only said "we must address run risk" with no convertibility/design feature, it
would be `mentions_design = false`.

**Example E — descriptive enumeration (0).**
Doc 8 (Brainard): *"a much broader set of institutions and individuals could access it,
[and] some types of balances might not pay interest."*
→ `mentions_design = true`, `design_stance = 0`. Design dimensions (access,
remuneration) are **listed descriptively** as possibilities, without endorsing a
limiting or expanding choice — neutral on the deposit-substitution axis.
