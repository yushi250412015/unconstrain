# Self-revision log — three cycles

The brief was: build something with a defensible novelty, and if the result is
weak, throw it away and rebuild. Three cycles were run. Cycle 1 was discarded in
full; cycles 2 and 3 were revisions.

Rubric, scored 1–5 on five axes (25 total). The bar for keeping a draft was 18.

| axis | question |
|---|---|
| novelty | is the central claim absent from the three concurrent 2025–26 papers? |
| rigour | does every claim have a derivation or a measurement behind it? |
| consistency | do abstract, tables and text agree numerically? |
| actionability | does it change a decision a practitioner makes? |
| clarity | does the abstract state the claim and the number? |

---

## Cycle 1 — "De-censoring is a bias intervention"

**Thesis.** A censored day carries only a fraction ρ of a clean day's Fisher
information; compute ρ, account for it on the benchmark, and use the resulting
7% floor to argue that accuracy has saturated while bias has not.

**Scores: novelty 3, rigour 2, consistency 4, actionability 2, clarity 4 → 14.**
Below the bar. Discarded.

**Why it failed.**

1. *Fatal, rigour.* The step from "the level has a 7% standard error" to "WAPE
   is saturated" was an assertion. The bound applies to a per-SKU level; WAPE is
   a daily prediction metric over a panel. No derivation connected them, so the
   central argument was a non sequitur dressed as a result.
2. *Fatal, novelty.* "De-censoring exists to remove bias" is the reason
   unconstraining was invented in 1994. Framing it as a discovery reads as
   ignorance of the field being addressed.
3. *Actionability.* "Budget de-censoring as a bias correction" is advice with no
   verb in it.
4. Two minor factual liabilities were also found in the section, not in the
   text: ρ's sensitivity to the dispersion parameter had never been checked, and
   the pooled-versus-local scope of the bound was unstated.

## Cycle 2 — "The ceiling is low"

**What changed.** The thesis moved from *which axis matters* to *how much is
available*. Two measurements were added before rewriting:

- ρ is insensitive to the dispersion (0.374 at k=10 → 0.334 as k→∞ at s=0.32),
  so the bound survives the obvious objection.
- Across the eleven methods, corr(|bias|, WAPE) = **0.94**, R² = **0.89**.

The second measurement destroyed cycle 1 and rebuilt the paper. If |bias|
explains 89% of the WAPE variance, then "bias and precision are separate axes"
is false; the WAPE differences *are* the bias differences. So the argument
became: bias is the whole of the accuracy gap, and the gap is small because the
information is not there. That is a ceiling claim, not a preference claim.

**Scores: novelty 4, rigour 3.5, consistency 3, actionability 4.5, clarity 4 →
19.**

**Defects found in review.**

1. *Consistency, hard error.* The abstract said the methods "differ by 20
   percentage points in bias"; the actual span is 35 (−7.1% to +28.0%).
2. *Consistency.* "7.0%" was reported as the population value when it is a
   median across series.
3. *Rigour.* The link from the bound to the floor is empirical (corr 0.94), not
   derived. Left in, but relabelled as an empirical regularity, and supplemented
   with two falsifiable predictions so a reader can check it rather than trust
   it.

## Cycle 3 — final

Fixes applied: the 35-point correction, "median 7.0%", the dispersion-sensitivity
sentence moved from limitations into §2 (a measured fact beats a caveat), and
the scope of the bound (per-SKU, not pooled) stated explicitly rather than
buried.

**Scores: novelty 4, rigour 4, consistency 5, actionability 4.5, clarity 4.5 →
22.**

**What is still weak, in order.**

1. The bound is exact; the bridge to WAPE is a correlation (0.94) on eleven
   points. A derivation of the achievable WAPE floor given ρ would be stronger.
2. One benchmark, one country, 90 days, seven holdout days.
3. The dual-control negative result (§6) is a simulation, not a field test.
4. The two falsifiable predictions have not been tested against the literature's
   claimed numbers, only against my own runs.

---

## What the three cycles produced

The version that survived is not the version that any single cycle aimed at.
Cycle 1 wanted to tell the field it was measuring the wrong axis; the data said
the axes were the same. Cycle 2 wanted to say the ceiling is low; that survived.
Cycle 3 found only bookkeeping errors, which is the signal that the argument had
stopped moving — the correct place to stop.

The one number that changed the paper is `corr(|bias|, WAPE) = 0.94`. It was not
planned. It was computed to fill a hole that a reviewer would have found in
cycle 1.
