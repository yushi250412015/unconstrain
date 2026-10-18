# Cover letter

> Fill `[Venue]`, `[Editor]`, `[Author]`. ~300 words, one page.

---

Dear `[Editor]`,

We submit "**De-censoring Has a Ceiling, and It Is Low: an Information Bound for
Censored Retail Demand**" for consideration at `[Venue]`.

**The problem.** Retail forecasting systems learn from sales, and sales are
censored by inventory. A 2025–2026 literature — including three papers on the
same FreshRetailNet-50K benchmark — competes on forecast accuracy.

**The question we ask instead.** How much accuracy is available? We compute how
much information a censored day carries, account for it on the benchmark, and
obtain a Cramér–Rao floor of 7% on any unbiased per-SKU level estimate. Two
consequences follow, both measured. A 7% level error can inflate WAPE by at most
2.3% — half a point at 20.5 — while the spread between the best and the worst
honest reconstruction method is 1.1 points. And rescaling the naive pipeline's
forecasts by 1.076 removes its entire bias while recovering 0.3 of a 2.1-point
gap. **Most of what the literature competes on is reconstruction artifact, not
information.**

**What it changes.** Measure aggregate bias, which is the whole of the
difference; correct it with a severity-weighted likelihood that takes fifteen
seconds; and spend the next dollar on the ordering quantile, worth 18.7×, where
no information bound applies. Our own estimator wins by 0.2 points, and we say
so in the abstract.

**Honest reporting.** The paper contains two results against our own interest:
the best deep imputer matches our method on accuracy, and a negative result on
whether the stock-out mask can be optimised directly. We report both, because
the field's practice of quoting one seed on one split is part of why this
problem has survived.

**Availability.** One command reproduces every number and figure on a laptop
CPU; no GPU, no training, twelve tests. The dataset is CC-BY-4.0, the code MIT.

Sincerely,

`[Author]`
