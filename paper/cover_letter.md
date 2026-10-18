# Cover letter

> 占位符：`[Venue]`、`[Editor]`、`[Author]` 需你填。全文约 330 词，一页。

---

Dear `[Editor]`,

We submit "**Not All Stock-outs Are Equal: Severity-Weighted Partial Censoring
for Retail Demand**" for consideration at `[Venue]`.

**The problem.** Retail forecasting systems learn from sales, and sales are
censored by inventory. On the best public benchmark for this problem —
FreshRetailNet-50K, 50,000 store–product series with *hourly* stock-status
annotations — the standard remedy is to impute the missing days or to fit a
censored-demand model under the assumption `demand > sales`. We show that the
second assumption is wrong for fresh retail: the median stock-out day loses only
24.5% of its volume, and a dark hour at 09:00 costs 2.6× one at 20:00. Treating
such days as fully censored overcorrects demand by 9.4% to 28.0%.

**What we contribute.** (i) A severity-weighted partial-censoring likelihood,
with closed-form posterior mean, that restores the bias to −0.3% at negligible
cost. (ii) A measurement of what the expensive alternative buys: SAITS,
TimesNet and ImputeFormer land in the same accuracy band as a two-second linear
interpolation, while their run-to-run variance (≈1 pp for SAITS) exceeds the gap
between methods. (iii) A demonstration that the cost of censoring is dominated
not by the forecast but by the ordering quantile: 18.7× fewer lost units from
moving to the newsvendor fractile, against 10–21% for the choice of
reconstruction. (iv) Evidence that zero-shot foundation models inherit the bias
(−9.8%) and recover 2.83 pp when fed a de-censored history.

**Why it is useful.** The paper ends with an ordered list of interventions a
practitioner can act on this quarter, and every number reproduces from one
command on a laptop CPU.

**A note on reporting.** Two of our results argue against our own method: the
best deep imputer matches it on accuracy, and a single-seed run initially
supported a stronger claim that four seeds do not. We report both, because the
field's current practice of quoting one seed on one split is part of why this
problem has survived.

Data and code are public (CC-BY-4.0 dataset; MIT-licensed repository), and the
full pipeline runs without a GPU.

Sincerely,

`[Author]`
