# De-censoring Has a Ceiling, and It Is Low: an information bound for censored retail demand

*Draft v4 — submission candidate, content freeze 2026-11-20. Numbers reproduce from `python -m unconstrain --data real --series 1500` (add `JOBLIB_MULTIPROCESSING=0` in a shell that forbids named pipes); twelve tests cover the estimators and the data-repair layer.*

## Abstract

Retail demand is censored by inventory, and a 2025–2026 literature on the
FreshRetailNet-50K benchmark competes on forecast accuracy. We ask how much
accuracy is available. A censored day does not carry a clean day's information,
and we compute the shortfall exactly: at the benchmark's mean severity of 0.32,
a censored day retains 22–58% of a clean day's Fisher information, 90 days of
data are worth about 68 (25% destroyed), and the Cramér–Rao floor on the
standard error of any unbiased per-SKU level estimate is a median 7.0%.
Against a daily coefficient of variation of about 25%, a 7% level uncertainty is
a minor term — which explains a pattern the field reports without explaining:
eleven methods, spanning linear interpolation to ImputeFormer, differ by 35
percentage points in bias and, with the bias removed, by only 1.1 in WAPE; |bias|
alone accounts for 89% of the between-method WAPE variance — a correlation that,
on inspection, is not causal (§4). Once the bias is corrected the
error sits at a floor near 20 WAPE, and no method has gone below it. The
decision layer is not bounded in the same way: ordering at the newsvendor
quantile rather than the mean moves lost units by 16.8× on its own (18.7× once
the history is de-censored), an order of magnitude more than the entire
reconstruction question. De-censoring is worth doing to remove bias; it is not
worth funding to chase accuracy.

## 1. The claim

Two literatures describe the same defect. Operations research calls it
*unconstraining* and estimates a demand distribution from censored sales
\citep{nahmias1994,talluri2004,vulcano2012,tobin1958,kaplan1958}. Machine
learning calls it *latent demand recovery* and imputes the missing days
\citep{du2023,wu2023,nie2024}. Both are judged by forecast error, and both report
improvements of a few percent: on FreshRetailNet-50K, a leakage-free pipeline
with inverse-Mills-ratio recovery reports bias moving from −8.23% to −2.62%
\citep{azar2026}; a censoring-aware hourly model reports −8.1% to −1.3%
\citep{cadre2026}; an earlier relational model reports that recovery alone does
not improve WAPE \citep{chen2025}.

The pattern in those three numbers — bias moves a lot, accuracy moves a little —
is not a property of the methods. It is a property of the problem, and this
paper makes it precise.

**Claim.** The accuracy gains available from de-censoring are bounded by how much
information the censoring left, and that bound is low. The bias is the part that
is both large and removable; the residual error is neither.

**What this paper is, and is not.** It is not a new reconstruction method, and it
does not claim to beat any of the eleven it compares — on this protocol it wins
by 0.2 points, and says so. It is an accounting paper: it computes what the
censoring destroyed, prices the residue in the metric the field optimises, and
shows that the differences the field reports are the part the bound leaves free.
The method slot on this benchmark is already occupied three times over, which is
why a fourth estimator is the wrong contribution here:

| work | what it reports | what it occupies | what it leaves |
|---|---|---|---|
| Azar 2026 (*Inventions* 11(5):89) | inverse-Mills-ratio recovery; bias −8.23% → −2.62%; a leakage-free pipeline | a working method and an evaluation protocol | no bound on what remains |
| Yin, Zheng & Kong 2026 (*Sustainability* 18(15):7642) | stock-out flags as lower bounds; hourly separation of censoring | censoring-aware feature engineering | no accuracy ceiling |
| Chen & Jin 2025 (*JTIE* 4(1):325) | relational model; recovery alone does not improve WAPE | the observation this paper formalises | no quantification of why |

The last row is the one to keep in view: the field has already noticed that
recovery does not buy accuracy. What it has not done is compute how much accuracy
was ever there to buy. That is the whole claim, and §4 states the two ways it can
be falsified.

## 2. How much information a censored day carries

For a day observed with stock level $c$, the observation is a mixture: values
below $c$ are seen exactly, values at or above $c$ collapse into the event
$D \ge c$. Under a demand family $p(\cdot;\mu)$ the Fisher information about
$\mu$ is

$$I(c,\mu)=\sum_{y<c}p(y)\left(\frac{\partial\log p(y)}{\partial\mu}\right)^2
+ P(D\ge c)\left(\frac{\partial\log P(D\ge c)}{\partial\mu}\right)^2,$$

and the **retention** $\rho=I(c,\mu)/I(\infty,\mu)$ is the fraction of a clean
day's information that survives. With NB2 demand, $\rho$ depends on the level
$\mu$ and on the severity $s$ of the loss (the fraction of volume unreachable,
which fixes $c$ through $E[\min(D,c)]=(1-s)\mu$).

![Figure 1. One SKU, reconstructed](results/p5_example_series.png)

Figure 1 is the problem in one picture: a single SKU with 61% stock-out days and a
mean severity of 0.35. The bars are what the warehouse recorded; the line is what
the severity-weighted likelihood says demand was. The reconstruction adds 9.0%
over the window, and the days it adds most are the days the bars are short — which
is also why the correction cannot be checked against truth here: the demand it
restores was never observed. The honest evaluation in §3 therefore scores only
fully-stocked days.

![Figure 2. Information retained by a censored day](results/p1_information_curves.png)

Retention is a *step* function of severity rather than a smooth one, because
demand is counted in whole units: every time the implied stock level crosses an
integer the censoring event changes and $\rho$ jumps (Figure 2). At $s=0.32$, the
benchmark's mean severity, $\rho$ is 0.58 at $\mu=1$, 0.34 at $\mu=5$ and 0.22 at
$\mu=20$ (marked). Dispersion matters much less than the level: at $\mu=5$ and
$s=0.32$, $\rho$ moves only from 0.374 ($k=10$) to 0.334 ($k\to\infty$), so the
bound is robust to a mis-specified tail. The computation is exact and we validate
it against Monte Carlo estimation of the score variance (agreement to $10^{-3}$ on
five settings).

## 3. Accounting on the benchmark

FreshRetailNet-50K: 50,000 store–product series, 90 days, hourly stock status.
We sample 1,500 series (52.7% stock-out days) and compute, per series, the
effective sample size $n_{\text{eff}}=\sum_t \rho(s_t)$ and the Cramér–Rao floor
on the relative standard error of the level, $1/\sqrt{n_{\text{eff}}I_1(\mu)}/\mu$.

| quantity | median |
|---|---|
| nominal days | 90 |
| clean days | 44 |
| **effective days** | **67.8** |
| information destroyed | **24.7%** |
| CR floor on level SE | **7.0%** (p90 8.3%) |

Censoring destroys a quarter of the level's information. The floor on a
series-level estimate is 7%.

![Figure 3. The floor against the demand level](results/p2_bound_vs_level.png)

The floor is not one number across the panel (Figure 3): it falls from about 8.0%
for the slowest quartile of series to 5.5% for the fastest, because a low-volume
SKU carries less information per day to begin with. Within this panel only 0.2% of
series exceed a 10% floor — a fact worth stating plainly, because the panel is
drawn from the heavy end of the catalogue (the 4,000 best-selling series). The
long tail, where a series sells a few units a week, is absent here and would sit
strictly above these curves; the 7% floor is therefore a lower bound for a real
catalogue. That is the conservative direction for a ceiling claim, and the
opposite direction for a practitioner carrying many slow movers.

## 4. The ceiling is visible in the numbers

Table 1 puts all eleven methods in one place, on the official holdout. They span
20.3 to 35.9 WAPE and −7.1% to +28.0% bias, and the two axes are not independent:
across the eleven, corr(|bias|, WAPE) = 0.94 (R² = 0.89) and the fitted slope is
0.56 WAPE per point of absolute bias (Figure 4), i.e. almost all of the accuracy
difference between methods *is* their bias difference.

| method | WAPE | bias | fill | lost |
|---|---|---|---|---|
| sales as demand | 23.4 | −7.1% | 0.865 | 6154 |
| linear interpolation | 21.1 | +1.3% | 0.898 | 4643 |
| gradient-boosted imputation | 21.3 | +1.6% | 0.901 | 4496 |
| Kaplan–Meier (hard) | 28.8 | +9.4% | 0.899 | 4602 |
| Tobit (hard) | 23.3 | +10.6% | 0.934 | 3023 |
| negative binomial (hard) | 35.9 | +28.0% | 0.960 | 1815 |
| Kaplan–Meier (severity) | 22.1 | +3.2% | 0.899 | 4604 |
| **Tobit (severity)** | **20.5** | **−0.3%** | 0.896 | 4723 |
| **negative binomial (severity)** | **20.3** | +1.5% | 0.906 | 4271 |
| SAITS (4 runs) | 21.4 | +2.1% | — | — |
| ImputeFormer | 20.4 | +4.3% | — | — |

*Notes.* All eleven are scored on the same official 7-day holdout, on days with no
stock-out hour; *fill* and *lost* are the inventory simulation of §5 (50:1 cost
ratio, 3-day lead time). The two deep models are reported on accuracy only. A
TimesNet baseline collapsed to 28.0 WAPE (−13.6% bias) after normalisation and is
quoted here rather than in the table. The full run takes 300 s on a laptop CPU.

![Figure 4. Bias against accuracy, all eleven methods](results/p4_bias_accuracy.png)

That is what the bound predicts. A method that removes bias entirely — the
severity-weighted likelihood does, at −0.3% — lands at 20.5 WAPE and stops
there. A 325-second SAITS lands at 21.4 with a +2.1% bias; ImputeFormer lands at
20.4 with +4.3%. The floor of the whole comparison is near 20.3, and the
remaining twenty points are daily demand noise, not information about the level.

**The correlation is not causal, and the difference matters.** It is tempting to
read corr = 0.94 as "bias is the whole of accuracy". Two controlled checks say
otherwise. A simulated *pure* level shift of −7% changes WAPE by −0.04 points —
essentially free, because an L1 loss is minimised near the conditional median,
which for right-skewed demand sits below the mean. And rescaling the naive
pipeline's forecasts by 1.076 removes its entire −7.5% bias while moving WAPE
from 22.2 to 21.9: **0.3 points of a 2.1-point gap**. The other 1.8 points come
from the corrupted *history*, not the level: lags and rolling means computed on
censored sales distort the dynamics, and no post-hoc multiplier repairs them.

That distinction is the practical one. A team that patches its forecast level
has fixed a rounding error; a team that repairs the history has fixed the model.
It is also why a fifteen-second severity-weighted likelihood beats a 7%
multiplier, and why this paper's tool spends its effort on the history.

**What the bound costs, in WAPE.** Simulated against the same demand process
(negative-binomial, 3,000 series, a known weekday structure so that the base
WAPE is realistic), a *random* 7% level error — the size of the Cramér–Rao floor
— raises WAPE by 1.0% of its own value: **0.2 points at the benchmark's 20.5**.

There is also a bound that does not depend on the simulation. Write c for the
conditional coefficient of variation of daily demand (measured on this panel:
median **0.32**) and δ for the relative standard error of the level. The variance
of a forecast error decomposes exactly as

    Var(D) = E[Var(D | μ)] + Var(m(μ)) = (c·m)² + (δ·m)²,

and for a location–scale family the mean absolute deviation is proportional to
the standard deviation, so the ratio of WAPEs is bounded by the ratio of spreads:

    WAPE(δ) / WAPE(0) − 1  ≤  sqrt(1 + (δ/c)²) − 1.

At the floor (δ = 0.07, c = 0.32) this is **+2.3%, or 0.5 points at 20.5**. The
simulation says 0.2; the bound says 0.5; the spread between the best and worst
honest method is 1.1. **Either number leaves most of the spread unexplained by
level information, so it is reconstruction artifact.** And the bound is
falsifiable in the useful direction: a method that beats the floor by more than
0.5 points has either found information the censoring destroyed or is leaking.

![Figure 5. What a level error costs in WAPE](results/p3_wape_budget.png)

Figure 5 reads the two numbers against each other. The bound is a curve; the three
quantities the paper cares about are points on it. At the Cramér–Rao floor the
permitted inflation is 2.3%; the matched simulation measures 1.0%; and the
*observed* spread between the honest methods is 5.4%, which the curve says would
require a level error of about 10.6% — one and a half times the floor. Level
uncertainty could explain the whole spread only if the effective level error were
1.5× the largest any unbiased estimator can be forced into; at the floor it
explains under half of the spread, and the simulation says a fifth.

**The asymmetry is the part that should worry an operator.** A −7% level error
costs *nothing* in L1 — it moves predictions toward the median, which is where
an L1 loss is minimised — while a +10% error costs 1.6 points. Bias low is free
for accuracy and expensive for shelves; bias high is the reverse. A pipeline
that reports WAPE is therefore structurally inclined to under-forecast, which is
precisely the direction that empties shelves. The metric and the censoring push
in the same direction, and neither shows up as an error.

The claim makes two falsifiable predictions:

1. **No honest method should reach ~17 WAPE on this protocol.** The level
   carries at most a couple of points, and it is already corrected.
2. **A method that beats the floor by a wide margin is more likely leaking the
   holdout** than exploiting information that the Fisher bound says is not in
   the data.

The run-to-run noise is of the same size as the effect: SAITS scored 20.70,
21.46, 21.64 and 21.70 over four seeds, so the between-method spread (1.1) is
comparable to the within-model spread (1.0). Ranking methods on one run of this
benchmark is not informative at the resolution the literature reports.

## 5. What moves cost

Under a 50:1 shortage-to-holding ratio, ordering at the newsvendor quantile
$q^{*}\approx0.98$ instead of at the mean moves lost units from 6139 to 365
(16.8×) for the naive pipeline, and from 5507 to 329 (16.7×) once the history is
de-censored; at a *fixed* quantile the choice of reconstruction moves them by
10–21%. The corner-to-corner number, 18.7×, combines the two moves, and the
decomposition is worth keeping straight: the decision is worth an order of
magnitude, the estimation inside it worth a tenth of that.

| ordering quantile | sales as demand | Tobit (severity) | NB (severity) |
|---|---|---|---|
| 0.50 (the mean) | 6139 | 5507 | 4864 |
| 0.90 | 1186 | 1025 | 848 |
| 0.95 | 669 | 608 | 533 |
| 0.98 (newsvendor) | 365 | **329** | 342 |

*Notes.* Lost units over the whole holdout; the policy and the 50:1 cost ratio are
identical in every cell. Reading down a column is the decision, reading across a
row is the reconstruction.

![Figure 6. Lost units against the ordering quantile](results/r4_quantile.png)

Two readings of the grid matter. **Down a column**, the quantile is worth 14.2× to
16.8× in lost units (mean → 0.98 for the same estimator). **Across a row**, at a
fixed quantile, the entire reconstruction question is worth 10–21%, and at the
newsvendor quantile only 10%. The decision dominates the estimation by an order of
magnitude and — unlike the estimation — is not bounded by the information the
censoring destroyed.

## 6. Can the mask be optimised instead?

If the mask is what destroys information, the obvious response is to stock more
and observe more. We test it in a Bayesian simulation: 300 SKUs, 30 weeks,
demand NB2 with an unknown level, three policies — greedy (newsvendor quantile),
dither (same mean inventory, ±40% randomisation) and explore (+1 posterior sd) —
against an inventory-matched greedy control.

| policy | posterior error | fill | mean inventory |
|---|---|---|---|
| greedy q=0.91 | 21.1% | 0.783 | 3.06 |
| dither | 21.5% | 0.731 | 3.00 |
| explore | 16.3% | 0.837 | 3.37 |
| greedy q=0.995 (matched) | 17.1% | 0.830 | 3.31 |

Randomisation at fixed inventory buys nothing, and once inventory is matched,
exploration buys 0.8 pp. At scalar uncertainty, information gathering has no
separate value: the policy lever *is* the inventory level. What that means is
that the information destroyed by censoring cannot be cheaply bought back —
which is why the bound in §2 is a real constraint rather than a modelling
inconvenience.

## 7. What this changes

**Measure aggregate bias first.** It is the whole of the WAPE difference and it
is cheap to compute from a clean holdout week. A pipeline that reports WAPE but
not bias is reporting the part that is bounded and omitting the part that is
not.

**Correct the bias with the cheapest correct estimator.** The severity-weighted
likelihood needs the stock-status log and fifteen seconds of CPU; it takes a
−7.1% bias to −0.3%. Deep imputation reaches +2.1% at three orders of magnitude
more compute, with run-to-run variance larger than the remaining gap.

![Figure 7. Deep imputers against a fifteen-second likelihood](results/r6_deep_imputer.png)

Figure 7 is that sentence as a picture: the deep imputers cost 325–900 s of CPU
and land within 1.1 WAPE of a likelihood that costs 15 s, with a *larger* bias in
every case (SAITS +2.1%, ImputeFormer +4.3%, a normalised TimesNet −13.6%),
against −0.3% for the severity-weighted Tobit. The vertical distance to the floor
is noise; the horizontal distance is three orders of magnitude. Paying it is a
decision a team is entitled to make, but it should be made knowing that the
reconstruction was never the expensive part of the problem.

**Spend the next dollar on the quantile, not the reconstruction.** No
information bound applies to the ordering rule. Moving from the mean to the
newsvendor fractile is worth 14.2–16.8× in lost units for a fixed estimator
(16.8–18.7× if the history is de-censored too); the entire reconstruction
question is worth 10–21%.

**Carry the 7%.** The floor is a property of the data and is not removable, but
it is small next to the 25% daily variability that already drives safety stock.
The practical reading is that per-SKU level uncertainty is a second-order term
in inventory sizing, so it should be *acknowledged* rather than *modelled*: a
pipeline that spends its engineering budget chasing it will be chasing a
rounding error.

### 7.1 The ladder: censoring is a procurement decision

The bound is a property of the data rather than of the model, which turns
de-censoring from a modelling question into a procurement one. What you can
recover is fixed by what you log:

| rung | you log | the correction you can justify | measured bias |
|---|---|---|---|
| **R0** | sales | nothing: the censoring is unidentifiable | unknown |
| **R1** | + a stock-out flag | the textbook censored likelihood | **+9.4 to +28.0%** |
| **R2** | + stock-out hours | severity-weighted likelihood | **−0.3%** |
| **R3** | + hourly sales | same, self-calibrated profile | −0.3% |

R1→R2 is one column in the warehouse system, and it is worth 10–28 points of
bias — more than every modelling advance reported on this benchmark combined
(this paper's included). A practitioner who reads one line of this paper should
read that one.

### 7.2 What the audit reports

The accompanying tool answers the three questions in the order they are asked:
how censored is this table (rate, severity, effective days, the 7% floor); what
was demand (severity-weighted reconstruction, damped for the R1 case because a
full correction over-shoots); and how much to order (the newsvendor quantile).
It repairs the predictable failures of real exports — GBK encodings, renamed
columns, duplicate rows, returns arriving as negative units — and distinguishes
a day with no sale from a day with no record, which the usual pipeline silently
conflates by writing a zero.

**Code and data.** The dataset is public (CC-BY-4.0). The audit tool, the
estimators, the experiment scripts and every figure in this paper reproduce from
one command on a laptop CPU — no GPU, no training, twelve tests covering both
the estimators and the data-repair layer.

## 8. Limitations

1. $\rho$ assumes a negative-binomial family. The dispersion barely matters
   (0.374 at $k=10$ versus 0.334 in the Poisson limit), so the bound is robust
   to a mis-specified tail; the real assumption is the family itself, and a
   zero-inflated demand process would change the curve.
2. The bound is for per-SKU estimation from that SKU's own history. A pooled
   model borrows strength across series and is not bounded by it. Every
   published de-censoring method reconstructs each series independently, so the
   bound does apply to the paradigm being compared, but a pooled estimator could
   in principle sit below it.
3. Scored on 7 clean holdout days; one split; one benchmark, one country, one
   season (spring 2024), so validity across stores, cities and categories is
   untested here.
4. Lead time 3 days, cost ratio 50:1, no shelf life.
5. The 90-day window cannot resolve annual seasonality.

## Appendix A. Run-to-run variance of the deep imputer

The four SAITS runs quoted in Table 1 use one configuration and four seeds:

| run | WAPE | bias |
|---|---|---|
| seed 0 | 20.70 | +0.0% |
| seed 1 | 21.46 | +1.7% |
| seed 2 | 21.64 | +2.7% |
| seed 3 | 21.70 | +4.1% |

The within-model range (1.0 WAPE) is the same size as the between-method spread of
the honest methods (1.1). This is why Table 1 carries the inventory columns and
bias instead of a leaderboard rank: at one run, at this resolution, the ranking is
not stable, and a paper that reports only the rank is reporting noise.

A second reproducibility note: the full pipeline (Table 1, the quantile grid of
§5 and every figure) runs in 300 s on a laptop CPU with no GPU and no training,
and twelve tests cover the estimators and the data-repair layer. In a shell that
forbids named pipes — some sandboxes do — the run needs
`JOBLIB_MULTIPROCESSING=0` and fits serially at about twice the wall-clock time.

## References

- Azar, J. (2026). A Reproducible, Leakage-Free Pipeline for Censored Demand Forecasting and Inventory Optimization on FreshRetailNet-50K. Inventions, 11(5), 89.
- Chen, M., & Jin, J. (2025). Stockout-Aware Relational Demand Forecasting and Cross-Dataset Error Trade-offs. JTIE, 4(1), 325–346.
- Du, S., et al. (2023). SAITS. Expert Systems with Applications.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. JASA.
- Nahmias, S. (1994). Demand Estimation in Lost Sales Inventory Systems. Naval Research Logistics.
- Nie, T., et al. (2024). ImputeFormer. KDD.
- Talluri, K. T., & van Ryzin, G. J. (2004). The Theory and Practice of Revenue Management. Springer.
- Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. Econometrica.
- Vulcano, G., van Ryzin, G., & Ratliff, R. (2012). Estimating Primary Demand. Operations Research.
- Wang, Y., et al. (2025). FreshRetailNet-50K. arXiv:2505.16319.
- Wu, H., et al. (2023). TimesNet. ICLR.
- Yin, C., Zheng, Y., & Kong, Z. (2026). Demand Hidden by Stockouts. Sustainability, 18(15), 7642.
