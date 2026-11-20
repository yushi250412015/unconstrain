# De-censoring Has a Ceiling, and It Is Low: an information bound for censored retail demand

*Draft v4 — submission candidate, content freeze 2026-11-20. Numbers reproduce from `python -m unconstrain --data real --series 1500` (add `JOBLIB_MULTIPROCESSING=0` in a shell that forbids named pipes); twelve tests cover the estimators and the data-repair layer.*

## Abstract

Retail demand is censored by inventory, and a 2025–2026 literature on the
FreshRetailNet-50K benchmark competes on forecast accuracy. We ask how much
accuracy is available. At the benchmark's mean severity, a censored day retains
22–58% of a clean day's Fisher information: 90 days are worth 68, a quarter of the
level's information is destroyed, and the Cramér–Rao floor on any unbiased per-SKU
level estimate is 7.0%. Eleven methods, from linear interpolation to a deep
imputer, span 35 percentage points of bias but, with the bias removed, only 1.1 of
WAPE; the correlation of |bias| with WAPE (0.94) is real and not causal. Once bias
is corrected, error sits at a floor near 20 WAPE and no method has gone below it.
The decision layer is not bounded this way: ordering at the newsvendor quantile
instead of the mean moves lost units by 16.8× (18.7× with a de-censored history),
an order of magnitude more than the entire reconstruction question, while the
floor's own level error is a second-order cost at the order point. De-censoring is
worth doing to remove bias; it is not worth funding to chase accuracy.

**Keywords:** censored demand; unconstraining; Fisher information; Cramér–Rao bound; relative efficiency; newsvendor quantile; pinball loss.

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

**Lineage, and what is new.** The arithmetic is not new. The relative efficiency
of a censored sample against a complete one is the classical Fisher-information
calculation (Lehmann & Casella, 1998), and the censored likelihoods compared here
are Tobin's and Kaplan–Meier's (1958). Reading a stock-out as *coarse* data
rather than as missing data is also established: a censored day is a coarsening of
the demand variable, and the ignorability theory for coarse data (Heitjan & Rubin,
1991) is precisely the garbling that Proposition 3 invokes; the lifetime-data
treatment of censored likelihoods is standard (Lawless, 2003). What is new is the
*accounting*, not the algebra: (i) the quantity is computed on this benchmark and
turned into a ladder of logging rungs whose next step is worth 10–28 points of
bias; (ii) the same level error is priced twice — second order in newsvendor cost,
first order through the ordering quantile (Proposition 5) — which separates the
two layers by an order of magnitude; and (iii) the ceiling is stated as a
falsifiable prediction, so a method that beats the floor is more likely to be
leaking the holdout than to be informative. A reader who knows the classical
literature should read the contribution as a measurement plus a decision-theoretic
reading, not as a new estimator.

## 2. How much information a censored day carries

**Model.** A SKU's demand on day $t$ is $D_t\sim\mathrm{NB2}(\mu,k)$, with mean
$\mu$ and dispersion $k$ (variance $\mu+\mu^2/k$); days are conditionally
independent given $\mu$. The warehouse stocks $C_t$ units before demand is
realised and records $Y_t=\min(D_t,C_t)$, together with the stock status that fixes
$C_t$. The object of interest is the *level* $\mu$; the estimator's job is to
report it, not to reconstruct the individual unobserved days.

We work under five assumptions, stated because each one is a scope restriction
that a reader is entitled to check:

- **(A1) Family.** $D_t\mid\mu$ is negative binomial with known dispersion
  $k$; $\mu\in(0,\infty)$; days independent given $\mu$.
- **(A2) Observation.** $Y_t=\min(D_t,C_t)$ with $C_t\perp D_t$ — the stock level
  is set by a policy, not by the demand it faces, so censoring is not
  informative about $\mu$ beyond the recorded severity.
- **(A3) Severity.** The data record
  $s_t=1-\mathbb{E}[\min(D_t,C_t)]/\mu\in[0,1]$, the share of the day's expected
  volume that was unreachable; $s_t=0$ on a fully-stocked day.
- **(A4) Level.** $\mu$ is constant over the accounting window; the estimand is
  the window-average level.
- **(A5) Regularity.** $p(\cdot;\mu)$ is differentiable in $\mu$ and dominated,
  so Fisher information and the Cramér–Rao bound apply.

*Scope remark.* (A1)–(A5) make the bound below a statement about **per-SKU
estimation from that SKU's own history**. A pooled estimator borrows strength
across series and is not covered; this is a restriction of the claim, not a
caveat to be argued away.

For a day observed with stock level $c$, the observation is a mixture: values
below $c$ are seen exactly, values at or above $c$ collapse into the event
$D \ge c$. Under (A1) the Fisher information about
$\mu$ is

$$I(c,\mu)=\sum_{y<c}p(y)\left(\frac{\partial\log p(y)}{\partial\mu}\right)^2
+ P(D\ge c)\left(\frac{\partial\log P(D\ge c)}{\partial\mu}\right)^2,$$

and the **retention** $\rho=I(c,\mu)/I(\infty,\mu)$ is the fraction of a clean
day's information that survives — in the classical language, the *relative
efficiency* of a censored observation against a complete one. With NB2 demand,
$\rho$ depends on the level $\mu$ and on the severity $s$ of the loss (the
fraction of volume unreachable, which fixes $c$ through
$E[\min(D,c)]=(1-s)\mu$).

**Proposition 1 (the cap is identified by severity).** Under (A1)–(A3), the map
$c\mapsto\mathbb{E}[\min(D,c)]$ is continuous and non-decreasing, with limit $\mu$
as $c\to\infty$; it is strictly increasing wherever $P(D\ge c)>0$. Hence for every
$s\in(0,1)$ there is a unique implicit cap $c(s)$ with
$\mathbb{E}[\min(D,c(s))]=(1-s)\mu$. *Proof:* appendix B.

**Proposition 2 (information is additive, so effective days are not a
heuristic).** Under (A1), (A2), (A5) the sample's Fisher information about $\mu$ is
$I_n(\mu)=I_1(\mu)\sum_t\rho(s_t)$, where $I_1$ is a clean day's information.
Writing $n_{\text{eff}}=\sum_t\rho(s_t)$, the sample is *exactly* informationally
equivalent to $n_{\text{eff}}$ clean days — the "effective days" of §3 are a
quantity, not an analogy. *Proof:* additivity of Fisher information for
independent observations, with Proposition 1 fixing each day's cap.

**Corollary (the floor).** Under (A1)–(A5), any unbiased estimator of $\mu$
satisfies $\mathrm{Var}(\hat\mu)\ge 1/(I_1(\mu)n_{\text{eff}})$, hence a relative
standard error of at least $1/(\mu\sqrt{I_1(\mu)n_{\text{eff}}})$. At the
benchmark medians ($n_{\text{eff}}=67.8$) this is 7.0%.

**Proposition 3 (retention falls monotonically with severity).** Under (A1)–(A2),
$\rho(s)$ is non-increasing in $s$, and strictly decreasing whenever
$P(D\ge c(s))>0$. *Proof sketch:* increasing severity replaces an exact
observation by a coarser one (either "below the cap" or "above it"), a garbling of
the original data; a garbling cannot increase Fisher information
(data-processing for experiments), so $\rho$ cannot rise. The dependence on
$\mu$ carries no such monotonicity: it is non-monotone in the sparse regime, and
we report it as computed (§3) rather than as proved.

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

### 3.1 The severity definition is a choice, and it moves bias

The accounting needs a number for the severity of a censored day, and the data
offer two: the *share of business hours* without stock, or the *share of the day's
volume* that was unreachable. The second requires the SKU's hourly sales profile;
the first requires nothing. The panel supports both, so the choice can be priced.
Three definitions are compared: hours out of 16; volume weighted with the global
hourly profile; volume weighted with the SKU's own profile (the one used throughout
this paper).

| severity definition | mean on censored days | KM (sev) bias | Tobit (sev) bias | NB (sev) bias | mean WAPE |
|---|---|---|---|---|---|
| hours / 16 | 0.388 | +4.08% | +0.92% | +4.30% | 21.24 |
| volume, global profile | 0.325 | +3.72% | +0.05% | +2.24% | 21.13 |
| **volume, own profile** | **0.318** | **+3.19%** | **−0.33%** | **+1.55%** | **20.98** |

*Same holdout, grader and forecaster in every row; only the severity column changes.
Mean WAPE is over the three severity-weighted estimators.*

Counting hours overstates severity by 22% (0.388 against 0.318), because sales
concentrate in the middle of the day: one missing hour at 9 a.m. destroys far more
demand than one at 8 p.m. The consequence is what the model predicts. Severity
enters through the cap (Proposition 1), so a mis-stated severity shows up as *bias*
first — the hour-count version carries +0.9 to +4.3 points of bias where the volume
version carries −0.3 to +3.2 — while accuracy moves by at most 0.5 WAPE, at the
resolution of the design (§8). Two conclusions follow. First, the severity
definition belongs in the bias column, not the accuracy column, exactly like the
reconstruction question it feeds. Second, the self-calibrated profile (own against
global) is worth 0.1–0.3 WAPE and up to 0.7 bias points: real, reproducible, and
small — which is the honest size of the "hourly sales" rung in the logging ladder
of §7.1.

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

There is also a bound that does not depend on the simulation.

**Proposition 4 (what a level error can cost, in WAPE).** Assume (i) the level
estimate carries relative error $\delta$, so that
$\mathrm{Var}(D)=\mathbb{E}[\mathrm{Var}(D\mid\mu)]+\mathrm{Var}(m(\mu))=(cm)^2+(\delta m)^2$
with $c$ the conditional coefficient of variation of daily demand (measured on
this panel: median **0.32**), and (ii) the demand family is location–scale, so its
mean absolute deviation is proportional to its standard deviation. Then the
relative inflation of the mean absolute error is
$\mathrm{MAD}(\delta)/\mathrm{MAD}(0)=\sqrt{1+(\delta/c)^2}$ exactly, and
$\mathrm{WAPE}$, being its sample analogue, inherits it up to normalisation.
*Why this is stated as a proposition about the MAD and not about WAPE:* WAPE
divides by realised sales rather than by expected volume, so it is not exactly a
location–scale functional; the ratio of realised to expected volume is 1.00 at the
median of this panel, which is why we report the MAD bound and use its sample
analogue.

At the floor ($\delta = 0.07$, $c = 0.32$) this is **+2.3%, or 0.5 points at 20.5**. The
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

### 4.1 The metric the decision uses

WAPE is a scaled L1 loss, and L1 is minimised at the conditional median — so it
scores the reconstruction where the order is *not* placed. The order is placed at a
high quantile, and the proper scoring rule for that is the pinball (quantile) loss
(Koenker & Bassett, 1978). Scored with it, and normalised by total demand so the
units match WAPE:

| method | pinball τ=0.50 | τ=0.90 | τ=0.98 |
|---|---|---|---|
| sales as demand | 11.35 | 5.78 | 1.96 |
| linear interpolation | 10.82 (−4.7%) | 5.83 (+0.9%) | 2.06 (+5.3%) |
| Tobit (severity) | **10.52 (−7.3%)** | **5.33 (−7.8%)** | **1.79 (−8.7%)** |

*One quantile-GBM fit per cell and one seed; the contrasts are much larger than the
0.4-WAPE resolution of the accuracy design, but a seed sweep of the quantile fits
is not reported here.*

Two things change relative to the WAPE ranking. First, linear interpolation is
−4.7% at the median, +0.9% at τ=0.9 and **+5.3% at τ=0.98**: it *degrades* exactly
where the order is placed, so its apparent parity with the severity-weighted
likelihood on WAPE is a median-metric artifact. Second, the reconstruction's value
*rises* with the quantile (−7.3%, −7.8%, −8.7%) — the benefit is larger on the
metric the decision uses than on the metric the field reports. That is the empirical
counterpart of Proposition 5, and it sharpens the recommendation in §7: if the order
is placed at a high quantile, the reconstruction should be scored there.

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

**Proposition 5 (the decision layer is first order in the quantile, the estimation
layer is second order in the level error).** Let
$C(q)=c_h\mathbb{E}[(q-D)^+]+c_s\mathbb{E}[(D-q)^+]$ be the per-period newsvendor
cost and $q^{*}=F^{-1}\!\left(c_s/(c_s+c_h)\right)$ its minimiser. For continuous
$F$, $C'(q)=(c_h+c_s)F(q)-c_s$ and $C''(q)=(c_h+c_s)f(q)$; hence $C'(q^{*})=0$ and
a level error of relative size $\varepsilon$ costs

$$C\big(q^{*}(1+\varepsilon)\big)-C(q^{*})
=\tfrac12(c_h+c_s)\,f(q^{*})\,(q^{*}\varepsilon)^2+o(\varepsilon^2),$$

while moving the order point from the mean $\mu$ to $q^{*}$ changes the cost at
first order. The estimation layer is *second order in the level error*; the
decision layer is *first order in the quantile gap*. That is the analytic content
of the ratio measured above.

**A worked evaluation at the median SKU** ($\mu=3.22$, $k=60$, $c_s/c_h=50$, so
$q^{*}=8$): $C(\mu)=37.4$ and $C(q^{*})=5.3$, a decision gain of **32.0**. A 7%
level error — the floor — costs **0.35**, so the decision is worth **92×** the
price of the floor's level error. The quadratic scaling is confirmed numerically:
the penalty at $\varepsilon=$ 3.5%, 7%, 14%, 28% is 0.17, 0.35, 0.73, 1.72, i.e.
it quadruples when the error doubles. The local Taylor constant is 3.4× smaller
than the exact penalty, because demand is integer and $C$ is piecewise linear with
kinks at every unit; the *scaling* survives and the constant does not, so we report
the exact figures and use the expansion only for the order statement.

The two numbers count different things and must not be added: the simulation's
18.7× is a lost-units ratio under a multi-period $(s,S)$ policy, while the 92× is a
cost ratio in a single-period newsvendor at the median SKU. What they share is the
ordering of the two layers — and that ordering is a theorem, not a coincidence.

**Across the panel the ratio is larger, not smaller.** Repeating the evaluation for
all 1,500 series (μ≥1; the optimal order point has median 8, p10–p90 6–11): the
decision gain has median **32.0**, the cost of a 7% level error has median
**0.21**, and their ratio has median **158** (p10 98, p90 755). Every series in the
panel has a ratio above 10. The median-SKU number above is therefore the
conservative case, not a cherry-picked one: on this panel the decision layer is
worth two orders of magnitude more than the floor's level error, and the ordering
is never close.

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

## 8. Statistical protocol

Four decisions were fixed before the numbers were read; they are what the
intervals below mean.

1. **Resampling unit.** The cluster bootstrap resamples *series*, not days: days
   inside a SKU share a level and are strongly dependent, so resampling days would
   understate every interval. 2,000 resamples, percentile intervals.
2. **Paired comparisons.** Methods are compared on per-series absolute-error sums,
   paired across the same series; a paired bootstrap on the difference is more
   powerful than comparing two overlapping marginal intervals. Two of the 1,500
   series have no fully-stocked holdout day and are excluded from the intervals,
   which moves point estimates by at most 0.12 WAPE.
3. **A pre-specified family, Holm-corrected.** Five comparisons were named before
   the intervals were computed. All five survive Holm correction at p ≤ 0.005. The
   two severity-weighted likelihoods are 0.12 WAPE apart with overlapping
   intervals, so we do not rank them.
4. **Descriptive statistics stay descriptive.** The correlation of 0.94 is computed
   across eleven method-level points, not across the 10,500 holdout observations.
   It is not a test, and §4 gives it no causal reading.

| method | WAPE | 95% CI | bias | 95% CI |
|---|---|---|---|---|
| sales as demand | 23.3 | 22.2 – 24.6 | −7.2 | −8.4 – −6.0 |
| linear interpolation | 21.1 | 20.0 – 22.2 | +1.2 | +0.1 – +2.4 |
| Kaplan–Meier (hard) | 28.6 | 26.8 – 30.7 | +9.2 | +7.3 – +11.4 |
| Tobit (hard) | 23.2 | 21.8 – 24.5 | +10.5 | +9.0 – +11.9 |
| negative binomial (hard) | 35.8 | 32.9 – 39.3 | +27.9 | +24.7 – +31.7 |
| Kaplan–Meier (severity) | 22.0 | 20.9 – 23.4 | +3.1 | +1.8 – +4.4 |
| **Tobit (severity)** | 20.4 | 19.3 – 21.5 | **−0.4** | −1.4 – +0.6 |
| negative binomial (severity) | 20.3 | 19.3 – 21.3 | +1.5 | +0.4 – +2.6 |

*Cluster-bootstrap intervals over 1,498 series. The point estimates differ from
Table 1 by at most 0.12 WAPE because two series have no fully-stocked holdout day.*

| comparison | ΔWAPE | 95% CI | Holm p |
|---|---|---|---|
| KM severity − KM hard | −6.59 | −7.62 – −5.71 | 0.0025 |
| Tobit severity − sales as demand | −2.95 | −3.38 – −2.55 | 0.0020 |
| Tobit severity − Tobit hard | −2.78 | −3.36 – −2.24 | 0.0015 |
| NB severity − interpolation | −0.81 | −1.23 – −0.40 | 0.0020 |
| Tobit severity − interpolation | −0.69 | −1.12 – −0.26 | 0.0040 |

*Paired cluster bootstrap over the same series; a negative ΔWAPE means the first
method is more accurate. Holm correction is over the pre-specified family of five.*

The smallest difference this design resolves at 95% confidence is about **0.4
WAPE**. Differences below that — including the 0.2-point gap between our estimator
and the strongest imputer — should be read as ties, not as a ranking.

## 9. Threats to validity

**Construct.** WAPE is not the decision cost. The paper keeps the two layers apart
and derives the mapping between them (Proposition 5) instead of letting one stand
for the other.

**Internal.** One split, seven holdout days, and no pre-registration beyond the
four protocol decisions in §8. The pipeline is deterministic given the seed and the
table reproduces exactly, but a different split would move the point estimates;
the intervals in §8 are the honest statement of that uncertainty.

**External.** One benchmark, one country, one season (spring 2024), and a panel
drawn from the heavy end of the catalogue. The long tail is absent, so the 7% floor
is a lower bound for a real catalogue rather than a typical value, and the
level-dependence in §3 must not be extrapolated to SKUs that sell a few units a
week.

**Statistical.** Eleven methods are compared on 1,498 clusters with effects of the
same order as the noise: the between-method spread among the honest methods is 1.1
WAPE, the within-model spread across four SAITS seeds is 1.0 (Appendix A). The
Fisher calculation assumes the NB2 family; the dispersion sensitivity (0.374 at
$k=10$ versus 0.334 as $k\to\infty$) says the bound is robust to that choice, but a
zero-inflated demand process would shift the curve and is not tested here.

**Model.** (A2) assumes the stock level is set by a policy rather than by the
demand it faces. In this dataset availability is independent of any forecaster
under test, but a real deployment that sizes inventory from its own forecast breaks
(A2), and the bound would then need the joint model of demand and policy. Lead
time 3 days, cost ratio 50:1, no shelf life; the 90-day window cannot resolve
annual seasonality.

**Scope of the bound.** Per-SKU, not pooled (the remark under A1–A5). Every
published de-censoring method reconstructs each series independently, so the bound
applies to the paradigm being compared; a pooled estimator could in principle sit
below it, and that is the one direction in which the ceiling claim could be
overturned by a method rather than by a leak.

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

## Appendix B. Proofs

**Proposition 1.** Write $\mathbb{E}[\min(D,c)] = c-\mathbb{E}[(c-D)^+]$. Both
terms are continuous in $c$ and the second is non-increasing, so the difference is
continuous and non-decreasing; it tends to $\mathbb{E}[D]=\mu$ as $c\to\infty$, and
its right derivative is $P(D\ge c)$, positive wherever the support carries mass
above $c$. A continuous non-decreasing function from $[0,\infty)$ onto $[0,\mu)$
therefore attains every value $(1-s)\mu$ with $s\in(0,1)$, uniquely except on flat
segments, where the value is attained at the left endpoint. ∎

**Proposition 2.** For independent observations with densities differentiable in
$\mu$, the joint score is the sum of the individual scores and the cross terms in
its variance vanish under independence (A1), so
$I_n(\mu)=\sum_t I(c_t,\mu)$. Each day contributes $I(c_t,\mu)=\rho(s_t)I_1(\mu)$ by
the definition of $\rho$, with $c_t$ recovered from the recorded severity by
Proposition 1. Summing gives $I_n=I_1(\mu)\sum_t\rho(s_t)$. ∎

**Proposition 3.** Lowering the cap from $c$ to $c'<c$ maps
$Y_c=\min(D,c)$ deterministically to $Y_{c'}=\min(Y_c,c')$: a garbling of the
observation, many-to-one wherever $P(D\ge c)>0$. By the data-processing inequality
for statistical experiments a garbling cannot raise Fisher information, so
$I(c',\mu)\le I(c,\mu)$ and $\rho$ is non-increasing in severity, strictly so
where the garbling is non-invertible on a set of positive probability. ∎

**Proposition 4.** Let the forecast be $p=\hat\mu$ with
$\mathbb{E}[\hat\mu]=\mu$ and relative standard deviation $\delta$, so
$\mathrm{Var}(p)=(\delta\mu)^2$. Then
$\mathrm{Var}(D-p)=\mathbb{E}[\mathrm{Var}(D\mid\mu)]+\mathrm{Var}(p)=(c\mu)^2+(\delta\mu)^2$,
which is $(c\mu)^2$ at $\delta=0$. The standard deviation — and, for a
location–scale family, the mean absolute deviation exactly — therefore inflates by
$\sqrt{1+(\delta/c)^2}$. ∎

**Proposition 5.** $C'(q)=(c_h+c_s)F(q)-c_s$ and $C''(q)=(c_h+c_s)f(q)>0$, so
$C'(q^{*})=0$ with $q^{*}=F^{-1}\!\big(c_s/(c_h+c_s)\big)$. Taylor's theorem at
$q^{*}$, with $q=q^{*}(1+\varepsilon)$, gives
$C(q)-C(q^{*})=\tfrac12C''(q^{*})(q^{*}\varepsilon)^2+o(\varepsilon^2)$. For integer
demand $C$ is piecewise linear and the expansion is an order statement only;
§5 reports the exact penalties. ∎

## Data, code and declarations

**Data availability.** FreshRetailNet-50K (Wang et al., 2025) is public under
CC-BY-4.0 and distributed on Hugging Face as `train.parquet` and `eval.parquet`;
no new data were generated. The official evaluation split (2024-06-26 to
2024-07-02) is used as the holdout and is never touched by model selection,
threshold choice or figure revision.

**Code availability.** The audit tool, estimators, experiment scripts and every
figure are in `github.com/yushi250412015/unconstrain` under the MIT licence.
`python -m unconstrain --data real --series 1500` reproduces Tables 1–2 and
Figures 1–7 in 300 s on a laptop CPU, and twelve tests cover the estimators and the
data-repair layer. In a shell that forbids named pipes, prefix the run with
`JOBLIB_MULTIPROCESSING=0`.

**Reproducibility.** Every empirical number is written by the pipeline to a JSON
file in `results/` (`theory.json`, `panel_theory.json`, `severity_ablation.json`);
the tables and the text quote only those files, and no number is typed by hand.

**Author contributions (CRediT).** [First Author]: conceptualization, methodology,
software, formal analysis, investigation, writing — original draft. [Co-author]:
[roles to be completed].

**Competing interests.** The author(s) declare no competing interests.

**Funding.** This research received no specific grant from any funding agency.

**Acknowledgements.** [To be completed: advisor, department and reviewers.]

## References

- Azar, J. (2026). A Reproducible, Leakage-Free Pipeline for Censored Demand Forecasting and Inventory Optimization on FreshRetailNet-50K. Inventions, 11(5), 89.
- Chen, M., & Jin, J. (2025). Stockout-Aware Relational Demand Forecasting and Cross-Dataset Error Trade-offs. JTIE, 4(1), 325–346.
- Du, S., et al. (2023). SAITS. Expert Systems with Applications.
- Heitjan, D. F., & Rubin, D. B. (1991). Ignorability and Coarse Data. Annals of Statistics, 19(4), 2244-2253.
- Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. JASA.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33-50.
- Lawless, J. F. (2003). Statistical Models and Methods for Lifetime Data (2nd ed.). Wiley.
- Lehmann, E. L., & Casella, G. (1998). Theory of Point Estimation (2nd ed.). Springer.
- Nahmias, S. (1994). Demand Estimation in Lost Sales Inventory Systems. Naval Research Logistics.
- Nie, T., et al. (2024). ImputeFormer. KDD.
- Talluri, K. T., & van Ryzin, G. J. (2004). The Theory and Practice of Revenue Management. Springer.
- Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. Econometrica.
- Vulcano, G., van Ryzin, G., & Ratliff, R. (2012). Estimating Primary Demand. Operations Research.
- Wang, Y., et al. (2025). FreshRetailNet-50K. arXiv:2505.16319.
- Wu, H., et al. (2023). TimesNet. ICLR.
- Yin, C., Zheng, Y., & Kong, Z. (2026). Demand Hidden by Stockouts. Sustainability, 18(15), 7642.

