# unconstrain

**Audit and undo demand censoring in retail sales data.**

When a shelf empties, the sale does not happen and the database records a smaller
number. Every downstream model learns from that truncated series. `unconstrain`
does three things about it:

1. **`diagnose`** — how censored is your data, what can it support, and which
   single column should you log next to make the correction possible.
2. **`decensor`** — reconstruct demand with a severity-weighted likelihood,
   which is what the stock-out log is *for*.
3. **`recommend_quantile`** — how much to order, because the mean forecast is
   the wrong target under asymmetric cost.

No GPU, no training, no config. The core is a few hundred lines of numpy/scipy.

---

## Install

```bash
pip install -e .          # numpy pandas scipy scikit-learn
```

On a shell that forbids named pipes (some sandboxes and restricted shells),
prefix any run with `JOBLIB_MULTIPROCESSING=0`: the pipeline then fits serially,
about twice as slow, and produces the same numbers.

## The data ladder

What you can recover depends on what you log. `diagnose` names your rung:

| rung | you log | what it costs you |
|---|---|---|
| **R0** | sales only | censoring is unidentifiable; bias unknown |
| **R1** | + a stock-out flag | the textbook correction **over-shoots 9–28%** |
| **R2** | + stock-out hours or severity | correction lands near **−0.3%** |
| **R3** | + hourly sales | same, and self-calibrated |

Moving from R1 to R2 is one column in the WMS and about 25 percentage points of
bias. That is the whole sales pitch.

## CLI

```bash
unconstrain diagnose sales.csv --stock-hours hours_out
unconstrain decensor sales.csv --stock-hours hours_out -o demand.csv
unconstrain decensor sales.csv --stockout flag --strength 0.4 -o demand.csv
unconstrain report   sales.csv --stock-hours hours_out -o audit.html
unconstrain quantile --shortage 1 --holding 0.02
unconstrain demo                     # runs the whole ladder on the real benchmark
```

`--stockout` accepts a boolean column. `--stock-hours` accepts hours and derives
severity from `--open-hours` (default 16). `--strength` exists for R1: when all
you have is a flag, a full correction over-corrects, so blend (`0.4` is a
measured compromise).

## Python

```python
import pandas as pd
from unconstrain import api

df = pd.read_csv("sales.csv")          # series, date, sales, hours_out

rep = api.diagnose(df, stock_hours="hours_out")
print(rep)                              # readiness, information retained, actions

dec = api.decensor(df, stock_hours="hours_out")
dec.to_csv("demand.csv", index=False)   # adds a demand_hat column

api.recommend_quantile(cost_shortage=1.0, cost_holding=0.02)
# {'quantile': 0.9804, ...}  -> order the 98th percentile, not the mean

api.cost_of_policy(dec)                 # fill rate, lost units, cost
```

## What it reports, and why you should believe it

Measured on FreshRetailNet-50K (50,000 store–product series, hourly stock status),
1,500-series sample, scored on the official holdout **only on fully-stocked days**
— where sales *are* demand, so no ground truth is invented:

| method | WAPE | bias |
|---|---|---|
| sales as demand | 23.4 | −7.1% |
| linear interpolation | 21.1 | +1.3% |
| SAITS / ImputeFormer (deep imputers) | 20.4–21.4 | +2.1…+4.3% |
| **severity-weighted tobit** | **20.5** | **−0.3%** |

A quarter of the demand-level information is destroyed by censoring, and any
unbiased per-series level estimate has a standard error of at least **7%** —
which is why no method gets far below 20 WAPE, and why the tool optimises bias
rather than chasing accuracy.

## Dirty data

Real exports are broken in predictable ways, and `diagnose` names every repair it
made rather than silently applying it:

| what breaks | what happens |
|---|---|
| GBK / BOM / latin-1 encoding | `read_table` retries four encodings |
| `Qty`, `Item`, `OrderDate`, `销量`, `缺货小时` | columns guessed from their names |
| transaction-level rows | same series-day summed, logged as `duplicates` |
| returns arriving as negative units | clipped to 0, logged as `negative_sales` |
| a SKU with no row that day | marked **unobserved**, counted in `coverage`; never read as a zero sale |
| products that launch late | counted as `ragged`, kept as missing rather than zero |

The distinction between "sold nothing" and "was not on the shelf" is the one that
matters: writing a zero where there is no row teaches the model that demand
collapsed. `decensor(..., complete=True)` returns the full rectangle with NaN in
those cells, so nothing is fabricated downstream.

On a deliberately dirty file (GBK, Chinese headers, 4 duplicates, 5 missing days,
4 negative rows):

```
censoring audit
  series                 3
  days                   60
  coverage               97.2% of series-days observed
  readiness              R2
  information retained   65%  (39 effective days)
  data issues:
    - negative_sales: 4 - returns/corrections clipped to 0
    - duplicates: 4 - same series-day appears more than once; summed
    - unobserved: 5 - 2.8% of series-days have no row; they are NOT treated as zero sales
```

## Tests

```bash
python tests/test_api.py      # 5 tests: ladder, recovery, strength, quantile, cost
python tests/test_dirty.py    # 7 tests: duplicates, holes, negatives, ragged, GBK, guessing
```

## Honest limits

- Severity is the share of the day's *volume* unreachable, estimated from the
  stock-out hours and the SKU's own hourly profile. It assumes the hourly demand
  rate is roughly stable within the day.
- The reconstruction is deliberately conservative: it does not assume that a
  blocked hour would have sold at the average rate, so its volume recovery is
  smaller than the naive `sales / (1 − s)` scaling.
- Cost model: lead time 3 days, (s,S), 50:1 shortage-to-holding by default; all
  assumptions are function arguments.
- Validated on one benchmark, one country, 90-day series.

## Citation

If you use this, cite the dataset:

```
Wang et al. (2025). FreshRetailNet-50K. arXiv:2505.16319
```

License: MIT. Dataset: CC-BY-4.0.
