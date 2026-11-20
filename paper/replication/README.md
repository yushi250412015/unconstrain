# Replication of What Hourly Stockout Records Reveal About Retail Demand

Unpack the full replication archive in an empty directory. The commands below run from its root. The archive also contains the numerical outputs used by the manuscript. The two data files are distributed by the data owner under CC BY 4.0. Place them in a directory named `data` at the root.

| File | Official download | SHA-256 |
|---|---|---|
| train.parquet | https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/train.parquet | 6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d |
| eval.parquet | https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/eval.parquet | 1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e |

Use Python 3.11.9. Package versions below match the run on Windows. The numerical analyses use CPU only.

```text
python -m pip install numpy==2.4.6 pandas==3.0.3 scipy==1.17.1 scikit-learn==1.9.0 pyarrow==20.0.0 matplotlib==3.11.0
python paper/replication/check_information.py
python paper/replication/analysis.py
python paper/replication/masking.py
```

| Output under paper/replication | Contents |
|---|---|
| window_90.json | Official week. Table 1 is in scenarios, Table 2 in forecasts, Table 3 in contrasts and Table 4 in quantiles. The groups field lists every city and category. |
| window_83.json | Earlier week. Separate sample selection, training and evaluation for Tables 2 and 3. |
| masking.json | Section 4.2. Reconstruction losses and paired uncertainty intervals on 5,711 complete days. |

The analysis also regenerates Figure 1 in `paper/figures/retention_v5.png`. Cite the dataset and its authors if reusing the numbers. The cap and thinning results are conditional scenarios with a plug-in mean. The two 95% intervals per comparison resample stores and products separately and retain the same evaluation dates. The inventory replay uses observed sales on all days.

The implementation includes the project’s original estimation and forecasting routines. The historical comparison point is local Git revision 2399446. The scripts and numerical results in this archive accompany the revised manuscript.

