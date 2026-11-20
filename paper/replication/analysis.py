"""Regenerate the manuscript tables and figure. Run from the repository root."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from unconstrain.realdata import select_series, volume_severity
from unconstrain.experiments_real import fit_predict, _fit_quantile
from unconstrain.simulate import Policy, policy_from_forecast, simulate_inventory
from unconstrain.metrics import inventory_cost

OUT = Path(__file__).resolve().parent
FIG = ROOT / "paper" / "figures" / "retention_v5.png"
FILES = [ROOT / "data" / name for name in ("train.parquet", "eval.parquet")]

def prepare(n_train):
    keys = ["store_id", "product_id", "dt", "sale_amount"]
    frame = pd.read_parquet(FILES[0], columns=keys)
    frame["sid"] = frame.store_id * 1000000 + frame.product_id
    days = sorted(frame.dt.unique())
    eligible = frame if n_train == 90 else frame[frame.dt <= days[n_train - 1]]
    ids = select_series(eligible, 1500, 4000, seed=0)
    del frame, eligible
    columns = keys + ["city_id", "first_category_id", "discount", "stock_hour6_22_cnt",
                      "hours_sale", "hours_stock_status"]
    chunks = []
    for path in (FILES if n_train == 90 else FILES[:1]):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns):
            raw = batch.to_pandas()
            raw["sid"] = raw.store_id * 1000000 + raw.product_id
            chunks.append(raw[raw.sid.isin(ids)])
    data = pd.concat(chunks).sort_values(["sid", "dt"])
    t = data.dt.nunique()
    assert data.groupby("sid").size().eq(t).all() and not data.duplicated(["sid","dt"]).any()
    assert data[["sale_amount","discount","stock_hour6_22_cnt"]].notna().all().all()
    n = len(ids)
    def matrix(key):
        return data[key].to_numpy().reshape(n, t)
    hours = np.stack(data.hours_sale).reshape(n,t,24)[:,:n_train]
    status = np.stack(data.hours_stock_status).reshape(n,t,24)[:,:n_train]
    mask = matrix("stock_hour6_22_cnt") > 0
    sales = matrix("sale_amount")
    promo = (matrix("discount") < .999).astype(float)
    panel = dict(sales=sales[:,:n_train], mask=mask[:,:n_train], promo=promo[:,:n_train])
    panel["severity"] = np.where(panel["mask"],volume_severity(hours,status,~panel["mask"]),0.)
    panel["hours"] = matrix("stock_hour6_22_cnt")[:,:n_train] / 16.
    hold = dict(sales=sales[:,n_train:],mask=mask[:,n_train:],promo=promo[:,n_train:])
    meta = data.drop_duplicates("sid").set_index("sid")[["store_id","product_id","city_id","first_category_id"]]
    return panel, hold, meta

def information(mu, k, severity):
    # Served volume at integer cap j is the sum of survival probabilities through j.
    # For an implied real-valued cap, searchsorted gives the first pooled integer.
    variance = mu + mu**2 / k
    kprob = k / (k+mu)
    j = np.arange(int(nbinom.ppf(1 - 1e-13, k, kprob)) + 2)
    pmf = nbinom.pmf(j,k,kprob)
    tail = nbinom.sf(j-1,k,kprob)
    served = np.r_[0., np.cumsum(tail[1:])]
    scores = (j - mu) / variance
    exact = np.r_[0.,np.cumsum(pmf[:-1]*scores[:-1]**2)]
    # Tail score numerator equals E[(D-mu) 1{D>=j}] / variance.
    numerator = mu * (nbinom.sf(j-2,k+1,kprob) - tail) / variance
    rho = (exact + numerator**2 / tail) * variance
    thresholds = np.searchsorted(served, (1-severity)*mu)
    return np.where(severity == 0, 1., rho[np.minimum(thresholds,len(j)-1)])

def scenarios(panel):
    valid = ~panel["mask"]
    means = (panel["sales"] * valid).sum(1)/valid.sum(1)
    assert np.isfinite(means).all() and min(means) > 0
    rows = []
    for scale,k in ((1,10),(1,60),(1,2000),(10,10),(10,60),(10,2000)):
        levels = scale*means
        effective = np.array([information(m,k,s).sum() for m,s in zip(levels,panel["severity"])])
        rel = np.sqrt((1/levels+1/k)/effective)
        assert np.all((effective >= valid.sum(1)-1e-8) & (effective <= valid.shape[1]+1e-8))
        available = 1. - panel["severity"]
        thinning = (available*(1+levels[:,None]/k)/(1+available*levels[:,None]/k)).sum(1)
        thin_rel = np.sqrt((1/levels+1/k)/thinning)
        rows.append(dict(scale=scale,k=k,thin_days=np.median(thinning),thin_rse=100*np.median(thin_rel),
                         days=np.median(effective),rse=100*np.median(rel),
                         p10=100*np.quantile(rel,.1),p90=100*np.quantile(rel,.9)))
    fig,ax = plt.subplots(figsize=(6.0,3.15),layout="constrained")
    for m,color in [(1,"#31353c"),(5,"#126579"),(20,"#993d29")]:
        s=np.linspace(0,1,1601)
        vals=information(m,60,s)
        assert np.isclose(vals[0],1) and np.isclose(vals[-1],0) and (np.diff(vals)<1e-8).all()
        ax.step(s,vals,where="post",color=color,label=f"Mean count {m}",lw=1.7)
    ax.set(xlabel="Expected share of demand lost at the cap",ylabel="Fraction of full information",xlim=(0,1),ylim=(-.02,1.03))
    ax.spines[["right","top"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(FIG,dpi=320)
    plt.close(fig)
    return rows

def cluster_interval(numerator, denominator, clusters):
    frame = pd.DataFrame(dict(num=numerator, den=denominator, cluster=clusters))
    grouped = frame.groupby("cluster")[["num","den"]].sum().to_numpy()
    rng = np.random.default_rng(2026)
    idx = rng.integers(len(grouped),size=(2000,len(grouped)))
    draws = grouped[idx].sum(axis=1)
    return list(np.percentile(100*draws[:,0]/draws[:,1],[2.5,97.5]))

def evaluate(n_train):
    panel,hold,meta=prepare(n_train)
    clean=~hold["mask"]
    y=hold["sales"]
    den=np.where(clean,y,0.).sum(1)
    result=dict(series=len(y),clean_days=int(clean.sum()),all_days=int(y.size),
                contributing=int(clean.any(1).sum()), stores=meta.store_id.nunique(), products=meta.product_id.nunique(),
                cities=meta.city_id.nunique(),cats=meta.first_category_id.nunique(),
                train_rate=panel["mask"].mean(),mean_severity=panel["severity"][panel["mask"]].mean(),
                hour_share=panel["hours"][panel["mask"]].mean(), positive_volume_series=int((den>0).sum()),
                median_clean=float(np.median((~panel["mask"]).sum(1))), forecasts={}, contrasts={}, quantiles=[])
    if n_train == 90:
        result["scenarios"]=scenarios(panel)
    errors={}
    pol=Policy()
    for name in ("naive","interp","tobit","hours","tobit_s"):
        method = "tobit_s" if name == "hours" else name
        inputs = {**panel, "severity":panel["hours"]} if name=="hours" else panel
        R,fc=fit_predict(inputs,hold,method)
        err=np.where(clean,fc-y,0.)
        errors[name]=abs(err).sum(1)
        result["forecasts"][name]=dict(wape=100*abs(err).sum()/den.sum(),bias=100*err.sum()/den.sum())
        print(n_train,name,result["forecasts"][name],flush=True)
        if n_train==90 and name in ("naive","interp","tobit_s"):
            for q in (.5,.9,.98):
                fq=_fit_quantile(panel,hold,R,q)
                d=y-fq
                pinball=100*np.maximum(q*d,(q-1)*d)[clean].sum()/den.sum()
                r,S=policy_from_forecast(fq.mean(1),pol)
                inv=simulate_inventory(y,r,S,pol)
                cost=inventory_cost(inv["sales"],inv["lost"],inv["inv"],pol.holding_cost,pol.stockout_cost)
                result["quantiles"].append(dict(method=name,q=q,pinball=pinball,lost=cost["lost_units"],
                                                cost=cost["total"],fill=cost["fill_rate"]))
    for name in ("naive","interp","tobit","hours"):
        delta=errors["tobit_s"]-errors[name]
        result["contrasts"][name]=dict(delta=100*delta.sum()/den.sum(),
                      store=cluster_interval(delta,den,meta.store_id.to_numpy()),
                      product=cluster_interval(delta,den,meta.product_id.to_numpy()))
    if n_train==90:
        details=[]
        for col in ("city_id","first_category_id"):
            for key in sorted(meta[col].unique()):
                subset=meta[col].to_numpy()==key
                details.append(dict(axis=col,id=int(key),series=int(subset.sum()),days=int(clean[subset].sum()),
                       delta=100*(errors["tobit_s"]-errors["naive"])[subset].sum()/den[subset].sum() if den[subset].sum() else None))
        result["groups"]=details
    return result

if __name__ == "__main__":
    for horizon in (90,83):
        result=evaluate(horizon)
        (OUT/f"window_{horizon}.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")

