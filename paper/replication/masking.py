"""Evaluate historical masks on days whose complete hourly sales can be observed."""
from analysis import FILES, OUT, cluster_interval, prepare
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    panel,hold,meta=prepare(90)
    sids=meta.index.to_numpy()
    arrays=[]
    for path,col in zip(FILES,("hours_stock_status","hours_sale")):
        chunks=[]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=["store_id","product_id","dt",col]):
            data=batch.to_pandas()
            data["sid"]=data.store_id*1000000+data.product_id
            chunks.append(data[data.sid.isin(sids)])
        data=pd.concat(chunks).sort_values(["sid","dt"])
        arrays.append(np.stack(data[col]).reshape(len(sids),-1,24))
    status,sales=arrays
    rng=np.random.default_rng(2026)
    # Draw one historical stockout mask per eligible clean date per repetition.
    # Keep the 20 draws together within series for all summaries.
    totals=np.zeros((len(sids),6))
    for i in range(len(sids)):
        candidates=np.flatnonzero(panel["mask"][i])
        dates=np.flatnonzero(~hold["mask"][i])
        clean_train=~panel["mask"][i]
        if not len(candidates) or not len(dates) or not clean_train.any():
            continue
        idx=rng.choice(candidates,(20,len(dates)),replace=True)
        blocked=status[i,idx,6:22]
        full=sales[i,dates].sum(axis=-1)
        hidden=(blocked*sales[i,dates,6:22]).sum(axis=-1)
        level=panel["sales"][i,clean_train].mean()
        totals[i,0]=full.sum()
        totals[i,1]=len(dates)
        totals[i,2]=hidden.sum()/20
        for j,field in enumerate(("hours","severity")):
            predicted=level*panel[field][i,idx]
            totals[i,3+j]=abs(predicted-hidden).sum()/20
        totals[i,5]=(level*panel["severity"][i,idx]-hidden).sum()/20
    denom=totals[:,0]
    delta=totals[:,4]-totals[:,3]
    out=dict(series=int((totals[:,1]>0).sum()),days=int(totals[:,1].sum()), volume=float(denom.sum()),
             no_fill=100*totals[:,2].sum()/denom.sum(),hours=100*totals[:,3].sum()/denom.sum(),
             volume_weight=100*totals[:,4].sum()/denom.sum(),bias=100*totals[:,5].sum()/denom.sum(),
             delta=100*delta.sum()/denom.sum(),store=cluster_interval(delta,denom,meta.store_id.to_numpy()),
             product=cluster_interval(delta,denom,meta.product_id.to_numpy()))
    (OUT/"masking.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()

