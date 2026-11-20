import gc,hashlib,time
import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier


def run(train,query,schema,weight):
    started=time.monotonic()
    given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    assert query[names].isna().all().all()
    mats,details={},{}
    torch.set_num_threads(2)
    for n in names:
        item=schema['items'][n];gate=item.get('gate')
        eligible=np.ones(len(train),dtype=bool);opened=np.ones(len(query),dtype=bool)
        known=False
        if gate and gate['parent'] in given:
            eligible=train[gate['parent']].isin(gate['observed_if']).to_numpy()
            known=np.array_equal(train[n].to_numpy()==schema['gated_value'],~eligible)
            if known:opened=query[gate['parent']].isin(gate['observed_if']).to_numpy()
            else:eligible[:]=True
        opts=item['values']+([schema['gated_value']] if gate else [])
        k=len(item['values']) if known else len(opts)
        rows=np.flatnonzero(eligible)
        lookup={v:i for i,v in enumerate(opts)}
        y=np.array([lookup[v] for v in train.loc[eligible,n]],dtype=int)
        prior=(np.bincount(y,minlength=k)+.5)/(len(y)+.5*k)
        x=train.loc[eligible,given].reset_index(drop=True).copy()
        q=query.loc[opened,given].reset_index(drop=True).copy()
        for field in given:
            rule=schema['items'][field]
            levels=rule['values']+([schema['gated_value']] if rule.get('gate') else [])
            x[field]=pd.Categorical(x[field],categories=levels)
            q[field]=pd.Categorical(q[field],categories=levels)
        tick=time.monotonic()
        if len(np.unique(y))<2 or not len(q):
            p=np.tile(prior,(len(q),1));classes=np.unique(y).tolist()
        else:
            model=TabICLClassifier(n_estimators=8,device='cuda',n_jobs=2,batch_size=4,
                  model_path=str(weight),allow_auto_download=False,offload_mode='gpu',
                  use_amp=False,softmax_temperature=.9,random_state=14721,verbose=False)
            model.fit(x,y)
            raw=model.predict_proba(q)
            classes=model.classes_.astype(int).tolist()
            p=np.zeros((len(q),k),dtype=float);p[:,classes]=raw
            p=.998*p+.002*prior
            del model,raw
            gc.collect();torch.cuda.empty_cache()
        if known:
            full=np.zeros((len(query),k+1));full[opened,:k]=p;full[~opened,-1]=1
        else:full=p
        full/=full.sum(1,keepdims=True)
        mats[n]=full
        details[n]=dict(context_rows=len(rows),context_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),known_gate=known,
                        inferred_rows=len(q),classes=classes,seconds=time.monotonic()-tick)

    return mats,dict(seconds=time.monotonic()-started,heads=details,n_estimators=8,device='cuda',use_amp=False)
