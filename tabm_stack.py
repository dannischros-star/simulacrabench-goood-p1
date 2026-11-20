"""Thirteen-expert UNICEF candidate; all adaptation uses supplied TRAIN only."""
import time
import numpy as np
import pandas as pd
import unicef_stack_v33_core as old
import unicef_stack_v33_gate as original
import probability_stack as stack
from tabm_expert import fit_predict as tabm

LAST_DIAGNOSTICS={}

def as_mats(z,names,indices,weights,power,prefix):
    return {n:old.base.transform(sum(weights[n][a]*z[f'{prefix}_{j}_{n}']
            for a,j in enumerate(indices) if weights[n][a]>0),power,0.) for n in names}

def cohere(mats,frame,schema):
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    raw,used=stack.postprocess(mats,frame,schema)
    return {n:np.asarray(raw[j::len(names)]) for j,n in enumerate(names)},used

def predict(frame,schema):
    global LAST_DIAGNOSTICS
    start=time.monotonic()
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    train=frame.loc[frame[names].notna().all(axis=1)].reset_index(drop=True)
    query=frame.loc[frame[names].isna().any(axis=1)].reset_index(drop=True)
    if not len(query):return []
    if len(train)<240 or not query[names].isna().all().all():return original.predict(frame,schema)
    indices=list(range(13));z={};labels={n:[] for n in names}
    preds={j:{n:[] for n in names} for j in indices};valid_frames=[];times=[]
    for fold in np.array_split(np.random.default_rng(14721).permutation(len(train)),3):
        fit=np.setdiff1d(np.arange(len(train)),fold)
        if len(fit)>6000:fit=np.random.default_rng(14722+len(fold)).permutation(fit)[:6000]
        tt=train.iloc[fit].reset_index(drop=True);vv=train.iloc[fold[:700]].reset_index(drop=True)
        qq=vv.copy();qq[names]=np.nan;valid_frames.append(qq)
        mats=old.fit_experts(tt,qq,schema)
        mats[12],diag=tabm(tt,qq,schema);times.append(diag['seconds'])
        for n in names:
            lookup={v:i for i,v in enumerate(old.base.util.options(schema,n))}
            labels[n].append(np.array([lookup[v] for v in vv[n]],int))
            for j in indices:preds[j][n].append(mats[j][n])
    for n in names:
        z['y_'+n]=np.concatenate(labels[n])
        for j in indices:z[f'oof_{j}_{n}']=np.concatenate(preds[j][n])
    base_indices=list(range(12))
    bw,bp,_=stack.select(z,names,base_indices);ew,ep,_=stack.select(z,names,indices)
    oof_frame=pd.concat([train,pd.concat(valid_frames,ignore_index=True)],ignore_index=True)
    base_oof,_=cohere(as_mats(z,names,base_indices,bw,bp,'oof'),oof_frame,schema)
    ext_oof,_=cohere(as_mats(z,names,indices,ew,ep,'oof'),oof_frame,schema)
    rr=np.arange(len(z['y_'+names[0]]))
    def losses(mats):
        return np.mean([-np.log(.001+(1-.001*mats[n].shape[1])*mats[n][rr,z['y_'+n]]) for n in names],axis=0)
    delta=losses(base_oof)-losses(ext_oof);se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    accepted=bool(delta.mean()>.0002+se)
    chosen=indices if accepted else base_indices;weights,power=(ew,ep) if accepted else (bw,bp)
    needed=[j for a,j in enumerate(chosen) if any(w[a]>0 for w in weights.values())]
    full=old.fit_experts(train,query,schema,[j for j in needed if j<12])
    if 12 in needed:
        full[12],diag=tabm(train,query,schema);times.append(diag['seconds'])
    for j,mats in full.items():
        for n in names:z[f'full_{j}_{n}']=mats[n]
    mats=as_mats(z,names,chosen,weights,power,'full')
    raw,used=stack.postprocess(mats,frame,schema)
    LAST_DIAGNOSTICS={'seconds':time.monotonic()-start,'accepted':accepted,'gain':float(delta.mean()),'se':se,
                      'needed':needed,'tabm_fit_seconds':times,'postprocess':used,'cpu_only':True,
                      'weights':{n:w.tolist() for n,w in weights.items()},'power':power}
    return raw
