"""TRAIN-only, cross-validated foundation extension of the protected stack."""
import time
import numpy as np
import pandas as pd
import unicef_stack_v33_core as old
import unicef_stack_v33_gate as original
import probability_stack as stack
import gpu_tabm_expert
import foundation_expert

LAST_DIAGNOSTICS={}
LAST_BASELINE=None
LAST_VALIDATION=None


def subset(z,rows):
    return {key:value[rows] for key,value in z.items()}


def mats(z,names,indices,weights,power,rows=None,prefix='oof'):
    values={n:old.base.transform(sum(weights[n][a]*z[f'{prefix}_{j}_{n}'] for a,j in enumerate(indices) if weights[n][a]>0),power,0.) for n in names}
    return values if rows is None else {n:p[rows] for n,p in values.items()}


def cohere(values,train,query,schema,names):
    raw,_=stack.postprocess(values,pd.concat([train,query],ignore_index=True),schema)
    return {n:np.asarray(raw[j::len(names)]) for j,n in enumerate(names)}


def losses(values,z,names):
    rr=np.arange(len(z['y_'+names[0]]))
    return np.mean([-np.log(.001+(1-.001*values[n].shape[1])*values[n][rr,z['y_'+n]]) for n in names],axis=0)


def choose_original(z,names,train,query,schema):
    base=list(range(12));extended=list(range(13))
    bw,bp,_=stack.select(z,names,base);ew,ep,_=stack.select(z,names,extended)
    b=cohere(mats(z,names,base,bw,bp),train,query,schema,names)
    e=cohere(mats(z,names,extended,ew,ep),train,query,schema,names)
    delta=losses(b,z,names)-losses(e,z,names)
    se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    accepted=bool(delta.mean()>.0002+se)
    return (extended,ew,ep) if accepted else (base,bw,bp)


def predict(frame,schema,weight_path=None):
    global LAST_DIAGNOSTICS,LAST_BASELINE,LAST_VALIDATION
    started=time.monotonic()
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    train=frame.loc[frame[names].notna().all(axis=1)].reset_index(drop=True)
    query=frame.loc[frame[names].isna().any(axis=1)].reset_index(drop=True)
    if not len(query):return []
    if len(train)<240 or not query[names].isna().all().all():return original.predict(frame,schema)
    indices=list(range(14));z={};labels={n:[] for n in names}
    preds={j:{n:[] for n in names} for j in indices};valid_frames=[];timings=[]
    for fold_number,fold in enumerate(np.array_split(np.random.default_rng(14721).permutation(len(train)),3)):
        allfit=np.setdiff1d(np.arange(len(train)),fold)
        fit=allfit
        if len(fit)>6000:fit=np.random.default_rng(14722+len(fold)).permutation(fit)[:6000]
        tt=train.iloc[fit].reset_index(drop=True);vv=train.iloc[fold[:700]].reset_index(drop=True)
        qq=vv.copy();qq[names]=np.nan;valid_frames.append(qq)
        tick=time.monotonic()
        out=old.fit_experts(tt,qq,schema)
        out[12],_=gpu_tabm_expert.fit_predict(tt,qq,schema)
        out[13],icldiag=foundation_expert.run(train.iloc[allfit].reset_index(drop=True),qq,schema,weight_path)
        timings.append(dict(fold=fold_number,seconds=time.monotonic()-tick,icl=icldiag['seconds']))

        for n in names:
            lookup={v:i for i,v in enumerate(old.base.util.options(schema,n))}
            labels[n].append(np.array([lookup[v] for v in vv[n]],int))
            for j in indices:preds[j][n].append(out[j][n])
    for n in names:
        z['y_'+n]=np.concatenate(labels[n])
        for j in indices:z[f'oof_{j}_{n}']=np.concatenate(preds[j][n])
    oq=pd.concat(valid_frames,ignore_index=True)
    chosen,bw,bp=choose_original(z,names,train,oq,schema)
    cut=sum(len(v) for v in valid_frames[:2]);learn=np.arange(cut);check=np.arange(cut,len(oq))
    lz=subset(z,learn)
    proxy,_,_=choose_original(lz,names,train,oq.iloc[learn],schema)
    rw,rp,_=stack.select(lz,names,proxy)
    ew,ep,_=stack.select(lz,names,proxy+[13])
    reference=cohere(mats(z,names,proxy,rw,rp,check),train,oq.iloc[check],schema,names)
    extended=cohere(mats(z,names,proxy+[13],ew,ep,check),train,oq.iloc[check],schema,names)
    cz=subset(z,check)
    delta=losses(reference,cz,names)-losses(extended,cz,names)
    se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    accepted=bool(delta.mean()>.0002+se)
    LAST_VALIDATION=dict(arrays={key:value.copy() for key,value in z.items()},
        learn=learn.copy(),check=check.copy(),proxy=proxy,chosen=chosen,
        reference_weights={n:w.tolist() for n,w in rw.items()},reference_power=rp,
        extended_weights={n:w.tolist() for n,w in ew.items()},extended_power=ep,
        original_weights={n:w.tolist() for n,w in bw.items()},original_power=bp)
    used=chosen+[13] if accepted else chosen
    weights,power,_=stack.select(z,names,used) if accepted else (bw,bp,None)
    needed=[j for a,j in enumerate(used) if any(w[a]>0 for w in weights.values())]
    baseline_needed=[j for a,j in enumerate(chosen) if any(w[a]>0 for w in bw.values())]
    union=sorted(set(needed+baseline_needed))
    full=old.fit_experts(train,query,schema,[j for j in union if j<12])
    if 12 in union:full[12],_=gpu_tabm_expert.fit_predict(train,query,schema)
    if 13 in union:full[13],_=foundation_expert.run(train,query,schema,weight_path)
    for j,values in full.items():
        for n in names:z[f'full_{j}_{n}']=values[n]
    baseline,_=stack.postprocess(mats(z,names,chosen,bw,bp,prefix='full'),frame,schema)
    raw,post=stack.postprocess(mats(z,names,used,weights,power,prefix='full'),frame,schema)
    LAST_BASELINE={n:np.asarray(baseline[j::len(names)]) for j,n in enumerate(names)}
    LAST_DIAGNOSTICS=dict(seconds=time.monotonic()-started,accepted=accepted,acceptance_gain=float(delta.mean()),
        acceptance_se=se,acceptance_rows=len(check),learning_rows=len(learn),original_experts=chosen,needed=needed,
        weights={n:w.tolist() for n,w in weights.items()},power=power,timings=timings,postprocess=post,query_labels_used=False)
    return raw
