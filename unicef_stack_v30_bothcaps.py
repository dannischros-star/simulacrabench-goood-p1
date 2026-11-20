"""UNICEF ten-expert cross-fitted probability stack.

Structure, weights, and calibration use supplied visible TRAIN only. The model
does not read generators, respondent identifiers, roles, files, or the network.
"""
import time
import numpy as np
import baseline_v11 as base
import context_expert
import uif_ratio3
import uif_ratio3_fine
import uif_ratio3_classwise

EXPERTS=base.EXPERTS+('context','ratio3_fine','ratio3','ratio3_classwise')
FIT_CAP=6000
SCORE_CAP=700
LAST_DIAGNOSTICS={}

def fit_experts(train,query,schema,needed=None):
    need=set(range(10)) if needed is None else set(needed)
    out=base.fit_experts(train,query,schema,need.intersection(range(6)))
    if 6 in need:out[6],_=context_expert.fit_predict(train,query,schema)
    if 7 in need:out[7],_=uif_ratio3_fine.fit_predict(train,query,schema,.15)
    if 8 in need:out[8],_=uif_ratio3.fit_predict(train,query,schema,.15)
    if 9 in need:out[9],_=uif_ratio3_classwise.fit_predict(train,query,schema,.15)
    return out

def predict(frame,schema):
    global LAST_DIAGNOSTICS
    started=time.monotonic()
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    train=frame.loc[frame[names].notna().all(axis=1)].reset_index(drop=True)
    query=frame.loc[frame[names].isna().any(axis=1)].reset_index(drop=True)
    if not len(query):return []
    if len(train)<240 or not query[names].isna().all().all():
        return base.predict(frame,schema)
    folds=np.array_split(np.random.default_rng(14721).permutation(len(train)),3)
    pred={j:{n:[] for n in names} for j in range(10)}
    labels={n:[] for n in names}
    for fold in folds:
        fit=np.setdiff1d(np.arange(len(train)),fold)
        if len(fit)>FIT_CAP:
            fit=np.random.default_rng(14722+len(fold)).permutation(fit)[:FIT_CAP]
        va=fold[:SCORE_CAP]
        tt=train.iloc[fit].reset_index(drop=True)
        vv=train.iloc[va].reset_index(drop=True)
        qq=vv.copy();qq[names]=np.nan
        mats=fit_experts(tt,qq,schema)
        for n in names:
            look={v:i for i,v in enumerate(base.util.options(schema,n))}
            labels[n].append(np.array([look[v] for v in vv[n]],int))
            for j in range(10):pred[j][n].append(mats[j][n])
    pred={j:{n:np.concatenate(v) for n,v in mat.items()} for j,mat in pred.items()}
    labels={n:np.concatenate(y) for n,y in labels.items()}
    nv=len(labels[names[0]]);rr=np.arange(nv)
    correct={n:np.column_stack([.001+(1-.001*len(base.util.options(schema,n)))*pred[j][n][rr,labels[n]] for j in range(10)]) for n in names}
    anchor=np.array([.11,.08,.10,.04,.05,.05,.12,.15,.15,.15])
    glob=base.optimize(np.concatenate(list(correct.values())),anchor,.001)
    weights={n:base.optimize(correct[n],glob,12./nv) for n in names}
    mix={n:sum(weights[n][j]*pred[j][n] for j in range(10)) for n in names}
    powers=(1.,.9,.95,1.05,1.1,1.15)
    curves=[]
    for power in powers:
        curves.append(np.mean([-np.log(.001+(1-.001*mix[n].shape[1])*base.transform(mix[n],power,0.)[rr,labels[n]]) for n in names],axis=0))
    best=min(range(len(curves)),key=lambda j:curves[j].mean())
    delta=curves[0]-curves[best]
    chosen=powers[best] if delta.mean()>.0002+delta.std(ddof=1)/np.sqrt(nv) else 1.
    power=1+.5*(chosen-1)
    for n,w in weights.items():
        w[w<.008]=0
        weights[n]=w/w.sum() if w.sum()>0 else glob.copy()
    needed=[j for j in range(10) if any(w[j]>0 for w in weights.values())]
    full=fit_experts(train,query,schema,needed)
    out={n:base.transform(sum(weights[n][j]*full[j][n] for j in needed if weights[n][j]>0),power,0.) for n in names}
    LAST_DIAGNOSTICS={'status':'completed','seconds':time.monotonic()-started,'experts':EXPERTS,'inner_rows':nv,'inner_train_cap':FIT_CAP,'global_weights':glob.tolist(),'item_weights':{n:w.tolist() for n,w in weights.items()},'power':power}
    return [out[n][i] for i in range(len(query)) for n in names]
