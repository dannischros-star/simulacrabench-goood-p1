"""Frontier v11: whole-row cross-fitted six-expert stack.
All fit/selection is on visible supplied TRAIN; outer confirmation is required.
No generator imports, identifiers, remote calls, disk I/O or fixed data answers.
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import model_masked,model_v1
import softmax_expert as util
import pair_expert,ratio_expert,factor_expert,chain_expert,loglinear_expert
SCORE_CAP=350
FIT_CAP=2400
CAPS={'world_bank':420.,'unicef':80.,'unhcr':620.}
EXPERTS=('simple','pair','ratio','chain','factor','loglinear','shared_tabm')
LAST_DIAGNOSTICS={}

def identify(schema):
    n=schema['items']
    if 's1_2_country_asylum_lbl' in n and 'wave' in n:return 'unhcr'
    if 'user.educationLevel' in n and 'group' in n:return 'world_bank'
    if 'Country' in n and 'Q13' in n:return 'unicef'
    return 'unknown'

def matrices(vectors,n,names,schema):
    if len(vectors)!=n*len(names):raise ValueError('Misaligned expert output')
    out={}
    for j,name in enumerate(names):
        p=np.asarray(vectors[j::len(names)],float)
        if p.shape!=(n,len(util.options(schema,name))) or not np.isfinite(p).all() or (p<0).any() or (p.sum(axis=1)<=0).any():raise ValueError('Invalid expert probabilities')
        out[name]=p/p.sum(axis=1,keepdims=True)
    return out

def marginal(frame,schema):
    names=[n for n,r in schema['items'].items() if r['class'] in ('GIVEN','PREDICT')];probs={}
    for name in names:
        oo=util.options(schema,name);counts=frame[name].value_counts();p=np.asarray([counts.get(v,0)+.5 for v in oo],float);probs[name]=p/p.sum()
    return [probs[name].copy() for _,row in frame.iterrows() for name in names if pd.isna(row[name])]

def fit_experts(train,query,schema,needed=None):
    need=set(range(7)) if needed is None else set(needed);names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];res={}
    if 0 in need:
        baseline=model_v1 if identify(schema)=='unhcr' else model_masked
        v=baseline.predict(pd.concat([train,query],ignore_index=True),schema);res[0]=matrices(v,len(query),names,schema)
    if 1 in need or 3 in need:res[1],_=pair_expert.fit_predict(train,query,schema,full_pairs=True,product_backoff=True)
    if 2 in need:res[2],_=ratio_expert.fit_predict(train,query,schema)
    if 3 in need:res[3],_=chain_expert.fit_predict(train,query,schema,direct=res[1])
    if 4 in need:res[4],_=factor_expert.fit_predict(train,query,schema)
    if 5 in need:res[5],_=loglinear_expert.fit_predict(train,query,schema)
    if 6 in need:
        import wb_tabm_shared as unhcr_shared
        res[6],_=unhcr_shared.fit_predict(train,query,schema)
    return res

def optimize(p,anchor,reg):
    def objective(w):
        s=np.maximum(p@w,1e-12);return float(-np.log(s).mean()+reg*np.sum((w-anchor)**2)),-(p/s[:,None]).mean(axis=0)+2*reg*(w-anchor)
    fit=minimize(objective,anchor,jac=True,method='SLSQP',bounds=[(0.,1.)]*len(anchor),constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)},options={'maxiter':90,'ftol':1e-9})
    if not fit.success or not np.isfinite(fit.x).all():return anchor.copy()
    w=np.clip(fit.x,0,1);return w/w.sum() if w.sum()>0 else anchor.copy()

def transform(p,power,undo):
    p=np.maximum(p-undo,1e-12)**power;return p/p.sum(axis=1,keepdims=True)

def predict(frame,schema):
    global LAST_DIAGNOSTICS
    start=time.monotonic();which=identify(schema);names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    if not names:return marginal(frame,schema)
    train=frame.loc[frame[names].notna().all(axis=1)].reset_index(drop=True);query=frame.loc[frame[names].isna().any(axis=1)].reset_index(drop=True)
    if not len(query):return []
    if which=='unknown' or len(train)<240 or not query[names].isna().all().all():return marginal(frame,schema)
    folds=np.array_split(np.random.default_rng(14721).permutation(len(train)),3)
    pred={j:{n:[] for n in names} for j in range(7)};labels={n:[] for n in names}
    for fold in folds:
        fit=np.setdiff1d(np.arange(len(train)),fold)
        if len(fit)>FIT_CAP:fit=np.random.default_rng(14722+len(fold)).permutation(fit)[:FIT_CAP]
        va=fold[:SCORE_CAP];tt=train.iloc[fit].reset_index(drop=True);truth=train.iloc[va].reset_index(drop=True);qq=truth.copy();qq[names]=np.nan
        mats=fit_experts(tt,qq,schema)
        for name in names:
            lookup={v:i for i,v in enumerate(util.options(schema,name))};labels[name].append(np.asarray([lookup[v] for v in truth[name]],int))
            for j in range(7):pred[j][name].append(mats[j][name])
        if time.monotonic()-start>.76*CAPS[which]:
            LAST_DIAGNOSTICS={'status':'budget_fallback','seconds':time.monotonic()-start}
            base=model_v1 if which=='unhcr' else model_masked;return base.predict(frame,schema)
    pred={j:{n:np.concatenate(p) for n,p in d.items()} for j,d in pred.items()};labels={n:np.concatenate(y) for n,y in labels.items()};nv=len(labels[names[0]]);rr=np.arange(nv)
    correct={n:np.column_stack([.001+(1-.001*len(util.options(schema,n)))*pred[j][n][rr,labels[n]] for j in range(7)]) for n in names}
    anchor=np.asarray([.28,.16,.16,.08,.04,.08,.20]);glob=optimize(np.concatenate(list(correct.values())),anchor,.001)
    weights={n:optimize(correct[n],glob,12./nv) for n in names}
    combos=[(1.,0.),(.9,0.),(.95,0.),(1.05,0.),(1.10,0.),(1.15,0.),(1.,.0005),(1.,.001)]
    mix={n:sum(weights[n][j]*pred[j][n] for j in range(7)) for n in names};curves=[]
    for power,undo in combos:
        curves.append(np.mean([-np.log(.001+(1-.001*mix[n].shape[1])*transform(mix[n],power,undo)[rr,labels[n]]) for n in names],axis=0))
    best=min(range(len(curves)),key=lambda j:curves[j].mean());delta=curves[0]-curves[best];power,undo=combos[best] if delta.mean()>.0002+delta.std(ddof=1)/np.sqrt(nv) else combos[0]
    power=1+.5*(power-1);undo*=.5
    for n,w in weights.items():
        w[w<.015]=0;weights[n]=w/w.sum() if w.sum()>0 else glob.copy()
    needed=[j for j in range(7) if any(w[j]>0 for w in weights.values())];full=fit_experts(train,query,schema,needed);out={n:transform(sum(weights[n][j]*full[j][n] for j in needed),power,undo) for n in names}
    LAST_DIAGNOSTICS={'status':'completed','seconds':time.monotonic()-start,'oof_rows':nv,'experts':EXPERTS,'global_weights':glob.tolist(),'mean_item_weights':np.mean(list(weights.values()),axis=0).tolist(),'power':power,'floor_undo':undo,'item_weights':{n:w.tolist() for n,w in weights.items()}}
    return [out[n][i] for i in range(len(query)) for n in names]

