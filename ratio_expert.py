"""Continuous, regularized likelihood-ratio expert combination.
Per-row training features use counts with that row removed. Interaction terms
are screened by LOO loss, not query answers. The whole algorithm needs an outer
holdout. This model does not read generators, seeds, roles or identifiers.
"""
import itertools,time
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
import softmax_expert as util

ALPHAS=(5.,20.,80.)

def node(x,y,q,sizes,k):
    n=len(y);p0=util.prior(y,k)
    if n<60 or len(np.unique(y))<2:return np.broadcast_to(p0,(len(q),k)).copy(),{'status':'marginal'}
    rr=np.arange(n);coef=1-k*.001;cy=np.bincount(y,minlength=k).astype(float)+.5
    prior=np.broadcast_to(cy,(n,k)).copy();prior[rr,y]-=1;prior/=(n-1+.5*k)
    lprior=np.log(prior);lp0=np.log(p0)
    def loss(py):return float(-np.log(.001+coef*py).mean())
    baseline=loss(prior[rr,y]);preds=[];qps=[];features=[];queries=[];singles=[]
    for f,lev in enumerate(sizes):
        c=np.bincount(x[:,f]*k+y,minlength=lev*k).reshape(lev,k).astype(float)
        tc=c[x[:,f]].copy();tc[rr,y]-=1;den=tc.sum(axis=1,keepdims=True)
        best=None
        for a in ALPHAS:
            p=(tc+a*prior)/(den+a);val=loss(p[rr,y])
            if best is None or val<best[0]:best=(val,a,p)
        score,a,p=best;qp=(c[q[:,f]]+a*p0)/(c.sum(axis=1)[q[:,f],None]+a);preds.append(p);qps.append(qp);singles.append(score)
        features.append(np.clip(np.log(p)-lprior,-4,4));queries.append(np.clip(np.log(qp)-lp0,-4,4))
    candidates=[]
    for f,g in itertools.combinations(range(len(sizes)),2):
        if sizes[f]*sizes[g]>3000:continue
        code=x[:,f]*sizes[g]+x[:,g];c=np.bincount(code*k+y,minlength=sizes[f]*sizes[g]*k).reshape(-1,k)
        den=c.sum(axis=1)[code]-1;tc=c[code,y]-1;bt=.5*(preds[f][rr,y]+preds[g][rr,y])
        for a in (5.,20.,80.):
            val=loss((tc+a*bt)/(den+a))
            if val<baseline-.003 and val<min(singles[f],singles[g])+.003:candidates.append((val,f,g,a))
    kept=[];seen=set()
    for record in sorted(candidates):
        _,f,g,a=record
        if (f,g) in seen:continue
        kept.append(record);seen.add((f,g))
        if len(kept)>=5:break
    for _,f,g,a in kept:
        code=x[:,f]*sizes[g]+x[:,g];qc=q[:,f]*sizes[g]+q[:,g];c=np.bincount(code*k+y,minlength=sizes[f]*sizes[g]*k).reshape(-1,k).astype(float)
        tc=c[code].copy();tc[rr,y]-=1;den=tc.sum(axis=1,keepdims=True);back=.5*(preds[f]+preds[g]);qb=.5*(qps[f]+qps[g])
        pp=(tc+a*back)/(den+a);qq=(c[qc]+a*qb)/(c.sum(axis=1)[qc,None]+a)
        features.append(np.clip(np.log(pp)-lprior-.5*(features[f]+features[g]),-4,4));queries.append(np.clip(np.log(qq)-lp0-.5*(queries[f]+queries[g]),-4,4))
    d=len(features);F=np.stack(features,axis=2).reshape(n*k,d);Q=np.stack(queries,axis=2).reshape(len(q)*k,d)
    penalty=np.r_[np.full(len(sizes),.004),np.full(len(kept),.008)]
    def objective(z):
        beta=z[:d];offset=z[d:];zz=lprior+(F@beta).reshape(n,k)+offset;norm=logsumexp(zz,axis=1);p=np.exp(zz-norm[:,None]);correct=p[rr,y];safe=.001+coef*correct
        value=-np.log(safe).mean()+penalty@beta+.002*np.sum(beta*beta)+.005*np.sum(offset*offset)
        p[rr,y]-=1;p*=((coef*correct/safe)/n)[:,None]
        g=np.r_[F.T@p.ravel()+penalty+.004*beta,p.sum(axis=0)+.01*offset]
        return float(value),g
    fit=minimize(objective,np.zeros(d+k),jac=True,method='L-BFGS-B',bounds=[(0.,3.)]*d+[(-3.,3.)]*k,options={'maxiter':75,'ftol':1e-9,'maxcor':8})
    zz=lp0+(Q@fit.x[:d]).reshape(len(q),k)+fit.x[d:];out=softmax(zz,axis=1)
    return out,{'status':'ratio','single_weights':fit.x[:len(sizes)].tolist(),'pairs':[(f,g,a) for _,f,g,a in kept],'pair_weights':fit.x[len(sizes):d].tolist(),'success':bool(fit.success)}

def fit_predict(train,query,schema):
    start=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given);out={};details={}
    for name in names:
        oo=util.options(schema,name);k=len(oo);lookup={v:i for i,v in enumerate(oo)};y=np.asarray([lookup[v] for v in train[name]],int)
        if schema['items'][name].get('gate'):
            asked=y<k-1;op,od=node(x,asked.astype(int),q,sizes,2);ordinary,dd=node(x[asked],y[asked],q,sizes,k-1);p=np.column_stack([op[:,1,None]*ordinary,op[:,0]]);details[name]={'open':od,'value':dd}
        else:p,dd=node(x,y,q,sizes,k);details[name]=dd
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-start,'items':details}
