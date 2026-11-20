"""Sparse multinomial helpers restored from the saved research source."""
from __future__ import annotations
import itertools,time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax

def options(schema,name):
    r=schema['items'][name]
    return list(r['values'])+([schema['gated_value']] if r.get('gate') else [])
def encode(frame,schema,given):
    xs=[];sizes=[]
    for f in given:
        o=options(schema,f);d={v:i for i,v in enumerate(o)}
        xs.append(np.array([d.get(v,len(o)) for v in frame[f]],np.int32));sizes.append(len(o)+1)
    return np.column_stack(xs),sizes

def design(x,sizes):
    off=np.r_[0,np.cumsum(sizes)]
    cols=(x+off[:-1]).ravel();rows=np.repeat(np.arange(len(x)),len(sizes))
    return sparse.csr_matrix((np.ones(len(cols)),(rows,cols)),shape=(len(x),int(off[-1])))

def prior(y,k):
    c=np.bincount(y,minlength=k).astype(float)+.5
    return c/c.sum()

def glm(x,y,q,k,reg=20.,maxiter=90):
    p0=prior(y,k);n,d=x.shape;off=np.log(p0);init=np.zeros((d+1,k),float);rr=np.arange(n)
    def objective(z):
        w=z.reshape(d+1,k);logit=x@w[:-1]+w[-1]+off;norm=logsumexp(logit,axis=1)
        loss=(norm-logit[rr,y]).sum()+.5*reg*np.sum(w[:-1]**2)+.5*np.sum(w[-1]**2)
        p=np.exp(logit-norm[:,None]);p[rr,y]-=1
        g=np.empty_like(w);g[:-1]=x.T@p+reg*w[:-1];g[-1]=p.sum(axis=0)+w[-1]
        return float(loss),g.ravel()
    fit=minimize(objective,init.ravel(),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':2e-8,'gtol':1e-5,'maxcor':8})
    w=fit.x.reshape(d+1,k)
    return .998*softmax(q@w[:-1]+w[-1]+off,axis=1)+.002*p0

def select_pairs(x,y,sizes,k,max_pairs=2):
    if len(x)<160:return []
    perm=np.random.default_rng(3027).permutation(len(x));cut=int(.72*len(x));tr,va=perm[:cut],perm[cut:]
    p0=prior(y[tr],k);yva=y[va];ref=-np.log(.001+(1-k*.001)*p0[yva]).mean();single=[]
    for f in range(x.shape[1]):
        c=np.bincount(x[tr,f]*k+y[tr],minlength=sizes[f]*k).reshape(sizes[f],k)
        p=(c[x[va,f]]+20*p0)/(c[x[va,f]].sum(axis=1,keepdims=True)+20)
        single.append(-np.log(.001+(1-k*.001)*p[np.arange(len(va)),yva]).mean())
    ref=min(ref,min(single));found=[]
    for f,g in itertools.combinations(range(x.shape[1]),2):
        lev=sizes[f]*sizes[g]
        if lev>2500:continue
        code=x[:,f]*sizes[g]+x[:,g];c=np.bincount(code[tr]*k+y[tr],minlength=lev*k).reshape(lev,k)
        p=(c[code[va]]+40*p0)/(c[code[va]].sum(axis=1,keepdims=True)+40)
        l=-np.log(.001+(1-k*.001)*p[np.arange(len(va)),yva]).mean()
        if l<ref-.006:found.append((l,f,g))
    return [(f,g) for _,f,g in sorted(found)[:max_pairs]]

def pair_design(x,sizes,pairs):
    codes=np.column_stack([x[:,f]*sizes[g]+x[:,g] for f,g in pairs])
    return design(codes,[sizes[f]*sizes[g] for f,g in pairs])

def calibrate_observed_gate(p,train,query,schema,name):
    gate=schema['items'][name].get('gate')
    if not gate or schema['items'][gate['parent']]['class']!='GIVEN':return p
    is_open=train[gate['parent']].isin(gate['observed_if']).to_numpy()
    query_open=query[gate['parent']].isin(gate['observed_if']).to_numpy()
    sentinel=(train[name]==schema['gated_value']).to_numpy()
    for state in (False,True):
        rows=is_open==state;query_rows=query_open==state;n=int(rows.sum());hits=int(sentinel[rows].sum())
        if n>=60 and hits in (0,n):
            prob=(hits+.1)/(n+.2);tmp=p[query_rows,:-1];z=tmp.sum(axis=1,keepdims=True)
            p[query_rows,:-1]=np.divide(tmp,z,out=np.full_like(tmp,1/tmp.shape[1]),where=z>0)*(1-prob)
            p[query_rows,-1]=prob
    return p

def ranked_features(x,y,sizes,k,max_features=4):
    n=len(y);cy=np.bincount(y,minlength=k).astype(float);vals=[]
    for f,levels in enumerate(sizes):
        c=np.bincount(x[:,f]*k+y,minlength=levels*k).reshape(levels,k).astype(float)
        cx=c.sum(axis=1);expected=cx[:,None]*cy[None,:]/max(n,1);nz=c>0
        gain=np.sum(c[nz]*np.log(c[nz]/np.maximum(expected[nz],1e-12)))
        df=max(0,(np.count_nonzero(cx)-1)*(np.count_nonzero(cy)-1));vals.append((gain-.5*df,f))
    return [f for value,f in sorted(vals,reverse=True)[:max_features] if value>0]

def sparse_glm(train,query,schema,reg=3.,interactions=False,hurdle=False,target_names=None):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];names=target_names or [n for n,r in schema['items'].items() if r['class']=='PREDICT']
    x,sizes=encode(train,schema,given);q,_=encode(query,schema,given);output={}
    def model(xx,yy,qq,k,pair=False):
        feats=ranked_features(xx,yy,sizes,k,max_features=4);pairs=select_pairs(xx,yy,sizes,k,max_pairs=2) if pair and len(xx)>=160 else []
        feats=sorted(set(feats+[f for ab in pairs for f in ab]))
        if len(xx)<30 or len(np.unique(yy))<2 or not feats:return np.broadcast_to(prior(yy,k),(len(qq),k)).copy()
        X=design(xx[:,feats],[sizes[f] for f in feats]);Q=design(qq[:,feats],[sizes[f] for f in feats])
        if pairs:
            X=sparse.hstack([X,pair_design(xx,sizes,pairs)],format='csr');Q=sparse.hstack([Q,pair_design(qq,sizes,pairs)],format='csr')
        return glm(X,yy,Q,k,reg=reg)
    for name in names:
        o=options(schema,name);k=len(o);look={v:i for i,v in enumerate(o)};y=np.array([look[v] for v in train[name]],int)
        if hurdle and schema['items'][name].get('gate'):
            asked=y!=k-1;openprob=model(x,asked.astype(int),q,2)[:,1]
            p=model(x[asked],y[asked],q,k-1,pair=interactions) if asked.sum() else np.full((len(q),k-1),1/(k-1))
            p=np.column_stack([openprob[:,None]*p,1-openprob])
        else:p=model(x,y,q,k,pair=interactions)
        output[name]=calibrate_observed_gate(p,train,query,schema,name)
    return output,{'seconds':time.monotonic()-started,'kind':'sparse_hurdle' if hurdle else 'sparse_direct'}
