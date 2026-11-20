"""UNICEF specialist with class-specific regularized likelihood ratios.
Fits only visible labels and GIVEN features. No generator, ids or role inputs.
"""
import itertools,time
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
import softmax_expert as util
import survey_expert

ALPHAS=(2.,5.,10.,20.,40.,80.,160.,320.,640.)
BETA_REG=0.002
OFF_REG=0.005
CLASS_REG=0.02

def node(x,y,q,sizes,k,penalty_scale=1.):
    n=len(y);p0=util.prior(y,k)
    if n<60 or len(np.unique(y))<2:return np.tile(p0,(len(q),1))
    active=[f for f in range(len(sizes)) if np.unique(x[:,f]).size>1]
    rr=np.arange(n);coef=1-k*.001;cy=np.bincount(y,minlength=k)+.5
    prior=np.tile(cy,(n,1));prior[rr,y]-=1;prior/=n-1+.5*k
    lp=np.log(prior);lq=np.log(p0)
    def loss(p):return float(-np.log(.001+coef*p[rr,y]).mean())
    def table(fs,back,qback):
        code=np.zeros(n,int);qc=np.zeros(len(q),int);lev=1
        for f in fs:code=code*sizes[f]+x[:,f];qc=qc*sizes[f]+q[:,f];lev*=sizes[f]
        c=np.bincount(code*k+y,minlength=lev*k).reshape(lev,k).astype(float)
        tc=c[code].copy();tc[rr,y]-=1;den=tc.sum(axis=1,keepdims=True)
        best=None
        for a in ALPHAS:
            p=(tc+a*back)/(den+a);val=loss(p)
            if best is None or val<best[0]:best=(val,a,p)
        val,a,p=best;qp=(c[qc]+a*qback)/(c.sum(axis=1)[qc,None]+a)
        return val,p,qp
    singles={f:table((f,),prior,p0) for f in active}
    pairs={}
    for f,g in itertools.combinations(active,2):
        if sizes[f]*sizes[g]>4500:continue
        pairs[(f,g)]=table((f,g),.5*(singles[f][1]+singles[g][1]),.5*(singles[f][2]+singles[g][2]))
    features=[];queries=[];penalties=[]
    for f in active:
        features.append(np.clip(np.log(singles[f][1])-lp,-5,5));queries.append(np.clip(np.log(singles[f][2])-lq,-5,5));penalties.append(.003)
    base_loss=loss(prior)
    kept=sorted(pairs,key=lambda fs:pairs[fs][0])[:8]
    for fs in kept:
        val,p,qp=pairs[fs]
        if val>=base_loss-.001:continue
        back=.5*(np.log(singles[fs[0]][1])+np.log(singles[fs[1]][1]))
        qb=.5*(np.log(singles[fs[0]][2])+np.log(singles[fs[1]][2]))
        features.append(np.clip(np.log(p)-back,-5,5));queries.append(np.clip(np.log(qp)-qb,-5,5));penalties.append(.006)
    triples=[]
    for fs in itertools.combinations(active,3):
        if np.prod([sizes[f] for f in fs])>4500:continue
        pp=list(itertools.combinations(fs,2))
        if not all(p in pairs for p in pp):continue
        back=sum(pairs[p][1] for p in pp)/3;qb=sum(pairs[p][2] for p in pp)/3
        val,p,qp=table(fs,back,qb)
        if val<base_loss-.001:triples.append((val,fs,p,qp,back,qb))
    for val,fs,p,qp,back,qb in sorted(triples,key=lambda a:a[0])[:10]:
        features.append(np.clip(np.log(p)-np.log(back),-5,5));queries.append(np.clip(np.log(qp)-np.log(qb),-5,5));penalties.append(.01)
    if not features:return np.tile(p0,(len(q),1))
    d=len(features);F=np.stack(features,axis=2);Q=np.stack(queries,axis=2);penalty=np.array(penalties)*penalty_scale
    def unpack(z):
        beta=z[:d];dev=z[d:d+k*d].reshape(k,d);offset=z[d+k*d:]
        return beta,dev,offset
    def objective(z):
        beta,dev,offset=unpack(z);centered=dev-dev.mean(axis=0)
        zz=lp+np.sum(F*(beta[None,None,:]+centered[None,:,:]),axis=2)+offset
        norm=logsumexp(zz,axis=1);p=np.exp(zz-norm[:,None]);correct=p[rr,y];safe=.001+coef*correct
        value=-np.log(safe).mean()+penalty@beta+BETA_REG*np.sum(beta**2)+OFF_REG*np.sum(offset**2)+CLASS_REG*np.sum(dev**2)
        p[rr,y]-=1;p*=((coef*correct/safe)/n)[:,None]
        gb=np.sum(F*p[:,:,None],axis=(0,1))+penalty+2*BETA_REG*beta
        gd=np.sum(F*p[:,:,None],axis=0);gd-=gd.mean(axis=0);gd+=2*CLASS_REG*dev
        go=p.sum(axis=0)+2*OFF_REG*offset
        return float(value),np.r_[gb,gd.ravel(),go]
    fit=minimize(objective,np.zeros(d+k*d+k),jac=True,method='L-BFGS-B',bounds=[(0.,3.)]*d+[(-1.5,1.5)]*(k*d)+[(-3.,3.)]*k,options={'maxiter':160,'ftol':1e-9,'maxcor':8})
    beta,dev,offset=unpack(fit.x);dev-=dev.mean(axis=0)
    return softmax(lq+np.sum(Q*(beta[None,None,:]+dev[None,:,:]),axis=2)+offset,axis=1)

def fit_predict(train,query,schema,penalty_scale=1.):
    start=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given);out={}
    for name,r in schema['items'].items():
        if r['class']!='PREDICT':continue
        oo=util.options(schema,name);k=len(oo);lookup={v:i for i,v in enumerate(oo)};y=np.array([lookup[v] for v in train[name]],int)
        if r.get('gate'):
            asked=y<k-1;op=node(x,asked.astype(int),q,sizes,2,penalty_scale);ordinary=node(x[asked],y[asked],q,sizes,k-1,penalty_scale);p=np.column_stack([op[:,1,None]*ordinary,op[:,0]])
        else:p=node(x,y,q,sizes,k,penalty_scale)
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-start,'penalty_scale':penalty_scale,'max_order':3}
