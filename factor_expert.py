"""Shared nominal factor encoder plus sparse conditional hurdle GLMs.
The factor is inferred from visible GIVEN fields, not synthetic latent variables.
Every fit is in-memory; query targets, roles and row IDs are not read.
"""
import time
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
from scipy import sparse
import softmax_expert as util

def encoder(train,query,schema,given,iterations=25):
    # Exclude declared GIVEN skip branches to avoid counting deterministic
    # routing as independent evidence for the latent respondent factor.
    fields=[f for f in given if not schema['items'][f].get('gate')]
    if len(fields)<3:return np.zeros((len(train),1)),np.zeros((len(query),1)),{'status':'too_few_fields'}
    x,sizes=util.encode(train,schema,fields);q,_=util.encode(query,schema,fields);n=len(x)
    ps=[(np.bincount(x[:,j],minlength=k)+.5)/(n+.5*k) for j,k in enumerate(sizes)]
    blocks=[(np.eye(k)[x[:,j]]-p)/np.sqrt(np.maximum(p,.005)) for j,(k,p) in enumerate(zip(sizes,ps))]
    M=np.column_stack(blocks);cov=M.T@M/n;offset=np.r_[0,np.cumsum(sizes)]
    for j in range(len(fields)):cov[offset[j]:offset[j+1],offset[j]:offset[j+1]]=0
    ev,vec=np.linalg.eigh(cov);score=M@vec[:,-1];score/=max(score.std(),1e-8)
    z,w=np.polynomial.hermite.hermgauss(13);z*=np.sqrt(2);w/=np.sqrt(np.pi);logw=np.log(w)
    params=[]
    for j,(k,p) in enumerate(zip(sizes,ps)):
        slopes=np.bincount(x[:,j],weights=score,minlength=k)/np.maximum(1,np.bincount(x[:,j],minlength=k))
        params.append(np.vstack([np.log(p),.35*slopes]))
    last=-np.inf
    for iteration in range(iterations):
        ll=np.tile(logw,(n,1));tables=[]
        for j,par in enumerate(params):
            lp=par[0]+z[:,None]*par[1];lp-=logsumexp(lp,axis=1)[:,None];tables.append(lp);ll+=lp[:,x[:,j]].T
        norm=logsumexp(ll,axis=1);resp=np.exp(ll-norm[:,None]);value=float(norm.mean())
        if iteration>8 and value-last<1e-6:break
        last=value
        for j,(k,p) in enumerate(zip(sizes,ps)):
            c=np.column_stack([np.bincount(x[:,j],weights=resp[:,t],minlength=k) for t in range(len(z))]).T+2*w[:,None]*p
            nz=c.sum(axis=1)
            def objective(v):
                b,a=v.reshape(2,k);zz=b+z[:,None]*a;lse=logsumexp(zz,axis=1);err=np.exp(zz-lse[:,None])*nz[:,None]-c
                loss=(nz*lse-(c*zz).sum(axis=1)).sum()+.5*np.sum(a*a)+.001*np.sum((b-np.log(p))**2)
                grad=np.vstack([err.sum(axis=0)+.002*(b-np.log(p)),(err*z[:,None]).sum(axis=0)+a])
                return float(loss),grad.ravel()
            fit=minimize(objective,params[j].ravel(),jac=True,method='L-BFGS-B',options={'maxiter':30,'ftol':1e-7,'maxcor':6})
            params[j]=fit.x.reshape(2,k)
    def transform(xx):
        ll=np.tile(logw,(len(xx),1))
        for j,par in enumerate(params):
            lp=par[0]+z[:,None]*par[1];lp-=logsumexp(lp,axis=1)[:,None];ll+=lp[:,xx[:,j]].T
        pz=np.exp(ll-logsumexp(ll,axis=1)[:,None]);return pz@np.column_stack([z,z*z-1,(z**3-3*z)/np.sqrt(6)])
    a,b=transform(x),transform(q);mu=a.mean(axis=0);sd=np.maximum(a.std(axis=0),.25)
    return np.clip((a-mu)/sd,-4,4),np.clip((b-mu)/sd,-4,4),{'status':'completed','iterations':iteration+1,'eigenvalue':float(ev[-1]),'train_likelihood':value}

def fit_predict(train,query,schema):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given)
    ft,fq,enc=encoder(train,query,schema,given);out={}
    def node(xx,y,qq,ff,qqf,k,pairs=False):
        p0=util.prior(y,k)
        if len(y)<50 or len(np.unique(y))<2:return np.broadcast_to(p0,(len(qq),k)).copy()
        fs=util.ranked_features(xx,y,sizes,k,max_features=4)
        combos=util.select_pairs(xx,y,sizes,k,max_pairs=1) if pairs else []
        fs=sorted(set(fs+[f for pair in combos for f in pair]))
        X=sparse.csr_matrix(ff*.55);Q=sparse.csr_matrix(qqf*.55)
        if fs:X=sparse.hstack([X,util.design(xx[:,fs],[sizes[f] for f in fs])],format='csr');Q=sparse.hstack([Q,util.design(qq[:,fs],[sizes[f] for f in fs])],format='csr')
        if combos:X=sparse.hstack([X,util.pair_design(xx,sizes,combos)],format='csr');Q=sparse.hstack([Q,util.pair_design(qq,sizes,combos)],format='csr')
        return util.glm(X,y,Q,k,reg=3.)
    for name in names:
        opts=util.options(schema,name);k=len(opts);lookup={v:i for i,v in enumerate(opts)};y=np.asarray([lookup[v] for v in train[name]],int)
        if schema['items'][name].get('gate'):
            asked=y<k-1;op=node(x,asked.astype(int),q,ft,fq,2);ordinary=node(x[asked],y[asked],q,ft[asked],fq,k-1,True);p=np.column_stack([op[:,1,None]*ordinary,op[:,0]])
        else:p=node(x,y,q,ft,fq,k,True)
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-started,'encoder':enc}
