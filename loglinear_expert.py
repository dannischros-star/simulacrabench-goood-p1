"""Selected pair log-linear MAP with separately regularized interactions.
Selection/calibration use a held-out portion of visible TRAIN. Outer evaluation
must assess the whole algorithm; no withheld competition answers are used.
"""
import time
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
import softmax_expert as util
import pair_expert as eb

CONFIGS=((1.,None),(3.,None),(3.,3.),(3.,20.),(3.,100.))

def fit_pair(x,y,q,sizes,k,reg_main=3.,reg_pair=None):
    a,b=sizes;n=len(y);p0=util.prior(y,k)
    code=x[:,0]*b+x[:,1];c=np.bincount(code*k+y,minlength=a*b*k).reshape(a,b,k).astype(float);total=c.sum(axis=2)
    # Work on aggregated sufficient statistics, independent of respondent count.
    d=1+a+b+(a*b if reg_pair is not None else 0);off=np.log(p0)
    def objective(z):
        w=z.reshape(d,k);zz=off+w[0]+w[1:1+a,None,:]+w[None,1+a:1+a+b,:]
        if reg_pair is not None:zz=zz+w[1+a+b:].reshape(a,b,k)
        norm=logsumexp(zz,axis=2);err=np.exp(zz-norm[:,:,None])*total[:,:,None]-c
        value=(total*norm-(c*zz).sum(axis=2)).sum()+.5*np.sum(w[0]**2)+.5*reg_main*np.sum(w[1:1+a+b]**2)
        grad=np.zeros_like(w);grad[0]=err.sum(axis=(0,1))+w[0];grad[1:1+a]=err.sum(axis=1)+reg_main*w[1:1+a];grad[1+a:1+a+b]=err.sum(axis=0)+reg_main*w[1+a:1+a+b]
        if reg_pair is not None:
            value+=.5*reg_pair*np.sum(w[1+a+b:]**2);grad[1+a+b:]=err.reshape(a*b,k)+reg_pair*w[1+a+b:]
        return float(value),grad.ravel()
    fit=minimize(objective,np.zeros((d,k)).ravel(),jac=True,method='L-BFGS-B',options={'maxiter':65,'ftol':1e-8,'maxcor':8})
    w=fit.x.reshape(d,k);zz=off+w[0]+w[1+q[:,0]]+w[1+a+q[:,1]]
    if reg_pair is not None:zz+=w[1+a+b:][q[:,0]*b+q[:,1]]
    p=softmax(zz,axis=1)
    return .998*p+.002*p0

def node(x,y,q,sizes,k):
    n=len(y)
    if n<100 or len(np.unique(y))<2:return eb.node(x,y,q,sizes,k,full_pairs=True,product_backoff=False)
    perm=np.random.default_rng(7712).permutation(n);cut=int(n*.78);tr,va=perm[:cut],perm[cut:]
    pb,detail=eb.node(x[tr],y[tr],x[va],sizes,k,full_pairs=True,product_backoff=False)
    candidates=[]
    for s in detail.get('structures',[]):
        if s[0] in ('pair','additive'):candidates.append(tuple(s[1:3]))
        elif s[0]=='single':candidates.append((s[1],))
    if not candidates:return eb.node(x,y,q,sizes,k,full_pairs=True,product_backoff=False)
    features=candidates[0]
    if len(features)==1:
        f=features[0];xx=np.column_stack([x[:,f],np.zeros(n,int)]);qq=np.column_stack([q[:,f],np.zeros(len(q),int)]);sz=[sizes[f],1]
    else:
        xx=x[:,features];qq=q[:,features];sz=[sizes[f] for f in features]
    rr=np.arange(len(va));coef=1-k*.001
    curves=[-np.log(.001+coef*pb[rr,y[va]])];configs=[None]
    for cfg in CONFIGS:
        if len(features)==1 and cfg[1] is not None:continue
        pr=fit_pair(xx[tr],y[tr],xx[va],sz,k,*cfg);curves.append(-np.log(.001+coef*pr[rr,y[va]]));configs.append(cfg)
    mean=np.mean(curves,axis=1);best=np.argsort(mean)[:3];w=np.exp(-.5*len(va)*(mean[best]-mean[best[0]]));w/=w.sum()
    result=np.zeros((len(q),k));used=[]
    for weight,j in zip(w,best):
        cfg=configs[j]
        if cfg is None:pr,_=eb.node(x,y,q,sizes,k,full_pairs=True,product_backoff=False)
        else:pr=fit_pair(xx,y,qq,sz,k,*cfg)
        result+=weight*pr;used.append({'config':cfg,'weight':float(weight)})
    return result,{'type':'loglinear','features':features,'selected':used}

def fit_predict(train,query,schema):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given);out={};details={}
    for name in names:
        opts=util.options(schema,name);k=len(opts);lookup={v:i for i,v in enumerate(opts)};y=np.asarray([lookup[v] for v in train[name]],int)
        if schema['items'][name].get('gate'):
            asked=y<k-1;op,od=node(x,asked.astype(int),q,sizes,2);ordinary,dd=node(x[asked],y[asked],q,sizes,k-1);p=np.column_stack([op[:,1,None]*ordinary,op[:,0]]);details[name]={'open':od,'value':dd}
        else:p,dd=node(x,y,q,sizes,k);details[name]=dd
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-started,'items':details}
