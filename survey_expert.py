"""Nonresponse-aware conditional experts. Fits ONLY supplied visible TRAIN.
No generator imports, respondent IDs, roles, hidden labels, or external I/O.
"""
import time,re
import numpy as np
from scipy import sparse
from scipy.special import softmax,logsumexp,expit
from scipy.optimize import minimize
import softmax_expert as util
import ratio_expert
NR={'Not answered','No response','Not recorded'}

def enrich(train,query,schema):
    given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    x,sz=util.encode(train,schema,given);q,_=util.encode(query,schema,given)
    names=given.copy();flags=[j for j,n in enumerate(given) if any(o in NR for o in util.options(schema,n))]
    for f in [train,query]:
        pass
    def extras(frame,encoded):
        miss=np.column_stack([frame[n].isin(NR).to_numpy() for n in given])
        extra=[miss.sum(axis=1).astype(int)];sizes=[len(given)+1];labels=['given_nonresponse_count']
        if flags:
            signature=(miss[:,flags]*(2**np.arange(len(flags)))).sum(axis=1).astype(int)
            extra.append(signature);sizes.append(2**len(flags));labels.append('given_nonresponse_pattern')
        a='user.wageOfSimilarEmployees';b='user.lowestAcceptablePay'
        if a in given and b in given:
            ia,ib=given.index(a),given.index(b);bad=miss[:,ia]|miss[:,ib]
            diff=np.clip(encoded[:,ia]-encoded[:,ib],-2,2)+2;diff[bad]=5
            extra.append(diff);sizes.append(6);labels.append('wage_bin_difference')
        return extra,sizes,labels
    ex,es,en=extras(train,x);eq,_,_=extras(query,q)
    return np.column_stack([x]+ex),np.column_stack([q]+eq),sz+es,names+en

def ordered(options):
    return len(options)>=3 and all(re.match(r'^[<>]?=?[0-9]',str(o)) for o in options)

def smooth_glm(x,y,q,sizes,k,smooth=8.):
    X=util.design(x,sizes);Q=util.design(q,sizes);n,d=X.shape;rr=np.arange(n);p0=util.prior(y,k);off=np.log(p0)
    if len(y)<60 or len(np.unique(y))<2:return np.tile(p0,(len(q),1))
    def obj(v):
        w=v.reshape(d+1,k);z=X@w[:-1]+w[-1]+off;normal=logsumexp(z,axis=1);err=np.exp(z-normal[:,None]);err[rr,y]-=1
        value=(normal-z[rr,y]).sum()+1.5*np.sum(w[:-1]**2)+.5*np.sum(w[-1]**2)
        grad=np.empty_like(w);grad[:-1]=X.T@err+3*w[:-1];grad[-1]=err.sum(axis=0)+w[-1]
        if k>=3:
            delta=w[:-1,2:]-2*w[:-1,1:-1]+w[:-1,:-2]
            value+=.5*smooth*np.sum(delta**2)
            grad[:-1,2:]+=smooth*delta;grad[:-1,1:-1]-=2*smooth*delta;grad[:-1,:-2]+=smooth*delta
        return float(value),grad.ravel()
    fit=minimize(obj,np.zeros((d+1,k)).ravel(),jac=True,method='L-BFGS-B',options={'maxiter':80,'ftol':1e-8,'maxcor':6})
    w=fit.x.reshape(d+1,k);return softmax(Q@w[:-1]+w[-1]+off,axis=1)

def tree_node(x,y,q,sizes,k):
    p0=util.prior(y,k)
    if len(y)<80 or len(np.unique(y))<2:return np.tile(p0,(len(q),1))
    import lightgbm as lgb
    clf=lgb.LGBMClassifier(n_estimators=90,num_leaves=15,max_depth=5,learning_rate=.05,min_child_samples=35,reg_lambda=12.,reg_alpha=.3,cat_smooth=25.,cat_l2=15.,max_cat_to_onehot=12,n_jobs=1,random_state=925,verbosity=-1,force_col_wise=True,deterministic=True)
    clf.fit(x,y,categorical_feature=list(range(x.shape[1])))
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names')
        p=clf.predict_proba(q)
    out=np.zeros((len(q),k));out[:,clf.classes_.astype(int)]=p
    return .998*out+.002*p0

def fractional_predict(x,target,q,sizes):
    X=util.design(x,sizes);Q=util.design(q,sizes);n,d=X.shape
    mean=np.clip(target.mean(),.001,.999);offset=np.log(mean/(1-mean))
    def obj(w):
        z=X@w[:-1]+w[-1]+offset;p=expit(z);err=p-target
        val=np.logaddexp(0,z).sum()-target@z+4*np.sum(w[:-1]**2)+.5*w[-1]**2
        grad=np.r_[X.T@err+8*w[:-1],err.sum()+w[-1]]
        return float(val),grad
    fit=minimize(obj,np.zeros(d+1),jac=True,method='L-BFGS-B',options={'maxiter':65,'ftol':1e-8})
    return expit(Q@fit.x[:-1]+fit.x[-1]+offset)

def shared_nonresponse(train,query,schema,x,q,sizes,names):
    groups={}
    for name in names:
        if 'Not answered' not in schema['items'][name]['values']:continue
        group='scale' if name.startswith('sc_') else ('tracer' if name.startswith('tr_') else 'post')
        groups.setdefault(group,[]).append(name)
    result={}
    for members in groups.values():
        if len(members)<4:continue
        y=train[members].eq('Not answered').to_numpy(float);frac=y.mean(axis=1)
        pred=fractional_predict(x,frac,q,sizes)
        for j,name in enumerate(members):
            disagreement=np.mean(np.abs(y[:,j]-frac))
            if disagreement>.075 or y[:,j].mean()<.01:continue
            # Fractional regression averages correlated labels per respondent;
            # it does not pretend the battery creates new independent people.
            shift=np.log((y[:,j].sum()+.5)/(len(y)-y[:,j].sum()+.5))-np.log((frac.sum()+.5)/(len(y)-frac.sum()+.5))
            result[name]=expit(np.log(np.maximum(pred,1e-9)/np.maximum(1-pred,1e-9))+shift)
    return result

def fit_predict(train,query,schema,mode='nr_ratio'):
    started=time.monotonic();x,q,sizes,features=enrich(train,query,schema)
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    shared=shared_nonresponse(train,query,schema,x,q,sizes,names) if mode=='shared' else {}
    def node(xx,y,qq,k,is_ordered=False):
        if k==1:return np.ones((len(qq),1))
        if not len(y):return np.full((len(qq),k),1/k)
        if mode=='tree':return tree_node(xx,y,qq,sizes,k)
        if mode=='ordinal' and is_ordered:return smooth_glm(xx,y,qq,sizes,k)
        return ratio_expert.node(xx,y,qq,sizes,k)[0]
    out={};used=0
    for name in names:
        opts=util.options(schema,name);k=len(opts);look={o:i for i,o in enumerate(opts)};y=np.array([look[o] for o in train[name]],int)
        nr=[j for j,o in enumerate(opts) if o in NR];gate=look.get(schema['gated_value']) if schema['items'][name].get('gate') else None
        special=nr+([gate] if gate is not None else []);normal=[j for j in range(k) if j not in special]
        if special and normal:
            status=np.zeros(k,int)
            for j,index in enumerate(special,1):status[index]=j
            yp=status[y];ps=node(x,yp,q,1+len(special));okay=yp==0
            lookup=np.full(k,-1,int);lookup[normal]=np.arange(len(normal))
            po=node(x[okay],lookup[y[okay]],q,len(normal),ordered([opts[j] for j in normal]));p=np.zeros((len(q),k))
            p[:,normal]=ps[:,0,None]*po
            for j,index in enumerate(special,1):p[:,index]=ps[:,j]
            if name in shared and 'Not answered' in opts:
                i=look['Not answered'];old=p[:,i].copy();new=.5*old+.5*shared[name];p*=((1-new)/np.maximum(1-old,1e-12))[:,None];p[:,i]=new;used+=1
        else:p=node(x,y,q,k,ordered(opts))
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'mode':mode,'features':features,'shared_targets':used,'seconds':time.monotonic()-started}
