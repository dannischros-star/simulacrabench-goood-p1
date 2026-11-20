"""Additive multinomial model with TRAIN-selected country subgroups.
Country interaction candidates are fixed from GIVEN fields; labels select them
only within visible TRAIN. No evaluator or population parameters are imported.
"""
import time,itertools
import numpy as np
from scipy import sparse
import softmax_expert as util

def code(x,sizes,fields):
    c=np.zeros(len(x),int);levels=1
    for f in fields:c=c*sizes[f]+x[:,f];levels*=sizes[f]
    return c,levels

def node(x,y,q,sizes,k,country):
    n=len(y);prior=util.prior(y,k)
    if n<100 or len(np.unique(y))<2:return np.tile(prior,(len(q),1))
    ix=np.random.default_rng(33301).permutation(n);cut=int(.78*n);tr,va=ix[:cut],ix[cut:]
    X=util.design(x,sizes);Q=util.design(q,sizes)
    pbase=util.glm(X[tr],y[tr],X,k,4.,100);rr=np.arange(len(va))
    def loss(p):return float(-np.log(.001+(1-.001*k)*p[rr,y[va]]).mean())
    ranking=[]
    pool=list(itertools.combinations(range(x.shape[1]),2))
    pool+=[f for f in itertools.combinations(range(x.shape[1]),3) if country in f]
    for fields in pool:
        c,levels=code(x,sizes,fields)
        if levels>4500:continue
        counts=np.bincount(c[tr]*k+y[tr],minlength=levels*k).reshape(levels,k)
        expected=np.zeros((levels,k));np.add.at(expected,c[tr],pbase[tr])
        smooth=80*prior
        ratio=(counts+smooth)/(expected+smooth)
        pred=pbase[va]*ratio[c[va]];pred/=pred.sum(axis=1,keepdims=True)
        ranking.append((loss(pred),fields))
    structures=[None]+[fields for _,fields in sorted(ranking)[:2]]
    configs=[(None,0.)]+[(fields,scale) for fields in structures[1:] for scale in (.3,.7,1.2)]
    losses=[];designs={None:(X,Q)}
    for fields,scale in configs:
        if fields is None:XX,QQ=X,Q
        else:
            c,lev=code(x,sizes,fields);cq,_=code(q,sizes,fields)
            XP=util.design(c[:,None],[lev]);QP=util.design(cq[:,None],[lev])
            XX=sparse.hstack([X,scale*XP],format='csr');QQ=sparse.hstack([Q,scale*QP],format='csr')
        p=util.glm(XX[tr],y[tr],XX[va],k,4.,100)
        losses.append(loss(p));designs[(fields,scale)]=(XX,QQ)
    best=np.argsort(losses)[:2];weights=np.exp(-.5*len(va)*(np.asarray(losses)[best]-min(losses)));weights/=weights.sum()
    out=np.zeros((len(q),k))
    for w,j in zip(weights,best):
        XX,QQ=designs[configs[j]];out+=w*util.glm(XX,y,QQ,k,4.,140)
    return out

def fit_predict(train,query,schema):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given)
    country=given.index('Country');out={}
    for name,r in schema['items'].items():
        if r['class']!='PREDICT':continue
        opts=util.options(schema,name);k=len(opts);look={o:i for i,o in enumerate(opts)};y=np.array([look[o] for o in train[name]])
        if r.get('gate'):
            asked=y<k-1;op=node(x,asked.astype(int),q,sizes,2,country)[:,1]
            po=node(x[asked],y[asked],q,sizes,k-1,country);p=np.column_stack([po*op[:,None],1-op])
        else:p=node(x,y,q,sizes,k,country)
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-started,'max_interaction_order':3,'selection':'TRAIN-only residual ranking then nested validation'}
