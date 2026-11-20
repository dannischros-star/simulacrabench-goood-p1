"""All-GIVEN multinomial regression, optional shrunken interactions.
Every configuration is selected inside supplied TRAIN. No external data.
"""
import time
import numpy as np
from scipy import sparse
import softmax_expert as util

def node(x,y,q,sizes,k,country):
    n=len(y);p0=util.prior(y,k)
    if n<70 or len(np.unique(y))<2:return np.tile(p0,(len(q),1))
    perm=np.random.default_rng(31581).permutation(n);cut=int(.8*n);tr,va=perm[:cut],perm[cut:]
    pairs=util.select_pairs(x[tr],y[tr],sizes,k,max_pairs=2)
    if country is not None:
        pairs=sorted(set(pairs+[(min(country,j),max(country,j)) for j in range(x.shape[1]) if j!=country]))
    configs=[(1.,0.),(4.,0.),(16.,0.),(4.,.25),(4.,.6)] if pairs else [(1.,0.),(4.,0.),(16.,0.)]
    X=util.design(x,sizes);Q=util.design(q,sizes)
    if pairs:XP=util.pair_design(x,sizes,pairs);QP=util.pair_design(q,sizes,pairs)
    def matrices(scale):
        if not scale:return X,Q
        return sparse.hstack([X,scale*XP],format='csr'),sparse.hstack([Q,scale*QP],format='csr')
    losses=[]
    for reg,scale in configs:
        XX,_=matrices(scale);p=util.glm(XX[tr],y[tr],XX[va],k,reg,maxiter=80)
        losses.append(float(-np.log(.001+(1-.001*k)*p[np.arange(len(va)),y[va]]).mean()))
    best=np.argsort(losses)[:2];weights=np.exp(-.5*len(va)*(np.asarray(losses)[best]-min(losses)));weights/=weights.sum()
    out=np.zeros((len(q),k))
    for w,j in zip(weights,best):
        reg,scale=configs[j];XX,QQ=matrices(scale);out+=w*util.glm(XX,y,QQ,k,reg,maxiter=110)
    return out

def fit_predict(train,query,schema):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given)
    country=given.index('Country') if 'Country' in given else None;out={}
    for name,r in schema['items'].items():
        if r['class']!='PREDICT':continue
        opts=util.options(schema,name);k=len(opts);look={o:i for i,o in enumerate(opts)};y=np.array([look[o] for o in train[name]])
        if r.get('gate'):
            asked=y<k-1;op=node(x,asked.astype(int),q,sizes,2,country)[:,1]
            po=node(x[asked],y[asked],q,sizes,k-1,country);p=np.column_stack([po*op[:,None],1-op])
        else:p=node(x,y,q,sizes,k,country)
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-started,'all_given_features':given,'selection':'TRAIN-only nested validation'}
