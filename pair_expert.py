"""Compact LOO empirical-Bayes tables, full-pair search, product backoff."""
import itertools,time
import numpy as np
import softmax_expert as util
ALPHAS=(1.,5.,20.,80.)
MAX_LEVELS=5000

def node(x,y,q,sizes,k,full_pairs=True,product_backoff=True):
    n=len(y);p0=util.prior(y,k)
    if n<40 or len(np.unique(y))<2:return np.broadcast_to(p0,(len(q),k)).copy(),{'type':'marginal'}
    rr=np.arange(n);cy=np.bincount(y,minlength=k).astype(float)+.5
    prior=np.broadcast_to(cy,(n,k)).copy();prior[rr,y]-=1;prior/=(n-1+.5*k)
    tp=prior[rr,y];coef=1-k*.001
    def loss(p):return float(-np.log(.001+coef*p).mean())
    records=[(loss(tp),('marginal',))];counts=[];preds=[];queries=[];ranking=[]
    for f,levels in enumerate(sizes):
        c=np.bincount(x[:,f]*k+y,minlength=levels*k).reshape(levels,k).astype(float);counts.append(c)
        den=c.sum(axis=1)[x[:,f]]-1;tc=c[x[:,f],y]-1;ss=[]
        for a in ALPHAS:
            score=loss((tc+a*tp)/(den+a))+np.log(max(2,len(sizes)))/n
            records.append((score,('single',f,a)));ss.append((score,a))
        score,a=min(ss);trc=c[x[:,f]].copy();trc[rr,y]-=1
        preds.append((trc+a*prior)/(den[:,None]+a));queries.append((c[q[:,f]]+a*p0)/(c.sum(axis=1)[q[:,f],None]+a));ranking.append((score,f))
    pool=list(range(len(sizes))) if full_pairs or len(sizes)<=10 else [f for _,f in sorted(ranking)[:7]]
    pairs=list(itertools.combinations(pool,2));penalty=np.log(max(2,len(pairs)*2))/n
    for f,g in pairs:
        if sizes[f]*sizes[g]>MAX_LEVELS:continue
        code=x[:,f]*sizes[g]+x[:,g];c=np.bincount(code*k+y,minlength=sizes[f]*sizes[g]*k).reshape(-1,k)
        den=c.sum(axis=1)[code]-1;tc=c[code,y]-1
        prod=preds[f]*preds[g]/np.maximum(prior,1e-12);prod/=prod.sum(axis=1,keepdims=True);pt=prod[rr,y]
        records.append((loss(pt)+penalty,('additive',f,g)))
        backs=[('arithmetic',.5*(preds[f][rr,y]+preds[g][rr,y]))]
        if product_backoff:backs.append(('product',pt))
        for mode,bt in backs:
            for a in ALPHAS:records.append((loss((tc+a*bt)/(den+a))+penalty,('pair',f,g,a,mode)))
    best=sorted(records,key=lambda z:z[0])[:4];weights=np.exp(-.5*n*(np.array([z[0] for z in best])-best[0][0]));weights/=weights.sum();out=np.zeros((len(q),k))
    for w,(_,s) in zip(weights,best):
        if s[0]=='marginal':qp=p0
        elif s[0]=='single':
            _,f,a=s;c=counts[f];qp=(c[q[:,f]]+a*p0)/(c.sum(axis=1)[q[:,f],None]+a)
        else:
            f,g=s[1:3];prod=queries[f]*queries[g]/p0;prod/=prod.sum(axis=1,keepdims=True)
            if s[0]=='additive':qp=prod
            else:
                _,f,g,a,mode=s;code=x[:,f]*sizes[g]+x[:,g];qc=q[:,f]*sizes[g]+q[:,g]
                c=np.bincount(code*k+y,minlength=sizes[f]*sizes[g]*k).reshape(-1,k)
                back=prod if mode=='product' else .5*(queries[f]+queries[g]);qp=(c[qc]+a*back)/(c.sum(axis=1)[qc,None]+a)
        out+=w*qp
    return out,{'type':best[0][1][0],'structures':[s for _,s in best],'weights':weights.tolist(),'pairs_examined':len(pairs)}

def fit_predict(train,query,schema,full_pairs=True,product_backoff=True):
    started=time.monotonic();given=[n for n,r in schema['items'].items() if r['class']=='GIVEN'];names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given);out={};details={}
    for name in names:
        opts=util.options(schema,name);k=len(opts);lookup={v:i for i,v in enumerate(opts)};y=np.asarray([lookup[v] for v in train[name]],int);kw={'full_pairs':full_pairs,'product_backoff':product_backoff}
        if schema['items'][name].get('gate'):
            asked=y<k-1;op,od=node(x,asked.astype(int),q,sizes,2,**kw);ordinary,dd=node(x[asked],y[asked],q,sizes,k-1,**kw);p=np.column_stack([op[:,1,None]*ordinary,op[:,0]]);details[name]={'open':od,'value':dd}
        else:p,dd=node(x,y,q,sizes,k,**kw);details[name]=dd
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'seconds':time.monotonic()-started,'items':details,'full_pairs':full_pairs,'product_backoff':product_backoff}
