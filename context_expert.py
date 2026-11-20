"""Hierarchical three-way categorical predictive tables with LOO shrinkage.
Structure/tuning sees only visible TRAIN. Requires full-program outer validation.
"""
import itertools,time
import numpy as np
import softmax_expert as util
import survey_expert as features
ALPHAS=(5.,20.,80.,320.)

def node(x,y,q,sizes,k):
    n=len(y);prior=util.prior(y,k)
    if n<60 or len(np.unique(y))<2:return np.tile(prior,(len(q),1))
    active=[i for i in range(x.shape[1]) if np.unique(x[:,i]).size>1]
    rr=np.arange(n);counts=np.bincount(y,minlength=k)+.5
    lp=np.tile(counts,(n,1));lp[rr,y]-=1;lp/=n-1+.5*k
    def loss(p):return float(-np.log(.001+(1-k*.001)*p[rr,y]).mean())
    def table(fs,a,back,qback):
        code=np.zeros(n,int);qc=np.zeros(len(q),int);levels=1
        for f in fs:code=code*sizes[f]+x[:,f];qc=qc*sizes[f]+q[:,f];levels*=sizes[f]
        c=np.bincount(code*k+y,minlength=levels*k).reshape(-1,k).astype(float)
        tr=c[code].copy();tr[rr,y]-=1
        return (tr+a*back)/(tr.sum(axis=1,keepdims=True)+a),(c[qc]+a*qback)/(c.sum(axis=1)[qc,None]+a)
    records=[(loss(lp),np.tile(prior,(len(q),1)),())];singles={};pairs={};ranking=[]
    for f in active:
        candidates=[]
        for a in ALPHAS:
            p,qp=table((f,),a,lp,prior);candidates.append((loss(p),p,qp))
        rec=min(candidates,key=lambda a:a[0]);singles[f]=rec
        records.append((rec[0]+np.log(max(2,len(active)))/n,rec[2],(f,)));ranking.append((rec[0],f))
    # Keep all pairs; triple pool includes country and at most 8 useful features.
    combos=list(itertools.combinations(active,2))
    for f,g in combos:
        if sizes[f]*sizes[g]>4500:continue
        back=.5*(singles[f][1]+singles[g][1]);qb=.5*(singles[f][2]+singles[g][2]);candidates=[]
        for a in ALPHAS:
            p,qp=table((f,g),a,back,qb);candidates.append((loss(p),p,qp))
        rec=min(candidates,key=lambda a:a[0]);pairs[f,g]=rec
        records.append((rec[0]+np.log(max(2,len(combos)*2))/n,rec[2],(f,g)))
    pool=sorted(f for _,f in sorted(ranking)[:8]);triples=list(itertools.combinations(pool,3))
    for f,g,h in triples:
        if sizes[f]*sizes[g]*sizes[h]>4500:continue
        ab=[tuple(sorted(z)) for z in ((f,g),(f,h),(g,h))]
        if not all(z in pairs for z in ab):continue
        back=sum(pairs[z][1] for z in ab)/3;qb=sum(pairs[z][2] for z in ab)/3
        for a in ALPHAS:
            p,qp=table((f,g,h),a,back,qb)
            records.append((loss(p)+np.log(max(2,len(triples)*4))/n+.001,qp,(f,g,h)))
    records.sort(key=lambda a:a[0]);best=records[:5]
    weights=np.exp(-.5*n*(np.array([r[0] for r in best])-best[0][0]));weights/=weights.sum()
    return sum(w*r[1] for w,r in zip(weights,best))

def fit_predict(train,query,schema,mode='context'):
    start=time.monotonic();x,q,sz,_=features.enrich(train,query,schema)
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];out={}
    for name in names:
        opts=util.options(schema,name);k=len(opts);look={o:i for i,o in enumerate(opts)};y=np.asarray([look[o] for o in train[name]],int)
        special=[j for j,o in enumerate(opts) if o in features.NR or o==schema['gated_value']];normal=[j for j in range(k) if j not in special]
        if special and normal:
            status=np.zeros(k,int)
            for j,z in enumerate(special,1):status[z]=j
            yp=status[y];ps=node(x,yp,q,sz,1+len(special));ok=yp==0
            lu=np.full(k,-1,int);lu[normal]=np.arange(len(normal))
            po=node(x[ok],lu[y[ok]],q,sz,len(normal));p=np.zeros((len(q),k));p[:,normal]=ps[:,0,None]*po
            for j,z in enumerate(special,1):p[:,z]=ps[:,j]
        else:p=node(x,y,q,sz,k)
        out[name]=util.calibrate_observed_gate(p,train,query,schema,name)
    return out,{'mode':'context','max_order':3,'seconds':time.monotonic()-start}
