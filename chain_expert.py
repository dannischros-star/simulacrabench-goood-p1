"""Empirical parent integration without enforcing questionnaire routing."""
import time
import numpy as np
import softmax_expert as util
import pair_expert as eb

def fit_predict(train,query,schema,direct=None):
    start=time.monotonic();names=[n for n,r in schema['items'].items() if r['class']=='PREDICT'];given=[n for n,r in schema['items'].items() if r['class']=='GIVEN']
    if direct is None:direct,_=eb.fit_predict(train,query,schema,full_pairs=True,product_backoff=True)
    x,sizes=util.encode(train,schema,given);q,_=util.encode(query,schema,given);out={n:p.copy() for n,p in direct.items()};todo=names.copy();done=set(given);details={}
    while todo:
        ready=[n for n in todo if not schema['items'][n].get('gate') or schema['items'][n]['gate']['parent'] in done]
        if not ready:raise ValueError('Gate cycle')
        for name in ready:
            todo.remove(name);done.add(name);gate=schema['items'][name].get('gate')
            if not gate or gate['parent'] in given:continue
            parent=gate['parent'];opts=util.options(schema,parent);k=len(opts);lookup={v:i for i,v in enumerate(opts)};yp=np.asarray([lookup[v] for v in train[parent]],int);asked=(train[name]!=schema['gated_value']).to_numpy().astype(int)
            xx=np.column_stack([x,yp]);qq=np.column_stack([np.repeat(q,k,axis=0),np.tile(np.arange(k),len(q))])
            pp,detail=eb.node(xx,asked,qq,sizes+[k+1],2,full_pairs=False,product_backoff=True)
            opened=(out[parent]*pp[:,1].reshape(len(q),k)).sum(axis=1)
            value=out[name][:,:-1];z=value.sum(axis=1,keepdims=True);value=np.divide(value,z,out=np.full_like(value,1/value.shape[1]),where=z>0)
            out[name]=np.column_stack([opened[:,None]*value,1-opened]);details[name]=detail
    return out,{'seconds':time.monotonic()-start,'nodes':details}
