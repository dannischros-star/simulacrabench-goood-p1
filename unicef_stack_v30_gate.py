"""UNICEF v26: preserve v22 forecasts but enforce observed questionnaire gates.

All postprocessing is determined by public schema, GIVEN values and visible
TRAIN consistency checks. No generator, IDs or hidden answers are referenced.
"""
import numpy as np
import os,sys
sys.path.insert(0,os.path.dirname(__file__))
import unicef_stack_v30_bothcaps as parent
LAST_DIAGNOSTICS={}

def predict(frame,schema):
    global LAST_DIAGNOSTICS
    names=[n for n,r in schema['items'].items() if r['class']=='PREDICT']
    tr=frame.loc[frame[names].notna().all(axis=1)]
    qu=frame.loc[frame[names].isna().any(axis=1)]
    raw=parent.predict(frame,schema)
    if not len(qu):return raw
    mats={n:np.asarray(raw[i::len(names)],float).copy() for i,n in enumerate(names)}
    nrows=len(qu)
    if any(p.shape[0]!=nrows for p in mats.values()):raise ValueError("Output rows misaligned")
    used=[]
    sentinel=schema['gated_value']
    if all(n in mats for n in ('Q08','Q09','Q10')) and len(tr)>100:
        good=tr['Q08'].isin(['Yes','No'])
        allowed=good.to_numpy()
        expected=(tr['Q08']!='No').to_numpy()
        correct=allowed.copy()
        for n in ('Q09','Q10'):
            correct &= (tr[n].to_numpy()==sentinel)==expected
        if int(good.sum())>=100 and correct[allowed].mean()>=.999:
            yes=schema['items']['Q08']['values'].index('Yes')
            no=schema['items']['Q08']['values'].index('No')
            gate=np.mean(np.column_stack((mats['Q08'][:,yes],mats['Q09'][:,-1],mats['Q10'][:,-1])),axis=1)
            gate=np.clip(gate,1e-9,1-1e-9)
            mats['Q08'][:,yes]=gate
            mats['Q08'][:,no]=1-gate
            for n in ('Q09','Q10'):
                ordinary=mats[n][:,:-1]
                mass=ordinary.sum(axis=1,keepdims=True)
                mats[n][:,:-1]=np.divide(ordinary,mass,out=np.full_like(ordinary,1/ordinary.shape[1]),where=mass>0)*(1-gate[:,None])
                mats[n][:,-1]=gate
            used.append('Q08_Q09_Q10_coherent')
    for name in names:
        gate_rule=schema['items'][name].get('gate')
        if not gate_rule or schema['items'][gate_rule['parent']]['class']!='GIVEN':continue
        parent_name=gate_rule['parent']
        eligible=tr[parent_name].isin(gate_rule['observed_if']).to_numpy()
        correct=(tr[name].to_numpy()==sentinel)==(~eligible)
        if len(correct)<100 or correct.mean()<.999:continue
        open_query=qu[parent_name].isin(gate_rule['observed_if']).to_numpy()
        mat=mats[name]
        ordinary=mat[:,:-1]
        mass=ordinary.sum(axis=1,keepdims=True)
        ordinary=np.divide(ordinary,mass,out=np.full_like(ordinary,1/ordinary.shape[1]),where=mass>0)
        mat[:,:-1]=ordinary*open_query[:,None]
        mat[:,-1]=(~open_query).astype(float)
        used.append(name+'_known_gate')
    out=[mats[n][r] for r in range(nrows) for n in names]
    LAST_DIAGNOSTICS={'status':'completed','postprocess':used,
                      'v22':dict(getattr(parent,'LAST_DIAGNOSTICS',{}))}
    return out
