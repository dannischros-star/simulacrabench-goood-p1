"""TRAIN-only adaptive tabular ensemble; all weights bundled locally."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import baseline_v11 as baseline

def predict(frame,schema):
    if baseline.identify(schema)!='unicef':
        return baseline.marginal(frame,schema)
    import torch
    if not torch.cuda.is_available():
        import tabm_stack
        return tabm_stack.predict(frame,schema)
    import adaptive_stack
    weight=Path(__file__).resolve().parent/'weights/tabicl-classifier-v2-20260212.ckpt'
    return adaptive_stack.predict(frame,schema,weight)
