"""One declared checkpoint, one atomic question at a time, never downloads."""
from pathlib import Path
import gc
import time
import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier
from f_budget import require
from f_selection import checked
import softmax_expert as util

WEIGHT = Path(__file__).resolve().parent / 'weights/tabicl-classifier-v2-20260212.ckpt'


def fit_item(train, query, schema, name, end, seed=63401, members=2, cap=768):
    require(end, 3.)
    given = [n for n, r in schema['items'].items() if r['class'] == 'GIVEN']
    options = util.options(schema, name)
    lookup = {v: j for j, v in enumerate(options)}
    rows = np.random.default_rng(seed).permutation(len(train))[:cap]
    tt = train.iloc[rows]
    y = tt[name].map(lookup).to_numpy(dtype=np.int64)
    prior = util.prior(y, len(options))
    if len(np.unique(y)) < 2:
        return np.tile(prior, (len(query), 1))
    # This checkpoint supports at most ten classes without a class hierarchy.
    if len(np.unique(y)) > 10:
        raise ValueError('Unsupported class count')
    x, q = tt[given].reset_index(drop=True).copy(), query[given].reset_index(drop=True).copy()
    for field in given:
        levels = util.options(schema, field)
        x[field] = pd.Categorical(x[field], categories=levels)
        q[field] = pd.Categorical(q[field], categories=levels)
    model = TabICLClassifier(n_estimators=members, device='cuda', n_jobs=1, batch_size=2,
                            model_path=str(WEIGHT), allow_auto_download=False,
                            offload_mode='gpu', use_amp=False,
                            softmax_temperature=.9, random_state=seed, verbose=False)
    try:
        require(end, 2.)
        model.fit(x, y)
        raw = model.predict_proba(q)
        p = np.zeros((len(q), len(options)), dtype=float)
        p[:, model.classes_.astype(int)] = raw
        return checked(.998*p+.002*prior, len(q), len(options))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def candidate(train, query, schema, name, plans, end, final):
    members, cap = (2, 768) if final else (1, 192)
    outputs = []
    for fit, hold in plans:
        require(end, 4.)
        outputs.append(fit_item(train.iloc[fit].reset_index(drop=True),
                                train.iloc[hold].reset_index(drop=True),
                                schema, name, end, members=members, cap=cap))
    require(end, 4.)
    full = fit_item(train, query, schema, name, end, members=members, cap=cap)
    return np.concatenate(outputs), full
