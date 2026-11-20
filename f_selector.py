"""Keep current predictions first; commit only complete fold/refit candidates."""
import gc
import time
import numpy as np
import pandas as pd
import baseline_v11 as baseline
import softmax_expert as util
import f_budget as budget
from f_selection import checked, correct, combine
import f_tabicl
import f_g_prior


def plans_for(train, score_cap, fit_cap):
    result = []
    for fold in np.array_split(np.random.default_rng(14721).permutation(len(train)), 3):
        fit = np.setdiff1d(np.arange(len(train)), fold)
        if len(fit) > fit_cap:
            fit = np.random.default_rng(14722+len(fold)).permutation(fit)[:fit_cap]
        result.append((fit, fold[:score_cap]))
    return result


def meta(module, pred, labels, names, rows):
    count = len(pred)
    anchor = (np.asarray([.25, .15, .60]) if count == 3
              else np.asarray([.28, .16, .16, .08, .04, .08, .20]))
    probs = {n: np.column_stack([correct(pred[j][n][rows], labels[n][rows])
                                for j in range(count)]) for n in names}
    global_w = module.optimize(np.concatenate(list(probs.values())), anchor, .001)
    weights = {n: module.optimize(probs[n], global_w, 12./len(rows)) for n in names}
    combos = [(1., 0.), (.9, 0.), (.95, 0.), (1.05, 0.), (1.10, 0.),
              (1.15, 0.), (1., .0005), (1., .001)]
    mixed = {n: sum(weights[n][j]*pred[j][n][rows] for j in range(count)) for n in names}
    losses = [np.mean([-np.log(correct(module.transform(mixed[n], p, u), labels[n][rows]))
                       for n in names], axis=0) for p, u in combos]
    best = min(range(len(losses)), key=lambda j: losses[j].mean())
    delta = losses[0]-losses[best]
    power, undo = combos[best] if delta.mean() > .0002+delta.std(ddof=1)/np.sqrt(len(rows)) else combos[0]
    for n, w in weights.items():
        w[w < .015] = 0.
        weights[n] = w/w.sum() if w.sum() else global_w.copy()
    return weights, 1+.5*(power-1), .5*undo


def assemble(module, pred, weights, power, undo, names, rows=None):
    return {n: module.transform(sum(weights[n][j]*(pred[j][n] if rows is None else pred[j][n][rows])
                                   for j in pred if weights[n][j] > 0), power, undo)
            for n in names}


def shared_candidate(train, query, schema, plans, cache, module, pred, labels,
                     names, fold_ids, end):
    # An independent seed and eight members; keep all existing experts fixed.
    which = baseline.identify(schema)
    if which == 'world_bank':
        from f_shared_wb import fit_predict
    else:
        from f_shared_unhcr import fit_predict
    shared_index = len(pred)-1
    parts = []
    for fit, hold in plans:
        budget.require(end, 25.)
        q = train.iloc[hold].reset_index(drop=True).copy()
        q[names] = np.nan
        current, _ = fit_predict(train.iloc[fit].reset_index(drop=True), q, schema,
                                 members=8, seed=63401, end=end)
        parts.append(current)
    budget.require(end, 25.)
    current, _ = fit_predict(train, query, schema, members=8, seed=63401, end=end)
    oof = {n: np.concatenate([p[n] for p in parts]) for n in names}
    changed = {j: dict(v) for j, v in pred.items()}
    changed[shared_index] = {n: .5*(pred[shared_index][n]+oof[n]) for n in names}
    out = {n: np.empty_like(oof[n]) for n in names}
    for fold in range(3):
        learn, hold = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        weights, power, undo = meta(module, pred, labels, names, learn)
        values = assemble(module, changed, weights, power, undo, names, hold)
        for n in names:
            out[n][hold] = values[n]
    # Use the exact current full-data stack parameters, including calibration.
    diag = module.LAST_DIAGNOSTICS
    full = {j: dict(v) for j, v in cache[-1].items()}
    if shared_index not in full:
        return {}
    full[shared_index] = {n: .5*(full[shared_index][n]+current[n]) for n in names}
    final = assemble(module, full, diag['item_weights'], diag['power'], diag['floor_undo'], names)
    return {n: (out[n], final[n]) for n in names}


def enhance(frame, schema, raw, module, cache, final):
    names = [n for n, r in schema['items'].items() if r['class'] == 'PREDICT']
    if len(cache) != 4 or module.LAST_DIAGNOSTICS.get('status') != 'completed':
        return raw
    end = budget.deadline(baseline.identify(schema))
    if time.monotonic()+12. >= end:
        return raw
    train = frame.loc[frame[names].notna().all(axis=1)].reset_index(drop=True)
    query = frame.loc[frame[names].isna().any(axis=1)].reset_index(drop=True)
    if not len(query) or not query[names].isna().all().all():
        return raw
    plans = plans_for(train, module.SCORE_CAP, module.FIT_CAP)
    labels = {n: np.concatenate([train.iloc[hold][n].map({v: j for j, v in enumerate(util.options(schema, n))})
                                 .to_numpy(dtype=np.int64) for _, hold in plans]) for n in names}
    fold_ids = np.concatenate([np.full(len(hold), f, dtype=int) for f, (_, hold) in enumerate(plans)])
    pred = {j: {n: np.concatenate([cache[f][j][n] for f in range(3)]) for n in names}
            for j in cache[0]}
    anchor_oof = {n: np.empty_like(pred[0][n]) for n in names}
    for fold in range(3):
        learn, hold = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        weights, power, undo = meta(module, pred, labels, names, learn)
        values = assemble(module, pred, weights, power, undo, names, hold)
        for n in names:
            anchor_oof[n][hold] = values[n]
    anchor_full = {n: checked(raw[j::len(names)], len(query), len(util.options(schema, n)))
                   for j, n in enumerate(names)}
    alternatives = {n: [] for n in names}
    # Dev has only the few seconds left after the protected baseline.
    if final:
        try:
            budget.require(end, 100.)
            candidate = shared_candidate(train, query, schema, plans, cache, module,
                                         pred, labels, names, fold_ids, end)
            for n, value in candidate.items():
                alternatives[n].append(value)
        except Exception:
            pass
    # Deterministic ranking from visible OOF loss; never from synthetic skill.
    ranked = sorted(names, key=lambda n: -float(-np.log(correct(anchor_oof[n], labels[n])).mean()))
    for name in ranked[:(8 if final else 1)]:
        try:
            budget.require(end, 18. if final else 10.)
            candidate = f_tabicl.candidate(train, query, schema, name, plans, end, final)
            alternatives[name].append(candidate)
        except budget.BudgetExpired:
            break
        except Exception:
            continue
    if f_g_prior.available():
        try:
            budget.require(end, 20.)
            parts = []
            for fit, hold in plans:
                q = train.iloc[hold].reset_index(drop=True).copy()
                q[names] = np.nan
                parts.append(f_g_prior.fit_predict(train.iloc[fit].reset_index(drop=True), q,
                                                    schema, names, end))
            full = f_g_prior.fit_predict(train, query, schema, names, end)
            for n in names:
                if n in full and all(n in p for p in parts):
                    oof = checked(np.concatenate([p[n] for p in parts]), len(fold_ids), anchor_full[n].shape[1])
                    pp = checked(full[n], len(query), anchor_full[n].shape[1])
                    alternatives[n].append((oof, pp))
        except Exception:
            pass
    if not any(alternatives.values()):
        return raw
    values = {n: combine(anchor_oof[n], anchor_full[n], alternatives[n], labels[n], fold_ids) for n in names}
    return [values[n][i] for i in range(len(query)) for n in names]


def predict(frame, schema, base):
    final = budget.configure(frame, schema)
    kind = baseline.identify(schema)
    if kind not in ('world_bank', 'unhcr'):
        return base(frame, schema)
    if kind == 'world_bank':
        import wb_shared_stack as module
    else:
        import unhcr_stack as module
    original = module.fit_experts
    cache = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        cache.append(result)
        return result

    module.fit_experts = capture
    try:
        raw = base(frame, schema)
    finally:
        module.fit_experts = original
    try:
        return enhance(frame, schema, raw, module, cache, final)
    except Exception:
        return raw
    finally:
        cache.clear()
        gc.collect()
