"""Frozen probability selection and questionnaire consistency; no evaluator imports."""
import numpy as np
import unicef_stack_v33_core as old
ANCHOR=np.r_[.9*np.array([.11,.08,.10,.04,.05,.05,.12,.15,.15,.15]),.05,.05]
def base_select(z, names, indices):
    nv = len(z['y_' + names[0]])
    rr = np.arange(nv)
    correct = {n: np.column_stack([0.001 + (1 - 0.001 * z[f'oof_{j}_{n}'].shape[1]) * z[f'oof_{j}_{n}'][rr, z['y_' + n]] for j in indices]) for n in names}
    anchor = ANCHOR.copy() if len(indices) == 12 else np.r_[0.97 * ANCHOR, 0.03]
    glob = old.base.optimize(np.concatenate(list(correct.values())), anchor, 0.001)
    ws = {n: old.base.optimize(p, glob, 12.0 / nv) for n, p in correct.items()}
    mixes = {n: sum((ws[n][a] * z[f'oof_{j}_{n}'] for a, j in enumerate(indices))) for n in names}
    powers = (1.0, 0.9, 0.95, 1.05, 1.1, 1.15)
    curves = [np.mean([-np.log(0.001 + (1 - 0.001 * mixes[n].shape[1]) * old.base.transform(mixes[n], p, 0.0)[rr, z['y_' + n]]) for n in names], axis=0) for p in powers]
    best = min(range(len(curves)), key=lambda j: curves[j].mean())
    delta = curves[0] - curves[best]
    chosen = powers[best] if delta.mean() > 0.0002 + delta.std(ddof=1) / np.sqrt(nv) else 1.0
    for n, w in ws.items():
        w[w < 0.008] = 0
        ws[n] = w / w.sum() if w.sum() > 0 else glob.copy()
    return (ws, 1 + 0.5 * (chosen - 1), glob)

def select(z, names, indices):
    if len(indices) == 12:
        return base_select(z, names, indices)
    nv = len(z['y_' + names[0]])
    rr = np.arange(nv)
    nnew = len(indices) - 12
    anchor = np.r_[(1 - 0.03 * nnew) * ANCHOR, np.full(nnew, 0.03)]
    correct = {n: np.column_stack([0.001 + (1 - 0.001 * z[f'oof_{j}_{n}'].shape[1]) * z[f'oof_{j}_{n}'][rr, z['y_' + n]] for j in indices]) for n in names}
    glob = old.base.optimize(np.concatenate(list(correct.values())), anchor, 0.001)
    weights = {n: old.base.optimize(p, glob, 12.0 / nv) for n, p in correct.items()}
    mixes = {n: sum((weights[n][a] * z[f'oof_{j}_{n}'] for a, j in enumerate(indices))) for n in names}
    powers = (1.0, 0.9, 0.95, 1.05, 1.1, 1.15)
    curves = [np.mean([-np.log(0.001 + (1 - 0.001 * mixes[n].shape[1]) * old.base.transform(mixes[n], p, 0.0)[rr, z['y_' + n]]) for n in names], axis=0) for p in powers]
    best = min(range(len(curves)), key=lambda j: curves[j].mean())
    delta = curves[0] - curves[best]
    chosen = powers[best] if delta.mean() > 0.0002 + delta.std(ddof=1) / np.sqrt(nv) else 1.0
    for n, w in weights.items():
        w[w < 0.008] = 0
        weights[n] = w / w.sum() if w.sum() else glob.copy()
    return (weights, 1 + 0.5 * (chosen - 1), glob)

def mix(z, names, indices, ws, power):
    return {n: old.base.transform(sum((ws[n][a] * z[f'full_{j}_{n}'] for a, j in enumerate(indices) if ws[n][a] > 0)), power, 0.0) for n in names}

def postprocess(mats, frame, schema, z=None, indices=None, ws=None, learned_gate=False):
    names = [n for n, r in schema['items'].items() if r['class'] == 'PREDICT']
    train = frame.loc[frame[names].notna().all(axis=1)]
    query = frame.loc[frame[names].isna().any(axis=1)]
    mats = {n: p.copy() for n, p in mats.items()}
    used = []
    sentinel = schema['gated_value']
    if all((n in mats for n in ('Q08', 'Q09', 'Q10'))) and len(train) > 100:
        good = train['Q08'].isin(['Yes', 'No']).to_numpy()
        expected = (train['Q08'] != 'No').to_numpy()
        correct = good.copy()
        for n in ('Q09', 'Q10'):
            correct &= (train[n].to_numpy() == sentinel) == expected
        if int(good.sum()) >= 100 and correct[good].mean() >= 0.999:
            yes = schema['items']['Q08']['values'].index('Yes')
            no = schema['items']['Q08']['values'].index('No')
            shared = np.mean(np.column_stack([mats['Q08'][:, yes], mats['Q09'][:, -1], mats['Q10'][:, -1]]), axis=1)
            if learned_gate and z is not None:
                cols, full_cols, anchor = ([], [], [])
                for n, c in [('Q08', yes), ('Q09', -1), ('Q10', -1)]:
                    for a, j in enumerate(indices):
                        cols.append(z[f'oof_{j}_{n}'][:, c])
                        full_cols.append(z[f'full_{j}_{n}'][:, c])
                        anchor.append(ws[n][a] / 3.0)
                oo = np.column_stack(cols)
                ff = np.column_stack(full_cols)
                y = z['y_Q08'] == yes
                safe = 0.001 + 0.998 * np.where(y[:, None], oo, 1 - oo)
                anchor = np.array(anchor)
                fit = old.base.optimize(safe, anchor, 0.01)
                shared = 0.5 * shared + 0.5 * (ff @ fit)
                used.append('TRAIN_crossfitted_shared_gate')
            shared = np.clip(shared, 1e-09, 1 - 1e-09)
            mats['Q08'][:, yes] = shared
            mats['Q08'][:, no] = 1 - shared
            for n in ('Q09', 'Q10'):
                value = mats[n][:, :-1]
                mass = value.sum(axis=1, keepdims=True)
                value = np.divide(value, mass, out=np.full_like(value, 1 / value.shape[1]), where=mass > 0)
                mats[n][:, :-1] = value * (1 - shared[:, None])
                mats[n][:, -1] = shared
            used.append('Q08_Q09_Q10_coherent')
    for n in names:
        rule = schema['items'][n].get('gate')
        if not rule or schema['items'][rule['parent']]['class'] != 'GIVEN':
            continue
        eligible = train[rule['parent']].isin(rule['observed_if']).to_numpy()
        correct = (train[n].to_numpy() == sentinel) == ~eligible
        if len(correct) < 100 or correct.mean() < 0.999:
            continue
        mass = query[rule['parent']].isin(rule['observed_if']).to_numpy()
        value = mats[n][:, :-1]
        total = value.sum(axis=1, keepdims=True)
        value = np.divide(value, total, out=np.full_like(value, 1 / value.shape[1]), where=total > 0)
        mats[n][:, :-1] = value * mass[:, None]
        mats[n][:, -1] = ~mass
        used.append(n + '_known_gate')
    return ([mats[n][r] for r in range(len(query)) for n in names], used)

