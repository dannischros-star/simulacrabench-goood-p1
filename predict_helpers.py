"""Shared in-memory encodings; no generators, IDs, roles or external data."""
import numpy as np
import softmax_expert as util


def metadata(train, schema):
    given = [n for n, r in schema['items'].items() if r['class'] == 'GIVEN']
    names = [n for n, r in schema['items'].items() if r['class'] == 'PREDICT']
    x, sizes = util.encode(train, schema, given)
    ys, widths, known = [], [], {}
    for name in names:
        oo = util.options(schema, name); widths.append(len(oo))
        lookup = {v: i for i, v in enumerate(oo)}
        y = np.array([lookup[v] for v in train[name]], dtype=np.int64); ys.append(y)
        rule = schema['items'][name].get('gate')
        if rule and schema['items'][rule['parent']]['class'] == 'GIVEN' and len(train) >= 100:
            opened = train[rule['parent']].isin(rule['observed_if']).to_numpy()
            if np.array_equal(y == len(oo) - 1, ~opened):
                known[name] = rule
    return given, names, x, sizes, np.column_stack(ys), widths, known


def open_masks(frame, names, known):
    return np.column_stack([frame[known[n]['parent']].isin(known[n]['observed_if']).to_numpy()
                            if n in known else np.ones(len(frame), bool) for n in names])


def correct_numpy(mats, frame, names, known):
    result = {}
    opened = open_masks(frame, names, known)
    for j, n in enumerate(names):
        p = np.maximum(np.asarray(mats[n], float), 0.)
        p = p / p.sum(axis=1, keepdims=True)
        if n in known:
            ordinary = p[:, :-1]
            z = ordinary.sum(axis=1, keepdims=True)
            ordinary = np.divide(ordinary, z, out=np.full_like(ordinary, 1 / ordinary.shape[1]), where=z > 0)
            p = np.column_stack([ordinary * opened[:, j, None], ~opened[:, j]])
        result[n] = p
    return result


def numpy_loss(mats, y, names, widths):
    rr = np.arange(len(y))
    return np.mean([-np.log(.001 + (1 - .001 * k) * mats[n][rr, y[:, j]])
                    for j, (n, k) in enumerate(zip(names, widths))], axis=0)
