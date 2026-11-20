"""Minimal conditional submission: gate rules + one best GIVEN feature.

Two levers over the crowd marginal (baseline/marginal_counts), both legal
because everything is fitted in-memory from the visible frame inside predict():

1. Gate rules. For a gated item whose parent cell is visible, the answer is
   forced: parent not in observed_if -> the respondent was never asked, so the
   truth is the sentinel (one-hot). Parent in observed_if -> the sentinel slot
   is impossible; predict the asked-population marginal instead.
2. Single-feature conditioning. For every other blank cell, the marginal is
   swapped for a distribution conditioned on the GIVEN feature that helped
   most on a held-out slice of the visible rows (Dirichlet-style shrinkage
   toward the global marginal, alpha = 20). An item keeps the plain marginal
   when no feature beat it out of sample.

Feature choice is picked per item on an 80/20 split of the complete rows
(rows whose PREDICT cells are all visible -- those are the TRAIN respondents);
the chosen tables are then refit on all complete rows. Deterministic: seed 0,
no network, no files.
"""

import numpy as np
import pandas as pd

SHRINK = 20.0


def _options(schema, name, gated_value):
    rec = schema["items"][name]
    opts = list(rec["values"])
    if rec.get("gate"):
        opts.append(gated_value)
    return opts


def _smoothed(counts, opts, prior):
    weights = np.array([counts.get(o, 0.0) for o in opts], float) + prior
    return weights / weights.sum()


def predict(frame, schema):
    gated_value = schema["gated_value"]
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    pred_items = [n for n in items if records[n]["class"] == "PREDICT"]
    given_items = [n for n in items if records[n]["class"] == "GIVEN"]

    # Complete rows are the TRAIN respondents; they carry every answer.
    train_mask = frame[pred_items].notna().all(axis=1)
    train = frame.loc[train_mask]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train))
    n_fit = int(0.8 * len(train))
    fit_df = train.iloc[perm[:n_fit]]
    val_df = train.iloc[perm[n_fit:]]

    def val_logloss(p, col_opts, rows):
        lookup = {o: i for i, o in enumerate(col_opts)}
        idx = rows.map(lambda v: lookup.get(v, -1)).to_numpy()
        known = idx >= 0
        if not known.any():
            return 0.0
        return -float(np.mean(np.log(np.clip(p[idx[known]], 1e-12, None))))

    dist = {}          # item -> global marginal over its option vector
    cond = {}          # item -> {feature_value: distribution} or None
    feat_of = {}       # item -> chosen GIVEN feature or None
    asked = {}         # gated item -> marginal over values on asked rows
    sentinel_vec = {}  # gated item -> one-hot on the sentinel slot

    for name in pred_items:
        rec = records[name]
        opts = _options(schema, name, gated_value)
        gate = rec.get("gate")

        p_fit = _smoothed(fit_df[name].value_counts(), opts, 0.5)
        ll_global = val_logloss(p_fit, opts, val_df[name])

        best_ll, best_f = ll_global, None
        for feat in given_items:
            f_fit_codes = fit_df[feat].fillna("__NA__").to_numpy()
            x_vals = fit_df[name].to_numpy()
            lookup = {o: i for i, o in enumerate(opts)}
            x_idx = np.array([lookup.get(v, -1) for v in x_vals])
            counts = {}
            for i, fval in enumerate(f_fit_codes):
                if x_idx[i] >= 0:
                    counts.setdefault(fval, np.zeros(len(opts)))[x_idx[i]] += 1
            v_lookup = {o: i for i, o in enumerate(opts)}
            v_idx = val_df[name].map(lambda v: v_lookup.get(v, -1)).to_numpy()
            f_vals = val_df[feat].fillna("__NA__").to_numpy()
            known = v_idx >= 0
            if not known.any():
                continue
            ll_sum, n_val = 0.0, int(known.sum())
            for fval in set(f_vals[known]):
                sel = known & (f_vals == fval)
                cnt = counts.get(fval)
                if cnt is None:
                    p_row = p_fit
                else:
                    p_row = (cnt + SHRINK * p_fit) / (cnt.sum() + SHRINK)
                ll_sum -= float(np.sum(np.log(
                    np.clip(p_row[v_idx[sel]], 1e-12, None))))
            ll = ll_sum / n_val
            if ll < best_ll - 1e-4:
                best_ll, best_f = ll, feat

        # Refit on all complete rows with the chosen structure.
        p_full = _smoothed(train[name].value_counts(), opts, 0.5)
        dist[name] = p_full
        feat_of[name] = best_f
        if best_f is not None:
            f_col = train[best_f].fillna("__NA__")
            lookup = {o: i for i, o in enumerate(opts)}
            x_idx = train[name].map(lambda v: lookup.get(v, -1)).to_numpy()
            table = {}
            for fval in f_col.unique():
                sel = (f_col == fval).to_numpy() & (x_idx >= 0)
                cnt = np.zeros(len(opts))
                np.add.at(cnt, x_idx[sel], 1.0)
                table[fval] = (cnt + SHRINK * p_full) / (cnt.sum() + SHRINK)
            cond[name] = table
        else:
            cond[name] = None

        if gate:
            width = len(opts)
            sv = np.zeros(width)
            sv[-1] = 1.0
            sentinel_vec[name] = sv
            parent = gate["parent"]
            obs_if = list(gate["observed_if"])
            asked_rows = train[train[parent].isin(obs_if)]
            a_opts = list(rec["values"])
            asked[name] = _smoothed(asked_rows[name].value_counts(), a_opts, 0.5)

    # ---- assemble one vector per blank cell, in canonical order ----
    values = frame[items].to_numpy(dtype=object)
    col_of = {n: i for i, n in enumerate(items)}
    out = []
    for row in range(values.shape[0]):
        for c, name in enumerate(items):
            if not pd.isna(values[row, c]):
                continue
            gate = records[name].get("gate")
            if gate:
                pv = values[row, col_of[gate["parent"]]]
                if not pd.isna(pv):
                    if pv in gate["observed_if"]:
                        vec = np.zeros(len(sentinel_vec[name]))
                        vec[:len(asked[name])] = asked[name]
                        out.append(vec)
                    else:
                        out.append(sentinel_vec[name])
                    continue
            table = cond[name]
            f = feat_of[name]
            if table is not None and f is not None:
                fv = values[row, col_of[f]]
                if not pd.isna(fv):
                    d = table.get(fv)
                    if d is not None:
                        out.append(d)
                        continue
            out.append(dist[name])
    return out
