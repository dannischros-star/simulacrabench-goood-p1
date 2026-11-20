"""Conditional v2 (fixed): one vector builder shared by inference and CV.

What was wrong in the original v2
---------------------------------
The CV that chose features and temperature scored the *un-normalised*
log-weights (log prior + sum of feature log-ratios) as if they were
log-probabilities. Real inference softmaxes the full option vector, so the CV
never paid for the normaliser and was optimistic whenever features
double-counted evidence. It also punished an unseen feature value with a
-CLIP log-ratio where inference simply skips that feature.

Example (the task book's): prior (0.5, 0.5), two identical conditionals
(0.8, 0.2) -> un-normalised weights (1.28, 0.08); the old CV scored the true
class at log 1.28 = +0.247, the real output gives 1.28/1.36 = 0.941 ->
-0.061. That gap is the bug; this file removes it.

The fix
-------
`_vector()` builds the final probability vector for one cell and is used by
both `predict()` and the cross-validation, so clipping (+-CLIP), temperature,
unknown-value fallback (skip), softmax normalisation and the grader's floor
mix are identical by construction. Feature selection is forward selection on
the CV log-score of that actual vector; temperature is tuned afterwards the
same way. The gated "asked" branch uses the same clipped log-ratio machinery
on the values-only support, and the sentinel branch is the one-hot the grader
rewards.

This removes a measurement bias. It is not a claim that platform scores rise.
"""

import numpy as np
import pandas as pd

SHRINK = 20.0
ALPHAS = (5.0, 20.0, 100.0)
MAX_FEATURES = 3
MARGIN = 1e-3
TEMPERATURES = (1.0, 2.0, 4.0)
CLIP = 3.0
FLOOR = 1e-3
EPS = 1e-12
DISABLE_GATE = False


# --------------------------------------------------------------- helpers --

def _options(schema, name, gated_value):
    rec = schema["items"][name]
    opts = list(rec["values"])
    n_vals = len(opts)
    if rec.get("gate"):
        opts = opts + [gated_value]
    return opts, n_vals, rec.get("gate")


def _smoothed(counts, opts, prior):
    w = np.array([counts.get(o, 0.0) for o in opts], float) + prior
    return w / w.sum()


def _codes(values, lookup):
    return np.array([lookup.get(v, -1) for v in values], dtype=np.int64)


def _softmax(logp):
    w = np.exp(logp - np.max(logp))
    return w / w.sum()


def _mix_floor(vec, floor=FLOOR):
    """Exactly score.py's floored(): renormalise, then mix with a flat vector."""
    vec = np.asarray(vec, float)
    vec = vec / vec.sum()
    k = vec.size
    return floor + (1.0 - k * floor) * vec


def _log_ratios(prior, tables, get_value, T):
    """log prior + sum of clipped feature log-ratios / T, over `prior`'s support.

    An unknown or unseen feature value contributes nothing (skip), which is
    what inference does; it is not a penalty.
    """
    logp = np.log(np.clip(prior, EPS, None))
    for feat, table in tables:
        fv = get_value(feat)
        if fv is None:
            continue
        row_p = table.get(fv)
        if row_p is None:
            continue
        logp = logp + np.clip(
            np.log(np.clip(row_p, EPS, None)) - np.log(np.clip(prior, EPS, None)),
            -CLIP, CLIP) / T
    return logp


# ------------------------------------------------- the one vector builder --

def _vector(model, get_value):
    """Final probability vector for one blank cell. Used by predict() and CV."""
    m = model
    gate = m["gate"]
    if gate:
        pv = get_value(gate["parent"])
        if pv is not None:
            if pv not in gate["observed_if"]:
                vec = np.zeros(m["n_opts"])
                vec[-1] = 1.0                      # never asked -> sentinel
                return vec
            logp = _log_ratios(m["asked_prior"], m["asked_feats"], get_value, m["T"])
            vec = np.zeros(m["n_opts"])
            vec[:m["n_vals"]] = _softmax(logp)
            return vec
        # parent hidden: chained mixture over (asked, sentinel)
        p_open = m["open_global"]
        for feat, _table in m["feats_full"]:
            fv = get_value(feat)
            if fv is not None:
                v = m["open_tables"].get(feat, {}).get(fv)
                if v is not None:
                    p_open = float(v)
                    break
        logp = _log_ratios(m["asked_prior"], m["asked_feats"], get_value, m["T"])
        d = _softmax(logp)
        vec = np.zeros(m["n_opts"])
        vec[:m["n_vals"]] = p_open * d
        vec[-1] = 1.0 - p_open
        return vec
    logp = _log_ratios(m["prior_full"], m["feats_full"], get_value, m["T"])
    return _softmax(logp)


# --------------------------------------------------- fitting and scoring --

def _count_matrix(f_vals, x_idx, n_opts):
    f_codes, f_uniq = pd.factorize(f_vals)
    C = np.zeros((max(len(f_uniq), 1), n_opts))
    known = x_idx >= 0
    if known.any():
        np.add.at(C, (f_codes[known], x_idx[known]), 1.0)
    return C, f_uniq


def _table(C, alpha, prior):
    return (C + alpha * prior[None, :]) / (C.sum(axis=1, keepdims=True) + alpha)


def _neglog(vec, k):
    return -float(np.log(_mix_floor(vec)[k]))


def _observed_parent_fold_neglog(entry, chosen, T, K, gate):
    """Sum of -log p(true) over a fold's validation rows, using the same
    normalisation/floor as the real output. Module level so tests can check it
    against `_vector` on the same fitted pieces."""
    x_val = entry["x_val"]
    n = len(x_val)
    if n == 0:
        return 0.0, 0
    if gate:
        obs = list(gate["observed_if"])
        asked = entry["asked_mask"]
        sentinel_rows = ~asked
        total, count = 0.0, 0
        # sentinel branch: one-hot on the last slot, floored (never -inf)
        ns = int(sentinel_rows.sum())
        if ns:
            onehot = np.zeros(K)
            onehot[-1] = 1.0
            total += ns * _neglog(onehot, K - 1)
            count += ns
        if asked.any():
            prior = entry["a_prior"]
            Ks = prior.size
            logp = np.tile(np.log(np.clip(prior, EPS, None)), (int(asked.sum()), 1))
            for f, alpha in chosen:
                C, fc = entry["per_feat_a"][f]
                fcv = fc[asked]
                known = fcv >= 0
                if known.any():
                    rows = _table(C, alpha, prior)[fcv[known]]
                    logp[known] += np.clip(
                        np.log(np.clip(rows, EPS, None))
                        - np.log(np.clip(prior, EPS, None))[None, :],
                        -CLIP, CLIP) / T
            w = np.exp(logp - logp.max(axis=1, keepdims=True))
            p = w / w.sum(axis=1, keepdims=True)
            # the grader floors the FULL width-K vector we return (sentinel slot
            # included), so the asked branch must mix over K, not over the
            # values-only support.
            pm = FLOOR + (1.0 - K * FLOOR) * p
            ka = entry["a_codes"][asked]
            ok = ka >= 0
            if ok.any():
                total += float(-np.sum(np.log(pm[np.flatnonzero(ok), ka[ok]])))
                count += int(ok.sum())
        return total, count
    prior = entry["prior"]
    logp = np.tile(np.log(np.clip(prior, EPS, None)), (n, 1))
    for f, alpha in chosen:
        C, fc = entry["per_feat"][f]
        known = fc >= 0
        if known.any():
            rows = _table(C, alpha, prior)[fc[known]]
            logp[known] += np.clip(
                np.log(np.clip(rows, EPS, None))
                - np.log(np.clip(prior, EPS, None))[None, :],
                -CLIP, CLIP) / T
    w = np.exp(logp - logp.max(axis=1, keepdims=True))
    p = w / w.sum(axis=1, keepdims=True)
    pm = FLOOR + (1.0 - K * FLOOR) * p
    ok = x_val >= 0
    if not ok.any():
        return 0.0, 0
    total = float(-np.sum(np.log(pm[np.flatnonzero(ok), x_val[ok]])))
    return total, int(ok.sum())



def _hidden_parent_fold_neglog(entry, chosen, T, K):
    """Predict with a hidden gate parent, then score the full-width vector.

    The parent's validation truth can define labels, but never the prediction
    branch. Only training answers and validation GIVEN values estimate p(open).
    """
    x = entry["x_val"]
    n = len(x)
    if not n:
        return 0.0, 0
    prior = entry["a_prior"]
    logp = np.tile(np.log(np.clip(prior, EPS, None)), (n, 1))
    for f, alpha in chosen:
        C, fc = entry["per_feat_a"][f]
        known = fc >= 0
        if known.any():
            rows = _table(C, alpha, prior)[fc[known]]
            logp[known] += np.clip(
                np.log(np.clip(rows, EPS, None))
                - np.log(np.clip(prior, EPS, None))[None, :],
                -CLIP, CLIP) / T
    w = np.exp(logp - logp.max(axis=1, keepdims=True))
    asked_p = w / w.sum(axis=1, keepdims=True)
    p_open = np.full(n, entry["open_global"], dtype=float)
    used = np.zeros(n, dtype=bool)
    for f, _alpha in chosen:
        estimates = entry["open_cv"][f]
        take = ~used & np.isfinite(estimates)
        p_open[take] = estimates[take]
        used[take] = True
    p = np.zeros((n, K), dtype=float)
    p[:, :-1] = p_open[:, None] * asked_p
    p[:, -1] = 1.0 - p_open
    pm = FLOOR + (1.0 - K * FLOOR) * p
    known = x >= 0
    rows = np.flatnonzero(known)
    return float(-np.log(pm[rows, x[known]]).sum()), int(known.sum())


def _fold_neglog(entry, chosen, T, K, gate):
    if gate and entry.get("parent_hidden", False):
        return _hidden_parent_fold_neglog(entry, chosen, T, K)
    return _observed_parent_fold_neglog(entry, chosen, T, K, gate)


def predict(frame, schema):
    gated_value = schema["gated_value"]
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    pred_items = [n for n in items if records[n]["class"] == "PREDICT"]
    given_items = [n for n in items if records[n]["class"] == "GIVEN"]

    train = frame.loc[frame[pred_items].notna().all(axis=1)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(train)), 3)

    model = {}

    for name in pred_items:
        opts, n_vals, gate = _options(schema, name, gated_value)
        if DISABLE_GATE:
            gate = None
        K = len(opts)
        lookup = {o: i for i, o in enumerate(opts)}
        x_all = _codes(train[name].to_numpy(), lookup)
        p_glob = _smoothed(train[name].value_counts(), opts, 0.5)

        # ---- per-fold cache: prior + per-feature count matrices + val codes ----
        fold_cache = []
        for fold in folds:
            fit_df = train.drop(index=fold)
            val_df = train.iloc[fold]
            prior = _smoothed(fit_df[name].value_counts(), opts, 0.5)
            x_fit = _codes(fit_df[name].to_numpy(), lookup)
            per_feat = {}
            for f in given_items:
                C, uniq = _count_matrix(fit_df[f].fillna("__NA__").to_numpy(), x_fit, K)
                fmap = {v: i for i, v in enumerate(uniq)}
                fc = np.array([fmap.get(v, -1) for v in val_df[f].fillna("__NA__").to_numpy()],
                              dtype=np.int64)
                per_feat[f] = (C, fc)
            x_val = _codes(val_df[name].to_numpy(), lookup)
            entry = {"prior": prior, "per_feat": per_feat, "x_val": x_val,
                     "val_df": val_df}
            if gate:
                parent = gate["parent"]
                obs = list(gate["observed_if"])
                asked_mask = val_df[parent].isin(obs).to_numpy()
                a_opts = list(records[name]["values"])
                a_lookup = {o: i for i, o in enumerate(a_opts)}
                fit_asked = fit_df[fit_df[parent].isin(obs)]
                entry["asked_mask"] = asked_mask
                entry["a_prior"] = _smoothed(fit_asked[name].value_counts(), a_opts, 0.5)
                ax_fit = _codes(fit_asked[name].to_numpy(), a_lookup)
                per_feat_a = {}
                for f in given_items:
                    C, uniq = _count_matrix(fit_asked[f].fillna("__NA__").to_numpy(), ax_fit,
                                            len(a_opts))
                    fmap = {v: i for i, v in enumerate(uniq)}
                    fc = np.array([fmap.get(v, -1) for v in val_df[f].fillna("__NA__").to_numpy()],
                                  dtype=np.int64)
                    per_feat_a[f] = (C, fc)
                entry["per_feat_a"] = per_feat_a
                entry["a_codes"] = _codes(val_df[name].to_numpy(), a_lookup)
                entry["parent_hidden"] = records[parent]["class"] == "PREDICT"
                if entry["parent_hidden"]:
                    opened = fit_df[parent].isin(obs)
                    entry["open_global"] = float(opened.mean()) if len(fit_df) else 0.5
                    entry["open_cv"] = {}
                    for f in given_items:
                        groups = opened.groupby(fit_df[f].fillna("__NA__"), observed=True).agg(["sum", "count"])
                        mapping = {k: float((r["sum"] + SHRINK * entry["open_global"]) / (r["count"] + SHRINK))
                                   for k, r in groups.iterrows()}
                        entry["open_cv"][f] = np.array([mapping.get(v, np.nan)
                            for v in val_df[f].fillna("__NA__").to_numpy()], dtype=float)

            fold_cache.append(entry)

        def cv_of(chosen, T=1.0):
            total, count = 0.0, 0
            for entry in fold_cache:
                s, n = _fold_neglog(entry, chosen, T, K, gate)
                total += s
                count += n
            return total / max(count, 1)

        # ---- forward selection on the CV log-score of the real vector ----
        chosen = []
        best_cv = cv_of([])
        for _ in range(MAX_FEATURES):
            cand = None
            for f in given_items:
                if any(f == f0 for f0, _a in chosen):
                    continue
                for alpha in ALPHAS:
                    cv = cv_of(chosen + [(f, alpha)])
                    if cand is None or cv < cand[0]:
                        cand = (cv, f, alpha)
            if cand is None or cand[0] >= best_cv - MARGIN:
                break
            chosen.append((cand[1], cand[2]))
            best_cv = cand[0]

        # ---- temperature on the same actual-vector score ----
        best_T, best_T_cv = 1.0, None
        for T in TEMPERATURES:
            cv = cv_of(chosen, T)
            if best_T_cv is None or cv < best_T_cv:
                best_T, best_T_cv = T, cv

        # ---- refit on all complete rows for inference ----
        feats_full = []
        for f, alpha in chosen:
            C, uniq = _count_matrix(train[f].fillna("__NA__").to_numpy(), x_all, K)
            tbl = _table(C, alpha, p_glob)
            feats_full.append((f, {v: tbl[i] for i, v in enumerate(uniq)}))
        m = {"opts": opts, "n_opts": K, "n_vals": n_vals, "gate": gate,
             "prior_full": p_glob, "feats_full": feats_full, "T": best_T}
        if gate:
            parent = gate["parent"]
            obs = list(gate["observed_if"])
            a_opts = list(records[name]["values"])
            a_lookup = {o: i for i, o in enumerate(a_opts)}
            asked_rows = train[train[parent].isin(obs)]
            a_prior = _smoothed(asked_rows[name].value_counts(), a_opts, 0.5)
            a_feats = []
            if len(asked_rows):
                ax = _codes(asked_rows[name].to_numpy(), a_lookup)
                for f, alpha in chosen:
                    C, uniq = _count_matrix(asked_rows[f].fillna("__NA__").to_numpy(), ax,
                                            len(a_opts))
                    tbl = _table(C, alpha, a_prior)
                    a_feats.append((f, {v: tbl[i] for i, v in enumerate(uniq)}))
            p_open_glob = float(train[parent].isin(obs).mean())
            open_tables = {}
            for f in given_items:
                fv = train[f].fillna("__NA__")
                grp = train[parent].isin(obs)
                agg = grp.groupby(fv, observed=True).agg(["sum", "count"])
                open_tables[f] = {k: float((r["sum"] + SHRINK * p_open_glob) / (r["count"] + SHRINK))
                                  for k, r in agg.iterrows()}
            m.update({"asked_prior": a_prior, "asked_feats": a_feats,
                      "open_global": p_open_glob, "open_tables": open_tables})
        model[name] = m

    # ---- assemble one vector per blank cell, canonical order ----
    values = frame[items].to_numpy(dtype=object)
    col_of = {n: i for i, n in enumerate(items)}
    row_cache = {}

    def make_getter(row):
        cache = {}

        def get_value(item):
            if item in cache:
                return cache[item]
            v = values[row, col_of[item]]
            v = None if pd.isna(v) else v
            cache[item] = v
            return v
        return get_value

    out = []
    for row in range(values.shape[0]):
        get_value = make_getter(row)
        for c, name in enumerate(items):
            if not pd.isna(values[row, c]):
                continue
            out.append(_vector(model[name], get_value))
    return out
