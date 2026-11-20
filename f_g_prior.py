"""Optional G-lane interface. Absence and incomplete items add no candidate."""
try:
    from g_prior_expert import fit_predict as _expert
except Exception:
    _expert = None


def available():
    return _expert is not None


def fit_predict(train, query, schema, target_names, deadline):
    if _expert is None:
        return {}
    return _expert(train, query, schema, target_names=target_names, deadline=deadline)
