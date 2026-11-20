"""Floor-aware convex mixing; held-out meta checks and an exact anchor option."""
import numpy as np
from scipy.optimize import minimize_scalar


def checked(p, rows, width):
    p = np.asarray(p, dtype=float)
    if p.shape != (rows, width) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid expert probabilities')
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError('Empty probability mass')
    return p / mass


def correct(p, y):
    return .001 + (1 - .001 * p.shape[1]) * p[np.arange(len(y)), y]


def coefficient(anchor, expert, y, folds, maximum=.35):
    """Select only with meta-held-out evidence; return zero on uncertainty."""
    a, b = correct(anchor, y), correct(expert, y)
    if len(y) < 90 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.

    def fit(rows):
        def objective(w):
            return -np.log((1-w)*a[rows]+w*b[rows]).mean() + .004*w*w
        result = minimize_scalar(objective, bounds=(0., maximum), method='bounded',
                                 options={'xatol': 1e-4, 'maxiter': 40})
        return float(result.x) if result.success and objective(result.x) < objective(0.) else 0.

    gains = np.zeros(len(y))
    for fold in np.unique(folds):
        learn, check = folds != fold, folds == fold
        w = fit(learn)
        gains[check] = np.log((1-w)*a[check]+w*b[check])-np.log(a[check])
    se = gains.std(ddof=1) / np.sqrt(len(gains))
    if gains.mean() <= .0002 + se:
        return 0.
    # Shrink even the accepted optimum toward the untouched current package.
    return .5 * fit(np.ones(len(y), dtype=bool))


def combine(anchor_oof, anchor_full, alternatives, y, folds):
    if not alternatives:
        return anchor_full
    # Each alternative is independently validated against the same anchor.
    weights = [coefficient(anchor_oof, p, y, folds) for p, _ in alternatives]
    total = sum(weights)
    if total <= 0:
        return anchor_full
    if total > .35:
        weights = [w * .35 / total for w in weights]
        total = .35
    out = (1-total)*anchor_full + sum(w*p for w, (_, p) in zip(weights, alternatives))
    return checked(out, len(anchor_full), anchor_full.shape[1])
