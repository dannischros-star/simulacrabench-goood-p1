"""One in-process wall clock; no platform-private environment variables."""
import time

START = time.monotonic()
FINAL = None


class BudgetExpired(Exception):
    pass


def configure(frame, schema):
    global FINAL
    names = [n for n, r in schema['items'].items() if r['class'] == 'PREDICT']
    visible = int(frame[names].notna().all(axis=1).sum()) if names else 0
    split = schema.get('split', {})
    final = visible > int(split.get('n_train', visible))
    if FINAL is None:
        FINAL = final
    return bool(FINAL)


def deadline(kind):
    # Reserve the measured baseline cost of the instruments still to come.
    offsets = ({'unicef': 820., 'world_bank': 1230., 'unhcr': 1740.}
               if FINAL else {'unicef': 390., 'world_bank': 535., 'unhcr': 740.})
    return START + offsets.get(kind, 0.)


def require(end, reserve=0.):
    if time.monotonic() + reserve >= end:
        raise BudgetExpired()
