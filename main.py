"""SMB-F: current package anchor with optional in-process OOF experts."""
import f_budget
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import c3_main
import f_selector
import u08_entry

def predict(frame, schema):
    if c3_main.baseline.identify(schema) == 'unhcr':
        try:
            return u08_entry.predict(frame, schema)
        except Exception:
            u08_entry.LAST_STATUS = 'fallback_to_U03'
            return c3_main.predict(frame, schema)
    try:
        return f_selector.predict(frame, schema, c3_main.predict)
    except Exception:
        return c3_main.baseline.marginal(frame, schema)
