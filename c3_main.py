"""Final candidate: frozen D231 UNICEF, verified WB01, selected UNHCR."""
import main_d231_original as unicef_model
import baseline_v11 as baseline
import wb_shared_stack
import unhcr_stack
from huggingface_hub import hf_hub_download as _preloaded_hf_download

def predict(frame,schema):
    kind=baseline.identify(schema)
    if kind=='unicef':return unicef_model.predict(frame,schema)
    if kind=='world_bank':return wb_shared_stack.predict(frame,schema)
    if kind=='unhcr':return unhcr_stack.predict(frame,schema)
    return baseline.marginal(frame,schema)
