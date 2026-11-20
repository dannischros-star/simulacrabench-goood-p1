# SimulacraBench P1 source

This archive contains the runnable source and dependency pins from the P1 Codabench submission. The original Final (Test) submission was `#1054016`, archive `F_selector_fea9e362eb.zip`, 102,932,430 bytes. Its local SHA-256 is `fea9e362ebd19f330a85efdfcd2f1ca0f7e7cb9296dc0a8668ceb0fd1d1a5524`. The Codabench receipt matches the original filename and byte count. The original submission was recorded by archive hash.

## Entry point and structure

| File or directory | Role |
| --- | --- |
| `main.py` | The harness imports `predict(frame, schema)`. It returns one vector per masked response in respondent and schema order. |
| `adaptive_stack.py`, `gpu_tabm_expert.py`, `foundation_expert.py` | UNICEF D231 stack, TabM training, and TabICL in-context classification. |
| `wb_shared_stack.py`, `wb_tabm_shared.py`, `f_selector.py`, `f_selection.py` | Frozen World Bank reference stack and OOF selection of optional candidates. `f_shared_wb.py` and `f_tabicl.py` implement those candidates. |
| `u08_entry.py`, `u08_stack.py`, `u08_shared.py`, `unhcr_stack.py` | UNHCR U08-blas1 and the U03 exception fallback. |
| `tabicl/`, `sources/tabm.py` | Vendored TabICL and TabM implementations with their upstream license notices. |
| `requirements.txt`, `MODEL_PROVENANCE.json` | Original runtime dependency pins and checkpoint identity. |

## Checkpoint setup

| Field | Value |
| --- | --- |
| Hugging Face repository | `jingang/TabICL` |
| Repository revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Filename | `tabicl-classifier-v2-20260212.ckpt` |
| Destination relative to `main.py` | `weights/tabicl-classifier-v2-20260212.ckpt` |
| Bytes | `110368038` |
| SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| License | BSD-3-Clause |

Download the checkpoint before running the harness. From the extracted directory, run the following commands in a POSIX shell. [Pinned download](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt)

```sh
mkdir -p weights
curl --fail --location \
  'https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt' \
  --output weights/tabicl-classifier-v2-20260212.ckpt
python - <<'PY'
import hashlib
from pathlib import Path

weight = Path('weights/tabicl-classifier-v2-20260212.ckpt')
expected = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
with weight.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != expected:
    raise SystemExit(f'Checkpoint checksum mismatch: {actual}')
print(actual)
PY
```

UNICEF invokes TabICL for the GPU OOF expert and for the full-data fit when that expert has positive selected weight. World Bank can use the same checkpoint for optional per-item experts with OOF-selected weights. UNHCR uses no pretrained checkpoint. TabM trains on the visible labeled rows in each bank. The two TabICL callers open the local checkpoint with automatic downloads disabled.

To rebuild a runnable ZIP, restore the checkpoint in a clean extraction and run this from the extracted directory.

```
python -c "from shutil import make_archive; make_archive('../p1-runnable', 'zip', '.')"
```

This produces `p1-runnable.zip` beside the extraction. The archived source and checkpoint are the inputs for reproducing predictions. The original submitted ZIP has the separately recorded digest above.

## Run with the SimulacraBench starter kit

Use Python 3.12 and a CUDA-capable PyTorch environment for the GPU paths. The original dependency versions are pinned in `requirements.txt`. For direct calls from your own harness, install them with `python -m pip install -r requirements.txt` and import `main.predict`. The official grader creates its own per-submission environment and installs these pins during setup.

Extract this archive as `p1/`. In its parent directory, clone the starter kit and run a synthetic Final-phase rehearsal. A full-scale GPU run needs a suitable CUDA environment. The examples use the starter version whose documented CLI was checked. [Official starter kit](https://github.com/SituatedEvals/public/tree/5c69e17b191e82fe867eea6daf06c8ba8feaf622)

```sh
git clone https://github.com/SituatedEvals/public.git smbench-starter
cd smbench-starter
git checkout 5c69e17b191e82fe867eea6daf06c8ba8feaf622
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
python score.py --submission ../p1 --data _sandbox/unicef \
  --schema data/unicef.json --phase 2
```

Use the matching schema under `data/` to rehearse another instrument. `--phase 2` makes TRAIN and DEV labels available and masks TEST. The same entry point runs with the organizer's private evaluation inputs to reproduce official scoring. Synthetic rehearsals check the interface and execution. Official scores require the organizer-held inputs.

## Licenses

The original submission code in this source bundle is provided under the PolyForm Noncommercial License 1.0.0 in `LICENSE`, which permits non-commercial research and reproduction. The bundled TabICL implementation retains BSD-3-Clause in `TABICL_LICENSE.txt`. TabM retains Apache-2.0 in `TABM_LICENSE.txt` and `sources/TABM_LICENSE`. The separate checkpoint retains its upstream BSD-3-Clause license.
