"""Original U08-blas1 settings, isolated after the unchanged v1 WB call."""
import random
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import u08_stack
import u08_shared

# Data-free discovery belongs to module import, outside predict.
_POOLS = threadpool_info()
LAST_STATUS = 'not_called'


def predict(frame, schema):
    global LAST_STATUS
    old = (torch.get_num_threads(), torch.get_float32_matmul_precision(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.are_deterministic_algorithms_enabled(),
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        random.seed(14721)
        np.random.seed(14721)
        torch.manual_seed(14721)
        torch.cuda.manual_seed_all(14721)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with threadpool_limits(limits=1):
            raw = u08_stack.predict(frame, schema)
        LAST_STATUS = u08_stack.LAST_DIAGNOSTICS.get('status', 'completed')
        if LAST_STATUS != 'completed':
            raise RuntimeError('U08 did not complete')
        return raw
    finally:
        threads, precision, matmul, cudnn, deterministic, cudnn_det, benchmark = old
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = cudnn_det
        torch.backends.cudnn.benchmark = benchmark
