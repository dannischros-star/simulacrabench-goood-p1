"""Official TabM backbone, adapted here to twelve categorical survey heads.

Original backbone: yandex-research/tabm, Apache-2.0; retained with its license.
Our multi-task/gating wrapper is an adaptation, not an official TabM benchmark.
"""
import time
import numpy as np
import torch
from sources.tabm import TabM
import softmax_expert as util
from predict_helpers import metadata, open_masks, numpy_loss


def probabilities(logits, names, widths, known, opened):
    parts = {}; pos = 0
    for j, (name, k) in enumerate(zip(names, widths)):
        values = logits[:, :, pos:pos+k]; pos += k
        if name in known:
            ordinary = torch.softmax(values[:, :, :-1], dim=-1)
            mass = opened[:, j, None, None]
            p = torch.cat([ordinary * mass, (1 - mass).expand(-1, values.shape[1], 1)], dim=-1)
        else:
            p = torch.softmax(values, dim=-1)
        parts[name] = p
    return parts


def member_loss(parts, y, names, widths):
    rr = torch.arange(len(y),device=y.device); result = 0.
    for j, (n, k) in enumerate(zip(names, widths)):
        correct = parts[n][rr, :, y[:, j]]
        result = result - torch.log(.001 + (1 - .001 * k) * correct).mean() / len(names)
    return result


def make_net(sizes, widths, k):
    return TabM.make(n_num_features=0, cat_cardinalities=sizes, d_out=sum(widths),
                     k=k, n_blocks=3, d_block=128, dropout=.1)


def fit_net(X, Y, sizes, names, widths, known, opened, epochs, k, validation=None):
    torch.manual_seed(73201)
    model = make_net(sizes, widths, k).to("cuda")
    x = torch.as_tensor(X, dtype=torch.long,device="cuda"); y = torch.as_tensor(Y, dtype=torch.long,device="cuda")
    mask = torch.as_tensor(opened, dtype=torch.float32,device="cuda")
    bias = torch.tensor(np.concatenate([np.log(util.prior(Y[:,j], w)) for j,w in enumerate(widths)]), dtype=torch.float32,device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    order_rng = np.random.default_rng(73202)
    best, best_loss, history = 0, float('inf'), []
    for epoch in range(1, epochs+1):
        model.train()
        order = order_rng.permutation(len(X))
        for start in range(0, len(order), 256):
            batch = order[start:start+256]
            out = model(None, x[batch]) + bias
            loss = member_loss(probabilities(out, names, widths, known, mask[batch]), y[batch], names, widths)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        if validation is not None and (epoch == 1 or epoch % 5 == 0):
            vx, vy, vm = validation
            mats = predict_net(model, bias, vx, names, widths, known, vm)
            value = float(numpy_loss(mats, vy, names, widths).mean())
            history.append([epoch, value])
            if value < best_loss - 1e-5:
                best, best_loss = epoch, value
            elif epoch - best >= 25:
                break
    return model, bias, best, history


def predict_net(model, bias, x, names, widths, known, opened):
    result = {n:[] for n in names}; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xx = torch.as_tensor(x[start:start+512], dtype=torch.long,device="cuda")
            mm = torch.as_tensor(opened[start:start+512], dtype=torch.float32,device="cuda")
            parts = probabilities(model(None, xx) + bias, names, widths, known, mm)
            for n in names:
                # Independent losses during training; probability mean at inference.
                result[n].append(parts[n].mean(dim=1).cpu().numpy().astype(float))
    return {n:np.concatenate(p) for n,p in result.items()}


def fit_predict(train, query, schema, k=8):
    started = time.monotonic(); torch.set_num_threads(2)
    given, names, x, sizes, y, widths, known = metadata(train, schema)
    q, _ = util.encode(query, schema, given)
    opened = open_masks(train, names, known); query_open = open_masks(query, names, known)
    order = np.random.default_rng(73203).permutation(len(train)); cut = int(.8 * len(train))
    fit, valid = order[:cut], order[cut:]
    _, _, epochs, history = fit_net(x[fit], y[fit], sizes, names, widths, known, opened[fit], 120, k,
                                    validation=(x[valid], y[valid], opened[valid]))
    epochs = max(1, epochs)
    model, bias, _, _ = fit_net(x, y, sizes, names, widths, known, opened, epochs, k)
    out = predict_net(model, bias, q, names, widths, known, query_open)
    return out, {'seconds':time.monotonic() - started, 'ensemble_members':k, 'epochs':epochs,
                 'inner_validation':history, 'joint_targets':len(names), 'threads':2,
                 'training':'mean independent member losses', 'inference':'mean probabilities'}
