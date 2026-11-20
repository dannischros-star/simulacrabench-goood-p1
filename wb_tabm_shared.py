"""Shared TabM categorical heads for UNHCR; trained only on supplied TRAIN.

TabM backbone: yandex-research/tabm, Apache-2.0 (notice bundled).
All categories, including NA_GATED, are learned rather than hard projected.
"""
import time
import numpy as np
import softmax_expert as util


def metadata(train, schema):
    given = [n for n, r in schema['items'].items() if r['class'] == 'GIVEN']
    names = [n for n, r in schema['items'].items() if r['class'] == 'PREDICT']
    x, sizes = util.encode(train, schema, given)
    widths = [len(util.options(schema, n)) for n in names]
    y = np.column_stack([
        train[n].map({v:i for i,v in enumerate(util.options(schema,n))}).to_numpy(dtype=np.int64)
        for n in names])
    return given, names, x, sizes, y, widths


def fit_predict(train, query, schema, members=4):
    import torch
    from sources.tabm import TabM
    if not torch.cuda.is_available():
        raise RuntimeError('U01 shared expert requires the declared H100 GPU')
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    given, names, x, sizes, y, widths = metadata(train, schema)
    q, _ = util.encode(query, schema, given)
    device = 'cuda'
    width = max(widths)
    offset = np.r_[0, np.cumsum(widths)]
    index = np.zeros((len(names), width), np.int64)
    legal = np.zeros_like(index, dtype=bool)
    for j, k in enumerate(widths):
        index[j,:k] = np.arange(offset[j], offset[j+1])
        legal[j,:k] = True
    index = torch.as_tensor(index.ravel(), device=device)
    legal = torch.as_tensor(legal, device=device)
    floor_coef = torch.as_tensor(1 - .001*np.asarray(widths), dtype=torch.float32, device=device)

    def parts(logits):
        z = logits.index_select(-1, index).reshape(len(logits), members, len(names), width)
        return torch.softmax(z.masked_fill(~legal, -1e9), dim=-1)

    def loss(p, yy):
        truth = yy[:,None,:,None].expand(-1, p.shape[1], -1, 1)
        correct = p.gather(-1, truth).squeeze(-1)
        return -torch.log(.001 + floor_coef * correct).mean()

    tx = torch.as_tensor(x, dtype=torch.long, device=device)
    ty = torch.as_tensor(y, dtype=torch.long, device=device)
    order = np.random.default_rng(927031).permutation(len(x))
    cut = max(1, int(.8*len(x)))
    fit, valid = order[:cut], order[cut:]
    if not len(valid):
        valid = fit
    rng = np.random.default_rng(927032)

    def make(rows):
        torch.manual_seed(927030)
        net = TabM.make(n_num_features=0, cat_cardinalities=sizes,
                        d_out=sum(widths), k=members, n_blocks=3,
                        d_block=128, dropout=.1).to(device)
        bias = np.concatenate([np.log(util.prior(y[rows,j], k)) for j,k in enumerate(widths)])
        bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
        return net, bias, opt

    def epoch(net, bias, opt, rows):
        net.train()
        for start in range(0, len(rows), 256):
            batch = rows[start:start+256]
            opt.zero_grad(set_to_none=True)
            value = loss(parts(net(None, tx[batch])+bias), ty[batch])
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()

    def evaluate(net, bias, rows):
        net.eval()
        total = 0.
        with torch.no_grad():
            for start in range(0,len(rows),512):
                batch = rows[start:start+512]
                p = parts(net(None,tx[batch])+bias).mean(dim=1,keepdim=True)
                total += float(loss(p,ty[batch]))*len(batch)
        return total/max(1,len(rows))

    net, bias, opt = make(fit)
    best_epoch, best_loss, history = 1, float('inf'), []
    for e in range(1,61):
        epoch(net,bias,opt,rng.permutation(fit))
        if e == 1 or e % 3 == 0:
            value = evaluate(net,bias,valid)
            history.append([e,value])
            if value < best_loss - 1e-5:
                best_epoch,best_loss = e,value
            elif e-best_epoch >= 15:
                break
    del net,opt,bias
    net,bias,opt = make(np.arange(len(x)))
    for e in range(best_epoch):
        epoch(net,bias,opt,rng.permutation(len(x)))
    net.eval()
    outputs = {n:[] for n in names}
    with torch.no_grad():
        for start in range(0,len(q),512):
            inputs = torch.as_tensor(q[start:start+512],dtype=torch.long,device=device)
            p = parts(net(None,inputs)+bias).mean(dim=1).cpu().numpy()
            for j,(n,k) in enumerate(zip(names,widths)):
                outputs[n].append(p[:,j,:k].astype(float))
    out = {n:np.concatenate(v) for n,v in outputs.items()}
    del net,opt,bias,tx,ty
    torch.cuda.synchronize()
    return out, {'seconds':time.monotonic()-started,'members':members,
                 'targets':len(names),'selected_epoch':best_epoch,
                 'inner_validation':history,'hard_gates':False,
                 'fit_rows':len(train),'input_features':len(given)}

