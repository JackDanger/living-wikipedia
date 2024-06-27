import torch
import os
import tqdm
from model import m, train_data, val_data, itos, config, load_checkpoint, save_checkpoint, optimizer

## Training
eval_iters = 200
eval_interval = 200
sample_interval = 50
sample_length = 500
iters_per_epoch = len(train_data)
epochs = 10
max_iters = iters_per_epoch * epochs


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in [train_data, val_data]:

        losses = torch.zeros(eval_iters)
        for k, (xb, yb) in enumerate(split):
            if k >= eval_iters:
                break

            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



best_val_loss = float('inf')
checkpoint = load_checkpoint()
if checkpoint is not None and 'iter' in checkpoint and checkpoint['iter'] > 0:
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
    iter_start = checkpoint['iter']
else:
    iter_start = 0


progress = tqdm.tqdm(range(iter_start, max_iters), total=max_iters, desc='training      ')
for iter in progress:
    if iter > 0 and iter % eval_interval == 0:
        loss = estimate_loss()
        if loss['val'] < best_val_loss:
            best_val_loss = loss['val']
            print(f"New best val loss {loss['val']}")
            save_checkpoint({
                    'config': config,
                    'iter': iter,
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'model': m.state_dict(),
                })
        else:
            print(f"val loss {loss['val']} is no better than best of {best_val_loss}")

    if iter > 0 and iter % sample_interval == 0:
        count = 0
        for token in m.generate():
            count += 1
            print(itos[token], end='', flush=True)
            if count > sample_length:
                break

    _, batch = next(enumerate(train_data))
    xb, yb = batch
    logits, loss = m(xb, yb)
    lossf = "{:.2f}".format(loss.mean())
    progress.set_description(f"training loss: {lossf}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()