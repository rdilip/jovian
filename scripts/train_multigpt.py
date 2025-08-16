"""barebones training script for gpt"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np
import wandb
import os
from functools import partial
from dataclasses import asdict

from jovian.models.multigpt import MultiGPT, MultiGPTConfig
import math

import argparse
from jovian.utils.utils import load_cfg


def get_lr_full(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def move(tensor, device):
    if "cuda" in device:
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def get_batch_pretrain(
    split, block_size, batch_size, device, data_dir
):
    # forward is 0, reverse is 1
    data = [
        np.memmap(
            os.path.join(data_dir, f"{split}_{mode}.bin"), dtype=np.uint16, mode="r"
        )
        for mode in ["forward", "reverse"]
    ]

    ix = torch.randint(0, len(data[0]) - block_size, (batch_size,))
    # asymmetric flip probability
    to_flip = (
        torch.rand(
            batch_size,
        )
        > 0.5
    ).long()
    x, y = [], []

    for m, i in enumerate(ix):
        tf = to_flip[m]
        data_mmap = data[tf]
        x.append(torch.from_numpy((data_mmap[i : i + block_size]).astype(np.int64)))
        y.append(
            torch.from_numpy((data_mmap[i + 1 : i + 1 + block_size]).astype(np.int64))
        )

    x = torch.stack(x)
    y = torch.stack(y)
    to_flip = to_flip.long()

    if "cuda" in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, to_flip, y = (
            x.pin_memory().to(device, non_blocking=True),
            to_flip.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, to_flip, y = x.to(device), to_flip.to(device), y.to(device)

    return x, to_flip, y


def get_batch_full(split, batch_size, pad, data_dir, device="cuda"):
    """
    start_indices: starting position of each element along memmap. You should PASS this because loading takes ~1ms
    ~ 1ms for batch_size = 16
    """
    data = [
        np.memmap(
            os.path.join(data_dir, f"{split}_{mode}.bin"), dtype=np.uint16, mode="r"
        )
        for mode in ["forward", "reverse"]
    ]
    start_indices = np.load(
        os.path.join(data_dir, f"{split}.starts.npy"), mmap_mode="r+"
    )
    idx = np.random.randint(0, len(start_indices) - 1, size=(batch_size,))
    r, c = start_indices[idx], start_indices[idx + 1]

    L = max(len(data[0][r[m] : c[m]]) for m in range(batch_size))

    X_for, X_rev, y_for, y_rev = torch.full(
        (4 * batch_size, L - 1), dtype=torch.long, fill_value=pad
    ).chunk(4, dim=0)
    dir_for = torch.zeros((batch_size,), dtype=torch.long)
    dir_rev = torch.ones((batch_size,), dtype=torch.long)

    for m in range(batch_size):
        forward_tokens = data[0][r[m] : c[m]]
        reverse_tokens = data[1][r[m] : c[m]]

        # fill forward half
        X_for[m, : len(forward_tokens) - 1] = torch.from_numpy(
            forward_tokens[:-1].copy()
        )
        y_for[m, : len(forward_tokens) - 1] = torch.from_numpy(
            forward_tokens[1:].copy()
        )

        # fill reverse half
        X_rev[m, : len(reverse_tokens) - 1] = torch.from_numpy(
            reverse_tokens[:-1].copy()
        )
        y_rev[m, : len(reverse_tokens) - 1] = torch.from_numpy(
            reverse_tokens[1:].copy()
        )

    return {
        "forward": (move(X_for, device), move(dir_for, device), move(y_for, device)),
        "reverse": (move(X_rev, device), move(dir_rev, device), move(y_rev, device)),
    }

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_default_dtype(torch.float32)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None,
                        help="YAML/JSON file with experiment overrides")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra KEY=VALUE pairs to override after YAML")
    args = parser.parse_args()

    # first load defaults, then YAML, then manual key=val tweaks
    cfg = load_cfg(args.cfg)
    for pair in args.override:
        k, v = pair.split("=", 1)
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown override key {k}")
        try:          
            setattr(cfg, k, eval(v))
        except Exception:
            setattr(cfg, k, v)

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    master_process = rank == 0
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    torch.manual_seed(1137 + rank)

    if master_process:
        wandb.init(
            project="disco",
            config=asdict(cfg),
            mode="disabled" if not cfg.use_wandb else "online",
        )
        run_name = wandb.run.name
        wandb.run.name = f"{run_name}__{cfg.descr}__{cfg.host}"
        Path(run_name).mkdir(exist_ok=True, parents=True)


    n_train = len(np.load(os.path.join(cfg.pth_to_tokens, f"train.starts.npy"), mmap_mode="r+"))
    n_val = len(np.load(os.path.join(cfg.pth_to_tokens, f"val.starts.npy"), mmap_mode="r+"))
    if master_process:
        print(f"Train size: {n_train}, Val size: {n_val}")

    get_batch = partial(
        get_batch_pretrain,
        data_dir=cfg.pth_to_tokens,
        block_size=512,
        device=device,
        batch_size=cfg.batch_size,
    )

    get_batch_from_start = partial(
        get_batch_full,
        data_dir=cfg.pth_to_tokens,
        pad=cfg.PAD,
        device=device,
        batch_size=cfg.batch_size,
    )

    ############### model setup #####################3
    model_cfg = MultiGPTConfig(
        vocab_size=cfg.vocab_size + 3,
        bos=cfg.BOS,
        eos=cfg.EOS,
        pad=cfg.PAD,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        block_size=514,
        dilation=cfg.dilation,
    )
    model = MultiGPT(model_cfg).to(device)

    if master_process:
        print(f"Number of parameters: {model.num_params():_}")
    if ddp:
        model = DDP(model, device_ids=[rank])

    ################## train loop #######
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    get_lr = partial(
        get_lr_full,
        warmup_iters=cfg.warmup_iters,
        learning_rate=cfg.lr,
        lr_decay_iters=cfg.lr_decay_iters,
        min_lr=cfg.min_lr,
    )
    it = 0
    batch = get_batch("train")
    raw_model = model if not ddp else model.module
    best_val_loss = float("inf")

    while True:
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % cfg.eval_every == 0 and master_process:
            model.eval()
            to_log = {
                "train/loss_forward": [],
                "val/loss_forward": [],
                "train/loss_reverse": [],
                "val/loss_reverse": [],
                "train/acc_forward": [],
                "val/acc_forward": [],
                "train/acc_reverse": [],
                "val/acc_reverse": [],
            }

            with torch.no_grad():
                for split in ["train", "val"]:
                    for m in range(cfg.n_eval_batches):
                        eval_batch = get_batch_from_start(split)
                        for mode in ["forward", "reverse"]:
                            logits, loss = model(*eval_batch[mode])
                            tgt_tokens = eval_batch[mode][-1]
                            pred = logits.argmax(-1).view(tgt_tokens.shape)
                            mask = tgt_tokens != cfg.PAD
                            acc = ((pred == tgt_tokens) * mask).sum() / mask.sum()

                            to_log[f"{split}/loss_{mode}"].append(loss.item())
                            to_log[f"{split}/acc_{mode}"].append(acc.item())

                    for mode in ["forward", "reverse"]:
                        to_log[f"{split}/loss_{mode}"] = np.mean(
                            to_log[f"{split}/loss_{mode}"]
                        )
                        to_log[f"{split}/acc_{mode}"] = np.mean(
                            to_log[f"{split}/acc_{mode}"]
                        )

                wandb.log(to_log)

            print(
                f"Evaluation: [Test forward]: {to_log['val/loss_forward']:.6f}, [Train forward]: {to_log['train/loss_forward']:.6f} "
                f"Evaluation: [Test reverse]: {to_log['val/loss_reverse']:.6f}, [Train reverse]: {to_log['train/loss_reverse']:.6f}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(*batch)
        batch = get_batch("train", batch_size=cfg.batch_size, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if master_process:
            wandb.log(
                {
                    "train/xent": loss.item(),
                    "train/epoch": it // (n_train / cfg.batch_size),
                },
                commit=False,
            )

        # Saving: keep separate from everything else
        # evals every and saves 2 checkpoints every 2000 iterations. 
        # every 5000 save to file. this is so we can get designability plot.
        if (it % cfg.save_every == 0) and master_process and ("dummy" not in run_name):
            avg_val_loss = 0.5 * (
                to_log["val/loss_forward"] + to_log["val/loss_reverse"]
            )
            ckpt = {
                "it": it,
                "model_state_dict": raw_model.state_dict(),
                "test_loss": avg_val_loss,
                "model_cfg": asdict(model_cfg),
                "cfg_": asdict(cfg),
                "optimizer": optimizer.state_dict()
            }
            save_best = best_val_loss > avg_val_loss
            best_val_loss = min(best_val_loss, avg_val_loss)
            if save_best:
                torch.save(ckpt, f"{run_name}/best_model.pt")
            torch.save(ckpt, f'{run_name}/latest_model.pt')
        
        if (it % 5_000 == 0) and master_process and ("dummy" not in run_name):
            torch.save(ckpt, f"{run_name}/model_{it}.pt")

        it += 1
