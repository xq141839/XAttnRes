"""
2D training script for XAttnRes on AMOS22.

Pipeline:
    PNG dataset (built by preprocess.py)
        -> Albumentations (HFlip/VFlip/Rotate/intensity tweaks)
        -> ImageNet-normalized (3, 256, 256) tensor
        -> XAttnRes U-Net with n_classes=16 head
        -> DiceCE loss
        -> AdamW + Poly LR
        -> per-epoch CSV with mean + per-class Dice
"""

import os
import time
import json
import csv
import math
import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete

from dataloader import (
    AMOS2DPNGDataset, load_splits, expand_cases_to_slices,
    build_train_transform, build_val_transform,
)
import XAttnResUnet

torch.set_num_threads(8)


# ======================================================================================
# Poly LR scheduler (nnU-Net's default — works well for segmentation)
# ======================================================================================
class PolyLRScheduler:
    def __init__(self, optimizer, initial_lr: float, max_epochs: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent

    def step(self, epoch: int):
        lr = self.initial_lr * (1 - epoch / self.max_epochs) ** self.exponent
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ======================================================================================
# CSV logger (per-class Dice columns)
# ======================================================================================
class EpochCSVLogger:
    def __init__(self, csv_path: str, class_names: Sequence[str]):
        self.csv_path = csv_path
        self.class_names = list(class_names)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            header = ['epoch', 'lr', 'train_loss', 'train_dice_mean']
            header += [f'train_dice_{n}' for n in self.class_names]
            header += ['val_loss', 'val_dice_mean']
            header += [f'val_dice_{n}' for n in self.class_names]
            header += ['time_sec']
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    def log(self, epoch, lr, tr_loss, tr_dice, tr_dice_pc,
            va_loss, va_dice, va_dice_pc, elapsed):
        def _fmt(xs):
            return ['nan' if (x is None or (isinstance(x, float) and math.isnan(x)))
                    else f'{x:.6f}' for x in xs]
        row = [epoch, f'{lr:.6e}', f'{tr_loss:.6f}', f'{tr_dice:.6f}']
        row += _fmt(tr_dice_pc)
        row += [f'{va_loss:.6f}', f'{va_dice:.6f}']
        row += _fmt(va_dice_pc)
        row += [f'{elapsed:.2f}']
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)


# ======================================================================================
# Single epoch
# ======================================================================================
def run_one_epoch(model, loader, criterion, dice_metric, optimizer,
                  device, post_pred, post_label, training: bool, local_rank: int, desc: str):
    """Returns (epoch_loss, mean_dice, per_class_dice_list)."""
    model.train() if training else model.eval()
    running_loss, steps = 0.0, 0
    pbar = tqdm(loader, desc=desc, disable=(local_rank != 0))
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img, lbl, _ in pbar:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)              # (B, H, W) long
            target_for_loss = lbl.unsqueeze(1)                   # (B, 1, H, W) for DiceLoss

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(img)                                    # (B, C, H, W)
            loss = criterion(pred, target_for_loss)

            if training:
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Warning: Loss contains NaN or Inf!")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()

            running_loss += loss.item()
            steps += 1

            with torch.no_grad():
                pred_bin = torch.stack([post_pred(p) for p in torch.softmax(pred, dim=1)])
                lbl_bin = torch.stack([post_label(t) for t in target_for_loss])
                dice_metric(y_pred=pred_bin, y=lbl_bin)

    epoch_loss = running_loss / max(steps, 1)
    dice_tensor = dice_metric.aggregate()
    if dice_tensor.ndim == 0:
        per_class = [dice_tensor.item()]
        mean_dice = dice_tensor.item()
    else:
        per_class = dice_tensor.detach().cpu().tolist()
        mean_dice = torch.nanmean(dice_tensor).item()
    dice_metric.reset()
    return epoch_loss, mean_dice, per_class


# ======================================================================================
# Main training routine
# ======================================================================================
def train(args, local_rank: int):
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # --------------- splits + manifest --------------
    train_ids, val_ids = load_splits(args.splits_json, args.fold)
    if local_rank == 0:
        print(f"Fold {args.fold}: {len(train_ids)} train cases, {len(val_ids)} val cases")

    preproc_split = Path(args.preproc_dir) / args.split_subdir
    manifest_path = preproc_split / "manifest.json"
    assert manifest_path.exists(), (
        f"Preprocessing manifest not found at {manifest_path}. "
        f"Did you run preprocess.py first?")
    with open(manifest_path) as f:
        manifest = json.load(f)

    train_samples = expand_cases_to_slices(train_ids, manifest)
    val_samples = expand_cases_to_slices(val_ids, manifest)
    if local_rank == 0:
        print(f"  → {len(train_samples)} train slices, {len(val_samples)} val slices")

    # --------------- datasets + loaders --------------
    train_ds = AMOS2DPNGDataset(
        preproc_split_dir=str(preproc_split),
        sample_ids=train_samples,
        transform=(build_val_transform(args.image_size) if args.no_aug
                   else build_train_transform(args.image_size)),
    )
    val_ds = AMOS2DPNGDataset(
        preproc_split_dir=str(preproc_split),
        sample_ids=val_samples,
        transform=build_val_transform(args.image_size),
    )

    if args.is_ddp:
        tr_sampler = DistributedSampler(train_ds, shuffle=True)
        va_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                                  pin_memory=True, sampler=tr_sampler, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers,
                                pin_memory=True, sampler=va_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                                  pin_memory=True, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers,
                                pin_memory=True)

    # --------------- model --------------
    model = XAttnResUnet.Model(n_classes=args.num_classes).to(device)

    # Sanity check — fail fast if the model head doesn't actually emit num_classes channels.
    if local_rank == 0:
        model.eval()
        with torch.no_grad():
            _x = torch.randn(2, 3, args.image_size, args.image_size, device=device)
            _y = model(_x)
        print(f"[sanity] model output: {tuple(_y.shape)}  "
              f"(expected (2, {args.num_classes}, {args.image_size}, {args.image_size}))")
        assert _y.shape[1] == args.num_classes
        _img, _lbl, _ = train_ds[0]
        print(f"[sanity] image: shape={tuple(_img.shape)}, range="
              f"[{_img.min():.3f}, {_img.max():.3f}]")
        print(f"[sanity] label: shape={tuple(_lbl.shape)}, unique="
              f"{torch.unique(_lbl).tolist()}")
        assert _lbl.min() >= 0 and _lbl.max() < args.num_classes

    if args.is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if local_rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Params = {n_train/1e6:.2f}M")

    # --------------- loss + metric + optim --------------
    _dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True,
                     reduction='mean', smooth_nr=1e-5, smooth_dr=1e-5)
    _ce = nn.CrossEntropyLoss()

    def criterion(pred_logits, target_bcw):
        d = _dice(pred_logits, target_bcw)
        c = _ce(pred_logits, target_bcw.squeeze(1).long())
        return d + c

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    post_label = AsDiscrete(to_onehot=args.num_classes)

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                              weight_decay=3e-5, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    scheduler = PolyLRScheduler(optimizer, initial_lr=args.lr, max_epochs=args.epoch)

    # --------------- logging --------------
    run_tag = f"{args.model_tag}_fold{args.fold}"
    ckpt_dir = Path(args.output_dir) / 'checkpoints'
    csv_dir = Path(args.output_dir) / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    foreground_names = [f"class_{i}" for i in range(1, args.num_classes)]
    ds_json = Path(args.dataset_dir) / 'dataset.json'
    if ds_json.exists():
        try:
            with open(ds_json) as f:
                meta = json.load(f)
            labels = meta.get('labels', {})
            id2name = {int(v): k for k, v in labels.items()}
            foreground_names = [
                id2name.get(i, f"class_{i}").replace(' ', '_').replace('/', '_').replace(',', '_')
                for i in range(1, args.num_classes)
            ]
        except Exception as e:
            if local_rank == 0:
                print(f"[warn] couldn't read class names from {ds_json}: {e}")
    csv_logger = (EpochCSVLogger(str(csv_dir / f"{run_tag}.csv"), foreground_names)
                  if local_rank == 0 else None)

    # --------------- epoch loop --------------
    best_dice = -1.0
    for epoch in range(args.epoch):
        epoch_start = time.time()
        if args.is_ddp:
            train_loader.sampler.set_epoch(epoch)
        lr_now = scheduler.step(epoch)

        if local_rank == 0:
            print(f"\nEpoch {epoch}/{args.epoch-1}   lr={lr_now:.6e}")
            print('-' * 40)

        tr_loss, tr_dice, tr_dice_pc = run_one_epoch(
            model, train_loader, criterion, dice_metric, optimizer,
            device, post_pred, post_label,
            training=True, local_rank=local_rank, desc='train',
        )
        va_loss, va_dice, va_dice_pc = run_one_epoch(
            model, val_loader, criterion, dice_metric, optimizer,
            device, post_pred, post_label,
            training=False, local_rank=local_rank, desc='valid',
        )

        if args.is_ddp:
            scalar = torch.tensor([tr_loss, tr_dice, va_loss, va_dice], device=device)
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            scalar /= dist.get_world_size()
            tr_loss, tr_dice, va_loss, va_dice = scalar.tolist()
            pc_t = torch.tensor(tr_dice_pc, device=device)
            pc_v = torch.tensor(va_dice_pc, device=device)
            dist.all_reduce(pc_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(pc_v, op=dist.ReduceOp.SUM)
            pc_t /= dist.get_world_size()
            pc_v /= dist.get_world_size()
            tr_dice_pc = pc_t.tolist()
            va_dice_pc = pc_v.tolist()

        elapsed = time.time() - epoch_start
        if local_rank == 0:
            print(f"train  loss={tr_loss:.4f}  mean_dice={tr_dice:.4f}")
            print(f"valid  loss={va_loss:.4f}  mean_dice={va_dice:.4f}   ({elapsed:.1f}s)")
            named = list(zip(foreground_names, va_dice_pc))
            named.sort(key=lambda kv: (float('nan') if kv[1] is None else kv[1]))
            def _fmt(p):
                n, v = p
                return f'{n}={"nan" if (v is None or v != v) else f"{v:.3f}"}'
            print("  val top3:   " + ', '.join(_fmt(p) for p in named[-3:][::-1]))
            print("  val bot3:   " + ', '.join(_fmt(p) for p in named[:3]))
            csv_logger.log(epoch, lr_now, tr_loss, tr_dice, tr_dice_pc,
                           va_loss, va_dice, va_dice_pc, elapsed)

            state = (model.module.state_dict()
                     if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict())
            torch.save(state, ckpt_dir / f"{run_tag}_latest.pth")
            if va_dice > best_dice:
                best_dice = va_dice
                torch.save(state, ckpt_dir / f"{run_tag}_best.pth")
                print(f"  ** new best val dice = {best_dice:.4f}, checkpoint saved. **")

    if local_rank == 0:
        print(f"\nDone. Best val dice = {best_dice:.4f}")


# ======================================================================================
# CLI
# ======================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Source dataset root (for reading dataset.json class names).')
    parser.add_argument('--preproc_dir', type=str, required=True,
                        help='Where preprocess.py wrote the PNGs.')
    parser.add_argument('--split_subdir', type=str, default='Tr',
                        help='Which preprocessed subset to use for train+val (Tr or Ts).')
    parser.add_argument('--splits_json', type=str, required=True,
                        help='Path to nnU-Net splits_final.json.')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)

    # training
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_aug', action='store_true',
                        help='Disable augmentation; use only resize + ImageNet normalize.')

    # outputs
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_tag', type=str, default='xattnres')

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.is_ddp = "WORLD_SIZE" in os.environ
    if args.is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    train(args, local_rank)

    if args.is_ddp:
        dist.destroy_process_group()