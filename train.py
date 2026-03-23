"""
Training script for DitherNet.

Checkpoints are saved to checkpoints/best.pt and checkpoints/latest.pt.
Resume training from latest checkpoint by passing --resume.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DitherDataset
from dither_net import DitherNet
from losses import HVSLoss

# ── config ────────────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 16
LR         = 1e-4
CROP_SIZE  = 128
CKPT_DIR   = Path("checkpoints")
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, device, train: bool, hvs_loss=None, hvs_weight=0.0):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss = 0.0
    desc = "  train" if train else "  valid"

    with ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss = F.binary_cross_entropy_with_logits(logits, y)
            if hvs_loss is not None:
                loss = loss + hvs_weight * hvs_loss(logits, x)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pt")
    parser.add_argument("--hvs-weight", type=float, default=0.0,
                        help="Weight for HVSLoss added on top of BCE (0 = disabled)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    CKPT_DIR.mkdir(exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_ds = DitherDataset(
        split="train", crop_size=CROP_SIZE,
        random_crop=True, hflip=True, vflip=True, rotate=True,
    )
    valid_ds = DitherDataset(
        split="valid", crop_size=CROP_SIZE,
        random_crop=False, hflip=False, vflip=False, rotate=False,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin)
    print(f"Train: {len(train_ds):,} pairs   Valid: {len(valid_ds):,} pairs\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model = DitherNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DitherNet: {n_params:,} parameters\n")

    # ── optimiser / scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── HVS loss (optional) ───────────────────────────────────────────────────
    hvs_loss = HVSLoss().to(device) if args.hvs_weight > 0 else None
    if hvs_loss:
        print(f"HVSLoss enabled  (weight={args.hvs_weight})\n")

    # ── optionally resume ─────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")

    if args.resume:
        ckpt_path = CKPT_DIR / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt.get("best_val", float("inf"))
            print(f"Resumed from epoch {ckpt['epoch']}  (best val={best_val:.4f})\n")
        else:
            print("No checkpoint found, starting from scratch.\n")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS}   lr={lr_now:.2e}")

        tr_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                            hvs_loss=hvs_loss, hvs_weight=args.hvs_weight)
        vl_loss = run_epoch(model, valid_loader, optimizer, device, train=False,
                            hvs_loss=hvs_loss, hvs_weight=args.hvs_weight)

        scheduler.step(vl_loss)

        print(f"  train  loss={tr_loss:.4f}")
        print(f"  valid  loss={vl_loss:.4f}")

        # Save best
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": vl_loss}, CKPT_DIR / "best.pt")
            print(f"  → best checkpoint (val={vl_loss:.4f})")

        # Save latest (always, for resuming)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        vl_loss,
            "best_val":        best_val,
        }, CKPT_DIR / "latest.pt")

        print()


if __name__ == "__main__":
    main()
