"""Train the 3-class U-Net (bioseg.unet) on the official BBBC039 train split.

Meant to run on Kaggle (GPU); see notebooks/train_kaggle.ipynb. Locally only a CPU smoke
test is allowed:
    python scripts/train.py --epochs 1 --limit 4 --val-limit 2 --out runs/unet_smoke

- Device: CUDA if available, else CPU (override with --device).
- Seed: fixed (--seed), cuDNN deterministic.
- Model selection: AP@0.5:0.95 ("all" variant, bioseg.evaluation) on the val split,
  computed every --val-every epochs on full images after watershed post-processing.
  Test is never read.
- Outputs in --out: best.pt (best val), last.pt, log.csv (one row per epoch), config.json.

Training choices (documented in docs/notas_noche.md):
- Targets: background / interior / boundary (bioseg.unet.labels_to_classes). All
  reference objects are kept, including those < 25 px (Caicedo et al. dropped them for
  training; we keep the ground truth untouched, consistent with step 1).
- Loss: cross-entropy with class weights 1 / 1 / 10 (boundary weight as in Caicedo et al.).
- Augmentation: random 256x256 crops, flips, 90-degree rotations, mild intensity scaling.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from bioseg.evaluation import image_counts, stems_of, summarize
from bioseg.unet import UNet, labels_to_classes, normalize, predict_labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(data: Path, split: str, limit: int | None):
    stems = stems_of(data, split)[:limit]
    imgs = [tifffile.imread(data / "images" / f"{s}.tif") for s in stems]
    labs = [tifffile.imread(data / "labels" / f"{s}.tif") for s in stems]
    return stems, imgs, labs


class CropDataset(Dataset):
    """Random augmented crops; `crops_per_image` samples per image per epoch."""

    def __init__(self, imgs, labs, crop: int, crops_per_image: int):
        self.imgs = [normalize(i) for i in imgs]
        self.cls = [labels_to_classes(l) for l in labs]
        self.crop, self.n = crop, crops_per_image

    def __len__(self):
        return len(self.imgs) * self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(torch.randint(0, 2**31, ()).item())  # seeded via torch
        img, cls = self.imgs[idx // self.n], self.cls[idx // self.n]
        y0 = rng.integers(0, img.shape[0] - self.crop + 1)
        x0 = rng.integers(0, img.shape[1] - self.crop + 1)
        x = img[y0:y0 + self.crop, x0:x0 + self.crop]
        y = cls[y0:y0 + self.crop, x0:x0 + self.crop]
        k = rng.integers(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def validate(model, imgs, labs, device) -> dict:
    model.eval()
    counts = [image_counts(gt, predict_labels(model, img, device)) for img, gt in zip(imgs, labs)]
    return summarize(counts)["all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/unet"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-image", type=int, default=4)
    ap.add_argument("--base", type=int, default=32, help="channels of the first U-Net level")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--limit", type=int, default=None, help="use only N train images (smoke test)")
    ap.add_argument("--val-limit", type=int, default=None, help="use only N val images")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()} | {"device_used": str(device)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print("Dispositivo:", device)

    _, tr_imgs, tr_labs = load_split(args.data, "train", args.limit)
    _, va_imgs, va_labs = load_split(args.data, "val", args.val_limit)
    ds = CropDataset(tr_imgs, tr_labs, args.crop, args.crops_per_image)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        num_workers=args.workers, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(args.seed),
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 10.0], device=device))

    log, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
        sched.step()
        row = {"epoch": epoch, "train_loss": total / n, "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = validate(model, va_imgs, va_labs, device)
            row |= {f"val_{k}": v for k, v in val.items()}
            if val["AP@0.5:0.95"] > best:
                best = val["AP@0.5:0.95"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                            "args": config}, args.out / "best.pt")
                row["best"] = True
        log.append(row)
        pd.DataFrame(log).to_csv(args.out / "log.csv", index=False)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()), flush=True)

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": config},
               args.out / "last.pt")
    print(f"Mejor val AP@0.5:0.95 = {best:.3f}. Pesos en {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
