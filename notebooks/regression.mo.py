import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Header."""
    import marimo as _mo

    _mo.md(
        """
        # Stage 2/3: Heuristic Labels + Multitask Training
        - Generates `data/raw/labels.csv` from satellite + terrain tiles.
        - Trains multitask regression/classification model.
        """
    )


@app.cell
def _():
    """Config."""
    from dataclasses import dataclass

    @dataclass
    class Config:
        # Paths
        raw_dir: str = "data/raw"
        satellite_dir: str = "data/raw/images/satellite/z16"
        terrain_dir: str = "data/raw/images/terrain/z16"
        labels_csv: str = "data/raw/labels.csv"
        processed_dir: str = "data/processed"
        model_dir: str = "models"
        classifier_best_ckpt: str = "models/classifier/best_model.pt"
        classifier_use_resisc45_stats: bool = True

        # Heuristic labels
        heuristic_labels: bool = False
        heuristic_max_tiles: int = 2000
        heuristic_use_classifier: bool = True
        heuristic_default_lat: float = 0.0
        heuristic_default_lon: float = 0.0

        # Multitask training
        batch_size: int = 32
        lr: float = 1e-4
        epochs: int = 5
        num_workers: int = 0
        image_size: int = 224
        train_split: float = 0.70
        val_split: float = 0.15
        num_classes: int = 45
        loss_weights: tuple[float, float] = (1.0, 1.0)
        best_ckpt: str = "scenic_multitask_best.pt"
        run_log: str = "train_run.json"
        run_name: str = "baseline"
        resume_from: str = ""

        # Repro
        seed: int = 42

    cfg = Config()
    cfg
    return (cfg,)


@app.cell
def _():
    """Imports."""
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import timm
    from tqdm import tqdm
    import json
    from datetime import datetime
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    torch, nn, Dataset, DataLoader, pd, np, Path, Image, transforms, timm, tqdm, json, datetime, make_grid, to_pil_image
    return (
        DataLoader,
        Dataset,
        Image,
        Path,
        datetime,
        json,
        make_grid,
        nn,
        np,
        pd,
        timm,
        to_pil_image,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(Path, cfg, np, pd):
    """Heuristic label generation (optional)."""
    if not cfg.heuristic_labels:
        "Heuristic label generation disabled. Set cfg.heuristic_labels = True to run."
    else:
        import sys as _sys
        from PIL import Image as _Image

        _project_root = Path.cwd()
        if str(_project_root) not in _sys.path:
            _sys.path.insert(0, str(_project_root))

        _sat_dir = Path(cfg.satellite_dir)
        _terrain_dir = Path(cfg.terrain_dir)
        if not _sat_dir.exists():
            raise FileNotFoundError(f"Satellite dir not found: {_sat_dir}")
        if not _terrain_dir.exists():
            raise FileNotFoundError(f"Terrain dir not found: {_terrain_dir}")

        _sat_files = sorted(
            [p for p in _sat_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        _terrain_index = {
            p.stem: p
            for p in _terrain_dir.glob("**/*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

        if not _sat_files:
            raise ValueError(f"No satellite images found in: {_sat_dir}")

        _rows: list[dict] = []
        _max_tiles = max(1, cfg.heuristic_max_tiles)

        _classifier = None
        _clf_transform = None
        if cfg.heuristic_use_classifier:
            _ckpt = Path(cfg.classifier_best_ckpt)
            if _ckpt.exists():
                from torchvision import transforms as _tv
                from src.classifier.model import LandscapeClassifier, TERRAIN_CLASSES, get_scenic_weight
                from src.classifier.inference import RESISC45_MEAN, RESISC45_STD, IMAGENET_MEAN, IMAGENET_STD

                _mean = RESISC45_MEAN if cfg.classifier_use_resisc45_stats else IMAGENET_MEAN
                _std = RESISC45_STD if cfg.classifier_use_resisc45_stats else IMAGENET_STD

                _classifier = LandscapeClassifier(
                    pretrained=False,
                    pretrained_path=_ckpt,
                    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                )
                _classifier.eval()
                _clf_transform = _tv.Compose([
                    _tv.Resize((224, 224)),
                    _tv.ToTensor(),
                    _tv.Normalize(mean=_mean, std=_std),
                ])
            else:
                print(f"Classifier checkpoint not found: {_ckpt}")

        def _decode_terrain_rgb(_terrain_img: _Image.Image) -> np.ndarray:
            _arr = np.array(_terrain_img).astype(np.float32)
            _r = _arr[..., 0]
            _g = _arr[..., 1]
            _b = _arr[..., 2]
            return -10000.0 + (_r * 256.0 * 256.0 + _g * 256.0 + _b) * 0.1

        def _safe_div(_num: np.ndarray, _den: np.ndarray) -> np.ndarray:
            return _num / (np.maximum(_den, 1e-6))

        _processed = 0
        for _sat_path in _sat_files:
            if _processed >= _max_tiles:
                break

            _terrain_path = _terrain_index.get(_sat_path.stem)
            if _terrain_path is None:
                continue

            _sat_img = _Image.open(_sat_path).convert("RGB")
            _terrain_img = _Image.open(_terrain_path).convert("RGB")

            _elev = _decode_terrain_rgb(_terrain_img)
            _gy, _gx = np.gradient(_elev)
            _slope = np.sqrt(_gx ** 2 + _gy ** 2)

            _relief = float(_elev.max() - _elev.min())
            _roughness = float(_elev.std())
            _slope_mean = float(_slope.mean())

            _low_elev = _elev < np.percentile(_elev, 10)
            _flat = _slope < np.percentile(_slope, 10)
            _water_proxy = float((_low_elev & _flat).mean())

            _sat_arr = np.array(_sat_img).astype(np.float32)
            _r = _sat_arr[..., 0]
            _g = _sat_arr[..., 1]
            _b = _sat_arr[..., 2]
            _veg_proxy = float(_safe_div(_g, _r + _g + _b).mean())

            _class_score = 0.0
            _class_id = 0
            if _classifier is not None and _clf_transform is not None:
                import torch as _torch

                _input = _clf_transform(_sat_img).unsqueeze(0).to(_classifier.device)
                with _torch.no_grad():
                    _logits = _classifier(_input)
                    _probs = _torch.softmax(_logits, dim=-1).cpu().numpy()[0]
                _class_id = int(_probs.argmax())
                _class_name = TERRAIN_CLASSES[_class_id]
                _class_score = float(get_scenic_weight(_class_name))
            else:
                _class_id = 0
                _class_score = 0.3

            _score = (
                2.5 * _class_score
                + 2.0 * np.tanh(_relief / 500.0)
                + 1.5 * np.tanh(_roughness / 200.0)
                + 1.5 * np.tanh(_slope_mean / 15.0)
                + 1.5 * _water_proxy
                + 1.0 * _veg_proxy
            )
            _score = float(np.clip(_score, 0.0, 10.0))

            _rows.append(
                {
                    "image_path": str(_sat_path.relative_to(Path(cfg.raw_dir))),
                    "scenic_score": _score,
                    "lat": cfg.heuristic_default_lat,
                    "lon": cfg.heuristic_default_lon,
                    "class_id": _class_id,
                }
            )
            _processed += 1

        if not _rows:
            raise ValueError("No paired satellite/terrain tiles found to label.")

        _out_path = Path(cfg.labels_csv)
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        _df = pd.DataFrame(_rows)
        _df.to_csv(_out_path, index=False)
        {
            "labels_written": str(_out_path),
            "rows": len(_df),
            "used_classifier": _classifier is not None,
        }


@app.cell
def _(DataLoader, Dataset, Image, Path, cfg, np, pd, torch, transforms):
    """Data loaders (multi-task)."""
    _labels_path = Path(cfg.labels_csv)
    if not _labels_path.exists():
        print(f"Missing labels file: {_labels_path}")
        train_loader = None
        val_loader = None
        test_loader = None
        train_loader, val_loader, test_loader
    else:
        _df = pd.read_csv(_labels_path)
        if _df.empty:
            raise ValueError("labels.csv is empty.")

        _missing_paths = []
        _image_root = Path(cfg.raw_dir)
        for _p in _df["image_path"].head(2000):
            _path = Path(_p)
            if not _path.is_absolute():
                _path = _image_root / _path
            if not _path.exists():
                _missing_paths.append(str(_path))
                if len(_missing_paths) >= 10:
                    break
        if _missing_paths:
            raise FileNotFoundError(
                "Missing image files (first 10 shown):\n" + "\n".join(_missing_paths)
            )

        _rng = np.random.default_rng(cfg.seed)
        _indices = _rng.permutation(len(_df))
        _n_train = int(len(_df) * cfg.train_split)
        _n_val = int(len(_df) * cfg.val_split)
        _train_idx = _indices[:_n_train]
        _val_idx = _indices[_n_train:_n_train + _n_val]
        _test_idx = _indices[_n_train + _n_val:]

        class ScenicDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, image_root: Path, transform):
                self.frame = frame.reset_index(drop=True)
                self.image_root = image_root
                self.transform = transform

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, idx):
                row = self.frame.iloc[idx]
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = self.image_root / image_path

                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)

                scenic_score = torch.tensor(float(row["scenic_score"]), dtype=torch.float32)
                class_id = torch.tensor(int(row["class_id"]), dtype=torch.long)

                return image, scenic_score, class_id

        _train_tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        _eval_tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        _image_root = Path(cfg.raw_dir)
        _train_ds = ScenicDataset(_df.iloc[_train_idx], _image_root, _train_tf)
        _val_ds = ScenicDataset(_df.iloc[_val_idx], _image_root, _eval_tf)
        _test_ds = ScenicDataset(_df.iloc[_test_idx], _image_root, _eval_tf)

        _loader_kwargs = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=(__import__("torch").cuda.is_available()),
        )
        train_loader = DataLoader(_train_ds, shuffle=True, **_loader_kwargs)
        val_loader = DataLoader(_val_ds, shuffle=False, **_loader_kwargs)
        test_loader = DataLoader(_test_ds, shuffle=False, **_loader_kwargs)

        train_loader, val_loader, test_loader
    return train_loader, val_loader


@app.cell
def _(Path, cfg, nn, timm, torch):
    """Model scaffold (ViT backbone + regression + classification)."""
    class ScenicMultiTask(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,
            )
            feat_dim = self.backbone.num_features
            self.reg_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 1),
            )
            self.cls_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, num_classes),
            )

        def forward(self, x):
            feats = self.backbone(x)
            scenic = self.reg_head(feats).squeeze(-1)
            logits = self.cls_head(feats)
            return scenic, logits

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScenicMultiTask(cfg.num_classes).to(_device)
    if cfg.resume_from:
        model.load_state_dict(torch.load(cfg.resume_from, map_location=_device))
    model
    return (model,)


@app.cell
def _(Path, cfg, datetime, json, model, nn, torch, tqdm, train_loader, val_loader):
    """Training loop placeholder."""
    if train_loader is None or val_loader is None:
        print("No data loaders available.")
        _metrics = {}
        _metrics
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

        _best_val = float("inf")
        _patience = 5
        _patience_left = _patience
        _metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_rmse": [],
        }

        _run_ckpt_path = Path(cfg.model_dir) / cfg.best_ckpt
        _run_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        _run_dir = Path(cfg.processed_dir)
        _run_dir.mkdir(parents=True, exist_ok=True)
        _run_path = _run_dir / f"{cfg.run_name}_{cfg.run_log}"

        for _epoch in range(cfg.epochs):
            model.train()
            _train_loss = 0.0
            for batch in tqdm(
                train_loader, desc=f"Epoch {_epoch+1}/{cfg.epochs} [train]"
            ):
                _images, _scenic_scores, _class_ids = batch
                _images = _images.to(device)
                _scenic_scores = _scenic_scores.to(device)
                _class_ids = _class_ids.to(device)

                _pred_scores, _logits = model(_images)
                _reg_loss = nn.functional.mse_loss(_pred_scores, _scenic_scores)
                _cls_loss = nn.functional.cross_entropy(_logits, _class_ids)
                loss = cfg.loss_weights[0] * _reg_loss + cfg.loss_weights[1] * _cls_loss

                _optimizer.zero_grad()
                loss.backward()
                _optimizer.step()

                _train_loss += loss.item()

            _train_loss /= max(1, len(train_loader))

            model.eval()
            _val_loss = 0.0
            _correct = 0
            _total = 0
            _mse_sum = 0.0
            with torch.no_grad():
                for _batch in tqdm(
                    val_loader, desc=f"Epoch {_epoch+1}/{cfg.epochs} [val]"
                ):
                    _images, _scenic_scores, _class_ids = _batch
                    _images = _images.to(device)
                    _scenic_scores = _scenic_scores.to(device)
                    _class_ids = _class_ids.to(device)

                    _pred_scores, _logits = model(_images)
                    _reg_loss = nn.functional.mse_loss(_pred_scores, _scenic_scores)
                    _cls_loss = nn.functional.cross_entropy(_logits, _class_ids)
                    loss = cfg.loss_weights[0] * _reg_loss + cfg.loss_weights[1] * _cls_loss

                    _val_loss += loss.item()
                    _mse_sum += _reg_loss.item() * _images.size(0)
                    _preds = _logits.argmax(dim=-1)
                    _correct += (_preds == _class_ids).sum().item()
                    _total += _images.size(0)

            _val_loss /= max(1, len(val_loader))
            _val_rmse = (_mse_sum / max(1, _total)) ** 0.5
            _val_acc = _correct / max(1, _total)

            _metrics["train_loss"].append(_train_loss)
            _metrics["val_loss"].append(_val_loss)
            _metrics["val_acc"].append(_val_acc)
            _metrics["val_rmse"].append(_val_rmse)

            if _val_loss < _best_val:
                _best_val = _val_loss
                _patience_left = _patience
                torch.save(model.state_dict(), _run_ckpt_path)
            else:
                _patience_left -= 1
                if _patience_left <= 0:
                    break

        _run_summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config": {
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "num_workers": cfg.num_workers,
                "image_size": cfg.image_size,
                "train_split": cfg.train_split,
                "val_split": cfg.val_split,
                "num_classes": cfg.num_classes,
                "loss_weights": cfg.loss_weights,
                "run_name": cfg.run_name,
                "resume_from": cfg.resume_from,
            },
            "best_val_loss": _best_val,
            "metrics": _metrics,
            "best_checkpoint": str(_run_ckpt_path),
        }

        with open(_run_path, "w", encoding="utf-8") as f:
            json.dump(_run_summary, f, indent=2)

        _run_summary


if __name__ == "__main__":
    app.run()
