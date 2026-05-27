"""
Holografikus alga osztályozó – ConvNeXt-Tiny kettős fejű modell

Futtatás:
    .venv\\Scripts\\python.exe train.py

Kimenet:
    submission_multiclass.csv  – TARGET ∈ {0,1,2,3,4}
    submission_binary.csv      – TARGET ∈ {0=chlorella, 1=nem-chlorella}
    best_model.pth             – checkpoint + optimális küszöbök
"""

# ── Az első print még a nehéz importok előtt fut ki ───────────────
import sys
print("[ 1/5 ] Csomagok betöltése...", flush=True)

import os
import glob
import copy
import inspect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

print("[ 2/5 ] PyTorch betöltése...", flush=True)

import torch
import torch.nn as nn

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

_GRADSCALER_HAS_DEVICE = "device_type" in inspect.signature(GradScaler.__init__).parameters
_AUTOCAST_HAS_DEVICE   = "device_type" in inspect.signature(autocast).parameters

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

print("[ 2/5 ] Csomagok betöltve.\n", flush=True)


# ──────────────────────────────────────────────────────────────────
# KONFIGURÁCIÓ
# ──────────────────────────────────────────────────────────────────

TRAIN_DIR    = "train"
TEST_DIR     = "test"
RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS     = 25
PATIENCE   = 7

# A bináris fej veszteségfüggvényének súlya a teljes loss-ban
BINARY_LOSS_WEIGHT = 2.0

# Threshold grid – CSAK a validációs halmazon, a tesztet nem látjuk
THRESHOLD_GRID = np.arange(0.01, 0.91, 0.02)   # p0 küszöb (multi) ill. bináris prob
MARGIN_GRID    = np.arange(0.00, 0.41, 0.05)   # extra margin a multi-class fejhez

# Minimum recall, ami alatt a score-t tizedeljük (verseny metrikája)
MIN_RECALL = 0.50

NUM_WORKERS = 0 if os.name == "nt" else 4

CLASS_NAMES = ["chlorella", "debris", "haematococcus", "small_haemato", "small_particle"]
LABEL_MAP   = {
    "class_chlorella":      0,
    "class_debris":         1,
    "class_haematococcus":  2,
    "class_small_haemato":  3,
    "class_small_particle": 4,
}


# ──────────────────────────────────────────────────────────────────
# ADATHALMAZ
# ──────────────────────────────────────────────────────────────────

class HoloDataset(Dataset):
    """
    Amplitúdó / fázis / maszk hármasokat tölt be a train mappából.

    Minden mintához 3 csatornás képet épít:
      R = *_amp.png   (amplitúdó)
      G = *_phase.png (fázis,   ha hiányzik → amplitúdó másolata)
      B = *_mask.png  (maszk,   ha hiányzik → amplitúdó másolata)
    """

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform

        self.amp_paths = sorted(
            glob.glob(os.path.join(root, "**", "*_amp.png"), recursive=True)
        )
        if not self.amp_paths:
            raise RuntimeError(f"Nem találhatók _amp.png fájlok itt: {root!r}")

        self.targets = np.array([
            LABEL_MAP[os.path.basename(os.path.dirname(p))]
            for p in self.amp_paths
        ])

    def __len__(self) -> int:
        return len(self.amp_paths)

    def _load_aux(self, amp_path: str, suffix: str, fallback: Image.Image) -> Image.Image:
        """Segédcsatorna betöltése; ha nem létezik, az amplitúdó másolatát adja vissza."""
        base = os.path.splitext(amp_path)[0]
        candidates = []
        if base.endswith("_amp"):
            candidates.append(base[:-4] + suffix + ".png")
        candidates.append(base + suffix + ".png")
        for path in candidates:
            if os.path.exists(path):
                return Image.open(path).convert("L")
        return fallback.copy()

    def __getitem__(self, idx: int):
        amp_path  = self.amp_paths[idx]
        amp_img   = Image.open(amp_path).convert("L")
        phase_img = self._load_aux(amp_path, "_phase", amp_img)
        mask_img  = self._load_aux(amp_path, "_mask",  amp_img)
        img = Image.merge("RGB", (amp_img, phase_img, mask_img))
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class SubsetDataset(Dataset):
    """Részhalmazt képez egy HoloDataset-ből index-lista alapján."""

    def __init__(self, base: HoloDataset, indices: List[int]):
        self.base    = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


# ──────────────────────────────────────────────────────────────────
# TRANSZFORMÁCIÓK
# ──────────────────────────────────────────────────────────────────

train_tf = T.Compose([
    T.RandomResizedCrop(256, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.RandomAutocontrast(p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.25, 0.25, 0.25]),
])

eval_tf = T.Compose([
    T.Resize(272),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.25, 0.25, 0.25]),
])


# ──────────────────────────────────────────────────────────────────
# MODELL
# ──────────────────────────────────────────────────────────────────

class AlgaeNet(nn.Module):
    """
    ConvNeXt-Tiny alapú kettős fejű hálózat.

    Kimenet:
      multi_logits : [B, 5]  – 5 osztály logitjai
      binary_logit : [B]     – chlorella vs. nem-chlorella logit
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        base          = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        in_features   = base.classifier[2].in_features
        self.dropout  = nn.Dropout(dropout)
        self.head_multi  = nn.Linear(in_features, 5)
        self.head_binary = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.pool(self.backbone(x)).flatten(1)
        feats = self.dropout(feats)
        return self.head_multi(feats), self.head_binary(feats).squeeze(1)


# ──────────────────────────────────────────────────────────────────
# VESZTESÉGFÜGGVÉNYEK
# ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss a bináris fejhez – a nehéz minták kapnak nagyobb súlyt."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ──────────────────────────────────────────────────────────────────
# PREDIKCIÓS SEGÉDFÜGGVÉNYEK
# ──────────────────────────────────────────────────────────────────

def predict_multiclass(
    multi_probs: np.ndarray,
    threshold: float,
    margin: float,
) -> np.ndarray:
    """
    5-osztályos predikció a multi-class fej valószínűségeiből.

    Egy minta chlorella (0), ha:
      P(chlorella) >= threshold  ÉS
      P(chlorella) - max(többi osztály valószínűsége) >= margin
    Különben a legjobb nem-chlorella osztályt kapja (1-4).
    """
    p0     = multi_probs[:, 0]
    p_rest = multi_probs[:, 1:]
    preds  = p_rest.argmax(axis=1) + 1   # alapértelmezett: legjobb nem-chlorella osztály
    chlorella_mask = (p0 >= threshold) & ((p0 - p_rest.max(axis=1)) >= margin)
    preds[chlorella_mask] = 0
    return preds


def predict_binary(binary_probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Bináris predikció a bináris fej valószínűségéből.
    TARGET: 0 = chlorella, 1 = nem-chlorella
    """
    return np.where(binary_probs >= threshold, 0, 1)


def contest_score(precision: float, recall: float) -> float:
    """Verseny metrikája: precision, ha recall >= MIN_RECALL, egyébként precision / 10."""
    return precision if recall >= MIN_RECALL else precision / 10.0


# ──────────────────────────────────────────────────────────────────
# KIÉRTÉKELÉS (csak validációs halmazon)
# ──────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Kiértékeli a modellt a validációs halmazon és megkeresi az optimális küszöbértékeket.
      Multi-class: 2D grid (threshold × margin)
      Bináris:     1D grid (threshold)
    """
    model.eval()
    all_multi, all_binary, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            m_logit, b_logit = model(images.to(device, non_blocking=True))
            all_multi.append(m_logit.cpu())
            all_binary.append(b_logit.cpu())
            all_labels.append(labels)

    multi_probs  = torch.softmax(torch.cat(all_multi),  dim=1).numpy()
    binary_probs = torch.sigmoid(torch.cat(all_binary)).numpy()
    labels_np    = torch.cat(all_labels).numpy()
    true_binary  = (labels_np == 0).astype(int)   # 1 = chlorella

    # ── Multi-class: 2D grid ──────────────────────────────────────
    best_multi = {"score": -np.inf, "threshold": 0.5, "margin": 0.0,
                  "precision": 0.0, "recall": 0.0}

    for thr in THRESHOLD_GRID:
        for marg in MARGIN_GRID:
            preds    = predict_multiclass(multi_probs, float(thr), float(marg))
            pred_bin = (preds == 0).astype(int)
            rec  = recall_score(true_binary, pred_bin, zero_division=0)
            prec = precision_score(true_binary, pred_bin, zero_division=0)
            s    = contest_score(prec, rec)
            if s > best_multi["score"]:
                best_multi = {"score": s, "threshold": float(thr), "margin": float(marg),
                              "precision": prec, "recall": rec}

    # ── Bináris: 1D grid ──────────────────────────────────────────
    best_binary = {"score": -np.inf, "threshold": 0.5, "precision": 0.0, "recall": 0.0}

    for thr in THRESHOLD_GRID:
        preds    = predict_binary(binary_probs, float(thr))
        pred_bin = (preds == 0).astype(int)
        rec  = recall_score(true_binary, pred_bin, zero_division=0)
        prec = precision_score(true_binary, pred_bin, zero_division=0)
        s    = contest_score(prec, rec)
        if s > best_binary["score"]:
            best_binary = {"score": s, "threshold": float(thr),
                           "precision": prec, "recall": rec}

    return {"best_multi": best_multi, "best_binary": best_binary}


# ──────────────────────────────────────────────────────────────────
# TANÍTÁS – EGY EPOCH
# ──────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion_multi: nn.Module,
    criterion_binary: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    bar = tqdm(loader, desc=f"  Epoch {epoch:02d} tanítás", leave=False,
               unit="batch", dynamic_ncols=True)

    for images, labels in bar:
        images        = images.to(device, non_blocking=True)
        labels        = labels.to(device)
        binary_labels = (labels == 0).float()

        optimizer.zero_grad(set_to_none=True)

        amp_kwargs = {"enabled": device.type == "cuda"}
        if _AUTOCAST_HAS_DEVICE:
            amp_kwargs["device_type"] = "cuda"

        with autocast(**amp_kwargs):
            m_logit, b_logit = model(images)
            loss = (criterion_multi(m_logit, labels)
                    + BINARY_LOSS_WEIGHT * criterion_binary(b_logit, binary_labels))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────
# ADATBETÖLTŐK FELÉPÍTÉSE
# ──────────────────────────────────────────────────────────────────

def build_dataloaders():
    """
    Train/val split (82%/18%), osztályarányos véletlen mintavételezővel.
    A chlorella osztálynak 2.5× extra veszteségsúlyt adunk.
    """
    full_train = HoloDataset(TRAIN_DIR, transform=train_tf)
    full_val   = HoloDataset(TRAIN_DIR, transform=eval_tf)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.18,
        random_state=RANDOM_STATE,
        stratify=full_train.targets,
    )

    train_targets  = full_train.targets[train_idx]
    class_counts   = np.maximum(np.bincount(train_targets, minlength=5), 1)
    sample_weights = 1.0 / class_counts[train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loss_weights    = (class_counts.sum() / class_counts).astype(np.float32)
    loss_weights   /= loss_weights.mean()
    loss_weights[0] *= 2.5
    loss_weights   /= loss_weights.mean()

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=NUM_WORKERS, pin_memory=pin)

    train_loader = DataLoader(
        SubsetDataset(full_train, train_idx.tolist()),
        batch_size=BATCH_SIZE, sampler=sampler, **kw,
    )
    val_loader = DataLoader(
        SubsetDataset(full_val, val_idx.tolist()),
        batch_size=BATCH_SIZE, shuffle=False, **kw,
    )

    print(f"  Tanítóhalmaz : {len(train_loader.dataset)} minta", flush=True)
    print(f"  Validáció    : {len(val_loader.dataset)} minta", flush=True)
    print(f"  Osztályeloszlás (train): "
          + ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, class_counts.tolist())),
          flush=True)

    return train_loader, val_loader, loss_weights


# ──────────────────────────────────────────────────────────────────
# TESZTKÉP BETÖLTÉSE
# ──────────────────────────────────────────────────────────────────

def load_test_image(path: str) -> Image.Image:
    """
    Tesztkép betöltése amp/phase/mask tripletként.
    Ha a phase/mask nem létezik, az amplitúdó másolatát használja.
    """
    amp  = Image.open(path).convert("L")
    d    = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]

    def find_aux(suffix: str) -> Image.Image:
        candidates = []
        if base.endswith("_amp"):
            candidates.append(os.path.join(d, base[:-4] + suffix + ".png"))
        candidates.append(os.path.join(d, base + suffix + ".png"))
        for p in candidates:
            if os.path.exists(p):
                return Image.open(p).convert("L")
        return amp.copy()

    return Image.merge("RGB", (amp, find_aux("_phase"), find_aux("_mask")))


# ──────────────────────────────────────────────────────────────────
# SUBMISSION GENERÁLÁS
# ──────────────────────────────────────────────────────────────────

def generate_submissions(
    model: nn.Module,
    device: torch.device,
    multi_threshold: float,
    multi_margin: float,
    binary_threshold: float,
) -> None:
    """
    Lefuttatja a modellt a teljes teszthalmazon és két CSV-t ír ki:
      submission_multiclass.csv  – 5-osztályos (TARGET: 0-4)
      submission_binary.csv      – bináris     (TARGET: 0=chlorella, 1=más)
    """
    test_paths = sorted(
        glob.glob(os.path.join(TEST_DIR, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not test_paths:
        raise RuntimeError(f"Nem találhatók PNG fájlok itt: {TEST_DIR!r}")

    model.eval()
    ids, all_multi, all_binary = [], [], []

    bar = tqdm(test_paths, desc="  Teszt inferencia", unit="kép", dynamic_ncols=True)
    with torch.no_grad():
        for path in bar:
            tensor = eval_tf(load_test_image(path)).unsqueeze(0).to(device)
            m, b   = model(tensor)
            all_multi.append(m.cpu())
            all_binary.append(b.cpu())
            ids.append(int(os.path.splitext(os.path.basename(path))[0]))

    multi_probs  = torch.softmax(torch.cat(all_multi),  dim=1).numpy()
    binary_probs = torch.sigmoid(torch.cat(all_binary)).numpy()
    ids_arr      = np.array(ids)
    n            = len(ids_arr)

    # ── Multi-class ───────────────────────────────────────────────
    mc_preds = predict_multiclass(multi_probs, multi_threshold, multi_margin)
    pd.DataFrame({"ID": ids_arr, "TARGET": mc_preds}).sort_values("ID") \
      .to_csv("submission_multiclass.csv", index=False)

    print("\n── Multi-class submission ───────────────────────────────", flush=True)
    for i, name in enumerate(CLASS_NAMES):
        cnt = int((mc_preds == i).sum())
        bar_fill = "█" * int(cnt / n * 30)
        print(f"  {name:20s}: {cnt:3d}  ({cnt/n:5.1%})  {bar_fill}", flush=True)
    print("  → submission_multiclass.csv", flush=True)

    # ── Bináris ───────────────────────────────────────────────────
    bin_preds = predict_binary(binary_probs, binary_threshold)
    pd.DataFrame({"ID": ids_arr, "TARGET": bin_preds}).sort_values("ID") \
      .to_csv("submission_binary.csv", index=False)

    n_chl = int((bin_preds == 0).sum())
    print("\n── Bináris submission ───────────────────────────────────", flush=True)
    print(f"  chlorella     : {n_chl:3d}  ({n_chl/n:5.1%})", flush=True)
    print(f"  nem-chlorella : {n-n_chl:3d}  ({(n-n_chl)/n:5.1%})", flush=True)
    print("  → submission_binary.csv", flush=True)


# ──────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ──────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ 3/5 ] Eszköz: {device}", flush=True)

    # ── Adat ──────────────────────────────────────────────────────
    print("\n[ 3/5 ] Adathalmaz betöltése...", flush=True)
    train_loader, val_loader, loss_weights = build_dataloaders()

    # ── Modell ────────────────────────────────────────────────────
    print("\n[ 4/5 ] Modell inicializálása (ImageNet súlyok betöltése)...", flush=True)
    model = AlgaeNet(dropout=0.25).to(device)
    print("        Modell kész.", flush=True)

    criterion_multi  = nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weights, dtype=torch.float32, device=device)
    )
    criterion_binary = FocalLoss(alpha=0.75, gamma=2.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=8e-4,
        epochs=EPOCHS,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e2,
    )

    sc_kw = {"enabled": device.type == "cuda"}
    if _GRADSCALER_HAS_DEVICE:
        sc_kw["device_type"] = "cuda"
    scaler = GradScaler(**sc_kw)

    # ── Tanítási loop ──────────────────────────────────────────────
    print(f"\n[ 5/5 ] Tanítás ({EPOCHS} epoch max, early stopping: {PATIENCE})...", flush=True)
    print("─" * 90, flush=True)

    best_state  = None
    best_score  = -np.inf
    best_stats  = None
    no_improve  = 0

    for epoch in range(1, EPOCHS + 1):
        loss  = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_multi, criterion_binary, device, scaler,
            epoch=epoch,
        )
        stats = evaluate(model, val_loader, device)
        mc    = stats["best_multi"]
        bi    = stats["best_binary"]

        # Javulás jelzése csillaggal
        improved = mc["score"] > best_score
        marker   = " ★" if improved else ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f} │ "
            f"multi  P={mc['precision']:.3f} R={mc['recall']:.3f} "
            f"score={mc['score']:.3f} "
            f"(thr={mc['threshold']:.2f} marg={mc['margin']:.2f}) │ "
            f"binary P={bi['precision']:.3f} R={bi['recall']:.3f} "
            f"score={bi['score']:.3f} "
            f"(thr={bi['threshold']:.2f})"
            f"{marker}",
            flush=True,
        )

        if improved:
            best_score = mc["score"]
            best_state = copy.deepcopy(model.state_dict())
            best_stats = stats
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Korai leállás ({PATIENCE} epoch javulás nélkül).", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Nem keletkezett érvényes checkpoint.")

    model.load_state_dict(best_state)
    mc = best_stats["best_multi"]
    bi = best_stats["best_binary"]

    print("\n" + "─" * 90, flush=True)
    print("Legjobb checkpoint összefoglaló:", flush=True)
    print(
        f"  Multi-class : thr={mc['threshold']:.3f}  margin={mc['margin']:.3f}  "
        f"P={mc['precision']:.3f}  R={mc['recall']:.3f}  score={mc['score']:.3f}",
        flush=True,
    )
    print(
        f"  Bináris     : thr={bi['threshold']:.3f}  "
        f"P={bi['precision']:.3f}  R={bi['recall']:.3f}  score={bi['score']:.3f}",
        flush=True,
    )

    torch.save({
        "state_dict":       best_state,
        "multi_threshold":  mc["threshold"],
        "multi_margin":     mc["margin"],
        "binary_threshold": bi["threshold"],
        "class_names":      CLASS_NAMES,
    }, "best_model.pth")
    print("\nCheckpoint mentve → best_model.pth", flush=True)

    print("\nSubmission CSV-k generálása...", flush=True)
    generate_submissions(
        model, device,
        multi_threshold  = mc["threshold"],
        multi_margin     = mc["margin"],
        binary_threshold = bi["threshold"],
    )

    print("\n✓ Kész. Fájlok: submission_multiclass.csv  submission_binary.csv", flush=True)


if __name__ == "__main__":
    main()
