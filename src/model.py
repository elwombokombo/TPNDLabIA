"""Modelo base (transfer learning): MobileNetV2 preentrenado en ImageNet.

Contrato de entrada: los lotes son tensores uint8 NHWC en **RGB** (N, 128, 128, 3).
La normalización con media/desvío de ImageNet se hace dentro del modelo.
Las etiquetas son índices enteros según CLASSES.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CLASSES = ["child", "young", "adult", "senior"]  # orden de los índices de clase
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AgeNet(nn.Module):
    """MobileNetV2 (features) + cabeza densa para clasificar rango de edad."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v2(weights=weights).features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES)),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: lote uint8 NHWC en RGB."""
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))

    def train(self, mode: bool = True):
        super().train(mode)
        # Mantiene fijas las estadísticas de BatchNorm (backbone congelado o lr bajo).
        self.backbone.eval()
        return self


def _batches(x: torch.Tensor, y: torch.Tensor, size: int, shuffle: bool):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), size):
        j = idx[i : i + size]
        yield x[j], y[j]


@torch.no_grad()
def predict_logits(
    model: AgeNet, x: torch.Tensor, device: torch.device, batch_size: int = 256
) -> torch.Tensor:
    """Logits (N, 4) en CPU para un tensor uint8 NHWC RGB."""
    model.eval()
    out = [
        model(xb.to(device)).cpu()
        for xb, _ in _batches(x, torch.zeros(len(x)), batch_size, shuffle=False)
    ]
    return torch.cat(out)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia (clase 'adult' domina)."""
    counts = torch.bincount(labels, minlength=len(CLASSES)).float()
    return counts.sum() / (len(CLASSES) * counts)


def _run_epochs(model, optimizer, loss_fn, data, epochs, device, history, tag):
    x_tr, y_tr, x_va, y_va = data
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in _batches(x_tr, y_tr, 64, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            flip = torch.rand(len(xb), device=device) < 0.5  # flip horizontal aleatorio
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(2), xb)  # NHWC: ancho = dim 2
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)

        val_logits = predict_logits(model, x_va, device)
        val_loss = loss_fn(val_logits.to(device), y_va.to(device)).item()
        val_acc = (val_logits.argmax(1) == y_va).float().mean().item()
        history["loss"].append(loss_sum / total)
        history["acc"].append(correct / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[{tag}] epoch {epoch}/{epochs}  loss {loss_sum / total:.4f}  "
            f"acc {correct / total:.3f}  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}"
        )


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 5,
    finetune_epochs: int = 3,
    device: torch.device | None = None,
) -> tuple[AgeNet, dict[str, list[float]]]:
    """Entrena en dos fases y devuelve (modelo, historial).

    Fase 1: backbone congelado, se entrena solo la cabeza (lr 1e-3).
    Fase 2: se descongelan los últimos 5 bloques con lr bajo (1e-5; cabeza 1e-4).
    """
    device = device or get_device()
    torch.manual_seed(42)
    data = (x_train, y_train, x_val, y_val)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    model = AgeNet().to(device)
    history: dict[str, list[float]] = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    _run_epochs(model, optimizer, loss_fn, data, epochs, device, history, "head")

    if finetune_epochs > 0:
        for p in model.backbone[-5:].parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            [
                {"params": model.backbone[-5:].parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": 1e-4},
            ]
        )
        _run_epochs(model, optimizer, loss_fn, data, finetune_epochs, device, history, "finetune")

    return model, history


def save_model(model: AgeNet, path: str | Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(path: str | Path = MODEL_PATH, device: torch.device | None = None) -> AgeNet:
    device = device or get_device()
    model = AgeNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()
