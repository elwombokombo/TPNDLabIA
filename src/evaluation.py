"""Evaluación del modelo y visualización de resultados."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # guarda las figuras en disco en lugar de abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.model import CLASSES, AgeNet, predict_logits

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def evaluate(
    model: AgeNet,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    out_dir: str | Path = REPORTS_DIR,
) -> tuple[str, np.ndarray]:
    """Evalúa en test, imprime y guarda el reporte; devuelve (reporte, matriz de confusión)."""
    preds = predict_logits(model, x_test, device).argmax(1).numpy()
    y_true = y_test.numpy()
    report = classification_report(
        y_true, preds, labels=range(len(CLASSES)), target_names=CLASSES, digits=3, zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=range(len(CLASSES)))
    print("\nTest set:\n" + report)
    print("Matriz de confusión (filas = real, columnas = predicho):\n", cm)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report + f"\n{cm}\n")
    return report, cm


def plot_training(
    history: dict[str, list[float]], cm: np.ndarray, out_dir: str | Path = REPORTS_DIR
) -> Path:
    """Curvas de accuracy y loss, y matriz de confusión, en una sola figura."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for key, label in [("acc", "train acc"), ("val_acc", "val acc")]:
        ax[0].plot(history[key], label=label)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    for key, label in [("loss", "train loss"), ("val_loss", "val loss")]:
        ax[1].plot(history[key], label=label)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()

    ax[2].imshow(cm, cmap="Blues")
    ax[2].set_xticks(range(len(CLASSES)), CLASSES)
    ax[2].set_yticks(range(len(CLASSES)), CLASSES)
    ax[2].set_xlabel("predicho")
    ax[2].set_ylabel("real")
    ax[2].set_title("Matriz de confusión")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "training.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def predict_examples(
    model: AgeNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n: int = 8,
    out_dir: str | Path = REPORTS_DIR,
) -> Path:
    """Guarda una grilla con n predicciones de ejemplo (verde = acierto, rojo = error).

    images: uint8 NHWC RGB. Se eligen n imágenes al azar (semilla fija).
    """
    n = min(n, len(images))
    idx = torch.randperm(len(images), generator=torch.Generator().manual_seed(42))[:n]
    probs = predict_logits(model, images[idx], device).softmax(1)

    cols = min(4, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, ax in zip(range(n), axes.ravel()):
        pred = int(probs[k].argmax())
        true = int(labels[idx[k]])
        ax.imshow(images[idx[k]].numpy())
        ax.set_title(
            f"real: {CLASSES[true]}\npred: {CLASSES[pred]} ({probs[k, pred]:.2f})",
            fontsize=9,
            color="green" if pred == true else "red",
        )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
