"""Visor interactivo: muestra una cara de UTKFace y lo que predice el modelo.

Uso:  python -m src.viewer [--model models/model.pt] [--data data/UTKFace]

Teclas: flecha derecha / izquierda = siguiente / anterior, "r" = aleatoria.
No importa src.evaluation a propósito: ese módulo fuerza el backend "Agg" de
matplotlib (sin ventanas) y acá se necesita el backend interactivo por defecto.
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

from src.categories import age_to_class
from src.model import CLASSES, MODEL_PATH, get_device, load_model, predict_logits

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "UTKFace"
_AGE_RE = re.compile(r"^(\d+)_")

COLOR_BAR = "#9aa5b1"
COLOR_PRED = "#2f6fdd"
COLOR_OK = "#2e8b57"
COLOR_BAD = "#c0392b"


def list_faces(data_dir: Path) -> list[Path]:
    """Imágenes de la carpeta cuyo nombre empieza con la edad (sin subcarpetas)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {data_dir}")
    paths = [
        p
        for p in sorted(data_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and _AGE_RE.match(p.name)
    ]
    if not paths:
        raise ValueError(f"No se encontraron imágenes válidas en: {data_dir}")
    return paths


def load_face(path: Path) -> np.ndarray | None:
    """Imagen 128 x 128 RGB uint8, o None si no se puede leer."""
    encoded = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if img is None:
        return None
    return cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)


class FaceViewer:
    def __init__(self, model, device: torch.device, paths: list[Path], seed: int = 42) -> None:
        self.model, self.device = model, device
        self.paths = paths[:]
        random.Random(seed).shuffle(self.paths)  # recorrido en orden aleatorio reproducible
        self.pos = 0

        self.fig = plt.figure(figsize=(9, 4.8))
        self.fig.canvas.manager.set_window_title("Predicción de rango etario")
        self.ax_img = self.fig.add_axes([0.04, 0.17, 0.38, 0.70])
        self.ax_bar = self.fig.add_axes([0.54, 0.22, 0.42, 0.60])
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._buttons = []  # referencias: matplotlib no mantiene vivos los widgets
        for x, label, step in [(0.04, "← Anterior", -1), (0.18, "Siguiente →", 1)]:
            self._add_button(x, label, lambda _e, s=step: self.move(s))
        self._add_button(0.32, "Aleatoria", lambda _e: self.jump(random.randrange(len(self.paths))))
        self.show()

    def _add_button(self, x: float, label: str, callback) -> None:
        button = Button(self.fig.add_axes([x, 0.04, 0.12, 0.07]), label)
        button.on_clicked(callback)
        self._buttons.append(button)

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.move(1)
        elif event.key == "left":
            self.move(-1)
        elif event.key == "r":
            self.jump(random.randrange(len(self.paths)))

    def move(self, step: int) -> None:
        self.jump((self.pos + step) % len(self.paths))

    def jump(self, pos: int) -> None:
        self.pos = pos
        self.show()

    def show(self) -> None:
        # Salta imágenes ilegibles; el tope evita un bucle si todas fallaran.
        for _ in range(len(self.paths)):
            path = self.paths[self.pos]
            face = load_face(path)
            if face is not None:
                break
            self.pos = (self.pos + 1) % len(self.paths)
        else:
            raise RuntimeError("Ninguna imagen se pudo leer.")

        age = int(_AGE_RE.match(path.name).group(1))
        true = age_to_class(age)
        batch = torch.from_numpy(face).unsqueeze(0)
        probs = predict_logits(self.model, batch, self.device).softmax(1)[0].numpy()
        pred = int(probs.argmax())

        self.ax_img.clear()
        self.ax_img.imshow(face)
        self.ax_img.axis("off")
        self.ax_img.set_title(
            f"Edad real: {age} años ({CLASSES[true]})", fontsize=11, loc="left"
        )

        self.ax_bar.clear()
        bars = self.ax_bar.barh(
            CLASSES, probs, color=[COLOR_PRED if i == pred else COLOR_BAR for i in range(len(CLASSES))]
        )
        self.ax_bar.invert_yaxis()
        self.ax_bar.set_xlim(0, 1.15)
        self.ax_bar.set_xlabel("probabilidad")
        for i, (bar, p) in enumerate(zip(bars, probs)):
            tag = f"{p:.1%}" + ("  ← real" if i == true else "")
            self.ax_bar.text(p + 0.01, bar.get_y() + bar.get_height() / 2, tag, va="center", fontsize=9)
        for side in ("top", "right"):
            self.ax_bar.spines[side].set_visible(False)
        self.ax_bar.set_title(
            f"Predicción: {CLASSES[pred]} ({probs[pred]:.1%})  "
            + ("✓" if pred == true else "✗"),
            fontsize=11,
            loc="left",
            color=COLOR_OK if pred == true else COLOR_BAD,
        )
        self.fig.suptitle(f"{self.pos + 1} / {len(self.paths)}  ·  {path.name}", fontsize=8, y=0.99)
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Pesos del modelo (.pt).")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Carpeta de imágenes.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del orden de las imágenes.")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.exit(1, f"Error: no existe el modelo entrenado: {args.model}\n")
    try:
        paths = list_faces(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    device = get_device()
    viewer = FaceViewer(load_model(args.model, device), device, paths, args.seed)
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
