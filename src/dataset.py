"""Carga del dataset UTKFace: imágenes y edades, sin preparar entrenamiento."""

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd


def load_dataset(dataset_dir: str | Path) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Devuelve imágenes BGR de 128 x 128 y un DataFrame con la columna age.

    La posición de cada imagen coincide con la fila de su edad. Se omiten
    archivos con nombres inválidos o imágenes que no se puedan decodificar.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {dataset_path}")

    images: list[np.ndarray] = []
    ages: list[int] = []
    skipped = 0

    for image_path in sorted(dataset_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        try:
            age = int(image_path.name.split("_", 1)[0])
        except ValueError:
            skipped += 1
            continue
        if age < 0:
            skipped += 1
            continue

        try:
            # fromfile + imdecode admite rutas con tildes en Windows.
            encoded_image = np.fromfile(image_path, dtype=np.uint8)
            if encoded_image.size == 0:
                skipped += 1
                continue
            img = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            skipped += 1
            continue
        if img is None:
            skipped += 1
            continue

        images.append(cv2.resize(img, (128, 128)))
        ages.append(age)

    if not images:
        raise ValueError(f"No se encontraron imágenes válidas en: {dataset_path}")
    if skipped:
        warnings.warn(
            f"Se omitieron {skipped} archivos con edad inválida o imagen ilegible.",
            UserWarning,
            stacklevel=2,
        )

    return images, pd.DataFrame({"age": ages})
