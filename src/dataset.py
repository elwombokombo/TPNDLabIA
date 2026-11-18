"""Carga del dataset UTKFace: imágenes, edades y géneros, sin preparar entrenamiento."""

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd


def load_dataset(dataset_dir: str | Path) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Devuelve imágenes BGR de 128 x 128 y un DataFrame con las columnas age y gender.

    La posición de cada imagen coincide con su fila. gender es el código de
    UTKFace (0 = hombre, 1 = mujer) o <NA> si el nombre no lo trae bien; esas
    imágenes se conservan porque la edad sigue siendo válida. Se omiten
    archivos con edad inválida o imágenes que no se puedan decodificar.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {dataset_path}")

    images: list[np.ndarray] = []
    ages: list[int] = []
    genders: list[int | None] = []
    skipped = 0

    for image_path in sorted(dataset_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        parts = image_path.name.split("_")
        try:
            age = int(parts[0])
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
        genders.append(_parse_gender(parts))

    if not images:
        raise ValueError(f"No se encontraron imágenes válidas en: {dataset_path}")
    if skipped:
        warnings.warn(
            f"Se omitieron {skipped} archivos con edad inválida o imagen ilegible.",
            UserWarning,
            stacklevel=2,
        )

    df = pd.DataFrame({"age": ages, "gender": pd.array(genders, dtype="Int64")})
    return images, df


def _parse_gender(parts: list[str]) -> int | None:
    """Código de género (0 o 1) del nombre age_gender_race_date, o None."""
    if len(parts) < 3 or parts[1] not in {"0", "1"}:
        return None
    return int(parts[1])
