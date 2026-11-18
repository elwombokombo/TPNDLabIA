"""Preparación de datos: etiquetas, conversión a tensores y división train/val/test."""

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

Split = tuple[torch.Tensor, torch.Tensor]  # (imágenes uint8 NHWC RGB, etiquetas)


def prepare_data(
    images: list[np.ndarray],
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[Split, Split, Split]:
    """Devuelve (train, val, test), estratificados por categoría de edad.

    Las imágenes de load_dataset son BGR; el modelo espera RGB, así que se
    convierten acá. La normalización con media/desvío de ImageNet la hace el modelo.
    """
    x = torch.from_numpy(np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]))
    y = torch.from_numpy(df["age_group"].cat.codes.to_numpy().astype(np.int64))

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y.numpy(), random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size / (1 - test_size),
        stratify=y.numpy()[idx_trainval],
        random_state=seed,
    )
    return tuple((x[i], y[i]) for i in (idx_train, idx_val, idx_test))
