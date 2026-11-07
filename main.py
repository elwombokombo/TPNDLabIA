"""Punto 1: cargar las imágenes de UTKFace y sus edades."""

import argparse
from pathlib import Path

from src.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "UTKFace",
        help="Carpeta de imágenes (por defecto: data/UTKFace).",
    )
    args = parser.parse_args()

    try:
        images, df = load_dataset(args.dataset_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Imágenes cargadas: {len(images)}")
    print(f"Dimensiones por imagen: {images[0].shape}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
