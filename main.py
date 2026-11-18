"""Puntos 1 y 2: cargar UTKFace y crear las categorías de edad y género."""

import argparse
from pathlib import Path

from src.categories import add_categories
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

    df = add_categories(df)

    print(f"Imágenes cargadas: {len(images)}")
    print(f"Dimensiones por imagen: {images[0].shape}")
    print(df.head().to_string(index=False))
    print("\nImágenes por categoría de edad:")
    print(df["age_group"].value_counts(sort=False).to_string())
    print("\nImágenes por género:")
    print(df["gender_group"].value_counts(sort=False, dropna=False).to_string())


if __name__ == "__main__":
    main()
