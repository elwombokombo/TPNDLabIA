"""Puntos 1 y 2: cargar UTKFace y crear las categorías de edad y género."""

import argparse
import os
import subprocess
import sys
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
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrena el modelo, lo evalúa y guarda gráficos y predicciones en reports/.",
    )
    parser.add_argument("--no-show", action="store_true", help="No abrir los gráficos al terminar.")
    parser.add_argument("--epochs", type=int, default=5, help="Épocas con el backbone congelado.")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Épocas de fine-tuning.")
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

    if args.train:
        train_and_evaluate(images, df, args.epochs, args.finetune_epochs, not args.no_show)


def open_file(path) -> None:
    """Abre un archivo con el visor predeterminado del sistema."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)], check=False)


def train_and_evaluate(images, df, epochs: int, finetune_epochs: int, show: bool = True) -> None:
    """Entrena, evalúa y guarda curvas, matriz de confusión y predicciones de ejemplo."""
    # Imports acá para que correr main.py sin --train no requiera torch.
    from src.evaluation import evaluate, plot_training, predict_examples
    from src.model import get_device, save_model, train_model
    from src.preprocessing import prepare_data

    device = get_device()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(images, df)
    print(f"\nTrain/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}  (dispositivo: {device})")

    model, history = train_model(
        x_train, y_train, x_val, y_val, epochs=epochs, finetune_epochs=finetune_epochs, device=device
    )
    print(f"Modelo guardado en {save_model(model)}")

    _, cm = evaluate(model, x_test, y_test, device)
    plots = [plot_training(history, cm), predict_examples(model, x_test, y_test, device)]
    print("Gráficos guardados en:", *plots, sep="\n  ")
    if show:
        for plot in plots:
            open_file(plot)


if __name__ == "__main__":
    main()
