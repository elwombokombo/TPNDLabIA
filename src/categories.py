"""Punto 2: categorías de edad y de género.

Edad: las cuatro categorías de la consigna (niño 0-12, joven 13-25,
adulto 26-59, mayor 60+). El ejemplo de código de la consigna corta "adult"
en 50, pero se usa la definición principal (59), como indica el README.

Género: UTKFace lo codifica en el nombre del archivo (age_gender_race_date.jpg)
como 0 = hombre y 1 = mujer.
"""

import pandas as pd

# Mismo orden que src.model.CLASSES: el índice de cada nombre es su clase.
AGE_GROUPS = ["child", "young", "adult", "senior"]
GENDERS = ["male", "female"]  # índice = código de UTKFace

_AGE_UPPER_BOUNDS = [(12, "child"), (25, "young"), (59, "adult")]


def categorize_age(age: int) -> str:
    """Nombre de la categoría de edad (child, young, adult o senior)."""
    if age < 0:
        raise ValueError(f"Edad inválida: {age}")
    for upper, name in _AGE_UPPER_BOUNDS:
        if age <= upper:
            return name
    return "senior"


def age_to_class(age: int) -> int:
    """Índice de la categoría de edad, para usar como etiqueta del modelo."""
    return AGE_GROUPS.index(categorize_age(age))


def categorize_gender(code) -> str | None:
    """'male' o 'female' según el código de UTKFace; None si falta o es inválido."""
    if code is None or pd.isna(code) or int(code) not in (0, 1):
        return None
    return GENDERS[int(code)]


def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con age_group y, si existe la columna gender,
    gender_group. Ambas son columnas categóricas con el orden de AGE_GROUPS y
    GENDERS (un género desconocido queda como NaN).
    """
    out = df.copy()
    out["age_group"] = pd.Categorical(
        out["age"].map(categorize_age), categories=AGE_GROUPS, ordered=True
    )
    if "gender" in out.columns:
        out["gender_group"] = pd.Categorical(
            out["gender"].map(categorize_gender), categories=GENDERS
        )
    return out
