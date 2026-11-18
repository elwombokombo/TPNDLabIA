# Predicción de rango etario

Proyecto en Python 3.10+ para clasificar rostros por rango de edad. Están implementadas la carga del dataset, las categorías de edad y género, el modelo, la evaluación y el visor; falta la preparación de datos (`src/preprocessing.py`).

## Estructura

```text
TPNDLabIA/
├── main.py                 # Ejecuta la carga del dataset.
├── requirements.txt        # Dependencias del punto 1.
├── data/
│   └── UTKFace/             # Colocar aquí las imágenes extraídas.
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Implementado: imágenes, edades y géneros.
│   ├── categories.py        # Implementado: categorías de edad y género.
│   ├── preprocessing.py     # Pendiente: preparar datos.
│   ├── model.py             # Implementado: MobileNetV2 (transfer learning) y entrenamiento.
│   ├── evaluation.py        # Implementado: métricas, gráficos y predicciones de ejemplo.
│   └── viewer.py            # Implementado: ventana interactiva con la cara y la predicción.
├── models/                 # Reservado para modelos entrenados.
└── reports/                # Reservado para resultados y gráficos.
```

## Preparar el entorno

Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Colocar el dataset

Descargar y extraer las imágenes de [UTKFace](https://susanqq.github.io/UTKFace/) dentro de `data/UTKFace/`. Las imágenes deben quedar directamente en esa carpeta, por ejemplo:

```text
data/UTKFace/25_0_0_20170116174525125.jpg
```

El dataset no está incluido en el repositorio. También se aceptan nombres que terminen en `.jpg.chip.jpg`.

## Ejecutar los puntos 1 y 2

```powershell
.\.venv\Scripts\python.exe main.py
```

Para usar otra carpeta:

```powershell
.\.venv\Scripts\python.exe main.py "C:\datasets\UTKFace"
```

La función `load_dataset` de `src/dataset.py` devuelve:

- `images`: una lista de imágenes de 128 × 128 píxeles, con tres canales BGR y valores enteros de 0 a 255.
- `df`: un DataFrame de pandas con las columnas `age` y `gender` (0 = hombre, 1 = mujer, `<NA>` si el nombre no trae un género válido), alineado con la lista de imágenes.

La carga recorre los archivos en orden, extrae la edad del nombre y omite imágenes ilegibles o nombres sin una edad válida, mostrando un aviso. No recorre subcarpetas. Si la carpeta no existe o no contiene imágenes válidas, muestra un error. Los datos se conservan en memoria durante la ejecución.

## Categorías de edad y género

`src/categories.py` crea las categorías a partir del DataFrame de `load_dataset`:

```python
from src.categories import add_categories

df = add_categories(df)  # agrega age_group y gender_group
```

| Columna | Valores |
|---|---|
| `age_group` | `child` (0–12), `young` (13–25), `adult` (26–59), `senior` (60+) |
| `gender_group` | `male` (código 0), `female` (código 1), vacío si el código falta o es inválido |

También expone `categorize_age`, `categorize_gender` y `age_to_class` (índice de clase para el modelo; `AGE_GROUPS` tiene el mismo orden que `CLASSES` en `src/model.py`). Los límites de edad siguen la definición principal de la consigna; el ejemplo de código de la consigna corta "adult" en 50, pero se usa 59. `main.py` muestra cuántas imágenes hay en cada categoría.

## Visor del modelo

Abre una ventana con una cara de UTKFace, su edad real y las probabilidades que predice el modelo para cada categoría (barra azul = categoría predicha, "← real" = categoría real). Requiere `models/model.pt` entrenado.

```powershell
.\.venv\Scripts\python.exe -m src.viewer
```

Botones **Anterior / Siguiente / Aleatoria**, o teclas ← → y `r`. Opciones: `--model`, `--data` y `--seed` (orden de las imágenes). Las imágenes se recorren de a una, sin cargar todo el dataset. Ojo: si el modelo se entrenó con parte de esas imágenes, también aparecerán las de entrenamiento.

## Etapas pendientes

Queda pendiente `src/preprocessing.py`: normalización, codificación de etiquetas y división de datos.
