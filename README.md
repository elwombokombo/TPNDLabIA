# Predicción de rango etario

Proyecto en Python 3.10+ para clasificar rostros por rango de edad. Por ahora está implementado **únicamente el punto 1: cargar el dataset**. Las demás etapas tienen su archivo reservado, sin implementación.

## Estructura

```text
TPNDLabIA/
├── main.py                 # Ejecuta la carga del dataset.
├── requirements.txt        # Dependencias del punto 1.
├── data/
│   └── UTKFace/             # Colocar aquí las imágenes extraídas.
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Implementado: imágenes y edades.
│   ├── categories.py        # Pendiente: categorías de edad.
│   ├── preprocessing.py     # Pendiente: preparar datos.
│   ├── model.py             # Implementado: MobileNetV2 (transfer learning) y entrenamiento.
│   └── evaluation.py        # Implementado: métricas, gráficos y predicciones de ejemplo.
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

## Ejecutar el punto 1

```powershell
.\.venv\Scripts\python.exe main.py
```

Para usar otra carpeta:

```powershell
.\.venv\Scripts\python.exe main.py "C:\datasets\UTKFace"
```

La función `load_dataset` de `src/dataset.py` devuelve:

- `images`: una lista de imágenes de 128 × 128 píxeles, con tres canales BGR y valores enteros de 0 a 255.
- `df`: un DataFrame de pandas con la columna `age`, alineada con la lista de imágenes.

La carga recorre los archivos en orden, extrae la edad del nombre y omite imágenes ilegibles o nombres sin una edad válida, mostrando un aviso. No recorre subcarpetas. Si la carpeta no existe o no contiene imágenes válidas, muestra un error. Los datos se conservan en memoria durante la ejecución.

## Etapas pendientes

Las etapas de categorización, normalización, división de datos, entrenamiento y evaluación quedan pendientes. Las dependencias de esas etapas se agregarán cuando se implementen.

Las categorías previstas son: niño de 0 a 12 años, joven de 13 a 25, adulto de 26 a 59 y mayor de 60 en adelante. Se toman estos límites de la definición principal de la consigna.
