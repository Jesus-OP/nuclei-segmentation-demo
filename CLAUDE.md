# CLAUDE.md — contexto del proyecto

## Qué es esto
Demo pública de portfolio para un servicio freelance de análisis de bioimagen
(solo investigación, "research use only"). Debe parecer un **entregable real a un
cliente**: no un notebook de Kaggle, sino pipeline reproducible + validación + informe.

Autor: estudiante de 4.º de Ing. Biomédica (UA), especializado en IA para imagen médica
(2 papers en workshops de MICCAI, 3.er puesto en BraTS-PEDs). Viene de MRI 3D, no de
microscopía: explica brevemente cualquier convención propia de microscopía.

## Alcance de la demo (semana 1, no ampliar sin preguntar)
Tarea: segmentación de instancias de núcleos en fluorescencia 2D + cuantificación por objeto.
Dataset: **BBBC039** (Broad Bioimage Benchmark Collection): 200 imágenes TIFF 520x696 de
16 bits, ~23.000 núcleos anotados a mano. https://bbbc.broadinstitute.org/BBBC039

Entregables, en este orden:
1. `scripts/download_data.py` — descarga imágenes, máscaras y los splits oficiales
   (train/val/test) desde la página del dataset a `data/`. Leer la página para las URLs
   y el formato de las máscaras (vienen como PNG coloreados: convertir a etiquetas).
2. `scripts/baseline_classical.py` — Otsu + watershed con scikit-image. Es el "antes".
3. `scripts/baseline_pretrained.py` — StarDist `2D_versatile_fluo` sin ajustar.
4. `scripts/train.py` — modelo entrenado en el split train (StarDist o U-Net propia).
5. `scripts/evaluate.py` — usa `bioseg.metrics` sobre el split test: AP@0.5, AP@0.5:0.95,
   F1, IoU medio. Tabla comparativa de los 3 métodos en `runs/results.csv`.
6. `scripts/quantify.py` — `bioseg.measure.per_object_table` → CSV por imagen.
7. `scripts/report.py` — informe HTML autocontenido (`runs/report.html`): resumen de
   una página, tabla de métricas, 6 ejemplos (imagen / GT / predicción), **casos de fallo**
   (núcleos pegados, bordes, desenfoque) y limitaciones. Tono: informe para un cliente.
8. README: rellenar resultados y una figura.

Extensión opcional (solo si sobra tiempo): versión 3D con otro dataset abierto.

## Reglas
- **Licencias (crítico, esto es un servicio comercial):**
  - No usar Cellpose ni sus pesos: sus modelos están entrenados con datos CC-BY-NC.
  - Antes de usar cualquier peso preentrenado o dataset, comprobar su licencia y
    anotarla en `LICENSES.md`. Si hay duda, entrenar desde cero.
  - Confirmar la licencia de BBBC039 en su página y citarlo como allí se indica.
- Nada de datos en git (`data/` y `runs/` están en .gitignore).
- Semillas fijas, splits oficiales, sin tocar test hasta la evaluación final.
- No inflar resultados: si el baseline preentrenado ya es casi perfecto, decirlo y
  centrar el valor en validación, cuantificación e informe.
- Código simple: scripts con `argparse`, funciones cortas, sin frameworks de configuración.
- `pytest -q` debe pasar siempre. `bioseg/metrics.py` y `measure.py` ya tienen tests.
- Hardware: PC Windows. La GPU local está ocupada: todo en CPU salvo train.py, que se
  ejecuta en Kaggle. No lanzar entrenamientos en local. El modelo propio es una U-Net
  en PyTorch; StarDist/TensorFlow solo para inferencia del baseline.

## Comandos
```
conda activate bioseg
pytest -q
```
