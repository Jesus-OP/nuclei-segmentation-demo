# Notas de la noche (2026-11-16 → 17)

Trabajo autónomo, pasos 2 (cierre) a 7. Límite: ≤ 4 workers de CPU, sin GPU.

## Registro
- Paso 2 cerrado: parámetros elegidos solo en train (3 grids sucesivos, el último rodea el
  óptimo por todos los lados), test evaluado una sola vez con los parámetros congelados.
- Git: `git init -b main`, `negocio/` añadido a `.gitignore` antes del primer commit;
  `git status` comprobado: no aparecen `negocio/`, `data/` ni `runs/`. Push inicial OK
  (commit 3b43d5f).
- Refactor: la evaluación (agregación, variantes all/ge25) y la figura de ejemplos pasan a
  `bioseg/evaluation.py` y `bioseg/viz.py`, con tests (`tests/test_evaluation.py`), para
  que pasos 2, 3 y 5 usen exactamente el mismo código. Métricas del paso 2 idénticas tras
  el refactor. Corregido el título solapado de la figura del paso 2.
- Paso 3: licencias comprobadas (ver LICENSES.md): código y pesos BSD-3, datos de
  entrenamiento DSB 2018 = BBBC038 CC0 → compatible con uso comercial, se usa.
- Paso 3: en Windows `StarDist2D.from_pretrained` falla al crear un symlink (WinError 1314,
  falta de privilegio). La descarga y el sha256 ya se habían verificado; el script carga la
  carpeta extraída directamente en ese caso (sha256 del zip comprobado a mano: coincide).
- Paso 3: CPU forzada (`CUDA_VISIBLE_DEVICES=-1`; además TF >= 2.11 en Windows nativo no
  usa GPU), TensorFlow limitado a 4 hilos intra-op y 1 inter-op, OMP 4 hilos.
  ~0,8 s/imagen.
- Paso 4 (preparado, NO entrenado): `src/bioseg/unet.py` (U-Net 3 clases + watershed),
  `scripts/train.py`, `scripts/predict_unet.py` (inferencia en CPU, añadido para que la
  U-Net entre en `evaluate.py` sin tocar código), `notebooks/train_kaggle.ipynb`.
  Prueba de humo en CPU: 1 época, 4 imágenes train + 2 val, ~17 s, genera best.pt,
  last.pt, log.csv y config.json; `predict_unet.py` probado sobre 2 imágenes de val.
  El clonado desde Kaggle NO se ha probado (como pediste).
- Decisiones de entrenamiento (conservadoras, siguiendo a Caicedo et al. donde aplica):
  borde de 2 px entre núcleos que se tocan / 1 px contra fondo; pesos de clase 1/1/10;
  recortes 256x256, flips, rotaciones 90°, escalado de intensidad ±20 %; AdamW lr 1e-3
  con coseno, 150 épocas, batch 8; selección por val AP@0.5:0.95 cada 5 épocas.
  Se entrenan TODOS los objetos (también < 25 px); Caicedo los quitaba. Post-proceso:
  manchas de primer plano sin semilla interior se conservan como objeto propio (si no,
  los micronúcleos, que son todo "borde", desaparecerían).
- `torch` añadido como extra opcional `train` en pyproject; `tests/test_unet.py` se salta
  si no hay torch.
