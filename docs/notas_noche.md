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
