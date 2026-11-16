# Notas de la noche (2026-11-16 → 17)

Trabajo autónomo, pasos 2 (cierre) a 7. Límite respetado: ≤ 4 workers/hilos de CPU en
todo lo lanzado, GPU sin tocar. `pytest -q`: 15 tests pasan. Todo está en GitHub
(`main`, último commit con los pasos 5-7; los pushes funcionaron todos).

## Métricas (split oficial de test, 50 imágenes, 5.720 núcleos)

Agregación: TP/FP/FN sumados sobre todas las imágenes. "≥ 25 px": filtro aplicado a
referencia **y** predicción. Fuente: `runs/results.csv`.

| Método | Variante | AP@0.5 | AP@0.5:0.95 | F1@0.5 | F1@0.5:0.95 | IoU medio (aciertos) |
|---|---|---|---|---|---|---|
| Otsu + watershed (paso 2) | todos | 0.831 | 0.684 | 0.908 | 0.798 | 0.915 |
| Otsu + watershed (paso 2) | ≥ 25 px | 0.862 | 0.709 | 0.926 | 0.814 | 0.915 |
| StarDist 2D_versatile_fluo sin ajustar (paso 3) | todos | 0.906 | 0.754 | 0.951 | 0.833 | 0.912 |
| StarDist 2D_versatile_fluo sin ajustar (paso 3) | ≥ 25 px | 0.940 | 0.782 | 0.969 | 0.849 | 0.912 |
| U-Net propia | — | pendiente | | | | |

Referencia en val (para ver que no hay sobreajuste): Otsu AP@0.5:0.95 = 0.662 (todos),
StarDist 0.751. Parámetros de Otsu (elegidos en train): sigma 0, otsu_factor 0.9,
min_distance 11, min_size 25.

Lectura honesta: StarDist sin ajustar ya es bueno (F1@0.5 = 0.95), como avisaba CLAUDE.md.
Cuando ambos métodos aciertan, el contorno es igual de bueno (IoU ≈ 0.91); la diferencia
está en separar núcleos pegados (detectados: 82 % StarDist frente a 70 % Otsu).

Desenfoque simulado (test, `runs/robustness_blur.csv`), AP@0.5:0.95 con todos los objetos:

| σ (px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| Otsu + watershed | 0.684 | 0.656 | 0.605 | 0.482 |
| StarDist | 0.754 | 0.753 | 0.733 | 0.616 |

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
- Pasos 5-7: `evaluate.py`, `quantify.py`, `report.py`, más `robustness_blur.py`
  (ver decisiones). Informe: `runs/report.html` (2,4 MB, autocontenido).

## Decisiones tomadas (lo ambiguo, resuelto de forma conservadora)
1. **Desenfoque.** BBBC039 no tiene campos desenfocados: miré en train/val las imágenes con
   menor varianza del laplaciano y eran nítidas (la métrica la movían objetos saturados y
   una franja de suciedad). En vez de inventar una categoría "desenfocada" con una métrica
   que no funciona, añadí `scripts/robustness_blur.py`: desenfoque gaussiano sintético
   sobre test, sin reajustar nada. Es un script extra que no estaba en la lista.
2. **"Bordes" en casos de fallo** lo interpreté como núcleos cortados por el borde de la
   imagen. Se cuantifica en la tabla de detección por grupo (89 % frente a 92 % interiores
   con StarDist: no es un fallo grave), sin recorte propio. En su lugar el informe muestra un
   recorte de sobresegmentación (un núcleo lobulado que StarDist parte en tres), que es más
   representativo.
3. **Ejemplos del informe:** 6 imágenes de test repartidas por percentiles del AP por imagen
   del mejor método (de la mejor a la peor), no elegidas a mano.
4. **Cuantificación:** núcleos que tocan el borde del campo se marcan (`touches_border`) y se
   excluyen de los resúmenes por imagen, pero no se borran de la tabla. Áreas en píxeles:
   la página de BBBC039 no da tamaño de píxel (opción `--pixel-size-um` si lo tienes).
   Por defecto `quantify.py` usa las predicciones de StarDist (el mejor método hoy).
5. **Informe solo en modo claro**: es un documento para cliente e imprimible; las figuras
   son PNG embebidos (sin interactividad) y cada gráfica tiene su tabla de datos al lado.
   Números en formato español (coma decimal) en texto y tablas.
6. **Textos del informe generados a partir de los datos**: el "mejor método", el sentido de
   los sesgos y las frases de desenfoque se calculan; al añadir la U-Net el resumen se
   reescribe solo. La fila de la U-Net sale como "pendiente".
7. `evaluate.py` acepta `--method "NOMBRE=CARPETA"` (repetible). Si falta alguna predicción
   en la carpeta, falla en vez de evaluar un split parcial.

## Qué debes revisar tú
1. **Abrir `runs/report.html`** y leer el resumen con ojos de cliente (tono, claims).
2. **Solapamiento DSB 2018 / BBBC039.** Ambos salen en parte de colecciones del Broad. Si
   alguna imagen de test de BBBC039 estuvo en el entrenamiento de StarDist, su resultado es
   optimista. No lo he podido comprobar (habría que cruzar `metadata.xlsx` de BBBC038 con
   los nombres de BBBC039). Está como limitación en el informe y en LICENSES.md.
3. **Licencia de los pesos de StarDist:** no hay licencia específica del modelo; asumo la
   BSD-3 del repo `stardist-models` que los aloja. Es la lectura habitual, pero no es una
   declaración explícita. Si quieres certeza total para uso comercial, pregunta a los
   autores o apóyate solo en la U-Net propia.
4. **Kaggle:** el notebook clona el repo público por HTTPS; si el repo es privado, el
   clonado falla (habría que usar un token como *secret* de Kaggle). Activar *Internet* y
   *GPU* en Settings. El tiempo de entrenamiento (< 1 h en T4) es una estimación sin medir.
5. **Hiperparámetros de la U-Net** (sección "Decisiones de entrenamiento" arriba): sobre
   todo mantener los objetos < 25 px en el entrenamiento, al contrario que Caicedo.
6. **`CLAUDE.md` está en el repo público** (se subió en el primer commit, como el resto
   salvo `negocio/`). Describe tu perfil y el plan de servicio; si no quieres que sea
   público, sácalo con `git rm --cached CLAUDE.md` y añádelo a `.gitignore`.
7. **README (paso 8)** no está hecho: la tabla del README sigue con guiones.
8. Comparación con Caicedo et al.: ver `docs/notes_evaluate.md`. Ellos reportan F1 medio en
   0.50:0.95 contra la referencia completa; lo comparable es nuestra variante "todos"
   (StarDist 0.833, Otsu 0.798). No he buscado su cifra exacta del artículo.

## Siguientes comandos (cuando tengas los pesos de Kaggle)
```
# descomprimir unet_weights.zip en runs/unet/
python scripts/predict_unet.py --weights runs/unet/best.pt --split test
python scripts/evaluate.py
python scripts/quantify.py --labels runs/unet/pred/test     # opcional
python scripts/report.py
```
