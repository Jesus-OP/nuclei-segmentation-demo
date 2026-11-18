# Notas NuInsSeg (segunda demo, rama `nuinsseg`), 2026-11-18

Trabajo autónomo nocturno. Límites respetados: ≤ 4 hilos/procesos de CPU, GPU sin tocar,
nada entrenado en local (solo la prueba de humo de 1 época que pediste). Cada paso tiene su
commit en la rama `nuinsseg` (no en `main`). Código en `nuinsseg/scripts/`, datos en
`data/nuinsseg/`, resultados en `runs/nuinsseg/` (estas dos no se versionan).

**El mensaje con las instrucciones llegó cortado en el punto 5** ("5." sin texto). He hecho
los pasos 1 a 4 y no he inventado un paso 5.

## Paso 1: descarga, splits y licencia

- Fuente: Zenodo 10.5281/zenodo.10518968 (`NuInsSeg.zip`, 1.627.427.342 bytes, el tamaño
  se comprueba). Licencia CC BY 4.0, cita y cambios en `LICENSES.md`.
- Contenido: 665 parches H&E RGB 512x512 de 31 órganos (23 humanos, 8 de ratón), 30.698
  núcleos. Etiquetas: carpeta `label masks modify` (los autores la recomiendan para
  evaluar: en solapes, el píxel va a una sola instancia). Mediana del área de un núcleo:
  ~750 px (5 %–95 %: 200–2.100 px); solo el 0,05 % mide < 25 px, así que la variante
  "≥ 25 px" de bioseg apenas cambia nada aquí.
- 208 instancias están partidas en más de un trozo (efecto de resolver solapes). **No las
  modifico**: se evalúan como vienen.
- Máscaras de "zonas dudosas" (`vague areas/mask binary`): el 5–8 % de la superficie de
  cada split. Se guardan en `data/nuinsseg/vague/`.

### Splits (decisión)
El dataset **no trae split train/test oficial**: el README de los autores recomienda un
KFold de 5 particiones *por imagen* (semilla 19), que pone parches del mismo corte en
train y en test. Cada carpeta de órgano es una WSI (31 WSIs, 31 órganos), así que divido
**por órgano** (= por corte/paciente; no hay identificador de paciente en los datos).
Por especie, los órganos se recorren en orden aleatorio (semilla 0) y cada uno va al split
más alejado de su cuota (20 % test / 15 % val / 65 % train, en parches).

| Split | Órganos | Parches | Núcleos |
|---|---|---|---|
| train | 19 (14 humanos, 5 ratón) | 429 | 20.835 |
| val | 4 (3 humanos, 1 ratón) | 102 | 5.117 |
| test | 8 (6 humanos, 2 ratón) | 134 | 4.746 |

- test: human bladder, brain, liver, melanoma, pylorus, rectum; mouse muscle_tibia, thymus.
- val: human jejunum, peritoneum, salivory gland; mouse liver.
- Consecuencia: test mide la generalización **a órganos no vistos**, más exigente (y más
  honesta para un cliente con su propio tejido) que el KFold por imagen de los autores.
  Nuestros números no son comparables con los del artículo.
- Riesgo residual: los órganos de ratón pueden venir del mismo animal; humano y ratón del
  mismo tipo (p. ej. hígado) van a splits distintos, pero son cortes y especies distintas.
- Primer intento descartado: asignar a test "hasta llegar al 20 %" se pasaba (38 % del
  ratón en test por órganos grandes). Corregido antes de mirar ningún resultado.
- Tabla versionada en `nuinsseg/splits.csv`; el script falla si al regenerarla no coincide.
- `train_rank`: permutación fija (semilla 0) del train. "Entrenar con N imágenes" = las de
  `train_rank < N`, así que el subconjunto de 10 está dentro del de 25.

### Evaluación (decisión)
Misma métrica que en BBBC039 (`bioseg.evaluation`: TP/FP/FN sumados, AP = TP/(TP+FP+FN)).
Resultado principal **con todos los núcleos de referencia**. Como secundario, ignorando
zonas dudosas: se quitan de referencia *y* predicción los objetos con ≥ 50 % de sus píxeles
dentro de la máscara (`bioseg.evaluation.drop_in_mask`, opción `ignore_dir` de
`evaluate_dir`, sin efecto en BBBC039).

### Cambios en `bioseg` (sin romper BBBC039)
- `evaluation.drop_in_mask` y `evaluate_dir(..., ignore_dir=None)`.
- `unet.predict_probs` acepta imágenes (H, W, C) para U-Net con `in_ch=3`. Comprobado:
  las predicciones de la U-Net de BBBC039 son idénticas píxel a píxel (3 imágenes).
- `pytest -q`: 19 tests pasan (3 nuevos en `tests/test_nuinsseg.py`).
