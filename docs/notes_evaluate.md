# Notas para `scripts/evaluate.py`

## Decisiones acordadas (2026-11-16)
1. **Métricas dobles**: con todos los objetos de referencia y con objetos >= 25 px.
2. En la variante >= 25 px el filtro se aplica a **referencia y predicción** (objetos con
   área < 25 px eliminados de ambas antes de emparejar). Si no, las detecciones pequeñas
   correctas contarían como falsos positivos.
3. Agregación: TP/FP/FN sumados sobre todas las imágenes del split en cada umbral de IoU
   (no media de métricas por imagen). AP = TP / (TP + FP + FN) (estilo DSB 2018).

## Qué hacían Caicedo et al. (repo `carpenterlab/unet4nuclei`, commit 5575e61, 2018-09-13)
Fuente: `unet4nuclei/config.py`, `01-preprocessing.ipynb`, `04-evaluation.ipynb`,
`utils/evaluation.py`.

| Etapa | Qué hacen | Valor |
|---|---|---|
| Preparación de etiquetas para **entrenar** | `remove_small_objects` sobre la referencia, comentado "filter small objects, e.g. micronuclei" | `min_nucleus_size = 25` px (quita área < 25) |
| Post-proceso de **predicciones** | `remove_small_objects` + `remove_small_holes` | `cell_min_size = 16` px |
| Post-proceso de predicciones antes de evaluar | dilatación con cuadrado 3x3 (compensa el borde de 2 px que su U-Net predice como clase aparte) | `object_dilation = 3` |
| **Evaluación** | la referencia se carga sin filtrar (canal R + `label`, conectividad 8) | sin umbral |
| Emparejamiento | `IoU > t` (estricto), sin húngaro (con IoU > 0.5 el emparejamiento es único) | t = 0.50:0.05:0.95 |
| Métrica principal | **F1** en cada umbral y su media ("AF1"); no AP estilo DSB | — |
| Otros | "Jaccard" = media sobre predicciones del IoU máximo; FN por tamaño: tiny < 250, small < 625, normal < 900 px | — |

Consecuencias para comparar con su artículo:
- Su umbral de 25 px coincide con el nuestro, pero ellos solo lo aplicaron al
  **entrenamiento**: evaluaban contra **toda** la referencia y su predicción descartaba
  objetos < 16 px. Lo comparable con sus números es nuestra variante "todos los objetos"
  (y aun así su predicción nunca podía acertar los objetos < 16 px).
- Para comparar hay que dar **F1 medio en 0.50:0.95**, además de nuestro AP.
- Diferencias menores: `>` frente a `>=` en el umbral de IoU; conectividad 8 frente a 4
  en la decodificación (afecta a 2 objetos en todo el dataset).
