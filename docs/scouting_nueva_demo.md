# Búsqueda de datasets para la segunda demo (2026-11-17)

Objetivo: datos públicos con anotación de instancias, licencia compatible con uso
comercial, donde StarDist `2D_versatile_fluo` sin ajustar rinda mal. Nada entrenado.

Scripts y resultados (no versionados, `data/` está en .gitignore):
`data/external/scout/_scripts/` (fetch.py, run_stardist.py, httpzip.py) y
`data/external/scout/_results/` (pooled.csv, per_image.csv, examples.png).

## Licencias (comprobadas en la fuente el 2026-11-17)

| Dataset | Licencia | Dónde consta |
|---|---|---|
| DSB 2018 stage1_train (BBBC038), parte no fluorescente | CC0 | https://bbbc.broadinstitute.org/BBBC038 (ya en LICENSES.md) |
| NuInsSeg (Mahbod et al., Sci Data 2024) | CC BY 4.0 | Zenodo 10.5281/zenodo.10518968, campo license = cc-by-4.0 |
| Kromp et al. 2020, fluorescencia DAPI (S-BSST265 / S-BIAD634) | CC0 | readme.txt del propio dataset ("open data license (CC0)") y ficha S-BIAD634 del BioImage Archive |
| NIS3D (NeurIPS 2023 D&B) | CC BY 4.0 | Zenodo 10.5281/zenodo.11456029; datos de origen IDR0044, IDR0051, IDR0118: CC BY 4.0 (fichas IDR) |

CC BY 4.0 exige atribución (citar el artículo/dataset) pero permite uso comercial.

Nota: StarDist `2D_versatile_he` se entrenó con MoNuSeg (CC BY-NC-SA), así que no sirve para
el servicio comercial; para H&E no hay un StarDist preentrenado utilizable.

## Prueba rápida: StarDist 2D_versatile_fluo sin ajustar, 5 imágenes por dataset

AP@0.5 = TP / (TP + FP + FN) sumando las 5 imágenes (misma definición que en BBBC039,
donde StarDist da 0,906). Imágenes elegidas al azar con semilla 0. Umbrales por defecto,
normalización 1–99,8, CPU 4 hilos. En campo claro/H&E se prueba la luminancia tal cual y
también invertida (núcleos brillantes, como en fluorescencia).

| Dataset | Entrada | AP@0.5 | Recall | Precisión | AP por imagen |
|---|---|---|---|---|---|
| DSB no fluorescente | tal cual | 0,01 | 0,02 | 0,03 | 0,02 0,00 0,00 0,01 0,04 |
| | invertida | 0,40 | 0,63 | 0,53 | 0,52 0,20 0,19 0,60 0,66 |
| NuInsSeg (H&E) | tal cual | 0,01 | 0,02 | 0,02 | 0,00 0,04 0,01 0,00 0,00 |
| | invertida | 0,25 | 0,48 | 0,35 | 0,06 0,28 0,45 0,16 0,09 |
| Kromp, secciones de tejido | fluorescencia | 0,64 | 0,86 | 0,71 | 0,85 0,69 0,69 0,42 0,36 |
| NIS3D (corte 2D de cada volumen) | fluorescencia | 0,26 | 0,32 | 0,60 | 0,77 0,81 0,38 0,31 0,00 |

Advertencias:
- 5 imágenes: estimación muy ruidosa (ver la dispersión por imagen).
- NuInsSeg trae máscaras de "zonas dudosas" que aquí no se excluyen (AP algo pesimista).
- Kromp: en algunas máscaras una misma etiqueta cubre dos manchas separadas; se separan
  en componentes conexas (otherspecimen_3: 55 etiquetas, 103 núcleos).
- NIS3D: no hay StarDist 3D preentrenado de uso general; se evalúa el modelo 2D en el corte
  z con más núcleos de cada volumen. Zebrafish_2 (núcleos de pocos píxeles) da 0,00: es
  sobre todo un problema de escala.

## Tamaños

| Dataset | Tipo de imagen | Tamaño |
|---|---|---|
| DSB no fluorescente | RGB 8 bits, histología teñida y campo claro, varios aumentos | 124 imágenes, 5.920 núcleos (stage1_train) |
| NuInsSeg | H&E RGB 512×512, 31 órganos humanos y de ratón | 665 parches, 30.698 núcleos, zip 1,6 GB |
| Kromp et al. | DAPI 8 bits, 10x–63x, preparaciones celulares y criosecciones | 79 imágenes, 7.813 núcleos, split train/test con 10 clases de test |
| NIS3D | 3D light-sheet, embriones de Drosophila, ratón y pez cebra (16 bits) | 6 volúmenes (cientos a miles de núcleos cada uno), zip 3,3 GB |

## Recomendación: NuInsSeg
Ver la respuesta en la conversación del 2026-11-17; resumen: mayor margen de mejora
(AP 0,25 en el mejor caso), anotación completa por instancia, 31 órganos, CC BY 4.0, y H&E
es el mercado donde no hay alternativa preentrenada comercial (versatile_he es NC).
