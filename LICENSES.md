# Licencias de datos y pesos

| Recurso | Licencia | Fuente | Comprobado el |
|---|---|---|---|
| BBBC039v1 (imagenes, mascaras, splits) | CC0 1.0 (dominio publico; los contribuidores renuncian a todos los derechos) | https://bbbc.broadinstitute.org/BBBC039 | 2026-11-16 |
| StarDist (codigo, v0.9.2) | BSD-3-Clause (LICENSE del paquete instalado: (c) 2018-2025 U. Schmidt, M. Weigert) | https://github.com/stardist/stardist | 2026-11-17 |
| StarDist 2D_versatile_fluo (pesos) | BSD-3-Clause: licencia del repo que los aloja (`stardist-models`, release v0.1); no hay licencia especifica del modelo. Se descargan en tiempo de ejecucion, no se redistribuyen | https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip (sha256 4ad678d0...c3dd) | 2026-11-17 |
| Datos de entrenamiento de 2D_versatile_fluo | Subconjunto de DSB 2018 stage1_train (segun el README de StarDist y la release 0.1.0) = BBBC038v1, **CC0** | https://bbbc.broadinstitute.org/BBBC038 | 2026-11-17 |

## Citas requeridas / recomendadas

**BBBC039v1** — texto indicado en la pagina del dataset:
> "We used image set BBBC039v1 Caicedo et al. 2018, available from the Broad Bioimage
> Benchmark Collection [Ljosa et al., Nature Methods, 2012]."

- Caicedo JC, Roth J, Goodman A, et al. Evaluation of Deep Learning Strategies for Nucleus
  Segmentation in Fluorescence Images. Cytometry Part A, 2019 (preprint bioRxiv 2018).
- Ljosa V, Sokolnicki KL, Carpenter AE. Annotated high-throughput microscopy image sets for
  validation. Nature Methods 9, 637 (2012).

Nota: el decodificado de mascaras de `scripts/download_data.py` es codigo propio (no copia
el gist de ejemplo de los autores, que no declara licencia).

**StarDist** (si se usa el baseline preentrenado), citar:
- Schmidt U, Weigert M, Broaddus C, Myers G. Cell Detection with Star-convex Polygons. MICCAI 2018.
- Datos de entrenamiento: "We used image set BBBC038v1, available from the Broad Bioimage
  Benchmark Collection [Caicedo et al., Nature Methods, 2019]."

Conclusion (2026-11-17): StarDist 2D_versatile_fluo es compatible con uso comercial
(codigo y pesos BSD-3, datos CC0). Riesgo (no de licencia) comprobado el 2026-11-17 con
`scripts/check_dsb_overlap.py`: 9 de las 50 imagenes de test de BBBC039 estan en DSB 2018
stage1_train (25 de train, 9 de val). Sin ellas, StarDist AP@0.5:0.95 = 0.754 (igual que con
las 50); ver README. stage1_train.zip (BBBC038, CC0) se descarga solo para esta comprobacion
y no se redistribuye.

## Segunda demo: NuInsSeg (H&E), rama `nuinsseg`

| Recurso | Licencia | Fuente | Comprobado el |
|---|---|---|---|
| NuInsSeg (665 parches H&E 512x512, 30.698 nucleos, mascaras de zonas dudosas) | CC BY 4.0 (uso comercial permitido; **atribucion obligatoria**) | Zenodo, campo license = cc-by-4.0: https://doi.org/10.5281/zenodo.10518968 | 2026-11-17 y 2026-11-18 |
| StarDist 2D_versatile_he | **No se usa**: entrenado con MoNuSeg (CC BY-NC-SA, no comercial) | https://github.com/stardist/stardist | 2026-11-17 |

Cita obligatoria (CC BY 4.0) de NuInsSeg:
> Mahbod A, Polak C, Feldmann K, Khan R, Gelles K, Dorffner G, Woitek R, Hatamikia S,
> Ellinger I. NuInsSeg: A fully annotated dataset for nuclei instance segmentation in
> H&E-stained histological images. Scientific Data 11, 295 (2024).
> https://doi.org/10.1038/s41597-024-03117-2 . Datos: https://doi.org/10.5281/zenodo.10518968

Cambios respecto al original (CC BY 4.0 pide indicarlos): las imagenes PNG se guardan como
TIFF sin perdida, las mascaras de `label masks modify` se guardan como uint16 sin tocar, y
se define una division train/val/test por organo propia (`nuinsseg/splits.csv`); los datos
no se redistribuyen, `nuinsseg/scripts/download_nuinsseg.py` los descarga de Zenodo.
