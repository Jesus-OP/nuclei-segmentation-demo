# Licencias de datos y pesos

| Recurso | Licencia | Fuente | Comprobado el |
|---|---|---|---|
| BBBC039v1 (imagenes, mascaras, splits) | CC0 1.0 (dominio publico; los contribuidores renuncian a todos los derechos) | https://bbbc.broadinstitute.org/BBBC039 | 2026-11-16 |
| StarDist (codigo) | BSD-3-Clause (confirmar) | https://github.com/stardist/stardist | |
| StarDist 2D_versatile_fluo (pesos) | (confirmar datos de entrenamiento) | | |

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
