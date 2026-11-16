# Nuclei segmentation & quantification — reproducible demo

End-to-end pipeline for instance segmentation of nuclei in fluorescence microscopy,
with validation against manual annotations and per-object measurements.
**Research use only.**

> Status: scaffold. Results below are filled in by `scripts/evaluate.py`.

## What you get
- Segmentation masks (one label per nucleus)
- A table with one row per nucleus: area, shape and intensity features (CSV)
- A validation report: accuracy against ground truth, examples, known failure cases

## Results on BBBC039 (official test split)

| Method | AP@0.5 | AP@0.5:0.95 | F1@0.5 | Mean IoU |
|---|---|---|---|---|
| Otsu + watershed | – | – | – | – |
| Pretrained model (no tuning) | – | – | – | – |
| Trained on this dataset | – | – | – | – |

## Reproduce
```
pip install -e ".[dev,models]"
python scripts/download_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/report.py   # -> runs/report.html
```

## Data
BBBC039v1, Broad Bioimage Benchmark Collection (Caicedo et al., Cytometry Part A, 2019;
Ljosa et al., Nature Methods, 2012). See `LICENSES.md`.

## Contact
<name> — <email> — custom segmentation and quantification pipelines for research imaging.
