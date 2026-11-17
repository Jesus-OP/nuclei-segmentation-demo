"""Self-contained HTML validation report -> runs/report.html.

Reads only files produced by the other scripts (no model is run here):
    runs/results.csv, runs/eval/*_per_image.csv, runs/eval/*_objects.csv   (evaluate.py)
    runs/robustness_blur.csv                                             (robustness_blur.py)
    runs/results_excl_dsb.csv  (optional; check_dsb_overlap.py, then evaluate.py)
    runs/quant/<method>/summary.csv, runs/quant/reference/summary.csv     (quantify.py)
Figures are embedded as base64 PNG, so the HTML is a single file that can be e-mailed.
"""
import argparse
import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage import measure, segmentation

from bioseg.evaluation import slug, stems_of
from bioseg.viz import normalize

# Reference categorical palette (dataviz skill, slots 1-3: validated all-pairs, light mode)
COLORS = {"Otsu + watershed": "#2a78d6",
          "StarDist 2D_versatile_fluo (sin ajustar)": "#eb6834",
          "U-Net (entrenada en BBBC039)": "#1baf7a"}
SHORT = {"Otsu + watershed": "Otsu + watershed",
         "StarDist 2D_versatile_fluo (sin ajustar)": "StarDist (preentrenado)",
         "U-Net (entrenada en BBBC039)": "U-Net propia"}
EXTRA_COLORS = ["#eda100", "#6250d6", "#e87ba4"]  # slots 4+, only for methods added later
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def short(m: str) -> str:
    return SHORT.get(m, m)


def color(m: str) -> str:
    if m not in COLORS:
        COLORS[m] = EXTRA_COLORS[(len(COLORS) - 3) % len(EXTRA_COLORS)]
    return COLORS[m]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


# ---------------------------------------------------------------- helpers
def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure_html(src: str, caption: str) -> str:
    return f'<figure><img src="{src}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def table_html(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                s = "—"
            elif c in fmt:
                s = fmt[c](v)
            else:
                s = html.escape(str(v))
            cells.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def es(v, d: int = 2) -> str:
    """Spanish decimal comma."""
    return f"{v:.{d}f}".replace(".", ",")


def pct(v) -> str:
    return es(100 * v, 1) + " %"


def signed_pct(v) -> str:
    return f"{100 * v:+.1f} %".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def load(data: Path, stem: str, pred_dirs: dict):
    img = normalize(tifffile.imread(data / "images" / f"{stem}.tif"))
    gt = tifffile.imread(data / "labels" / f"{stem}.tif")
    preds = {m: tifffile.imread(Path(d) / f"{stem}.tif") for m, d in pred_dirs.items()}
    return img, gt, preds


def filled(ax, img, labels, title):
    """Each instance a random colour over the image (legible at full-image scale)."""
    from skimage.color import label2rgb
    colors = np.random.default_rng(0).random((64, 3)) * 0.8 + 0.2
    ax.imshow(label2rgb(labels, img, bg_label=0, alpha=0.45, colors=colors))
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")


def overlay(ax, img, labels, color, title, crop=None):
    sl = crop or (slice(None), slice(None))
    ax.imshow(img[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if labels is not None:
        b = segmentation.find_boundaries(labels[sl], mode="inner")
        rgba = np.zeros(b.shape + (4,))
        rgba[b] = matplotlib.colors.to_rgba(color)
        ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")


# ---------------------------------------------------------------- figures
def fig_examples(data, pred_dirs, per_image) -> str:
    """6 test images spread over the per-image AP range of the best method."""
    ranked = per_image.sort_values("all_AP@0.5:0.95", ascending=False).reset_index(drop=True)
    idx = np.linspace(0, len(ranked) - 1, 6).round().astype(int)
    ncol = 2 + len(pred_dirs)
    fig, ax = plt.subplots(6, ncol, figsize=(3.6 * ncol, 6 * 2.85))
    for row, i in enumerate(idx):
        stem = ranked.loc[i, "stem"]
        img, gt, preds = load(data, stem, pred_dirs)
        overlay(ax[row, 0], img, None, None,
                f"Imagen · AP@0.5:0.95 = {es(ranked.loc[i, 'all_AP@0.5:0.95'], 2)}")
        filled(ax[row, 1], img, gt, f"Referencia ({len(np.unique(gt)) - 1})")
        for col, (m, p) in enumerate(preds.items(), start=2):
            filled(ax[row, col], img, p, f"{short(m)} ({len(np.unique(p)) - 1})")
    fig.tight_layout()
    return png(fig)


def _first_split(data, pred_dir, cands: pd.DataFrame):
    """Largest missed nucleus covered by >= 2 predicted objects (each >= 15 % of it)."""
    for _, r in cands.sort_values("area", ascending=False).iterrows():
        gt = tifffile.imread(data / "labels" / f"{r['stem']}.tif")
        pred = tifffile.imread(Path(pred_dir) / f"{r['stem']}.tif")
        inside = pred[gt == r["label"]]
        counts = np.bincount(inside[inside > 0])
        if (counts >= 0.15 * r["area"]).sum() >= 2:
            return r["stem"], int(r["label"])
    return None


def pick_failures(data, objects: pd.DataFrame, pred_dir) -> list[tuple[str, str, int, str]]:
    """(caption, stem, label, kind) for 3 representative misses of the best method."""
    miss = objects[~objects["found"]]
    out = []
    t = miss[miss["touching"] & ~miss["at_border"] & (miss["area"] >= 300)]
    if len(t):
        r = t.sort_values("area", ascending=False).iloc[0]
        out.append(("Núcleos pegados: un núcleo que se toca con otro no se separa bien",
                    r["stem"], int(r["label"]), "touching"))
    split = _first_split(data, pred_dir, miss[miss["area"] >= 300])
    if split is not None:
        out.append(("Sobresegmentación: un núcleo irregular se divide en varios objetos",
                    split[0], split[1], "split"))
    per_img = objects.groupby("stem")["found"].mean().sort_values()
    worst = per_img.index[0]
    w = miss[(miss["stem"] == worst) & (miss["area"] >= 25)]
    if len(w):
        gt = tifffile.imread(data / "labels" / f"{worst}.tif")
        props = {p.label: p.centroid for p in measure.regionprops(gt)}
        cents = np.array([props[l] for l in w["label"]])
        dens = [(np.linalg.norm(cents - c, axis=1) < 40).sum() for c in cents]
        out.append(("Núcleos fragmentados (apoptóticos): la referencia anota cada fragmento",
                    worst, int(w["label"].iloc[int(np.argmax(dens))]), "fragmented"))
    return out


def fig_failures(data, pred_dirs, failures) -> str:
    ncol = 2 + len(pred_dirs)
    fig, ax = plt.subplots(len(failures), ncol, figsize=(3.2 * ncol, 3.4 * len(failures)),
                           squeeze=False)
    for row, (caption, stem, label, _) in enumerate(failures):
        img, gt, preds = load(data, stem, pred_dirs)
        cy, cx = measure.regionprops((gt == label).astype(np.uint8))[0].centroid
        h = 45
        y0 = int(np.clip(cy - h, 0, gt.shape[0] - 2 * h))
        x0 = int(np.clip(cx - h, 0, gt.shape[1] - 2 * h))
        crop = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        overlay(ax[row, 0], img, None, None, caption.split(":")[0], crop)
        overlay(ax[row, 1], img, gt, "#ffffff", "Referencia", crop)
        for col, (m, p) in enumerate(preds.items(), start=2):
            overlay(ax[row, col], img, p, color(m), short(m), crop)
    fig.tight_layout()
    return png(fig)


def recall_by_category(objects: dict) -> pd.DataFrame:
    cats = {
        "Aislados": lambda o: ~o["touching"],
        "Pegados a otro núcleo": lambda o: o["touching"],
        "Interiores": lambda o: ~o["at_border"],
        "Cortados por el borde": lambda o: o["at_border"],
        "≥ 25 px": lambda o: o["area"] >= 25,
        "< 25 px (micronúcleos)": lambda o: o["area"] < 25,
    }
    rows = []
    for name, sel in cats.items():
        any_o = next(iter(objects.values()))
        row = {"Grupo de núcleos": name, "N (referencia)": int(sel(any_o).sum())}
        for m, o in objects.items():
            row[short(m)] = float(o.loc[sel(o), "found"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fig_recall(rec: pd.DataFrame, methods: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(rec))
    hgt = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = rec[short(m)].to_numpy()
        pos = y + (k - (len(methods) - 1) / 2) * hgt
        ax.barh(pos, vals, height=hgt - 0.04, color=color(m), label=short(m))
        for p, v in zip(pos, vals):
            ax.text(v + 0.01, p, f"{es(100 * v, 0)} %", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, rec["Grupo de núcleos"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Núcleos de la referencia detectados (IoU ≥ 0,5)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return png(fig)


def fig_blur(blur: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for m, g in blur.groupby("method", sort=False):
        ax.plot(g["sigma"], g["all_AP@0.5:0.95"], color=color(m), lw=2, marker="o", ms=6)
        ax.text(g["sigma"].iloc[-1] + 0.1, g["all_AP@0.5:0.95"].iloc[-1], short(m),
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("Desenfoque gaussiano añadido (σ, píxeles)")
    ax.set_ylabel("AP@0.5:0.95 (test)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: es(v, 2)))
    ax.set_xticks(sorted(blur["sigma"].unique()))
    ax.set_xlim(-0.2, blur["sigma"].max() + 1.9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    return png(fig)


def fig_counts(quant: dict, ref: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = ref["n_nuclei"].max() * 1.08
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    ax.text(lim * 0.97, lim * 0.93, "y = x", ha="right", fontsize=8, color=INK2)
    for m, q in quant.items():
        j = ref.merge(q, on="stem", suffixes=("_ref", ""))
        ax.scatter(j["n_nuclei_ref"], j["n_nuclei"], s=36, color=color(m), label=short(m),
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Núcleos por imagen (referencia)")
    ax.set_ylabel("Núcleos por imagen (método)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=GRID, lw=0.8)
    return png(fig)


def count_agreement(quant: dict, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, q in quant.items():
        j = ref.merge(q, on="stem", suffixes=("_ref", ""))
        nz = j[j["n_nuclei_ref"] > 0]
        rel = (nz["n_nuclei"] - nz["n_nuclei_ref"]) / nz["n_nuclei_ref"]
        area = (nz["median_area_px"] - nz["median_area_px_ref"]) / nz["median_area_px_ref"]
        rows.append({"Método": short(m),
                     "Correlación de recuentos (r)": float(np.corrcoef(j["n_nuclei_ref"], j["n_nuclei"])[0, 1]),
                     "Error de recuento mediano": float(np.median(rel)),
                     "Error de recuento máx. (abs.)": float(np.max(np.abs(rel))),
                     "Error de área mediana (mediana por imagen)": float(np.nanmedian(area))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page
CSS = """
:root{color-scheme:light;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;
--surface:#fcfcfb;--panel:#f4f3f0;--accent:#2a78d6;--warn:#8a5a00;--warnbg:#fdf4e1}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 4px}
h2{font-size:20px;margin:48px 0 8px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 6px}
.sub{color:var(--ink2);margin:0 0 24px}
.ruo{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
color:var(--warn);background:var(--warnbg);padding:2px 8px;border-radius:4px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--panel);border-radius:8px;padding:14px 16px}
.kpi b{display:block;font-size:26px;line-height:1.2;font-variant-numeric:tabular-nums}
.kpi span{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600;font-size:13px}
.tablewrap{overflow-x:auto}
figure{margin:16px 0}
figure img{max-width:100%;height:auto;display:block;border-radius:4px}
figcaption{color:var(--ink2);font-size:13px;margin-top:6px}
.pending{color:var(--muted);font-style:italic}
ul{padding-left:20px} li{margin:4px 0}
code{background:var(--panel);padding:1px 5px;border-radius:3px;font-size:13px}
footer{margin-top:48px;color:var(--muted);font-size:13px}
@media print{h2{break-before:page}figure{break-inside:avoid}}
"""


def metrics_table(results: pd.DataFrame, variant: str) -> str:
    r = results[results["variant"] == variant].copy()
    r["Método"] = r["method"].map(SHORT)
    cols = ["AP@0.5", "AP@0.5:0.95", "F1@0.5", "F1@0.5:0.95", "IoU_matched@0.5"]
    out = r[["Método", *cols]].rename(columns={"IoU_matched@0.5": "IoU medio (aciertos)"})
    html_t = table_html(out, {c: (lambda v: f"{es(v, 3)}") for c in out.columns[1:]})
    for m in r.loc[r["status"] == "pendiente", "Método"]:
        html_t = html_t.replace(f"<td>{m}</td>", f'<td>{m} <span class="pending">— pendiente '
                                                 f"(entrenamiento en curso)</span></td>")
    return f'<div class="tablewrap">{html_t}</div>'


def overlap_section(runs: Path, results: pd.DataFrame, n_test: int) -> tuple[str, str]:
    """Metrics without the test images that are also in DSB 2018 stage1_train.

    Returns (html for the metrics section, <li> for the limitations). Without
    runs/results_excl_dsb.csv (check_dsb_overlap.py + evaluate.py) only the caveat remains.
    """
    path = runs / "results_excl_dsb.csv"
    if not path.exists():
        return "", ("<li><b>Posible solapamiento de datos.</b> StarDist se entrenó con parte de "
                    "DSB 2018, que procede en parte de las mismas colecciones del Broad Institute; "
                    "si alguna imagen de test estuviera en su entrenamiento, su resultado sería "
                    "algo optimista.</li>")
    excl = pd.read_csv(path)
    n_ex = int(excl["n_excluded"].iloc[0])
    cols = ["AP@0.5", "AP@0.5:0.95", "F1@0.5"]
    full = results[(results["variant"] == "all") & (results["status"] == "ok")].set_index("method")
    sub = excl[excl["variant"] == "all"].set_index("method")
    rows = [{"Método": short(m), "Imágenes": label, **{c: src.loc[m, c] for c in cols}}
            for m in sub.index
            for label, src in [(f"las {n_test}", full), (f"{n_test - n_ex} (sin las {n_ex})", sub)]]
    table = table_html(pd.DataFrame(rows), {c: (lambda v: es(v, 3)) for c in cols})
    sd = next((m for m in sub.index if "StarDist" in m), None)
    delta = "" if sd is None else (
        f" Para StarDist, AP@0.5:0.95 es {es(full.loc[sd, 'AP@0.5:0.95'], 3)} con todas las "
        f"imágenes y {es(sub.loc[sd, 'AP@0.5:0.95'], 3)} sin ellas.")
    html_s = ("<h3>Imágenes de test que StarDist pudo ver al entrenarse</h3>"
              "<p>StarDist <code>2D_versatile_fluo</code> se entrenó con parte del conjunto "
              "DSB 2018 (stage1_train), que incluye campos de BBBC039. Comparando píxel a píxel "
              f"(correlación normalizada) hemos encontrado <b>{n_ex} de las {n_test} imágenes de "
              "test</b> en DSB 2018 stage1_train. No sabemos cuáles de ellas entraron en el "
              "subconjunto concreto de StarDist, así que las tratamos todas como vistas y repetimos "
              f"la evaluación sin ellas (todos los objetos).{delta}</p>"
              f'<div class="tablewrap">{table}</div>')
    li = (f"<li><b>Solapamiento de datos (comprobado).</b> {n_ex} imágenes de test están en DSB "
          "2018, de donde salen los datos de entrenamiento de StarDist. Sin ellas las métricas "
          "apenas cambian (ver «Métricas»), pero una validación sobre datos del cliente, que "
          "StarDist no ha visto, sigue siendo la prueba definitiva.</li>")
    return html_s, li


def build(args) -> str:
    results = pd.read_csv(args.runs / "results.csv")
    ok = results[(results["status"] == "ok") & (results["variant"] == "all")]
    methods = ok["method"].tolist()
    best = ok.sort_values("AP@0.5:0.95", ascending=False).iloc[0]
    second = ok.sort_values("AP@0.5:0.95", ascending=False).iloc[1]
    best25 = results[(results["method"] == best["method"]) & (results["variant"] == "ge25")].iloc[0]
    per_image = {m: pd.read_csv(args.runs / "eval" / f"{slug(m)}_per_image.csv") for m in methods}
    objects = {m: pd.read_csv(args.runs / "eval" / f"{slug(m)}_objects.csv") for m in methods}
    pred_dirs = dict(zip(ok["method"], ok["pred_dir"]))
    blur = pd.read_csv(args.runs / "robustness_blur.csv")
    ref_q = pd.read_csv(args.runs / "quant" / "reference" / "summary.csv")
    # quantify.py default output: runs/quant/<name of the folder above pred/>
    qdirs = {m: args.runs / "quant" / Path(d).parent.parent.name / "summary.csv"
             for m, d in pred_dirs.items()}
    quant = {m: pd.read_csv(p) for m, p in qdirs.items() if p.exists()}
    n_test = len(stems_of(args.data, "test"))
    n_ref = int(objects[best["method"]].shape[0])
    rec = recall_by_category(objects)
    agree = count_agreement(quant, ref_q)
    a_best = agree[agree["Método"] == short(best["method"])].iloc[0]
    failures = pick_failures(args.data, objects[best["method"]], pred_dirs[best["method"]])
    b0 = blur[blur["sigma"] == 0].set_index("method")["all_AP@0.5:0.95"]
    bmax = blur[blur["sigma"] == blur["sigma"].max()].set_index("method")["all_AP@0.5:0.95"]
    drop = (b0 - bmax).sort_values()
    blur_sentence = (f"Ante desenfoque simulado (σ = {blur['sigma'].max():g} px), "
                     f"{short(drop.index[0])} pierde {es(drop.iloc[0], 2)} puntos de AP@0.5:0.95 y "
                     f"{short(drop.index[-1])} {es(drop.iloc[-1], 2)}.")
    how = ("usado tal cual, sin ajustar a estos datos" if "sin ajustar" in best["method"]
           else "ajustado con train/val")
    pending = results.loc[results["status"] == "pendiente", "method"].unique()
    pending_li = "".join(f"<li><b>{short(m)}</b>: <b>pendiente</b>; su fila se añadirá a la "
                         "tabla sin cambiar el resto del análisis.</li>" for m in pending)
    count_dir = ("tiende a contar de menos" if a_best["Error de recuento mediano"] < 0
                 else "tiende a contar de más")
    area_dir = ("se sobreestima" if a_best["Error de área mediana (mediana por imagen)"] > 0
                else "se subestima")
    border_share = float(objects[best["method"]]["at_border"].mean())
    rb = rec.set_index("Grupo de núcleos")[short(best["method"])]
    overlap_html, overlap_li = overlap_section(args.runs, results, n_test)

    kpis = f"""<div class="kpis">
<div class="kpi"><b>{es(best['AP@0.5'], 2)}</b><span>AP@0.5 del mejor método ({short(best['method'])})</span></div>
<div class="kpi"><b>{pct(best['F1@0.5'])}</b><span>F1@0.5: núcleos bien detectados y separados</span></div>
<div class="kpi"><b>{es(a_best['Correlación de recuentos (r)'], 3)}</b><span>correlación del recuento por imagen con la referencia</span></div>
<div class="kpi"><b>{signed_pct(a_best['Error de área mediana (mediana por imagen)'])}</b><span>sesgo típico del área mediana por imagen frente a la referencia</span></div>
</div>"""

    summary = f"""
<p>Hemos evaluado la segmentación automática de núcleos (tinción de ADN, Hoechst) en
<b>{n_test} imágenes de test</b> de BBBC039 con <b>{thousands(n_ref)} núcleos anotados a mano</b>,
comparando cada método con la anotación de referencia, núcleo a núcleo.</p>
<ul>
<li><b>Mejor método hoy: {short(best['method'])}</b>, {how}: detecta y separa correctamente el {pct(best['F1@0.5'])} de los núcleos (F1 con IoU ≥ 0,5),
frente al {pct(second['F1@0.5'])} del siguiente método ({short(second['method'])}).</li>
<li>Si se excluyen los objetos de menos de 25 píxeles (micronúcleos y restos), el F1 sube a
{pct(best25['F1@0.5'])}: buena parte de los errores restantes son objetos diminutos.</li>
<li>Para la <b>cuantificación</b>, que es lo que se entrega, el recuento de núcleos por imagen
correlaciona r = {es(a_best['Correlación de recuentos (r)'], 3)} con el recuento manual, aunque
{count_dir} (mediana {signed_pct(a_best['Error de recuento mediano'])} por imagen: micronúcleos no
detectados y núcleos pegados o fragmentados que se agrupan en un solo objeto), y el área mediana por imagen
{area_dir} típicamente un {pct(abs(a_best['Error de área mediana (mediana por imagen)']))}.</li>
<li>Los fallos se concentran en <b>núcleos pegados</b> ({pct(rb['Pegados a otro núcleo'])} detectados frente
a {pct(rb['Aislados'])} de los aislados), <b>núcleos fragmentados</b> y <b>objetos diminutos</b>.
{blur_sentence}</li>
{pending_li}
</ul>
{kpis}"""

    sections = [
        f'<h1>Segmentación y cuantificación de núcleos</h1>'
        f'<p class="sub">Informe de validación sobre BBBC039 (fluorescencia 2D, U2OS, Hoechst) · '
        f'<span class="ruo">Solo para investigación</span></p>',
        "<h2>Resumen</h2>" + summary,
        "<h2>Métricas en el conjunto de test</h2>"
        f"<p>Split oficial de test ({n_test} imágenes), evaluado una sola vez al final; todos los "
        "parámetros se fijaron solo con train/val. Los aciertos se cuentan con emparejamiento "
        "uno a uno entre núcleos de la referencia y de la predicción, sumando sobre todas las "
        "imágenes.</p>"
        "<h3>Todos los objetos de la referencia</h3>" + metrics_table(results, "all") +
        "<h3>Solo objetos ≥ 25 px (filtro aplicado a referencia y predicción)</h3>" +
        metrics_table(results, "ge25") +
        "<p><b>Cómo leerlo.</b> <i>AP@0.5</i> = aciertos / (aciertos + falsos positivos + "
        "núcleos no detectados), contando como acierto un solapamiento (IoU) ≥ 0,5. "
        "<i>AP@0.5:0.95</i> promedia exigencias de solapamiento de 0,5 a 0,95, así que premia "
        "contornos precisos. <i>IoU medio (aciertos)</i> mide lo bien que se ajusta el contorno "
        "de los núcleos que sí se detectan.</p>" + overlap_html,
        "<h2>Ejemplos</h2>" + figure_html(
            fig_examples(args.data, pred_dirs, per_image[best["method"]]),
            f"Seis imágenes de test repartidas desde la mejor a la peor según "
            f"{short(best['method'])}. Cada color es un objeto distinto; entre paréntesis, número "
            f"de objetos. Los recortes de la sección siguiente muestran el detalle."),
        "<h2>Casos de fallo</h2>"
        "<p>Porcentaje de núcleos de la referencia que cada método detecta (IoU ≥ 0,5), por "
        "tipo de núcleo:</p>" +
        figure_html(fig_recall(rec, methods), "Detección por grupo de núcleos (test).") +
        '<div class="tablewrap">' + table_html(rec, {short(m): pct for m in methods}) + "</div>" +
        figure_html(fig_failures(args.data, pred_dirs, failures),
                    "Recortes de 90×90 px con fallos típicos del mejor método. "
                    + " · ".join(f"<b>{c.split(':')[0]}</b>:{c.split(':')[1]}" for c, *_ in failures)) +
        "<h3>Desenfoque</h3>"
        "<p>BBBC039 no contiene campos claramente desenfocados, así que lo hemos simulado con "
        "un desenfoque gaussiano sobre las imágenes de test, sin reajustar ningún método:</p>" +
        figure_html(fig_blur(blur), "AP@0.5:0.95 frente al desenfoque añadido (σ en píxeles).") +
        f"<p>Con σ = {blur['sigma'].max():g} px el método clásico pasa de "
        f"{es(b0['Otsu + watershed'], 2)} a {es(bmax['Otsu + watershed'], 2)} y StarDist de "
        f"{es(b0['StarDist 2D_versatile_fluo (sin ajustar)'], 2)} a "
        f"{es(bmax['StarDist 2D_versatile_fluo (sin ajustar)'], 2)}.</p>",
        "<h2>Validez de la cuantificación</h2>"
        "<p>La tabla por núcleo (área, posición, excentricidad, solidez, intensidad media y "
        "máxima) se entrega en CSV, uno por imagen. Comparando los resúmenes por imagen con los "
        "calculados sobre la referencia manual:</p>" +
        figure_html(fig_counts(quant, ref_q), "Núcleos por imagen: método frente a referencia "
                                              "(cada punto es una imagen de test).") +
        '<div class="tablewrap">' + table_html(agree, {
            "Correlación de recuentos (r)": lambda v: f"{es(v, 3)}",
            "Error de recuento mediano": signed_pct,
            "Error de recuento máx. (abs.)": pct,
            "Error de área mediana (mediana por imagen)": signed_pct}) + "</div>"
        f"<p>Un {pct(border_share)} de los núcleos de referencia están cortados por el borde del "
        "campo; en la tabla van marcados (<code>touches_border</code>) y se excluyen de los "
        "resúmenes de área e intensidad porque su forma está truncada.</p>",
        "<h2>Limitaciones</h2><ul>"
        "<li><b>Un solo tipo de muestra.</b> BBBC039 es una línea celular (U2OS), una tinción "
        "(Hoechst) y un microscopio. Otros tejidos, aumentos o tinciones necesitan su propia "
        "validación con unas pocas imágenes anotadas.</li>"
        "<li><b>Referencia ambigua en núcleos fragmentados.</b> En células apoptóticas la "
        "anotación separa cada fragmento; los métodos tienden a agruparlos. Parte de ese error "
        "es discutible incluso entre anotadores.</li>"
        "<li><b>Objetos diminutos.</b> Los objetos &lt; 25 px son micronúcleos o restos anotados a "
        "propósito; ningún método los detecta bien. Si son relevantes para el estudio, hace "
        "falta un modelo específico.</li>"
        + overlap_li +
        "<li><b>Desenfoque solo simulado.</b> El desenfoque gaussiano es una aproximación; el "
        "desenfoque óptico real y la saturación pueden comportarse distinto.</li>"
        "<li><b>Sin tamaño de píxel.</b> La página del dataset no lo indica: las áreas van en "
        "píxeles. Con el tamaño de píxel del microscopio se convierten a µm².</li>"
        "<li><b>U-Net pendiente.</b> La fila del modelo entrenado se completará cuando termine "
        "el entrenamiento.</li></ul>",
        "<h2>Método y reproducibilidad</h2><ul>"
        "<li>Datos: BBBC039v1 (CC0), splits oficiales train/val/test (100/50/50). "
        "Máscaras de referencia sin filtrar.</li>"
        "<li>Otsu + watershed (scikit-image): umbral de Otsu, transformada de distancia y "
        "watershed; parámetros elegidos por búsqueda en rejilla en train.</li>"
        "<li>StarDist <code>2D_versatile_fluo</code> (BSD-3), umbrales por defecto del modelo, "
        "normalización por percentiles 1–99,8.</li>"
        "<li>Código: scripts con semilla fija; cada cifra de este informe se regenera con "
        "<code>evaluate.py</code>, <code>robustness_blur.py</code>, <code>quantify.py</code> y "
        "<code>report.py</code>.</li></ul>",
        '<footer>Datos: "We used image set BBBC039v1 Caicedo et al. 2018, available from the '
        'Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]." · '
        "StarDist: Schmidt et al., MICCAI 2018. · Solo para investigación; no apto para "
        "diagnóstico.</footer>",
    ]
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Informe de segmentación de núcleos</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(sections)}</main></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/report.html"))
    args = ap.parse_args()
    args.out.write_text(build(args), encoding="utf-8")
    print(f"Informe: {args.out} ({es(args.out.stat().st_size / 1e6, 1)} MB)")


if __name__ == "__main__":
    main()
