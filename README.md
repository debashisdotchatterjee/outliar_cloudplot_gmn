# outliar_cloudplot_gmn

# GMN Cloud-Plot Pipeline (S² Radiant Outlier Detection) — Reproducible Bundle

This repository / folder contains a **Colab-ready** pipeline to:
1. **Download** Global Meteor Network (GMN) *trajectory-summary* data,
2. Filter to a chosen **IAU shower code** (default: `PER` for Perseids),
3. Convert geocentric radiant **(RA, Dec)** to unit vectors on the **unit sphere** \(\mathbb S^2\),
4. Compute the **intrinsic spherical geometric median** (Riemannian Weiszfeld),
5. Construct a **Cloud-plot inner fence** for robust outlier detection using **vMF-calibrated** constants,
6. Produce **many publication-quality plots + tables**, an **HTML report**, and a **ZIP** bundle of all outputs.

---

## Data sources (with proper citation expectations)

### Global Meteor Network (GMN) public data products
- GMN trajectory-summary archive index (daily/monthly/yearly/all-time):  
  https://globalmeteornetwork.org/data/traj_summary_data/
- Example monthly file used in the script (Aug 2019):  
  https://globalmeteornetwork.org/data/traj_summary_data/monthly/traj_summary_monthly_201908.txt
- Official column definitions (PDF):  
  https://globalmeteornetwork.org/data/media/GMN_orbit_data_columns.pdf
- GMN data portal (licensing + general info):  
  https://globalmeteornetwork.org/data/

**Licensing:** GMN public data products are distributed under **CC BY 4.0** (see GMN portal / documentation).  
**When you publish:** cite the **GMN portal** and the **GMN methodology papers**.

---

## What the pipeline does (methodology summary)

### 1) Spherical embedding
For each meteor radiant direction \((\mathrm{RA}_i, \mathrm{Dec}_i)\) (degrees), form the unit vector:
\[
\mathbf{x}_i =
\begin{pmatrix}
\cos(\mathrm{Dec}_i)\cos(\mathrm{RA}_i)\\
\cos(\mathrm{Dec}_i)\sin(\mathrm{RA}_i)\\
\sin(\mathrm{Dec}_i)
\end{pmatrix}\in\mathbb S^2.
\]

### 2) Robust spherical centre (intrinsic geometric median)
Compute
\[
\widehat{\mathbf{m}}
\in
\arg\min_{\mathbf{y}\in\mathbb S^2}\sum_{i=1}^n d_g(\mathbf{x}_i,\mathbf{y}),
\]
where \(d_g(\mathbf{x},\mathbf{y})=\arccos(\mathbf{x}^\top \mathbf{y})\) is the geodesic distance.

### 3) Cloud-plot distances and quantiles
Compute \(d_i=d_g(\mathbf{x}_i,\widehat{\mathbf{m}})\) and empirical quantiles:
- \(\widehat q_{0.50}\) (median distance; used as SIQR),
- \(\widehat q_{0.75}\).

### 4) vMF-calibrated fence
Use the vMF reference family on \(\mathbb S^2\) to interpret the fence as a **coverage target** \(\alpha\).
Estimate \(\widehat\kappa\) **robustly** by matching the empirical median distance:
\[
q_{0.50}(\widehat\kappa) \approx \widehat q_{0.50}.
\]
Define the resistant constant
\[
k(\widehat\kappa;\alpha)=\frac{q_\alpha(\widehat\kappa)-q_{0.75}(\widehat\kappa)}{q_{0.50}(\widehat\kappa)},
\]
and inner fence radius
\[
\widehat{\mathrm{IF}}=\widehat q_{0.75}+k(\widehat\kappa;\alpha)\,\widehat q_{0.50}.
\]
Flag outliers if \(d_i>\widehat{\mathrm{IF}}\).

---

## Outputs

After a run, you get:

### `cloudplot_gmn_outputs/`
- `data/`  
  downloaded GMN file(s)
- `docs/`  
  `GMN_orbit_data_columns.pdf`
- `tables/` (CSV)
  - `table_summary.csv`
  - `table_alpha_sensitivity.csv`
  - `table_top_outliers.csv`
  - `table_group_stats.csv`
- `figures/` (PNG)
  - multiple single-figure plots (RA–Dec scatter, Mollweide map, histograms, ECDF vs vMF, sensitivity curves, QC plots, etc.)
- `report.html`  
  local HTML report embedding figures and tables
- `reproducibility.json`  
  timezone-aware UTC timestamp, dataset URL, file SHA256 hash, key estimates, fence, counts

### `cloudplot_gmn_outputs.zip`
A ZIP bundle of the entire output folder for easy sharing / archiving.

---

## How to run (Google Colab)

1. Open a new Colab notebook.
2. Paste the **single-cell script** (the one provided in chat) into a cell.
3. Run the cell.

The cell will:
- install dependencies,
- download data,
- run the analysis,
- save figures/tables/report,
- produce `cloudplot_gmn_outputs.zip`.

---

## Customisation

### Change shower code
Edit:
```python
SHOWER_CODE = "GEM"   # e.g. Geminids
