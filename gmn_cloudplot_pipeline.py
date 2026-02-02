#!/usr/bin/env python3
"""
GMN Cloud-Plot Pipeline (S^2 radiants) — reproducible analysis bundle.

This script:
1) Downloads a chosen GMN trajectory-summary file (monthly/daily/yearly/all-time).
2) Filters to an IAU shower code (default PER).
3) Computes spherical geometric median and intrinsic distances.
4) Calibrates a Cloud-plot outlier fence using a vMF reference distribution.
5) Saves figures + tables + a self-contained HTML report.
6) Zips the whole output folder.

Data source: Global Meteor Network public data products (CC BY 4.0).
Please cite the GMN papers and the GMN data portal in publications.

Author: (generated pipeline)
"""

from __future__ import annotations
import os, re, io, math, zipfile, hashlib, datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import requests
except ImportError:
    requests = None


# ----------------------------
# Configuration
# ----------------------------
GMN_MONTHLY_URL = "https://globalmeteornetwork.org/data/traj_summary_data/monthly/traj_summary_monthly_201908.txt"
GMN_COLUMNS_PDF = "https://globalmeteornetwork.org/data/media/GMN_orbit_data_columns.pdf"

OUTDIR = Path("cloudplot_gmn_outputs")
DATA_DIR = OUTDIR / "data"
FIG_DIR  = OUTDIR / "figures"
TAB_DIR  = OUTDIR / "tables"
DOC_DIR  = OUTDIR / "docs"

SHOWER_CODE = os.environ.get("GMN_SHOWER", "PER").upper()
ALPHA = float(os.environ.get("CLOUDPLOT_ALPHA", "0.99"))


# ----------------------------
# Utilities
# ----------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    if requests is None:
        raise RuntimeError("requests is not installed; in Colab: pip install requests")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)

def robust_unique_names(tokens: List[str]) -> List[str]:
    names = []
    last_base = None
    sigma_count = 0
    for t in tokens:
        t0 = re.sub(r"\s+", " ", t.strip())
        if not t0:
            t0 = "EMPTY"
        if "+/-" in t0 or t0 == "+/-" or t0.lower().startswith("+/-"):
            sigma_count += 1
            nm = f"{last_base}_sigma" if last_base else f"sigma_{sigma_count}"
            if nm in names:
                k = 2
                while f"{nm}_{k}" in names:
                    k += 1
                nm = f"{nm}_{k}"
            names.append(nm)
            continue

        nm = t0.replace(" ", "_")
        nm = nm.replace("(", "").replace(")", "")
        nm = nm.replace("/", "_per_")
        nm = nm.replace("+", "plus")
        nm = nm.replace("-", "_")
        nm = re.sub(r"[^A-Za-z0-9_]+", "", nm).strip("_")
        nm = nm or "COL"
        if nm in names:
            k = 2
            while f"{nm}_{k}" in names:
                k += 1
            nm = f"{nm}_{k}"
        names.append(nm)
        last_base = nm
    return names

def load_gmn_traj_summary(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    header_line = None
    for ln in lines:
        s = ln.strip().lstrip("\ufeff").lstrip("\r")
        if s.startswith("#") and "Unique trajectory" in s:
            header_line = s.lstrip("#").strip()
            break
    if header_line is None:
        raise RuntimeError("Header line with column names not found.")

    tokens = header_line.split(";")
    colnames = robust_unique_names(tokens)

    data_lines = []
    for ln in lines:
        s = ln.lstrip("\ufeff").lstrip("\r")
        if s.strip().startswith("#") or s.strip() == "":
            continue
        data_lines.append(s)

    buf = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(buf, sep=";", header=None, names=colnames, engine="python")

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df

def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ----------------------------
# Spherical geometry
# ----------------------------
def radec_deg_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    X = np.vstack([x, y, z]).T
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def geodesic_dist(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    dots = np.clip((X*y).sum(axis=1), -1.0, 1.0)
    return np.arccos(dots)

def expmap(y: np.ndarray, v: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return y.copy()
    return np.cos(nv)*y + np.sin(nv)*(v/nv)

def logmap(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dot = float(np.clip(np.dot(y, x), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-12:
        return np.zeros(3)
    u = x - dot*y
    su = np.linalg.norm(u)
    if su < 1e-12:
        return np.zeros(3)
    return (theta/su)*u

def spherical_geometric_median(X: np.ndarray, max_iter: int = 500, tol: float = 1e-10, delta: float = 1e-10) -> np.ndarray:
    m = X.mean(axis=0)
    nm = np.linalg.norm(m)
    y = X[0].copy() if nm < 1e-12 else m/nm
    obj_prev = float("inf")

    for _ in range(max_iter):
        V = np.array([logmap(y, X[i]) for i in range(X.shape[0])])
        d = np.linalg.norm(V, axis=1)
        w = 1.0/(d + delta)
        vbar = (w[:,None]*V).sum(axis=0) / w.sum()

        if np.linalg.norm(vbar) > 0.5:
            vbar = vbar*(0.5/np.linalg.norm(vbar))

        y_new = expmap(y, vbar)
        y_new = y_new/np.linalg.norm(y_new)

        d_new = geodesic_dist(X, y_new)
        obj = float(d_new.sum())

        move = math.acos(float(np.clip(np.dot(y, y_new), -1.0, 1.0)))
        if abs(obj_prev - obj) < tol and move < tol:
            y = y_new
            break
        y, obj_prev = y_new, obj

    return y


# ----------------------------
# vMF reference calibration
# ----------------------------
def q_p_vmf(p: float, kappa: float) -> float:
    p = float(p); kappa = float(kappa)
    if kappa < 1e-10:
        return math.acos(max(-1.0, min(1.0, 1.0 - 2.0*p)))
    if kappa > 50:
        logval = kappa + math.log(1.0 - p)
    else:
        logval = kappa + math.log((1.0 - p) + p*math.exp(-2.0*kappa))
    cosq = max(-1.0, min(1.0, logval/kappa))
    return math.acos(cosq)

def kappa_from_median_distance(d50: float, max_kappa: float = 1e6) -> float:
    target = float(d50)
    lo, hi = 0.0, 1.0
    while q_p_vmf(0.5, hi) > target and hi < max_kappa:
        hi *= 2.0
    if hi >= max_kappa:
        return max_kappa
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if q_p_vmf(0.5, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def k_resistant(kappa: float, alpha: float) -> float:
    q50 = q_p_vmf(0.50, kappa)
    q75 = q_p_vmf(0.75, kappa)
    qa  = q_p_vmf(alpha, kappa)
    return (qa - q75)/q50

def F_vmf(d: np.ndarray, kappa: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    if kappa < 1e-10:
        return 0.5*(1.0 - np.cos(d))
    denom = 1.0 - np.exp(-2.0*kappa) if kappa < 50 else 1.0
    return (1.0 - np.exp(kappa*(np.cos(d)-1.0))) / denom


# ----------------------------
# Reporting
# ----------------------------
def make_html_report(summary_df: pd.DataFrame, tables: Dict[str, Path], figures: List[Path], out: Path) -> None:
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>GMN Cloud-Plot Report</title>")
    parts.append("<style>body{font-family:Arial, sans-serif; margin:22px;} img{max-width:980px; width:100%; border:1px solid #ddd; margin:10px 0;} table{border-collapse:collapse;} td,th{border:1px solid #ccc; padding:6px 8px;}</style></head><body>")
    parts.append(f"<h1>GMN Cloud-Plot Report</h1>")
    parts.append(f"<p><b>Shower code:</b> {SHOWER_CODE} &nbsp;&nbsp; <b>alpha:</b> {ALPHA}</p>")
    parts.append("<h2>Summary</h2>")
    parts.append(summary_df.to_html(index=False))
    parts.append("<h2>Figures</h2>")
    for fp in figures:
        parts.append(f"<h3>{fp.name}</h3>")
        parts.append(f"<img src='figures/{fp.name}'/>")
    parts.append("<h2>Tables (CSV)</h2><ul>")
    for name, p in tables.items():
        parts.append(f"<li>{name}: <code>tables/{p.name}</code></li>")
    parts.append("</ul>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main():
    for d in [DATA_DIR, FIG_DIR, TAB_DIR, DOC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Download inputs
    download(GMN_MONTHLY_URL, DATA_DIR / Path(GMN_MONTHLY_URL).name)
    download(GMN_COLUMNS_PDF, DOC_DIR / Path(GMN_COLUMNS_PDF).name)

    # Load
    df = load_gmn_traj_summary(DATA_DIR / Path(GMN_MONTHLY_URL).name)

    # Coerce numerics
    for c in ["Sol_lon","RAgeo","DECgeo","Vgeo","Qc","MedianFitErr","Num","RAgeo_sigma","DECgeo_sigma"]:
        if c in df.columns:
            df[c] = to_numeric(df[c])

    # Filter shower
    df = df[df["IAU_2"].str.upper().eq(SHOWER_CODE)].copy()
    df = df.dropna(subset=["RAgeo","DECgeo","Sol_lon"])
    if len(df) < 30:
        raise RuntimeError(f"Too few records after filtering to {SHOWER_CODE}: n={len(df)}")

    # Spherical median + distances
    X = radec_deg_to_unitvec(df["RAgeo"].to_numpy(), df["DECgeo"].to_numpy())
    m = spherical_geometric_median(X)
    d = geodesic_dist(X, m)
    d_deg = np.rad2deg(d)

    q50 = float(np.quantile(d, 0.50))
    q75 = float(np.quantile(d, 0.75))

    # Calibrate kappa and fence
    kappa = kappa_from_median_distance(q50)
    k = k_resistant(kappa, ALPHA)
    IF = q75 + k*q50
    IF_deg = float(np.rad2deg(IF))

    df["dist_deg"] = d_deg
    df["is_outlier"] = d > IF

    # Tables
    summary = pd.DataFrame([{
        "shower": SHOWER_CODE,
        "n": len(df),
        "median_dist_deg": float(np.rad2deg(q50)),
        "q75_dist_deg": float(np.rad2deg(q75)),
        "IF_deg": IF_deg,
        "kappa_rob": kappa,
        "k(alpha)": k,
        "outliers": int(df["is_outlier"].sum()),
        "outlier_rate": float(df["is_outlier"].mean())
    }])
    summary.to_csv(TAB_DIR/"table_summary.csv", index=False)

    top_out = (df[df["is_outlier"]]
               .sort_values("dist_deg", ascending=False)
               [["Unique_trajectory","Beginning_2","IAU","IAU_2","Sol_lon","RAgeo","DECgeo","dist_deg",
                 "Vgeo","Qc","MedianFitErr","Num"]]
               .head(30))
    top_out.to_csv(TAB_DIR/"table_top_outliers.csv", index=False)

    # Figures (save-only, no subplots)
    # 1) RA/Dec scatter
    plt.figure(figsize=(8,6))
    inl = df[~df["is_outlier"]]
    out = df[df["is_outlier"]]
    plt.scatter(inl["RAgeo"], inl["DECgeo"], s=10, alpha=0.45, label="inlier")
    plt.scatter(out["RAgeo"], out["DECgeo"], s=25, alpha=0.9, label="outlier")
    plt.xlabel(r"$\mathrm{RA}_{\rm geo}$ (deg, J2000)")
    plt.ylabel(r"$\mathrm{Dec}_{\rm geo}$ (deg, J2000)")
    plt.title(f"{SHOWER_CODE} radiants: inliers vs outliers (Cloud-plot fence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig01_ra_dec_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Distance histogram + fence
    plt.figure(figsize=(8,6))
    plt.hist(d_deg, bins=40, alpha=0.85)
    plt.axvline(np.rad2deg(q50), linestyle="--", linewidth=2, label="q50")
    plt.axvline(np.rad2deg(q75), linestyle="--", linewidth=2, label="q75")
    plt.axvline(IF_deg, linestyle="-", linewidth=2, label=f"IF (α={ALPHA})")
    plt.xlabel("Distance to spherical median (deg)")
    plt.ylabel("Count")
    plt.title("Distance distribution and fence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig02_distance_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) ECDF vs vMF
    dd = np.sort(d)
    ecdf = np.arange(1, len(dd)+1)/len(dd)
    tv = F_vmf(dd, kappa)
    plt.figure(figsize=(8,6))
    plt.plot(np.rad2deg(dd), ecdf, label="Empirical CDF")
    plt.plot(np.rad2deg(dd), tv, label=f"vMF CDF (κ≈{kappa:.1f})")
    plt.xlabel("Distance (deg)")
    plt.ylabel("CDF")
    plt.title("Empirical distance CDF vs vMF reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig03_ecdf_vs_vmf.png", dpi=200, bbox_inches="tight")
    plt.close()

    # HTML report
    figs = sorted(FIG_DIR.glob("fig*.png"))
    tables = {"summary": TAB_DIR/"table_summary.csv", "top_outliers": TAB_DIR/"table_top_outliers.csv"}
    make_html_report(summary, tables, figs, OUTDIR/"report.html")

    # Reproducibility metadata
    meta = {
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "gmn_file": str((DATA_DIR / Path(GMN_MONTHLY_URL).name).name),
        "sha256": sha256_of_file(DATA_DIR / Path(GMN_MONTHLY_URL).name),
        "columns_pdf": str((DOC_DIR / Path(GMN_COLUMNS_PDF).name).name),
        "shower": SHOWER_CODE,
        "alpha": ALPHA,
    }
    (OUTDIR/"reproducibility.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Zip bundle
    zip_path = Path(f"{OUTDIR.name}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUTDIR.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(OUTDIR.parent)))
    print(f"Saved: {zip_path.resolve()}")

if __name__ == "__main__":
    main()
