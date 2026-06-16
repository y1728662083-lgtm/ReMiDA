#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml


PALETTES = {
    "mystic_romance": ["#16058b", "#6200aa", "#9e169d", "#cc4a74", "#eb7852", "#fcb431"],
    "yellow_twilight": ["#274753", "#297270", "#299d8f", "#8ab07c", "#e7c66b", "#f3a361", "#e66d50"],
    "natural_energy": ["#a30543", "#f36f43", "#fbda83", "#e9f4a3", "#80cba4", "#4965b0"],
    "bio_figure": ["#2f2f2f", "#cc0967", "#bc6a17", "#87309e", "#010086", "#109163"],
}

EXP2_VIOLIN_STYLE = {
    "outer_bg": "#E6C8C8",
    "inner_bg": "#F7F5F3",
    "title": "#B88282",
    "subtitle": "#C7A7A7",
    "B1_edge": "#5698A9",
    "B1_fill": "#BCD5DC",
    "M1_edge": "#D83738",
    "M1_fill": "#ECB1B1",
}


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return sorted(obj.get("experiments", []), key=lambda x: int(x.get("order", 0)))


def summary_path(run_dir: Path, doc: str) -> Path:
    return run_dir / "logs" / doc / f"{doc}_summary.json"


def metric_mean_std(seed_metrics: List[Dict[str, Any]], key: str) -> tuple[float, float]:
    vals = [float(m[key]) for m in seed_metrics if key in m and m[key] is not None]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def collect_rows(manifest: List[Dict[str, Any]], run_dir: Path):
    rows = []
    missing = []
    for item in manifest:
        sp = summary_path(run_dir, item["doc"])
        if not sp.exists():
            missing.append({"doc": item["doc"], "config": item["config"], "summary_path": str(sp)})
            continue
        with open(sp, "r", encoding="utf-8") as f:
            summary = json.load(f)
        seed_metrics = summary.get("seed_metrics", [])
        full_mcc_mean = float(summary.get("full_mcc_mean", float("nan")))
        full_mcc_std = float(summary.get("full_mcc_std", float("nan")))
        perm_mean, perm_std = metric_mean_std(seed_metrics, "perm_agreement")
        sign_mean, sign_std = metric_mean_std(seed_metrics, "sign_agreement")
        probe_mean, probe_std = metric_mean_std(seed_metrics, "probe_acc")
        posthoc_mean, posthoc_std = metric_mean_std(seed_metrics, "posthoc_anchor_aligned_mcc")
        row = {
            **item,
            "full_mcc_mean": full_mcc_mean,
            "full_mcc_std": full_mcc_std,
            "perm_agreement_mean": perm_mean,
            "perm_agreement_std": perm_std,
            "sign_agreement_mean": sign_mean,
            "sign_agreement_std": sign_std,
            "probe_acc_mean": probe_mean,
            "probe_acc_std": probe_std,
            "posthoc_anchor_aligned_mcc_mean": posthoc_mean,
            "posthoc_anchor_aligned_mcc_std": posthoc_std,
            "n_sims": int(summary.get("n_sims", len(seed_metrics))),
            "summary_path": str(sp),
        }
        rows.append(row)
    return pd.DataFrame(rows), missing


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def set_matplotlib_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def method_color_map_linear():
    pal = PALETTES["yellow_twilight"]
    return {"B0": pal[0], "B1": pal[2], "M1": pal[5]}


def method_color_map_nonlinear():
    pal = PALETTES["mystic_romance"]
    return {"B0": pal[0], "B1": pal[2], "M1": pal[4], "M2": pal[5]}


def method_color_map_full():
    pal = PALETTES["bio_figure"]
    return {"M2": pal[3], "M3": pal[5]}


def make_curve(df: pd.DataFrame, methods: List[str], x_col: str, metric_mean: str, metric_std: str,
               color_map: Dict[str, str], title: str, xlabel: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for method in methods:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by=x_col)
        ax.errorbar(
            sub[x_col].astype(float).values,
            sub[metric_mean].astype(float).values,
            yerr=sub[metric_std].astype(float).values,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=color_map.get(method, "#333333"),
            label=method,
            capsize=3,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_bar(df: pd.DataFrame, label_col: str, metric_mean: str, metric_std: str,
             palette: List[str], title: str, xlabel: str, ylabel: str, out_path: Path):
    df = df.copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(x, df[metric_mean].astype(float).values, yerr=df[metric_std].astype(float).values,
           color=palette[:len(df)], edgecolor="black", linewidth=0.4, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col].tolist(), rotation=12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_heatmap(matrix: np.ndarray, labels: List[str], title: str, out_path: Path):
    cmap = LinearSegmentedColormap.from_list("mystic_romance", PALETTES["mystic_romance"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap=cmap, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    threshold = (np.nanmin(matrix) + np.nanmax(matrix)) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < threshold else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("similarity", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_aggregate_tables(df: pd.DataFrame, out_dir: Path, missing: List[Dict[str, Any]]):
    if not df.empty:
        df = df.sort_values(by="order")
        df.to_csv(out_dir / "planned_suite_aggregated.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "planned_suite_aggregated.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "planned_suite_missing.json", "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)


def plot_default_suite(df: pd.DataFrame, out_dir: Path):
    if df.empty or "benchmark" not in df.columns:
        return
    linear_df = df[df["benchmark"] == "linear_drift"].copy()
    if not linear_df.empty:
        cmap = method_color_map_linear()
        for metric in [("full_mcc_mean", "full_mcc_std", "FULL MCC"),
                       ("perm_agreement_mean", "perm_agreement_std", "Perm agreement"),
                       ("sign_agreement_mean", "sign_agreement_std", "Sign agreement")]:
            make_curve(
                linear_df,
                methods=["B0", "B1", "M1"],
                x_col="x_value",
                metric_mean=metric[0],
                metric_std=metric[1],
                color_map=cmap,
                title=f"Linear benchmark: {metric[2]} vs drift strength",
                xlabel="drift strength",
                ylabel=metric[2],
                out_path=out_dir / f"linear_{metric[0].replace('_mean','')}.png",
            )

    exp3 = df[df["experiment"] == "Exp3"].copy()
    if not exp3.empty:
        order_map = {"u": 0, "z": 1, "joint": 2, "none": 3}
        label_map = {"u": "U_raw", "z": "z_mix", "joint": "joint", "none": "none"}
        exp3["sort_key"] = exp3["x_value"].map(order_map).fillna(999)
        exp3 = exp3.sort_values("sort_key")
        exp3["plot_label"] = exp3["x_value"].map(label_map)
        make_bar(exp3, "plot_label", "full_mcc_mean", "full_mcc_std",
                 PALETTES["natural_energy"],
                 "Aux-routing ablation under linear drift",
                 "aux mode", "FULL MCC",
                 out_dir / "exp3_aux_full_mcc.png")

    nonlin_df = df[df["benchmark"] == "hierarchical_nonlinear"].copy()
    if not nonlin_df.empty:
        cmap = method_color_map_nonlinear()
        for metric in [("full_mcc_mean", "full_mcc_std", "FULL MCC"),
                       ("perm_agreement_mean", "perm_agreement_std", "Perm agreement"),
                       ("sign_agreement_mean", "sign_agreement_std", "Sign agreement")]:
            make_curve(
                nonlin_df,
                methods=["B0", "B1", "M1", "M2"],
                x_col="x_value",
                metric_mean=metric[0],
                metric_std=metric[1],
                color_map=cmap,
                title=f"Hierarchical nonlinear benchmark: {metric[2]} vs subject nonlinearity",
                xlabel="subject nonlinear strength",
                ylabel=metric[2],
                out_path=out_dir / f"hierarchical_nonlinear_{metric[0].replace('_mean','')}.png",
            )

    full_df = df[df["benchmark"] == "hierarchical_nonlinear_shift"].copy()
    if not full_df.empty:
        cmap = method_color_map_full()
        for metric in [("full_mcc_mean", "full_mcc_std", "FULL MCC"),
                       ("perm_agreement_mean", "perm_agreement_std", "Perm agreement"),
                       ("sign_agreement_mean", "sign_agreement_std", "Sign agreement")]:
            make_curve(
                full_df,
                methods=["M2", "M3"],
                x_col="x_value",
                metric_mean=metric[0],
                metric_std=metric[1],
                color_map=cmap,
                title=f"Residual conditional shift benchmark: {metric[2]} vs shift strength",
                xlabel="conditional shift strength",
                ylabel=metric[2],
                out_path=out_dir / f"full_model_shift_{metric[0].replace('_mean','')}.png",
            )

        xvals = sorted(full_df["x_value"].astype(float).unique().tolist())
        methods = ["M2", "M3"]
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        width = 0.36
        xpos = np.arange(len(xvals))
        for i, method in enumerate(methods):
            sub = full_df[full_df["method"] == method].sort_values("x_value")
            if sub.empty:
                continue
            ax.bar(
                xpos + (i - 0.5) * width,
                sub["full_mcc_mean"].astype(float).values,
                width=width,
                yerr=sub["full_mcc_std"].astype(float).values,
                color=method_color_map_full()[method],
                edgecolor="black",
                linewidth=0.4,
                capsize=3,
                label=method,
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{v:.2f}" for v in xvals])
        ax.set_xlabel("conditional shift strength")
        ax.set_ylabel("FULL MCC")
        ax.set_title("Full model improvement under residual conditional shift")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.legend(frameon=False)
        fig.tight_layout()
        out = out_dir / "full_model_shift_full_mcc_bar.png"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)


def load_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        obj = np.load(path)
        if isinstance(obj, np.lib.npyio.NpzFile):
            if "matrix" in obj.files:
                return obj["matrix"]
            return obj[obj.files[0]]
        return obj
    if suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "matrix" in obj:
            return np.asarray(obj["matrix"], dtype=float)
        return np.asarray(obj, dtype=float)
    raise ValueError(f"Unsupported matrix file suffix: {suffix}")


def load_labels(path: Optional[Path], n: int) -> List[str]:
    if path is None:
        return [f"D{i}" for i in range(n)]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "labels" in obj:
            return list(obj["labels"])
        return list(obj)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != n:
        raise ValueError(f"Expected {n} labels, got {len(lines)}")
    return lines


def _load_summary_seed_metrics(sp: Path) -> List[Dict[str, Any]]:
    with open(sp, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("seed_metrics", [])


def collect_exp2_violin_rows(manifest: List[Dict[str, Any]], run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in manifest:
        if item.get("experiment") != "Exp2":
            continue
        if item.get("method") not in {"B1", "M1"}:
            continue
        sp = summary_path(run_dir, item["doc"])
        if not sp.exists():
            continue
        seed_metrics = _load_summary_seed_metrics(sp)
        for sm in seed_metrics:
            if "full_mcc" not in sm:
                continue
            rows.append({
                "doc": item["doc"],
                "order": int(item.get("order", 0)),
                "method": str(item.get("method", "")),
                "label": str(item.get("label", "")),
                "drift_strength": float(item.get("x_value", np.nan)),
                "seed": int(sm.get("seed", len(rows))),
                "full_mcc": float(sm["full_mcc"]),
                "summary_path": str(sp),
            })
    if len(rows) == 0:
        return pd.DataFrame(columns=["doc", "order", "method", "label", "drift_strength", "seed", "full_mcc", "summary_path"])
    return pd.DataFrame(rows).sort_values(["drift_strength", "method", "seed", "doc"]).reset_index(drop=True)


def save_exp2_violin_palette_csv(out_csv: Path):
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "edge_color_hex", "fill_color_hex"])
        writer.writerow(["B1", EXP2_VIOLIN_STYLE["B1_edge"], EXP2_VIOLIN_STYLE["B1_fill"]])
        writer.writerow(["M1", EXP2_VIOLIN_STYLE["M1_edge"], EXP2_VIOLIN_STYLE["M1_fill"]])


def _draw_single_violin(ax, values: np.ndarray, position: float, edge_color: str, fill_color: str):
    vp = ax.violinplot([values], positions=[position], widths=0.72,
                       showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(fill_color)
        body.set_edgecolor(edge_color)
        body.set_linewidth(1.2)
        body.set_alpha(0.65)

    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr_low = np.min(values)
    iqr_high = np.max(values)
    ax.plot([position, position], [iqr_low, iqr_high], color=edge_color, linewidth=1.0, alpha=0.9)
    ax.plot([position - 0.12, position + 0.12], [iqr_low, iqr_low], color=edge_color, linewidth=1.0)
    ax.plot([position - 0.12, position + 0.12], [iqr_high, iqr_high], color=edge_color, linewidth=1.0)
    ax.plot([position - 0.18, position + 0.18], [med, med], color=edge_color, linewidth=2.6)
    ax.plot([position - 0.12, position + 0.12], [q1, q1], color=edge_color, linewidth=1.6, alpha=0.95)
    ax.plot([position - 0.12, position + 0.12], [q3, q3], color=edge_color, linewidth=1.6, alpha=0.95)


def plot_exp2_grouped_violin(manifest: List[Dict[str, Any]], run_dir: Path, out_dir: Path):
    df = collect_exp2_violin_rows(manifest, run_dir)
    df.to_csv(out_dir / "exp2_grouped_violin_data.csv", index=False, encoding="utf-8-sig")
    save_exp2_violin_palette_csv(out_dir / "exp2_grouped_violin_palette.csv")
    if df.empty:
        return

    drifts = sorted(df["drift_strength"].dropna().unique().tolist())
    positions: Dict[tuple[float, str], float] = {}
    xticks = []
    xticklabels = []
    group_centers = []
    gap = 2.25
    inner_offset = 0.42
    for i, drift in enumerate(drifts):
        center = i * gap
        positions[(drift, "B1")] = center - inner_offset / 2.0
        positions[(drift, "M1")] = center + inner_offset / 2.0
        xticks.extend([positions[(drift, "B1")], positions[(drift, "M1")]])
        xticklabels.extend(["B1", "M1"])
        group_centers.append(center)

    fig = plt.figure(figsize=(9.6, 6.8), facecolor=EXP2_VIOLIN_STYLE["outer_bg"])
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.76])
    ax.set_facecolor(EXP2_VIOLIN_STYLE["inner_bg"])

    # Rounded inner card for closer style to the user's reference.
    card = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=EXP2_VIOLIN_STYLE["inner_bg"],
        edgecolor="none",
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(card)

    for drift in drifts:
        for method in ["B1", "M1"]:
            sub = df[(df["drift_strength"] == drift) & (df["method"] == method)].copy()
            if sub.empty:
                continue
            vals = sub["full_mcc"].astype(float).values
            pos = positions[(drift, method)]
            edge = EXP2_VIOLIN_STYLE[f"{method}_edge"]
            fill = EXP2_VIOLIN_STYLE[f"{method}_fill"]
            _draw_single_violin(ax, vals, pos, edge_color=edge, fill_color=fill)
            rng = np.random.RandomState(int(round(drift * 1000)) + (0 if method == "B1" else 1000))
            jitter = rng.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=26,
                facecolors="none",
                edgecolors=edge,
                linewidths=0.9,
                alpha=0.75,
                zorder=3,
            )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("FULL MCC")
    ax.set_xlabel("method grouped by drift strength")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.7)

    ymin = float(df["full_mcc"].min())
    ymax = float(df["full_mcc"].max())
    pad = max(0.03, 0.08 * (ymax - ymin + 1e-8))
    ax.set_ylim(ymin - 2.8 * pad, min(1.02, ymax + 1.5 * pad))

    for center, drift in zip(group_centers, drifts):
        ax.text(center, ymin - 1.7 * pad, f"drift={drift:.2f}", ha="center", va="top",
                fontsize=10, color=EXP2_VIOLIN_STYLE["subtitle"])

    ax.set_title("Exp2 grouped violin plot", color=EXP2_VIOLIN_STYLE["title"], pad=14)

    legend_handles = [
        Patch(facecolor=EXP2_VIOLIN_STYLE["B1_fill"], edgecolor=EXP2_VIOLIN_STYLE["B1_edge"], label="B1"),
        Patch(facecolor=EXP2_VIOLIN_STYLE["M1_fill"], edgecolor=EXP2_VIOLIN_STYLE["M1_edge"], label="M1"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    out_png = out_dir / "exp2_grouped_violin_full_mcc.png"
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Planned-suite plotting helper.")
    ap.add_argument("--manifest", type=str, default="configs/experiments/experiment_manifest.yaml")
    ap.add_argument("--run-dir", type=str, default="run/planned_suite")
    ap.add_argument("--out-dir", type=str, default="run/planned_suite/figures")
    ap.add_argument("--mode", type=str, default="all", choices=["all", "summary", "curves", "bars", "heatmap"])
    ap.add_argument("--matrix-file", type=str, default=None, help="Optional matrix file (.npy/.npz/.csv/.txt/.json) for a domain-similarity heatmap.")
    ap.add_argument("--labels-file", type=str, default=None, help="Optional labels file for the heatmap.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    ensure_dir(out_dir)
    set_matplotlib_style()

    manifest = load_manifest(manifest_path)
    df, missing = collect_rows(manifest, run_dir)
    save_aggregate_tables(df, out_dir, missing)

    if args.mode in {"all", "summary", "curves", "bars"}:
        plot_default_suite(df, out_dir)
        plot_exp2_grouped_violin(manifest, run_dir, out_dir)

    if args.matrix_file is not None and args.mode in {"all", "heatmap"}:
        matrix_path = Path(args.matrix_file)
        if not matrix_path.is_absolute():
            matrix_path = repo_root / matrix_path
        labels_path = None if args.labels_file is None else Path(args.labels_file)
        if labels_path is not None and not labels_path.is_absolute():
            labels_path = repo_root / labels_path
        matrix = load_matrix(matrix_path)
        labels = load_labels(labels_path, matrix.shape[0])
        make_heatmap(matrix, labels, "Domain similarity", out_dir / "domain_similarity_heatmap.png")

    print(f"Saved aggregated tables and figures to: {out_dir}")
    if missing:
        print(f"Missing {len(missing)} summaries. See planned_suite_missing.json for details.")


if __name__ == "__main__":
    raise SystemExit(main())
