"""Make the RAID v8 manuscript figures from paper/v8_numbers.json and the collected run outputs only.

Run from the repository root:  python3 paper/v8_numbers.py && python3 paper/figures/make_figures.py [LIBERO_GOAL_HDF5]
The optional HDF5 (a LIBERO-Goal demo file) supplies the two example frames of Figure 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
N = json.loads((HERE.parent / "v8_numbers.json").read_text())
ENC = ("mae", "dinov2", "siglip")
ENC_NAME = {"mae": "MAE", "dinov2": "DINOv2", "siglip": "SigLIP"}

# Validated categorical palette (light mode); low-contrast slots carry direct labels.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID, SURFACE = "#9a9994", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ENC_COLOR = {"mae": BLUE, "dinov2": ORANGE, "siglip": AQUA}
ENC_MARK = {"mae": "o", "dinov2": "s", "siglip": "^"}

plt.rcParams.update({
    "font.family": ["Helvetica", "DejaVu Sans"], "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, fs=6.6, ls="-"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=INK2, lw=0.9,
                                ls=ls, transform=ax.transAxes, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            transform=ax.transAxes, zorder=3, linespacing=1.25)


def _arrow(ax, a, b, ls="-"):
    ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=4)


def fig1_overview(hdf5: str | None):
    """(a) The RAID v8 inverse-dynamics model; (b) the study protocol."""
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0, 0.30, 1, 0.70]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.005, 0.975, "(a) RAID v8: retrieval-augmented inverse dynamics with a disagreement-aware gate",
            fontsize=8.5, color=INK, va="top")
    if hdf5:
        import h5py
        with h5py.File(hdf5, "r") as h:
            frames = np.asarray(h["data/demo_0/obs/agentview_rgb"])[:, ::-1]  # OpenGL image convention -> upright
        for y, img in ((0.60, frames[0]), (0.08, frames[min(8, len(frames) - 1)])):
            a = fig.add_axes([0.015, 0.30 + 0.70 * y, 0.112, 0.175]); a.imshow(img); a.set_axis_off()
    ax.text(0.071, 0.875, "current frame oₜ, state qₜ", ha="center", fontsize=6.3, color=INK2)
    ax.text(0.071, 0.065, "target frame oₜ₊₈ (recorded here;\nfrom a video planner when deployed)", ha="center",
            va="top", fontsize=5.8, color=INK2)
    bus = 0.165
    for y in (0.72, 0.20):
        ax.plot([0.13, bus], [y, y], color=INK2, lw=1.0, transform=ax.transAxes)
    ax.plot([bus, bus], [0.20, 0.72], color=INK2, lw=1.0, transform=ax.transAxes)
    _box(ax, (0.185, 0.05), 0.69, 0.87, "", SURFACE, ls="--")
    ax.text(0.193, 0.895, "inverse-dynamics model (frozen image encoder; head adapted from K = 5 demonstrations)",
            fontsize=6.8, color=INK2, va="top")
    _box(ax, (0.20, 0.60), 0.12, 0.20, "frozen encoder\n(MAE, DINOv2\nor SigLIP)", "#eef3fb")
    _box(ax, (0.20, 0.17), 0.12, 0.20, "same frozen\nfeatures as\nretrieval keys", "#e8f6f0")
    _arrow(ax, (bus, 0.70), (0.20, 0.70)); _arrow(ax, (bus, 0.27), (0.20, 0.27))
    _box(ax, (0.35, 0.60), 0.12, 0.20, "direct head\nf_θ(s)", "#fdf0ea")
    _box(ax, (0.35, 0.17), 0.12, 0.20, "support bank\ntop-16\nneighbours", "#e8f6f0")
    _arrow(ax, (0.32, 0.70), (0.35, 0.70)); _arrow(ax, (0.32, 0.27), (0.35, 0.27))
    _box(ax, (0.50, 0.17), 0.13, 0.20, "attention over\nneighbours\n→ prior", "#e8f6f0")
    _arrow(ax, (0.47, 0.27), (0.50, 0.27))
    _box(ax, (0.50, 0.60), 0.13, 0.20, "confidence and\ndisagreement\n|direct − prior|", "#fff6dc")
    _arrow(ax, (0.47, 0.70), (0.50, 0.70)); _arrow(ax, (0.565, 0.37), (0.565, 0.60))
    _box(ax, (0.66, 0.40), 0.20, 0.24, "gate g\nâ = g·direct + (1 − g)·prior\ntrained with corrupted\ndirect predictions", "#fff6dc",
         fs=6.3)
    _arrow(ax, (0.63, 0.70), (0.70, 0.64)); _arrow(ax, (0.63, 0.27), (0.70, 0.40))
    _box(ax, (0.895, 0.42), 0.10, 0.20, "action\nchunk\n8 × 7", "#eef3fb")
    _arrow(ax, (0.86, 0.52), (0.895, 0.52))
    bx = fig.add_axes([0, 0, 1, 0.29]); bx.set_axis_off(); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    bx.text(0.005, 0.97, "(b) Protocol: design on development tasks, freeze, then one run on 30 sealed tasks",
            fontsize=8.5, color=INK, va="top")
    steps = [("Source: LIBERO-90\n90 tasks, 45 + 5 demos each\nsource heads, 3 encoders × 3 seeds", "#eef3fb"),
             ("Development: LIBERO-Goal\n10 tasks: gate variant,\nlearning rates, gripper rule", "#fdf0ea"),
             ("Frozen amendment\nmethods, settings, H1–H3,\ncode digest", "#fff6dc"),
             ("Final: 30 sealed tasks\nLIBERO-10, Spatial, Object\nrun once, nothing tuned after", "#e8f6f0")]
    w, gap, x = 0.225, 0.023, 0.01
    for i, (t, fc) in enumerate(steps):
        _box(bx, (x, 0.12), w, 0.62, t, fc, fs=6.4)
        if i < len(steps) - 1:
            _arrow(bx, (x + w, 0.43), (x + w + gap, 0.43))
        x += w + gap
    save(fig, "fig1_overview")


def fig2_zero_per_task():
    """Zero-training error per final task: RAID v8 against k-NN (a) and against RAID v7 (b), three encoders."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, other, label in ((axes[0], "knn", "VINN-style k-NN"), (axes[1], "v7", "RAID v7 (no disagreement input)")):
        lo, hi = 1e9, 0
        for e in ENC:
            z = N["final"][e]["zero_per_task"]
            x = np.array([z[other][t] for t in z["knn"]]); y = np.array([z["v8_c25"][t] for t in z["knn"]])
            ax.scatter(x, y, s=16, color=ENC_COLOR[e], marker=ENC_MARK[e], alpha=0.85, lw=0, label=ENC_NAME[e])
            lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        lo, hi = lo * 0.92, hi * 1.05
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"{label}: chunk MSE")
        b = sum(N["final"][e][f"v8_minus_{other}"]["tasks_better"] for e in ENC)
        ax.text(0.03, 0.96, f"below the diagonal: RAID v8 better\n{b} of 90 task × encoder pairs", transform=ax.transAxes,
                fontsize=7, color=INK2, va="top")
    axes[0].set_ylabel("RAID v8 (ours): chunk MSE")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Final tasks, no target training: one point per task and encoder (mean of 3 seeds)", fontsize=9,
                 x=0.02, ha="left", color=INK)
    save(fig, "fig2_zero_per_task")


def fig3_hypotheses():
    """Pre-registered comparisons H1-H3 for each encoder: paired difference with task-level bootstrap intervals."""
    labels = {"H1": "H1: RAID v8 − RAID v7 (no training)", "H2": "H2: RAID v8 − k-NN (no training)",
              "H3": "H3: RAID v8 − head-only (25 updates)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y, ticks = 0, []
    for h in ("H1", "H2", "H3"):
        for e in ENC:
            t = next(x for x in N["final"][e]["tests"] if x["hypothesis"] == h)
            c = ENC_COLOR[e]
            ax.plot(t["ci95"], [y, y], color=c, lw=2.2, solid_capstyle="round")
            ax.plot([t["mean_diff"]], [y], ENC_MARK[e], color=c, mec=SURFACE, mew=1.0, ms=6.5)
            ax.text(1.02, y, f"{t['mean_diff']:+.3f}  {t['tasks_better']}/{t['tasks']}  {t['verdict']}", va="center",
                    fontsize=7, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
            ticks.append((y, f"{labels[h] if e == 'mae' else ''}  {ENC_NAME[e]}"))
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([s for _, s in ticks], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Paired difference in chunk MSE (negative favours RAID v8)\n95% bootstrap interval over 30 tasks")
    ax.text(1.02, 0.6, "mean   better   Holm verdict", fontsize=7, color=INK2, transform=ax.get_yaxis_transform(),
            clip_on=False)
    ax.set_title("Pre-registered comparisons on the 30 sealed tasks", fontsize=9, color=INK, loc="left")
    save(fig, "fig3_hypotheses")


def fig4_compute():
    """MAE: chunk MSE against cumulative adaptation GPU-seconds for head updates and encoder tuning."""
    f = N["final"]["mae"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    series = [("raid_v8", "RAID v8 head (ours)", BLUE, "o"), ("raid_v7", "RAID v7 head", GRAY, "x"),
              ("head_only", "Head-only", YELLOW, "D")]
    for key, name, color, mk in series:
        a = f["adapted"][key]
        xs = [a["gpu_seconds"][str(n)] for n in (0, 10, 25, 50, 100, 200)]
        ys = [a["mse"][str(n)] for n in (0, 10, 25, 50, 100, 200)]
        ax.plot(xs, ys, color=color, marker=mk, ms=5, mec=SURFACE, mew=0.8, label=name, zorder=3)
    for key, name, color, mk in (("lora", "LoRA (encoder)", AQUA, "^"), ("full", "Full fine-tuning", MAGENTA, "v")):
        k = next(x for x in N["encoder_tuning"] if x.startswith(key))
        a = N["encoder_tuning"][k]
        xs = [a["gpu_seconds"][str(n)] for n in (0, 25, 50, 100, 200)]
        ys = [a["mse"][str(n)] for n in (0, 25, 50, 100, 200)]
        ax.plot(xs, ys, color=color, marker=mk, ms=5, mec=SURFACE, mew=0.8, label=name, zorder=3)
    ax.axhline(f["zero_mean"]["knn"], color=INK2, lw=1.0, ls="--")
    ax.annotate(f"k-NN, no training ({f['zero_mean']['knn']:.3f})", (3.0, f["zero_mean"]["knn"]), xytext=(0, 3),
                textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(-0.005, 60); ax.set_ylim(0.27, 0.56)
    ax.set_xticks([0, 0.1, 1, 10]); ax.set_xticklabels(["0", "0.1", "1", "10"])
    ax.set_xlabel(f"Cumulative adaptation GPU-seconds (symlog; {N['timing']['gpu']})")
    ax.set_ylabel("Chunk MSE (lower is better)")
    ax.set_title("Final tasks, MAE: error against adaptation compute (checkpoints 0–200 updates)", fontsize=9,
                 color=INK, loc="left")
    ax.legend(loc="upper center", fontsize=7, ncol=5, bbox_to_anchor=(0.5, -0.17), handlelength=2.2)
    ax.text(0.0, -0.33, "Untrained starting points above 0.56 are clipped (head-only and encoder tuning start at "
            f"{f['zero_mean']['head_only']:.3f}). Head methods: mean of 3 seeds; encoder tuning: seed 42.",
            transform=ax.transAxes, fontsize=6.5, color=INK2)
    save(fig, "fig5_compute")


def fig5_closed_loop():
    """Closed-loop success with recorded targets, per suite and overall."""
    c = N["closed_loop"]
    order = [("knn", "k-NN"), ("v7_zero", "RAID v7, no training"), ("v8_c25_zero", "RAID v8, no training"),
             ("head_only@25", "Head-only, 25"), ("raid_v8@25", "RAID v8, 25"), ("lora@200", "LoRA, 200"),
             ("full@200", "Full FT, 200")]
    suites = ["libero_10", "libero_spatial", "libero_object"]
    names = {"libero_10": "LIBERO-10", "libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object"}
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    width = 0.11
    colors = [GRAY, "#c9c8c3", BLUE, YELLOW, "#1d5fae", AQUA, MAGENTA]
    groups = suites + ["all"]
    for i, (m, label) in enumerate(order):
        vals = [c["by_suite"][s][m] for s in suites] + [c["mean"][m]]
        xs = np.arange(len(groups)) + (i - 3) * width
        ax.bar(xs, vals, width * 0.92, color=colors[i], label=label, zorder=2)
        lo, hi = c["ci95"][m]
        ax.plot([xs[-1], xs[-1]], [lo, hi], color=INK2, lw=0.9, zorder=3)
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels([names.get(g, "All 30 tasks") for g in groups])
    ax.set_ylabel("Success rate (20 episodes per task)"); ax.set_ylim(0, 0.75)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.8, ncol=4)
    ax.set_title("Closed loop with recorded target frames (seed 42; rounded gripper for every method)", fontsize=9,
                 color=INK, loc="left")
    ax.text(0.0, -0.17, "Bars for all 30 tasks show a 95% bootstrap interval over tasks.", transform=ax.transAxes,
            fontsize=6.5, color=INK2)
    save(fig, "fig6_closed_loop")


def fig6_dev_ablation():
    """Development tasks, no training: the disagreement gate helps only with corruption training."""
    d = N["dev"]["zero_mean"]
    methods = [("knn", "k-NN", GRAY), ("v7", "RAID v7", "#c9c8c3"), ("v8_c0", "v8 gate, no corruption", YELLOW),
               ("v8_c25", "v8 gate + corruption (ours)", BLUE)]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    w = 0.2
    for i, (k, label, color) in enumerate(methods):
        xs = np.arange(len(ENC)) + (i - 1.5) * w
        ax.bar(xs, [d[e][k] for e in ENC], w * 0.92, color=color, label=label, zorder=2)
    ax.set_xticks(np.arange(len(ENC))); ax.set_xticklabels([ENC_NAME[e] for e in ENC])
    ax.set_ylabel("Chunk MSE, no training"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 0.9)
    ax.legend(loc="upper center", fontsize=6.8, ncol=4, bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Development tasks (10 × 3 seeds): which part of the gate matters", fontsize=9, color=INK,
                 loc="left", pad=22)
    save(fig, "fig4_dev_ablation")


def main():
    fig1_overview(sys.argv[1] if len(sys.argv) > 1 else None)
    fig2_zero_per_task()
    fig3_hypotheses()
    fig4_compute()
    fig5_closed_loop()
    fig6_dev_ablation()


if __name__ == "__main__":
    main()
