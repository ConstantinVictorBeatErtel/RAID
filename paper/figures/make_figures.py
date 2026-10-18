"""Make the four manuscript figures from collected JSON only (no models, no datasets).

Inputs (all under artifacts/raid_hybrid_libero_goal_v6/):
  compute_bench_final_20261017/final_analysis.json            mean curves, frozen P1/P2 comparisons
  compute_bench_final_20261017/collection/bench/unit_*.json    per-setting and per-demo errors (final)
  compute_bench_baselines_20261017/collection/bench/unit_*.json per-demo errors (development)

Run from the repository root:  python3 paper/figures/make_figures.py
Writes fig1..fig4 as PDF and PNG next to this script.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE.parent.parent / "artifacts/raid_hybrid_libero_goal_v6"
FINAL = ART / "compute_bench_final_20261017"

# Reference categorical palette (light mode), validated with the dataviz validator:
# slots 1-5 pass the adjacent-pair gates; slots 1-3 pass all-pairs. Low-contrast slots carry direct labels.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID, SURFACE = "#9a9994", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})

NAMES = {
    "raid_lora_v7": "RAID+LoRA v7 (ours)", "raid_v7_head": "RAID v7 head-only (ours)", "lora_mlp": "LoRA",
    "head_only": "Head-only FT", "full_finetune": "Full FT", "probe_fresh": "Frozen-encoder probe",
    "scratch_lora": "LoRA, no source", "strap_style": "STRAP-style", "raid_lora_mlp": "RAID+LoRA v6",
}
TASK = {"put_the_bowl_on_top_of_the_cabinet": "Bowl on cabinet (final)",
        "put_the_wine_bottle_on_top_of_the_cabinet": "Wine bottle on cabinet (final)",
        "open_the_top_drawer_and_put_the_bowl_inside": "Drawer + bowl (dev.)",
        "put_the_cream_cheese_in_the_bowl": "Cream cheese in bowl (dev.)"}


def load(path: Path):
    return json.loads(path.read_text())


def units(run: str):
    files = sorted(glob.glob(str(ART / run / "collection/bench/unit_*.json")))
    assert len(files) == 6, files
    return [load(Path(f)) for f in files]


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_error_vs_compute(analysis: dict):
    """Mean chunk MSE against cumulative adaptation GPU-seconds (symlog x; 0 = no training)."""
    curves, zero = analysis["mean_curve"], {k: m for k, m, _ in analysis["zero"]}
    styled = [("raid_lora_v7", BLUE, "o"), ("raid_v7_head", ORANGE, "s"), ("lora_mlp", AQUA, "^"),
              ("head_only", YELLOW, "D"), ("full_finetune", MAGENTA, "v")]
    others = ["probe_fresh", "scratch_lora", "strap_style", "raid_lora_mlp"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    gray_styles = {"probe_fresh": (":", "."), "scratch_lora": ("-.", "."), "strap_style": ("--", "."),
                   "raid_lora_mlp": ("-", "x")}
    for key in others:
        xs, ys = zip(*[(s, m) for s, m, _, _ in curves[key]])
        ls, mk = gray_styles[key]
        ax.plot(xs, ys, color=GRAY, lw=1.1, ls=ls, marker=mk, ms=4, zorder=1, label=NAMES[key])
    for key, color, marker in styled:
        xs, ys = zip(*[(s, m) for s, m, _, _ in curves[key]])
        ax.plot(xs, ys, color=color, marker=marker, ms=5.5, mec=SURFACE, mew=1.0, label=NAMES[key], zorder=3)
    ax.axhline(zero["support_knn_average"], color=INK2, lw=1.0, ls="--", zorder=2)
    ax.annotate("VINN-style k-NN, no training (0.376)", (0.06, zero["support_knn_average"]), xytext=(0, 3),
                textcoords="offset points", fontsize=7, color=INK2)
    target = curves["lora_mlp"][-1][1]
    ax.axhline(target, color=INK2, lw=0.8, ls=":", zorder=2)
    ax.annotate(f"LoRA, 500 updates ({target:.3f})", (0.002, target), xytext=(0, -9), textcoords="offset points",
                fontsize=7, color=INK2, ha="left")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(-0.005, 1500)
    ax.set_ylim(0.25, 0.56)
    ax.set_xticks([0, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0", "0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("Cumulative adaptation GPU-seconds (symmetric log scale; A100-SXM4-80GB)")
    ax.set_ylabel("Chunk MSE, normalized actions (lower is better)")
    ax.set_title("Final tasks: action error vs adaptation compute (mean of 6 task × seed settings)",
                 fontsize=9, color=INK, loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2, handlelength=2.6)
    ax.text(0.0, -0.22, "Untrained starting points above 0.56 are clipped (1.055 source head; 1.339 fresh head). "
            "Checkpoints at 0, 25, 50, 100, 200, 350, 500 updates.", transform=ax.transAxes, fontsize=6.5,
            color=INK2)
    save(fig, "fig3_error_vs_gpu_seconds")


def fig2_frozen_comparisons(analysis: dict):
    """Forest plot of the frozen P1 and P2 comparisons with cluster bootstrap intervals and settings won."""
    rows = analysis["primary"]
    verdict_color = {"better": BLUE, "worse": ORANGE, "no clear difference": GRAY}
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    labels, y = [], 0
    groups = [("P1: no training", [r for r in rows if r["tag"] == "P1"]),
              ("P2: 25 updates", [r for r in rows if r["tag"] == "P2" and r["comparison"].endswith("25 updates")]),
              ("P2: 500 updates", [r for r in rows if r["tag"] == "P2" and r["comparison"].endswith("500 updates")])]
    ticks = []
    for title, group in groups:
        ax.text(-0.62, y - 0.1, title, fontsize=8, fontweight="bold", color=INK, va="bottom")
        y -= 1
        for r in group:
            lo, hi = r["cluster_ci"]
            c = verdict_color[r["verdict"]]
            ax.plot([lo, hi], [y, y], color=c, lw=2.0, solid_capstyle="round")
            ax.plot([r["mean"]], [y], "o", color=c, mec=SURFACE, mew=1.0, ms=6)
            name = r["comparison"].split(" vs ", 1)[1].rsplit(",", 1)[0]
            ticks.append((y, ("RAID v7 vs " if r["tag"] == "P1" else "vs ") + name))
            ax.text(1.02, y, f"{r['mean']:+.3f}   {r['won']}/{r['settings']}   {r['verdict']}", va="center",
                    fontsize=7, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([s for _, s in ticks], fontsize=7.5)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(-0.62, 0.30)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Paired difference in chunk MSE (ours minus baseline; negative favours ours)\n"
                  "95% percentile bootstrap over 6 task × seed clusters")
    ax.set_title("Frozen comparisons on the final tasks (P2 uses RAID+LoRA v7)", fontsize=9, color=INK, loc="left")
    ax.text(1.02, 0.25, "mean      won   verdict", fontsize=7, color=INK2, transform=ax.get_yaxis_transform(),
            clip_on=False)
    save(fig, "fig2_frozen_comparisons")


def fig1_per_setting(final_units: list[dict]):
    """Per-setting chunk MSE: two facets (no training; 25 updates), three series each."""
    def at25(u, key):
        return next(p["mse_macro"] for p in u["curves"][key] if p["updates"] == 25)

    order = sorted(final_units, key=lambda u: (u["task"] != "put_the_bowl_on_top_of_the_cabinet", u["seed"]))
    labels = [f"{'Bowl' if 'bowl' in u['task'] else 'Wine bottle'}, seed {u['seed']}" for u in order]
    facets = [("No target training", [("raid_bank_v7", "RAID v7 (ours)", BLUE, "o"),
                                      ("support_knn_average", "VINN-style k-NN", ORANGE, "s"),
                                      ("regent_blend", "REGENT-style blend", AQUA, "^")],
               lambda u, k: u["zero_adaptation"][k]["mse_macro"]),
              ("After 25 updates", [("raid_lora_v7", "RAID+LoRA v7 (ours)", BLUE, "o"),
                                    ("lora_mlp", "LoRA", ORANGE, "s"),
                                    ("raid_v7_head", "RAID v7 head-only (ours)", AQUA, "^")], at25)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, (title, series, get) in zip(axes, facets):
        for j, (key, name, color, marker) in enumerate(series):
            xs = [get(u, key) for u in order]
            ax.plot(xs, [i + (j - 1) * 0.18 for i in range(len(order))], marker, color=color, mec=SURFACE,
                    mew=0.8, ms=6, label=name, ls="none")
        ax.axhline(2.5, color=GRID, lw=1.0)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Chunk MSE (lower is better)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=6.8, ncol=1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.suptitle("Final tasks, per task × seed setting", fontsize=9, x=0.02, y=1.06, ha="left", color=INK)
    save(fig, "fig1_per_setting")


def fig4_posthoc(final_units: list[dict], dev_units: list[dict]):
    """POST HOC: per-demo RAID v7 error against frozen-source error and against k-NN error (no training)."""
    pts = []
    for u in final_units + dev_units:
        z = u["zero_adaptation"]
        for d, v7 in z["raid_bank_v7"]["mse_by_demo"].items():
            pts.append((u["task"], v7, z["source_frozen"]["mse_by_demo"][d], z["support_knn_average"]["mse_by_demo"][d]))
    assert len(pts) == 240, len(pts)
    wine = "put_the_wine_bottle_on_top_of_the_cabinet"
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, idx, xlabel in ((axes[0], 2, "Frozen source controller error"), (axes[1], 3, "VINN-style k-NN error")):
        rest = [p for p in pts if p[0] != wine]
        hit = [p for p in pts if p[0] == wine]
        ax.scatter([p[idx] for p in rest], [p[1] for p in rest], s=12, color=BLUE, alpha=0.55, lw=0,
                   label="Other 3 tasks (180 demos)")
        ax.scatter([p[idx] for p in hit], [p[1] for p in hit], s=14, color=ORANGE, alpha=0.85, lw=0, marker="s",
                   label="Wine bottle, final (60 demos)")
        lim = max(max(p[idx] for p in pts), max(p[1] for p in pts)) * 1.05
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(xlabel)
        ax.set_aspect("equal")
    axes[0].set_ylabel("RAID v7 error (no training)")
    axes[0].text(0.03, 0.95, "wine bottle: r = 0.97", transform=axes[0].transAxes, fontsize=7, color=INK2, va="top")
    axes[1].text(0.03, 0.95, "wine bottle: r = 0.31", transform=axes[1].transAxes, fontsize=7, color=INK2, va="top")
    axes[1].legend(loc="lower right", fontsize=6.8)
    fig.suptitle("POST HOC: per-demonstration errors without training (6 final + 6 development settings)",
                 fontsize=9, x=0.02, ha="left", color=INK)
    save(fig, "fig4_posthoc_scatter")


def main():
    analysis = load(FINAL / "final_analysis.json")
    final_units = units("compute_bench_final_20261017")
    dev_units = units("compute_bench_baselines_20261017")
    fig3_error_vs_compute(analysis)
    fig2_frozen_comparisons(analysis)
    fig1_per_setting(final_units)
    fig4_posthoc(final_units, dev_units)


if __name__ == "__main__":
    main()
