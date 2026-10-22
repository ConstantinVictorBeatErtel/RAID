"""Make the six manuscript figures from collected artifacts only (no models, no datasets).

Inputs (all under artifacts/raid_hybrid_libero_goal_v6/):
  compute_bench_final_20261017/final_analysis.json            mean curves, frozen P1/P2 comparisons
  compute_bench_final_20261017/collection/bench/unit_*.json    per-setting and per-demo errors (final)
  compute_bench_baselines_20261017/collection/bench/unit_*.json per-demo errors (development)
  vpp2_pilot_v5_20261014/collection/pilot/.../episodes/*_plan_vs_rollout.mp4  pilot frames (Figures 1, 6)

Run from the repository root:  python3 paper/figures/make_figures.py
Writes fig1..fig6 as PDF and PNG next to this script.
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
PILOT = ART / "vpp2_pilot_v5_20261014/collection/pilot"
# Fixed example rule for pilot frames: the earliest success of the pilot (cream cheese, start 18, RAID+LoRA v6) and
# the frozen source controller on the same start, which shares the first VPP2 plan.
PILOT_TASK = "put_the_cream_cheese_in_the_bowl"
PILOT_CASE = "put_the_cream_cheese_in_the_bowl_d18_s42"

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


def fig3_learning_curves(analysis: dict):
    """Mean chunk MSE against target updates (left) and cumulative adaptation GPU-seconds (right, symlog)."""
    curves, zero = analysis["mean_curve"], {k: m for k, m, _ in analysis["zero"]}
    styled = [("raid_lora_v7", BLUE, "o"), ("raid_v7_head", ORANGE, "s"), ("lora_mlp", AQUA, "^"),
              ("head_only", YELLOW, "D"), ("full_finetune", MAGENTA, "v")]
    others = ["probe_fresh", "scratch_lora", "strap_style", "raid_lora_mlp"]
    gray_styles = {"probe_fresh": (":", "."), "scratch_lora": ("-.", "."), "strap_style": ("--", "."),
                   "raid_lora_mlp": ("-", "x")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    target = curves["lora_mlp"][-1][1]
    for ax, xi in ((axes[0], 3), (axes[1], 0)):
        for key in others:
            xs, ys = zip(*[(p[xi], p[1]) for p in curves[key]])
            ls, mk = gray_styles[key]
            ax.plot(xs, ys, color=GRAY, lw=1.1, ls=ls, marker=mk, ms=4, zorder=1, label=NAMES[key])
        for key, color, marker in styled:
            xs, ys = zip(*[(p[xi], p[1]) for p in curves[key]])
            ax.plot(xs, ys, color=color, marker=marker, ms=5.5, mec=SURFACE, mew=1.0, label=NAMES[key], zorder=3)
        ax.axhline(zero["support_knn_average"], color=INK2, lw=1.0, ls="--", zorder=2)
        ax.axhline(target, color=INK2, lw=0.8, ls=":", zorder=2)
        ax.set_ylim(0.25, 0.56)
    a0, a1 = axes
    a0.set_xlim(-15, 515)
    a0.set_xticks([0, 100, 200, 350, 500])
    a0.set_xlabel("Target updates (batch 64)")
    a0.set_ylabel("Chunk MSE, normalized actions (lower is better)")
    a0.set_title("(a) Error per update", fontsize=8.5, color=INK, loc="left")
    a0.axvline(25, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=0)
    a0.annotate("25 updates (P2)", (25, 0.545), xytext=(3, 0), textcoords="offset points", fontsize=6.8,
                color=INK2, va="top")
    a0.annotate("VINN-style k-NN, no training (0.376)", (40, zero["support_knn_average"]), xytext=(0, 3),
                textcoords="offset points", fontsize=6.8, color=INK2)
    a1.annotate(f"LoRA after 500 updates ({target:.3f})", (0.002, target), xytext=(0, -9),
                textcoords="offset points", fontsize=6.8, color=INK2)
    a1.set_xscale("symlog", linthresh=0.05)
    a1.set_xlim(-0.005, 1500)
    a1.set_xticks([0, 0.1, 1, 10, 100, 1000])
    a1.set_xticklabels(["0", "0.1", "1", "10", "100", "1000"])
    a1.set_xlabel("Cumulative adaptation GPU-seconds (symlog; A100)")
    a1.set_title("(b) Error per GPU-second", fontsize=8.5, color=INK, loc="left")
    handles, labels = a1.get_legend_handles_labels()
    order = list(range(len(others), len(labels))) + list(range(len(others)))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=5, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.13), handlelength=2.4, columnspacing=1.2)
    fig.suptitle("Final tasks: action error during target adaptation (mean of 6 task × seed settings)",
                 fontsize=9, x=0.07, y=1.0, ha="left", color=INK)
    fig.text(0.07, -0.19, "Untrained starting points above 0.56 are clipped (1.055 source head; 1.339 fresh head). "
             "Checkpoints at 0, 25, 50, 100, 200, 350, 500 updates.", fontsize=6.5,
             color=INK2)
    save(fig, "fig3_learning_curves")


def fig4_frozen_comparisons(analysis: dict):
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
    save(fig, "fig4_frozen_comparisons")


def fig2_per_setting(final_units: list[dict]):
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
    save(fig, "fig2_per_setting")


def fig5_posthoc(final_units: list[dict], dev_units: list[dict]):
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
    save(fig, "fig5_posthoc_scatter")


def pilot_frames(condition: str) -> list:
    """RGB frames of a pilot plan_vs_rollout video: (plan, executed) pairs, one per executed step."""
    import cv2

    path = PILOT / PILOT_TASK / "rollout/episodes" / f"{PILOT_CASE}_{condition}_plan_vs_rollout.mp4"
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = frame[:, :, ::-1]
        out.append((rgb[:, :128], rgb[:, 128:]))
    assert out, path
    return out


def _box(ax, xy, w, h, text, fc, ec=INK2, fs=7.2, bold=False, ls="-"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=0.9,
                                ls=ls, transform=ax.transAxes, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", transform=ax.transAxes, zorder=3, linespacing=1.25)


def _arrow(ax, a, b, color=INK2, ls="-"):
    ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8), zorder=4)


def fig1_overview():
    """Schematic: (a) pipeline with RAID v7 IDM, using real pilot frames; (b) development / freeze / final protocol."""
    frames = pilot_frames("raid_lora_mlp")
    current, future = frames[0][1], frames[8][0]
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0, 0.30, 1, 0.70])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.005, 0.975, "(a) Pipeline and RAID v7 inverse-dynamics model", fontsize=8.5, color=INK, va="top")
    # left column: current frame -> frozen generator -> future frame; both frames feed the IDM through a bus
    ia = fig.add_axes([0.015, 0.30 + 0.70 * 0.60, 0.112, 0.175])
    ia.imshow(current)
    ia.set_axis_off()
    ib = fig.add_axes([0.015, 0.30 + 0.70 * 0.08, 0.112, 0.175])
    ib.imshow(future)
    ib.set_axis_off()
    ax.text(0.071, 0.875, "current frame oₜ, state qₜ", ha="center", fontsize=6.3, color=INK2)
    ax.text(0.071, 0.065, "future oₜ₊₈: VPP2 plan (pilot)\nor recorded frame (offline)", ha="center", va="top",
            fontsize=6.3, color=INK2)
    _box(ax, (0.015, 0.375), 0.112, 0.17, "frozen video\ngenerator (VPP2)\n+ instruction", "#eef3fb", fs=6.2)
    _arrow(ax, (0.071, 0.60), (0.071, 0.545))
    _arrow(ax, (0.071, 0.375), (0.071, 0.33))
    bus = 0.215
    ax.plot([0.13, bus], [0.72, 0.72], color=INK2, lw=1.0, transform=ax.transAxes)
    ax.plot([0.13, bus], [0.20, 0.20], color=INK2, lw=1.0, transform=ax.transAxes)
    ax.plot([bus, bus], [0.20, 0.72], color=INK2, lw=1.0, transform=ax.transAxes)
    ax.text(bus + 0.006, 0.46, "oₜ, oₜ₊₈, qₜ", fontsize=6.3, color=INK2, rotation=90, va="center")
    # IDM container
    _box(ax, (0.305, 0.04), 0.56, 0.88, "", SURFACE, ec=INK2, ls="--")
    ax.text(0.318, 0.895, "RAID v7 IDM (adapted from K = 5 target demonstrations)", fontsize=7, color=INK2,
            va="top")
    _box(ax, (0.33, 0.60), 0.15, 0.20, "controller encoder φ\nMAE ViT-B/16\n(frozen or LoRA)", "#fdf0ea", fs=6.6)
    _box(ax, (0.33, 0.17), 0.15, 0.20, "frozen reference\nencoder φ_ref\n(retrieval keys)", "#e8f6f0", fs=6.6)
    _arrow(ax, (bus, 0.70), (0.33, 0.70))
    _arrow(ax, (bus, 0.27), (0.33, 0.27))
    _box(ax, (0.51, 0.60), 0.13, 0.20, "direct head\nf_θ(s)\nMLP 512–256", "#fdf0ea", fs=6.6)
    _box(ax, (0.51, 0.17), 0.13, 0.20, "support bank\ntop-k = 16\ncandidates (φ_ref, aⁱ)", "#e8f6f0", fs=6.6)
    _arrow(ax, (0.48, 0.70), (0.51, 0.70))
    _arrow(ax, (0.48, 0.27), (0.51, 0.27))
    _box(ax, (0.67, 0.17), 0.17, 0.20, "attention over\ncandidates → prior\n(selection loss)", "#e8f6f0", fs=6.6)
    _arrow(ax, (0.64, 0.27), (0.67, 0.27))
    _arrow(ax, (0.575, 0.60), (0.73, 0.37), color=INK2, ls=":")
    ax.text(0.665, 0.475, "query s", fontsize=6.2, color=INK2, rotation=0)
    _box(ax, (0.67, 0.60), 0.17, 0.20, "per-dimension gate g\nâ = g ⊙ direct\n+ (1 − g) ⊙ prior", "#fff6dc", fs=6.6,
         bold=False)
    _arrow(ax, (0.64, 0.70), (0.67, 0.70))
    _arrow(ax, (0.755, 0.37), (0.755, 0.60))
    _box(ax, (0.89, 0.60), 0.10, 0.20, "action chunk\n8 × 7\n→ simulator", "#eef3fb", fs=6.6)
    _arrow(ax, (0.84, 0.70), (0.89, 0.70))
    # protocol strip
    bx = fig.add_axes([0, 0, 1, 0.29])
    bx.set_axis_off()
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.text(0.005, 0.97, "(b) Protocol: design on development tasks, freeze, then one run on sealed final tasks",
            fontsize=8.5, color=INK, va="top")
    steps = [("6 source tasks\nsource controller training\n(2,000 updates per head and seed)", "#eef3fb"),
             ("2 development tasks\nmethod design, 3 benchmarks\n(v6/v7, 9 baselines, v7 head-only)", "#fdf0ea"),
             ("Frozen amendment\n17 Oct 2026, 20:15 UTC\nmethods, comparisons P1–P3, rule", "#fff6dc"),
             ("2 sealed final tasks × 3 seeds\nrun once, nothing tuned after\n15 methods, 7 checkpoints", "#e8f6f0")]
    w, gap, x = 0.225, 0.023, 0.01
    for i, (t, fc) in enumerate(steps):
        _box(bx, (x, 0.12), w, 0.62, t, fc, fs=6.4)
        if i < len(steps) - 1:
            _arrow(bx, (x + w, 0.43), (x + w + gap, 0.43))
        x += w + gap
    save(fig, "fig1_overview")


def fig6_pilot_filmstrip():
    """Closed-loop pilot frames: VPP2 plan targets vs executed frames for a success and a failure on one start."""
    cases = [("raid_lora_mlp", "RAID+LoRA v6: success at step 84"),
             ("source_frozen", "Frozen source controller, same start: timeout at 300 steps")]
    ncol = 6
    fig, grid = plt.subplots(5, ncol, figsize=(7.2, 5.9),
                             gridspec_kw={"hspace": 0.18, "wspace": 0.04, "height_ratios": [1, 1, 0.42, 1, 1]})
    for a in grid[2]:
        a.set_axis_off()
    axes = grid[[0, 1, 3, 4]]
    for c, (cond, title) in enumerate(cases):
        frames = pilot_frames(cond)
        n = len(frames)
        idx = [round(i * (n - 1) / (ncol - 1)) for i in range(ncol)]
        for j, t in enumerate(idx):
            for r, (img, rowname) in enumerate(((frames[t][0], "VPP2 plan"), (frames[t][1], "executed"))):
                a = axes[2 * c + r, j]
                a.imshow(img, interpolation="bilinear")
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    a.set_ylabel(rowname, fontsize=7.5, color=INK)
                if r == 0:
                    a.set_title(f"step {t}", fontsize=7, color=INK2, pad=2)
        axes[2 * c, 0].text(0, 1.22, f"({'ab'[c]}) {title}", transform=axes[2 * c, 0].transAxes, fontsize=8.5,
                            color=INK, va="bottom")
    fig.text(0.125, 0.075, "Task: put the cream cheese in the bowl (development task), start 18, seed 42. "
             "Both conditions share the first VPP2 plan.\nTop row of each panel: generated target frame used "
             "at that step; bottom row: simulator frame actually reached.", fontsize=6.5, color=INK2, va="top")
    save(fig, "fig6_pilot_filmstrip")


def main():
    analysis = load(FINAL / "final_analysis.json")
    final_units = units("compute_bench_final_20261017")
    dev_units = units("compute_bench_baselines_20261017")
    fig1_overview()
    fig2_per_setting(final_units)
    fig3_learning_curves(analysis)
    fig4_frozen_comparisons(analysis)
    fig5_posthoc(final_units, dev_units)
    fig6_pilot_filmstrip()


if __name__ == "__main__":
    main()
