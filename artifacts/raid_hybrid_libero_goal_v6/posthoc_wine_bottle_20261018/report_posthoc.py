"""POST HOC diagnosis of the RAID v7 zero-training failure on the wine-bottle final task.

Reads only collected per-demo errors (development baseline run + the one final run). No model is run, no dataset
is opened, nothing is re-selected. Gate values, attention weights and retrieval distances were NOT collected, so
they are not available here. Writes POSTHOC_WINE_BOTTLE.md, posthoc_wine_bottle.html, posthoc_analysis.json."""
import glob, html, json, math, statistics as st
from pathlib import Path
P = Path(__file__).resolve().parent
A = P.parent
FIN = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(A / "compute_bench_final_20261017/collection/bench/unit_*.json")))]
DEV = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(A / "compute_bench_baselines_20261017/collection/bench/unit_*.json")))]
assert len(FIN) == 6 and len(DEV) == 6
SHORT = {"put_the_wine_bottle_on_top_of_the_cabinet": "Wine bottle on cabinet (final)",
         "put_the_bowl_on_top_of_the_cabinet": "Bowl on cabinet (final)",
         "open_the_top_drawer_and_put_the_bowl_inside": "Drawer + bowl (development)",
         "put_the_cream_cheese_in_the_bowl": "Cream cheese in bowl (development)"}
ORDER = list(SHORT)  # fixed categorical order: slot 1..4
SHAPE = ["circle", "square", "triangle", "diamond"]
ZERO = [("source_frozen", "Frozen source controller"), ("raid_bank_mlp", "RAID v6"), ("raid_bank_v7", "RAID v7"),
        ("support_knn_average", "VINN-style k-NN"), ("retrieve_and_play", "REGENT retrieve-and-play"),
        ("regent_blend", "REGENT-style blend")]


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(order):
        r[i] = k
    return r


spearman = lambda x, y: pearson(ranks(x), ranks(y))
units = FIN + DEV
demos = [{"task": u["task"], "seed": u["seed"], "demo": d,
          **{k: u["zero_adaptation"][k]["mse_by_demo"][d] for k, _ in ZERO}}
         for u in units for d in u["zero_adaptation"]["raid_bank_v7"]["mse_by_demo"]]
per_task, corr = {}, {}
for t in ORDER:
    uu = [u for u in units if u["task"] == t]
    per_task[t] = {**{k: st.mean(u["zero_adaptation"][k]["mse_macro"] for u in uu) for k, _ in ZERO},
                   "r2_v7": st.mean(u["zero_adaptation"]["raid_bank_v7"]["r2"] for u in uu),
                   "r2_knn": st.mean(u["zero_adaptation"]["support_knn_average"]["r2"] for u in uu),
                   **{f"{k}_25": st.mean(u["curves"][k][1]["mse_macro"] for u in uu) for k in ("head_only", "lora_mlp", "raid_lora_v7")}}
    dd = [r for r in demos if r["task"] == t]
    f, v, k, v6 = ([r[c] for r in dd] for c in ("source_frozen", "raid_bank_v7", "support_knn_average", "raid_bank_mlp"))
    corr[t] = {"n": len(dd), "pearson_v7_frozen": pearson(f, v), "pearson_v7_knn": pearson(k, v),
               "spearman_v7_frozen": spearman(f, v), "spearman_v7_knn": spearman(k, v),
               "pearson_v6_frozen": pearson(f, v6), "share_v7_worse_than_knn": sum(a > b for a, b in zip(v, k)) / len(v)}
settings = [{"task": u["task"], "seed": u["seed"], "frozen": u["zero_adaptation"]["source_frozen"]["mse_macro"],
             "gap": u["zero_adaptation"]["raid_bank_v7"]["mse_macro"] - u["zero_adaptation"]["support_knn_average"]["mse_macro"]}
            for u in units]
others = [s for s in settings if s["task"] != ORDER[0]]
L = lambda t: SHORT[t]
md = ["# POST HOC: why RAID v7 fails without training on the wine-bottle task", "",
      "POST HOC ANALYSIS. It was not in the frozen plan and was done after the final results were known. It changes no frozen result and selects nothing. It uses only the collected per-demo errors (6 final + 6 development settings, 20 query demos each). No model was run and no dataset was opened.",
      "Not available: gate values, attention weights and retrieval distances were not saved in the run, and the pod storage is deleted. The statements below about the gate are therefore inferences from errors, not direct measurements.", "",
      "## Finding 1: the failure is specific to one task, and retrieval itself is not the problem", "",
      "| Task | Frozen source | RAID v6 | RAID v7 | k-NN | Retrieve-and-play | REGENT blend | R2 RAID v7 | R2 k-NN |",
      "|---|---|---|---|---|---|---|---|---|"] + \
     [f"| {L(t)} | " + " | ".join(f"{per_task[t][k]:.3f}" for k, _ in ZERO) + f" | {per_task[t]['r2_v7']:.2f} | {per_task[t]['r2_knn']:.2f} |" for t in ORDER] + \
     ["", f"On the wine-bottle task, plain k-NN over the same support bank still works ({per_task[ORDER[0]]['support_knn_average']:.3f}, similar to the other tasks). The two learned RAID heads both fail ({per_task[ORDER[0]]['raid_bank_mlp']:.3f} and {per_task[ORDER[0]]['raid_bank_v7']:.3f}), and RAID v7 has a negative R2 (worse than predicting the mean action). The frozen source controller is much worse on this task ({per_task[ORDER[0]]['source_frozen']:.3f}) than on any other held-out task (0.63 to 0.88).", "",
      "## Finding 2: on this task, RAID v7 errors follow the source controller, not retrieval", "",
      "Correlation over the query demos of one task (3 seeds x 20 demos).", "",
      "| Task | Demos | Pearson RAID v7 vs frozen | Pearson RAID v7 vs k-NN | Spearman RAID v7 vs frozen | Spearman RAID v7 vs k-NN | Pearson RAID v6 vs frozen | Demos where RAID v7 is worse than k-NN |",
      "|---|---|---|---|---|---|---|---|"] + \
     [f"| {L(t)} | {c['n']} | {c['pearson_v7_frozen']:.2f} | {c['pearson_v7_knn']:.2f} | {c['spearman_v7_frozen']:.2f} | {c['spearman_v7_knn']:.2f} | {c['pearson_v6_frozen']:.2f} | {c['share_v7_worse_than_knn']:.0%} |" for t, c in corr.items()] + \
     ["", "On the wine-bottle task, the demos where the source controller is bad are exactly the demos where RAID v7 is bad (r = 0.97), and the k-NN error hardly predicts it (r = 0.31). On the other three tasks the RAID v7 error follows the k-NN error more closely. Inference: on the wine-bottle task the RAID heads give too much weight to their source-trained direct path, and the retrieved support actions do not correct it enough.", "",
      "## Finding 3: the failure appears only when the source controller is far outside its usual range", "",
      "| Setting group | Frozen source error (range) | RAID v7 minus k-NN (range) |", "|---|---|---|",
      f"| 9 other settings (3 tasks) | {min(s['frozen'] for s in others):.2f} to {max(s['frozen'] for s in others):.2f} | {min(s['gap'] for s in others):+.3f} to {max(s['gap'] for s in others):+.3f} |",
      f"| 3 wine-bottle settings | {min(s['frozen'] for s in settings if s['task'] == ORDER[0]):.2f} to {max(s['frozen'] for s in settings if s['task'] == ORDER[0]):.2f} | {min(s['gap'] for s in settings if s['task'] == ORDER[0]):+.3f} to {max(s['gap'] for s in settings if s['task'] == ORDER[0]):+.3f} |",
      "", "Within the 9 other settings, a larger source error does not make RAID v7 worse than k-NN. The pattern looks like a threshold, not a smooth trend, and it rests on one task. One task cannot establish a rule.", "",
      "## Finding 4: a few target updates repair it", "",
      "| Task | Head-only, 25 updates | LoRA, 25 updates | RAID+LoRA v7, 25 updates |", "|---|---|---|---|"] + \
     [f"| {L(t)} | {per_task[t]['head_only_25']:.3f} | {per_task[t]['lora_mlp_25']:.3f} | {per_task[t]['raid_lora_v7_25']:.3f} |" for t in ORDER] + \
     ["", "After 25 updates, RAID+LoRA v7 is the best of these methods on every task, and its largest gain over LoRA is on the wine-bottle task. Training on five support demos quickly corrects the direct path, and the retrieved candidates then help.", "",
      "## What this means", "",
      "- The zero-training RAID heads depend on the source-trained controller. When that controller is very wrong for a new task, the learned gate does not fall back to retrieval. k-NN has no such dependence, so it is more robust without training.",
      "- The cause of the poor source controller on this task is not tested. One hypothesis: the source set contains 'put the wine bottle on the rack', so the controller may map wine-bottle images to rack motions. Testing this needs the task data, which this analysis does not open.",
      "- Possible design change for future work: make the gate depend on a retrieval-confidence or disagreement signal (for example the distance between the direct prediction and the retrieved actions), so that it can fall back to retrieval. Design it on development tasks only and test it once on new held-out tasks with a new frozen plan. The current final tasks are used.",
      "- To measure the gate directly, a new diagnostic run must log gate values, attention weights and retrieval distances. That run reads the final tasks again, with no tuning. It needs your approval (about $1 to $2 on Runpod).", "",
      "## Limits", "",
      "Post hoc; 4 tasks only (one failure case); errors are correlated within task and seed; the frozen v6 controller is a proxy for the v7 direct path (same source data, different head). Development tasks were used for design.", ""]
(P / "POSTHOC_WINE_BOTTLE.md").write_text("\n".join(md))

# ---- charts (inline SVG, CSS tokens for light/dark; shapes + legend as secondary encoding; native hover titles)
def marker(shape, x, y, cls, tip):
    t = f"<title>{html.escape(tip)}</title>"
    if shape == "circle":
        return f"<circle class='{cls} m' cx='{x:.1f}' cy='{y:.1f}' r='4'>{t}</circle>"
    if shape == "square":
        return f"<rect class='{cls} m' x='{x-4:.1f}' y='{y-4:.1f}' width='8' height='8' rx='1'>{t}</rect>"
    if shape == "triangle":
        return f"<polygon class='{cls} m' points='{x:.1f},{y-5:.1f} {x+5:.1f},{y+4:.1f} {x-5:.1f},{y+4:.1f}'>{t}</polygon>"
    return f"<polygon class='{cls} m' points='{x:.1f},{y-5:.1f} {x+5:.1f},{y:.1f} {x:.1f},{y+5:.1f} {x-5:.1f},{y:.1f}'>{t}</polygon>"


def scatter(xkey, xlabel, title):
    W, H, Lm, R, T, B = 330, 300, 46, 10, 24, 44
    lo, hi = 0.0, 2.2
    X = lambda v: Lm + (min(v, hi) - lo) / (hi - lo) * (W - Lm - R)
    Y = lambda v: T + (hi - min(v, hi)) / (hi - lo) * (H - T - B)
    s = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='{html.escape(title)}'><text x='{Lm}' y='14' class='tt'>{html.escape(title)}</text>"]
    for g in (0.5, 1.0, 1.5, 2.0):
        s.append(f"<line class='grid' x1='{Lm}' x2='{W-R}' y1='{Y(g):.1f}' y2='{Y(g):.1f}'/><text class='ax' x='{Lm-6}' y='{Y(g)+4:.1f}' text-anchor='end'>{g:.1f}</text>"
                 f"<text class='ax' x='{X(g):.1f}' y='{H-B+16}' text-anchor='middle'>{g:.1f}</text>")
    s.append(f"<line class='ref' x1='{X(0):.1f}' y1='{Y(0):.1f}' x2='{X(hi):.1f}' y2='{Y(hi):.1f}'/>")
    s.append(f"<text class='ax' x='{(Lm+W-R)/2}' y='{H-8}' text-anchor='middle'>{html.escape(xlabel)}</text>")
    s.append(f"<text class='ax' transform='rotate(-90)' x='{-(T+H-B)/2}' y='12' text-anchor='middle'>RAID v7 error</text>")
    for i, t in enumerate(ORDER):
        for r in (r for r in demos if r["task"] == t):
            s.append(marker(SHAPE[i], X(r[xkey]), Y(r["raid_bank_v7"]), f"s{i+1}",
                            f"{L(t)}, seed {r['seed']}, demo {r['demo']}: {xlabel} {r[xkey]:.3f}, RAID v7 {r['raid_bank_v7']:.3f}"))
    return "".join(s) + "</svg>"


def setting_chart():
    W, H, Lm, R, T, B = 660, 260, 56, 16, 16, 44
    X = lambda v: Lm + (v - 0.5) / (1.6 - 0.5) * (W - Lm - R)
    Y = lambda v: T + (0.4 - v) / (0.5) * (H - T - B)
    s = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='RAID v7 minus k-NN against frozen source error, per setting'>"]
    for g in (-0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
        s.append(f"<line class='{'zero' if g == 0 else 'grid'}' x1='{Lm}' x2='{W-R}' y1='{Y(g):.1f}' y2='{Y(g):.1f}'/><text class='ax' x='{Lm-6}' y='{Y(g)+4:.1f}' text-anchor='end'>{g:+.1f}</text>")
    for g in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        s.append(f"<text class='ax' x='{X(g):.1f}' y='{H-B+16}' text-anchor='middle'>{g:.1f}</text>")
    s.append(f"<text class='ax' x='{(Lm+W-R)/2}' y='{H-8}' text-anchor='middle'>Frozen source controller error (per setting)</text>"
             f"<text class='ax' x='{Lm+4}' y='{Y(0)-6:.1f}'>above 0: RAID v7 worse than k-NN</text>")
    for i, t in enumerate(ORDER):
        for p in (p for p in settings if p["task"] == t):
            s.append(marker(SHAPE[i], X(p["frozen"]), Y(p["gap"]), f"s{i+1}", f"{L(t)}, seed {p['seed']}: frozen {p['frozen']:.3f}, RAID v7 minus k-NN {p['gap']:+.3f}"))
    return "".join(s) + "</svg>"


SYM = {"circle": "&#9679;", "square": "&#9632;", "triangle": "&#9650;", "diamond": "&#9670;"}
legend = "<p class='lg'>" + " &nbsp; ".join(f"<span class='s{i+1}t'>{SYM[SHAPE[i]]}</span> {html.escape(L(t))}" for i, t in enumerate(ORDER)) + "</p>"


def md_html(lines):
    out, rows = [], []
    for line in lines + [""]:
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")]); continue
        if rows:
            out.append("<div class=t><table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{html.escape(c)}</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>" for i, r in enumerate(rows)) + "</table></div>"); rows = []
        if line.startswith("<"): out.append(line)
        elif line.startswith("## "): out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "): out.append(f"<p>&bull; {html.escape(line[2:])}</p>")
        elif line: out.append(f"<p>{html.escape(line)}</p>")
    return "".join(out)


i2 = md.index("## Finding 2: on this task, RAID v7 errors follow the source controller, not retrieval")
i3 = md.index("## Finding 3: the failure appears only when the source controller is far outside its usual range")
charts2 = "<div class=pair>" + scatter("source_frozen", "Frozen source error", "RAID v7 vs frozen source (per demo)") + \
          scatter("support_knn_average", "k-NN error", "RAID v7 vs k-NN (per demo)") + "</div>" + legend + \
          "<p class=note>Each mark is one query demo; the diagonal is equal error. Hover a mark for its values.</p>"
body = md[:i2 + 2] + [charts2] + md[i2 + 2:i3 + 2] + [setting_chart() + legend] + md[i3 + 2:]
css = """:root{--bg:#fcfcfb;--fg:#1a1a19;--muted:#5f5e5a;--line:#d3d1c7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3d3d3a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
:root[data-theme=dark]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3d3d3a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}
body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:auto;padding:16px;background:var(--bg);color:var(--fg)}
.t{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:8px 0 18px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:13px}
svg{width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:640px){.pair{grid-template-columns:1fr}}
.grid{stroke:var(--line);stroke-width:1}.zero{stroke:var(--muted);stroke-width:1.5}.ref{stroke:var(--muted);stroke-dasharray:4 3}
.ax{fill:var(--muted);font-size:11px}.tt{fill:var(--fg);font-size:12px;font-weight:600}.note,.lg{font-size:13px;color:var(--muted)}
.m{stroke:var(--bg);stroke-width:1.5;fill-opacity:.85}.m:hover{fill-opacity:1;stroke:var(--fg)}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}.s1t{color:var(--s1)}.s2t{color:var(--s2)}.s3t{color:var(--s3)}.s4t{color:var(--s4)}"""
(P / "posthoc_wine_bottle.html").write_text("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                                            f"<title>Wine-bottle post hoc</title><style>{css}</style><body>" + md_html(body) + "</body></html>")
json.dump({"post_hoc": True, "per_task": per_task, "correlations": corr, "settings": settings,
           "not_available": ["gate values", "attention weights", "retrieval distances"]},
          open(P / "posthoc_analysis.json", "w"), indent=2)
print("written")
