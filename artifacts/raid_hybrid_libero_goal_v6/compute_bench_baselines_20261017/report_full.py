"""BASELINE_RESULTS.md and baseline_results.html from the verified collection (no network)."""
import json, glob, html, math, random, statistics as st
from pathlib import Path
P = Path(__file__).resolve().parent
units = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(P / "collection/bench/unit_*.json")))]
u0 = units[0]; mean = st.mean
L = {"source_frozen": "Frozen source controller", "raid_bank_mlp": "RAID v6 (ours)", "raid_bank_v7": "RAID v7 (ours)",
     "support_knn_average": "VINN-style k-NN", "retrieve_and_play": "REGENT retrieve-and-play", "regent_blend": "REGENT-style blend",
     "lora_mlp": "LoRA (TAIL/FADA style)", "raid_lora_mlp": "RAID+LoRA v6 (ours)", "raid_lora_v7": "RAID+LoRA v7 (ours)",
     "head_only": "Head-only fine-tuning", "probe_fresh": "Frozen-encoder probe (fresh head)", "scratch_lora": "LoRA, no source training",
     "full_finetune": "Full fine-tuning", "strap_style": "STRAP-style retrieved training data"}
def get(c, upd=None):
    out = {}
    for u in units:
        d = (u["zero_adaptation"][c] if upd is None else next(p for p in u["curves"][c] if p["updates"] == upd))["mse_by_demo"]
        out.update({(u["task"], u["seed"], k): v for k, v in d.items()})
    return out
def paired(a, b):
    keys = sorted(a); diffs = [b[k] - a[k] for k in keys]; rng = random.Random(0)
    boots = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(4000))
    cl = {}
    for k, d in zip(keys, diffs): cl.setdefault(k[:2], []).append(d)
    return mean(diffs), boots[100], boots[3899], sum(mean(v) < 0 for v in cl.values())
zero = [(k, mean(u["zero_adaptation"][k]["mse_macro"] for u in units), mean(u["zero_adaptation"][k]["r2"] for u in units))
        for k in u0["zero_adaptation"]]
zero.sort(key=lambda r: r[1])
curves = {k: [(mean(u["curves"][k][i]["adaptation_seconds"] for u in units), mean(u["curves"][k][i]["mse_macro"] for u in units),
               mean(u["curves"][k][i]["r2"] for u in units), u0["curves"][k][i]["updates"]) for i in range(len(u0["curves"][k]))]
          for k in u0["curves"]}
order = sorted(curves, key=lambda k: curves[k][-1][1])
dc = {k: (mean(u["decision_cost"][k]["seconds_per_decision"] for u in units) * 1000,
          mean(u["decision_cost"][k]["flops_per_decision"] for u in units) / 1e9) for k in u0["decision_cost"] if u0["decision_cost"][k]}
ad = {k: (mean(u["adaptation"][k]["adaptation_seconds"] for u in units), mean(u["adaptation"][k]["flops_per_update"] or 0 for u in units) / 1e12,
          u0["adaptation"][k]["trainable_parameters"]) for k in u0["adaptation"]}
comps = [("RAID v7 vs VINN-style k-NN (no training)", get("support_knn_average"), get("raid_bank_v7")),
         ("RAID v7 vs REGENT-style blend (no training)", get("regent_blend"), get("raid_bank_v7")),
         ("RAID v7 vs REGENT retrieve-and-play (no training)", get("retrieve_and_play"), get("raid_bank_v7")),
         ("RAID v7 vs RAID v6 (no training)", get("raid_bank_mlp"), get("raid_bank_v7"))]
for upd in (25, 500):
    for b in ("head_only", "lora_mlp", "full_finetune", "strap_style"):
        comps.append((f"RAID+LoRA v7 vs {L[b]}, {upd} updates", get(b, upd), get("raid_lora_v7", upd)))
pairs = [(n,) + paired(a, b) for n, a, b in comps]
upds = [p[3] for p in curves["lora_mlp"]]
md = ["# Few-shot inverse-dynamics benchmark: our methods vs re-implemented baselines", "",
      "Development tasks (2) x registered support seeds 42/43/44, five demos per task; 20 query demos per task with real recorded future frames.",
      "One pod (A100-SXM4-80GB, 16 CPU threads) for all methods. Metrics: chunk MSE on normalized actions (lower is better) and R2 (higher is better).",
      "Baselines are faithful re-implementations of each method family in this setting, not reproductions of the papers.", "",
      "## Without any target training", "", "| Rank | Method | MSE | R2 |", "|---|---|---|---|"] + \
     [f"| {i+1} | {L[k]} | {m:.3f} | {r:.3f} |" for i, (k, m, r) in enumerate(zero)] + \
     ["", "## With target training: error after N updates (same schedule for all)", "",
      "| Method | " + " | ".join(str(u) for u in upds) + " | GPU s (500 updates) | TFLOP / update | Trainable params |",
      "|---|" + "---|" * (len(upds) + 3)] + \
     [f"| {L[k]} | " + " | ".join(f"{p[1]:.3f}" for p in curves[k]) + f" | {ad[k][0]:.0f} | {ad[k][1]:.2f} | {ad[k][2]:,} |" for k in order] + \
     ["", "## Paired differences (second method minus first; negative = ours better), 95% bootstrap CI, settings won of 6", "",
      "| Comparison | Mean | 95% CI | Settings won |", "|---|---|---|---|"] + \
     [f"| {n} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {w}/6 |" for n, m, lo, hi, w in pairs] + \
     ["", "## Cost per control decision (batch 1)", "", "| Method | ms | GFLOP |", "|---|---|---|"] + \
     [f"| {L[k]} | {v[0]:.1f} | {v[1]:.0f} |" for k, v in sorted(dc.items(), key=lambda kv: kv[1][0])] + \
     ["", "## Reading", "",
      "- Without training, RAID v7 has the lowest error of all methods and beats REGENT-style blending and retrieve-and-play clearly; against VINN-style k-NN the gain is small (-0.017, CI touches 0, 3/6 settings).",
      "- With few updates, RAID+LoRA v7 is the best method (25 updates: better than every baseline in 6/6 settings).",
      "- With full training, full fine-tuning is best (0.260); RAID+LoRA v7 is equal to head-only and slightly worse than LoRA.",
      "- Per GPU-second, head-only training on cached frozen features is extremely cheap (about 1 s for 500 updates) and reaches 0.288; LoRA-based methods need about 200 s because every update runs the image encoder.",
      "", "## Limits", "",
      "Offline recorded-transition error on development tasks only (final tasks sealed). Three registered seeds. Bootstrap intervals treat demos as independent (optimistic). v7 and these baselines were chosen after seeing earlier development results, so this is exploratory until a frozen final-task run.", ""]
(P / "BASELINE_RESULTS.md").write_text("\n".join(md))
W, H, Lm, R, T, B = 680, 380, 56, 16, 16, 56
xs = lambda s: math.log10(max(s, 0.01))
xmin, xmax = xs(0.01), xs(max(c[-1][0] for c in curves.values()) * 1.3)
ymin, ymax = 0.24, 0.80
X = lambda s: Lm + (xs(s) - xmin) / (xmax - xmin) * (W - Lm - R)
Y = lambda e: T + (ymax - min(e, ymax)) / (ymax - ymin) * (H - T - B)
col = {"raid_lora_v7": "#0F6E56", "raid_lora_mlp": "#5DCAA5", "lora_mlp": "#534AB7", "full_finetune": "#D85A30",
       "head_only": "#BA7517", "strap_style": "#993556", "scratch_lora": "#888780", "probe_fresh": "#B4B2A9"}
svg = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img'><title>Action error vs adaptation GPU-seconds</title>"]
for e in (0.3, 0.4, 0.5, 0.6, 0.7):
    svg.append(f"<line x1='{Lm}' x2='{W-R}' y1='{Y(e):.1f}' y2='{Y(e):.1f}' stroke='currentColor' stroke-opacity='.12'/><text x='{Lm-6}' y='{Y(e)+4:.1f}' text-anchor='end' font-size='11' fill='currentColor'>{e:.1f}</text>")
for s in (0.01, 0.1, 1, 10, 100):
    svg.append(f"<text x='{X(s):.1f}' y='{H-B+16}' text-anchor='middle' font-size='11' fill='currentColor'>{s:g}</text>")
svg.append(f"<text x='{(Lm+W-R)/2}' y='{H-20}' text-anchor='middle' font-size='12' fill='currentColor'>Adaptation GPU-seconds per task (log scale; 0 plotted at 0.01)</text>")
for k, dash in (("raid_bank_v7", "1 0"), ("support_knn_average", "5 3")):
    m = dict((z[0], z[1]) for z in zero)[k]
    svg.append(f"<line x1='{Lm}' x2='{W-R}' y1='{Y(m):.1f}' y2='{Y(m):.1f}' stroke='#888780' stroke-dasharray='{dash}'/><text x='{W-R-4}' y='{Y(m)-5:.1f}' text-anchor='end' font-size='11' fill='currentColor'>{html.escape(L[k])} (no training) {m:.3f}</text>")
for k, c in col.items():
    svg.append(f"<polyline points='{' '.join(f'{X(s):.1f},{Y(e):.1f}' for s, e, _, _ in curves[k])}' fill='none' stroke='{c}' stroke-width='{2.5 if k == 'raid_lora_v7' else 1.5}'/>")
svg.append("</svg>")
legend = " &nbsp; ".join(f"<span style='color:{c}'>&#9632;</span> {html.escape(L[k])}" for k, c in col.items())
def md2html(lines):
    out, rows = [], []
    for line in lines + [""]:
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-"}: continue
            rows.append([c.strip() for c in line.strip("|").split("|")]); continue
        if rows:
            out.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{html.escape(c)}</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>" for i, r in enumerate(rows)) + "</table>"); rows = []
        if line.startswith("## "): out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "): out.append(f"<p>&bull; {html.escape(line[2:])}</p>")
        elif line: out.append(f"<p>{html.escape(line)}</p>")
    return "".join(out)
page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>RAID baseline benchmark</title>"
        "<style>:root{--bg:#fff;--fg:#111;--line:#ccc}@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--line:#333}}"
        "body{font:15px/1.5 system-ui;max-width:960px;margin:auto;padding:16px;background:var(--bg);color:var(--fg)}table{border-collapse:collapse;width:100%;margin:8px 0 18px}"
        "td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:13px}</style>"
        + md2html(md[:5]) + "<h2>Error vs adaptation compute</h2>" + "".join(svg) + f"<p style='font-size:13px'>{legend}</p>" + md2html(md[5:]))
(P / "baseline_results.html").write_text(page)
print("written")
