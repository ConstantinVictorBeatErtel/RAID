"""Write BENCH_RESULTS.md and bench_results.html (chart: action error vs adaptation GPU-seconds)."""
import json, glob, html, random, statistics as st
from pathlib import Path
P = Path(__file__).resolve().parent; B = P / "collection/bench"
units = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(B / "unit_*.json")))]
AD = ["lora_mlp", "raid_lora_mlp", "raid_lora_v7"]; ZERO = ["source_frozen", "raid_bank_mlp", "raid_bank_v7", "support_knn_average"]
LABEL = {"source_frozen": "Frozen", "raid_bank_mlp": "RAID v6", "raid_bank_v7": "RAID v7", "support_knn_average": "k-NN average",
         "lora_mlp": "LoRA", "raid_lora_mlp": "RAID+LoRA v6", "raid_lora_v7": "RAID+LoRA v7"}
idx = {u: i for i, u in enumerate(p["updates"] for p in units[0]["curves"]["lora_mlp"])}
def demos(u, cond, upd=None):
    d = (u["zero_adaptation"][cond] if upd is None else u["curves"][cond][idx[upd]])["mse_by_demo"]
    return {(u["task"], u["seed"], k): v for k, v in d.items()}
def paired(a, b):  # mean(b - a) over all demo x setting pairs, 95% bootstrap CI
    keys = sorted(a); diffs = [b[k] - a[k] for k in keys]; rng = random.Random(0)
    boots = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(4000))
    return st.mean(diffs), boots[100], boots[3899], len(diffs)
def collect(cond, upd=None):
    out = {}
    for u in units: out.update(demos(u, cond, upd))
    return out
comparisons = [("RAID v7 vs RAID v6 (no training)", collect("raid_bank_mlp"), collect("raid_bank_v7")),
               ("RAID+LoRA v7 vs LoRA, 25 updates", collect("lora_mlp", 25), collect("raid_lora_v7", 25)),
               ("RAID+LoRA v7 vs LoRA, 100 updates", collect("lora_mlp", 100), collect("raid_lora_v7", 100)),
               ("RAID+LoRA v7 vs LoRA, 500 updates", collect("lora_mlp", 500), collect("raid_lora_v7", 500)),
               ("RAID+LoRA v6 vs LoRA, 500 updates", collect("lora_mlp", 500), collect("raid_lora_mlp", 500)),
               ("RAID+LoRA v7 vs RAID+LoRA v6, 500 updates", collect("raid_lora_mlp", 500), collect("raid_lora_v7", 500))]
pairs = [(n,) + paired(a, b) for n, a, b in comparisons]
mean = lambda xs: st.mean(xs)
curve = {k: [(mean(u["curves"][k][i]["adaptation_seconds"] for u in units), mean(u["curves"][k][i]["mse_macro"] for u in units),
              units[0]["curves"][k][i]["updates"]) for i in range(len(units[0]["curves"][k]))] for k in AD}
zero = {k: mean(u["zero_adaptation"][k]["mse_macro"] for u in units) for k in ZERO}
dc = {k: (mean(u["decision_cost"][k]["seconds_per_decision"] for u in units) * 1000,
          mean(u["decision_cost"][k]["flops_per_decision"] for u in units) / 1e9,
          max(u["decision_cost"][k]["peak_gpu_bytes"] for u in units) / 1e9) for k in units[0]["decision_cost"]}
ad = {k: (mean(u["adaptation"][k]["adaptation_seconds"] for u in units), mean(u["adaptation"][k]["flops_per_update"] for u in units) / 1e12,
          units[0]["adaptation"][k]["trainable_parameters"]) for k in AD}
src = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(B / "source_cost_s*.json")))]
src_s = {k: mean(s[k]["training_seconds"] for s in src) for k in src[0]}
cache_s = json.loads((B / "frame_cache_cost.json").read_text())["seconds"]
bank_s = mean(u["bank_construction_seconds"] for u in units)
budget_rows = [[f"{c[0][2] if False else ''}"] for c in []]
upd_list = [p[2] for p in curve["lora_mlp"]]
table_curve = [[str(upd), f"{curve['lora_mlp'][i][0]:.0f}"] + [f"{curve[k][i][1]:.3f}" for k in AD] for i, upd in enumerate(upd_list)]
lines = ["# Compute-aware benchmark: RAID, LoRA, RAID+LoRA (v6) and RAID+LoRA (v7)",
         "",
         "Offline recorded-transition action error (chunk MSE, lower is better) on the 20 registered query demos of each development task;",
         "registered support seeds 42/43/44; five-shot; one pod, NVIDIA A100 80GB PCIe (the pilot used A100-SXM4-80GB). Mean over 6 settings (2 tasks x 3 seeds).",
         "",
         "## No target training (instant)", "", "| Method | Error |", "|---|---|"] + [f"| {LABEL[k]} | {zero[k]:.3f} |" for k in ZERO] + [
         "", "## Error vs adaptation budget (updates; GPU-seconds, evaluation excluded)", "",
         "| Updates | Seconds | LoRA | RAID+LoRA v6 | RAID+LoRA v7 |", "|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in table_curve] + [
         "", "## Paired differences (second minus first; negative = second is better), 95% bootstrap CI over demo x setting pairs", "",
         "| Comparison | Mean difference | 95% CI | n |", "|---|---|---|---|"] + [f"| {n} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {k} |" for n, m, lo, hi, k in pairs] + [
         "", "## Compute", "",
         f"- Shared one-time costs: reference feature cache {cache_s:.0f} s; source training {src_s['mlp_v6']:.0f} s (v6 head) and {src_s['attention_gate_v7']:.0f} s (v7 head), 2,000 updates each.",
         f"- RAID bank build per task: {bank_s:.3f} s (466-981 entries). No training.", "",
         "| Method | Adaptation s (500 updates) | TFLOP per update | Trainable parameters |", "|---|---|---|---|"] + [
         f"| {LABEL[k]} | {ad[k][0]:.0f} | {ad[k][1]:.2f} | {ad[k][2]:,} |" for k in AD] + [
         "", "| Method | ms per decision | GFLOP per decision | Peak GPU GB |", "|---|---|---|---|"] + [
         f"| {LABEL[k]} | {dc[k][0]:.1f} | {dc[k][1]:.0f} | {dc[k][2]:.2f} |" for k in ["source_frozen", "raid_bank_mlp", "raid_bank_v7", "lora_mlp", "raid_lora_mlp", "raid_lora_v7"]] + [
         "", "RAID+LoRA needs two encodes per decision (LoRA controller features plus frozen retrieval keys), so its decision cost is about twice that of the other methods.",
         "", "## Limits", "", "Offline error on recorded transitions, not task success. Development tasks only; final tasks sealed; held_out_claims_eligible = false. 3 registered seeds. The bootstrap intervals treat demos as independent although they share tasks, seeds and source controllers, so they are optimistic. v7 was designed after seeing v6 pilot results (exploratory).", ""]
(P / "BENCH_RESULTS.md").write_text("\n".join(lines))
# chart
W, H, L, R, T, Bm = 640, 360, 60, 20, 20, 50
xmax = max(c[-1][0] for c in curve.values()) * 1.05; ymin, ymax = 0.25, 0.80
X = lambda s: L + s / xmax * (W - L - R); Y = lambda e: T + (ymax - e) / (ymax - ymin) * (H - T - Bm)
color = {"lora_mlp": "#534AB7", "raid_lora_mlp": "#D85A30", "raid_lora_v7": "#0F6E56"}
svg = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img'><title>Action error vs adaptation seconds</title>"]
for e in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    svg.append(f"<line x1='{L}' x2='{W-R}' y1='{Y(e):.1f}' y2='{Y(e):.1f}' stroke='currentColor' stroke-opacity='.12'/><text x='{L-8}' y='{Y(e)+4:.1f}' text-anchor='end' font-size='11' fill='currentColor'>{e:.1f}</text>")
for s in (0, 50, 100, 150, 200):
    svg.append(f"<text x='{X(s):.1f}' y='{H-Bm+18}' text-anchor='middle' font-size='11' fill='currentColor'>{s}</text>")
svg.append(f"<text x='{(L+W-R)/2}' y='{H-8}' text-anchor='middle' font-size='12' fill='currentColor'>Adaptation GPU-seconds per task</text>")
for k, dash in (("source_frozen", "2 3"), ("raid_bank_mlp", "6 3"), ("raid_bank_v7", "1 0")):
    svg.append(f"<line x1='{L}' x2='{W-R}' y1='{Y(zero[k]):.1f}' y2='{Y(zero[k]):.1f}' stroke='#888780' stroke-dasharray='{dash}'/><text x='{W-R-4}' y='{Y(zero[k])-5:.1f}' text-anchor='end' font-size='11' fill='currentColor'>{LABEL[k]} (no training) {zero[k]:.3f}</text>")
for k in AD:
    pts = " ".join(f"{X(s):.1f},{Y(e):.1f}" for s, e, _ in curve[k])
    svg.append(f"<polyline points='{pts}' fill='none' stroke='{color[k]}' stroke-width='2'/>")
    svg += [f"<circle cx='{X(s):.1f}' cy='{Y(e):.1f}' r='3' fill='{color[k]}'/>" for s, e, _ in curve[k]]
svg.append("</svg>")
legend = " ".join(f"<span style='color:{color[k]}'>&#9632; {LABEL[k]}</span>" for k in AD)
body = "".join(f"<p>{html.escape(x)}</p>" if not x.startswith(("|", "#", "-")) and x else "" for x in [])
def md2html(md):
    out, rows = [], []
    for line in md.split("\n"):
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
page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>RAID compute benchmark</title>"
        "<style>:root{--bg:#fff;--fg:#111;--line:#ccc}@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--line:#333}}"
        "body{font:15px/1.5 system-ui;max-width:900px;margin:auto;padding:16px;background:var(--bg);color:var(--fg)}table{border-collapse:collapse;width:100%;margin:8px 0 18px}"
        "td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:14px}</style>"
        + md2html("\n".join(lines[:4])) + "<h2>Error vs adaptation compute</h2>" + "".join(svg) + f"<p>{legend}</p>" + md2html("\n".join(lines[4:])))
(P / "bench_results.html").write_text(page)
print("\n".join(lines[5:40]))
