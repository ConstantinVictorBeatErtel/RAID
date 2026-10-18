"""Write PILOT_RESULTS.md and pilot_results.html from the verified collection (no network)."""
import json, glob, base64, html
from pathlib import Path
P = Path(__file__).resolve().parent; c = P / "collection"
s = json.loads((c / "pilot_summary.json").read_text()); a = s["analysis"]; bc = s["by_condition"]
fin = json.loads((P / "final_status.json").read_text())
conds = ["source_frozen", "raid_bank_mlp", "lora_mlp", "raid_lora_mlp"]
name = lambda t: "drawer" if t.startswith("open") else "cream cheese"
eps = []
for f in sorted(glob.glob(str(c / "pilot/*/rollout/episodes/*_s42.json"))):
    e = json.loads(Path(f).read_text()); e["_dir"] = Path(f).parent; eps.append(e)
rec = {}
for f in glob.glob(str(c / "pilot/*/recorded/*.json")):
    r = json.loads(Path(f).read_text()); rec[name(r["task"])] = {k: v["chunk_mse_macro"] for k, v in r["results"].items()}
cost = 64.3522442697 - float(fin["balance_usd"])
rows_success = [[k, f"{bc[k]['successes']} / {bc[k]['episodes']}", str(bc[k]["plans"]), f"{bc[k]['generation_seconds']/3600:.1f}"] for k in conds]
rows_cases = [[f"{name(e['task'])}, start {e['query_demo']}"] + [("success (step %d)" % e["conditions"][k]["steps"]) if e["conditions"][k]["success"] else "timeout" for k in conds] for e in eps]
rows_mse = [[k, f"{rec['drawer'].get(k, float('nan')):.3f}", f"{rec['cream cheese'].get(k, float('nan')):.3f}"] for k in conds + ["support_knn_average"]]
S = a["macro"]["S"]; pw = a["paired_within_task"]
facts = [f"Run: pod kyeyj9c2t6i4tv (A100-SXM4-80GB), 16 Oct 05:56 UTC to 17 Oct about 00:35 UTC. "
         f"{fin['verified_result']['files']} files collected and SHA-256 verified; pod and volume deleted; Runpod empty.",
         f"Session cost (balance difference, incl. safety checks): about ${cost:.2f}. Balance after: ${float(fin['balance_usd']):.2f}.",
         "Setup: VPP2 v5 judged prompt (2 prompts + judge), 1 video per plan, 2 chunks (16 steps) per video, frames 9 and 18, "
         "300 steps max, 5 support demos, seed 42, one shared first plan per case. User waiver of the 15-case human panel gate recorded."]
contrasts = [f"Macro success: frozen {S['S_00']}, RAID {S['S_10']}, LoRA {S['S_01']}, RAID+LoRA {S['S_11']}.",
             f"Hybrid over RAID +{a['macro']['hybrid_over_raid']}; hybrid over LoRA +{a['macro']['hybrid_over_lora']}; interaction +{a['macro']['interaction']}.",
             f"Paired hybrid minus LoRA: drawer {pw['open_the_top_drawer_and_put_the_bowl_inside']['paired_diffs_lora']}, cream cheese {pw['put_the_cream_cheese_in_the_bowl']['paired_diffs_lora']}.",
             "3 successes in 40 episodes: no difference is statistically meaningful; the direction favours RAID+LoRA.",
             "Offline, LoRA alone has the lowest action error; in closed loop RAID+LoRA has the most successes. Both are small samples."]
limits = "Development tasks only (final tasks sealed; held_out_claims_eligible = false). One seed, 5 starts per task. VPP2 timing evidence is weak (v3 15-case calibration). The human panel gate was waived by the user after seeing results."
def mdtable(head, rows): return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
md = ("# VPP2 v5 pilot: results (development, 40 episodes)\n\n" + "\n".join("- " + f for f in facts) +
      "\n\n## Task success (registered primary outcome)\n\n" + mdtable(["Condition", "Successes", "Plans", "VPP2 hours"], rows_success) +
      "\n\n" + mdtable(["Case"] + conds, rows_cases) + "\n\n## Registered contrasts (descriptive)\n\n" + "\n".join("- " + x for x in contrasts) +
      "\n\n## Offline recorded-transition error (chunk MSE, lower is better)\n\n" + mdtable(["Condition", "drawer", "cream cheese"], rows_mse) +
      "\n\n## Limits\n\n" + limits + "\n")
(P / "PILOT_RESULTS.md").write_text(md)
def htable(head, rows): return "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td{' class=ok' if 'success' in x else ''}>{html.escape(x)}</td>" for x in r) + "</tr>" for r in rows) + "</table>"
videos = ""
for e in eps:
    for k in conds:
        if e["conditions"][k]["success"]:
            v = e["_dir"] / Path(e["conditions"][k]["video"]).name
            if v.is_file():
                videos += f"<section><h3>{name(e['task'])}, start {e['query_demo']}: {k}</h3><video src='data:video/mp4;base64,{base64.b64encode(v.read_bytes()).decode()}' controls loop muted playsinline></video></section>"
page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>VPP2 pilot results</title>
<style>:root{{--bg:#fff;--fg:#111;--ok:#0a7a2f;--line:#ccc}}@media(prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--ok:#5fd38a;--line:#333}}}}
body{{font:15px/1.5 system-ui;max-width:900px;margin:auto;padding:16px;background:var(--bg);color:var(--fg)}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:14px}}.ok{{color:var(--ok);font-weight:500}}video{{width:100%;background:#000;border-radius:6px}}</style>
<h1>VPP2 v5 pilot: 40 development episodes</h1><ul>{''.join(f'<li>{html.escape(f)}</li>' for f in facts)}</ul>
<h2>Task success</h2>{htable(["Condition", "Successes", "Plans", "VPP2 hours"], rows_success)}{htable(["Case"] + conds, rows_cases)}
<h2>Registered contrasts (descriptive)</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in contrasts)}</ul>
<h2>Offline action error (chunk MSE)</h2>{htable(["Condition", "drawer", "cream cheese"], rows_mse)}
<h2>Limits</h2><p>{html.escape(limits)}</p><h2>Successful episodes (left: VPP2 plan, right: robot)</h2>{videos}"""
(P / "pilot_results.html").write_text(page)
print("written", len(eps), "cases")
