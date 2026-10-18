# RAID paper: current version and provenance

Updated 18 October 2026. The current paper is a **results paper**: *Retrieval Accelerates but Does Not Replace Few-Shot Adaptation of Visual Inverse Dynamics: An Action-Error–Compute Benchmark on LIBERO-Goal* (Constantin Ertel, Phillip Kerger; University of California, Berkeley).

Its niche is few-shot inverse-dynamics adaptation on held-out LIBERO-Goal tasks: action error (chunk MSE on normalized actions, R²) against adaptation compute (GPU-seconds, FLOPs per update). The primary evidence is one frozen offline run on the two sealed final tasks × seeds 42/43/44 (analysis plan frozen before the data were read). Task success from the 40-episode VPP2 pilot is secondary evidence. `held_out_claims_eligible = false`.

## Current writing

- [Readable manuscript](raid_manuscript.md): the source of truth for local edits. Every table carries an HTML comment naming its source file under `artifacts/raid_hybrid_libero_goal_v6/`.
- [Standalone LaTeX manuscript](raid_manuscript.tex): generated from the Markdown with `python3 paper/md_to_tex.py` (needs pandoc). Do not edit the TeX by hand; edit the Markdown and regenerate. It embeds the numbered bibliography and uses the PDF figures. It was not compiled locally (no TeX installation on this Mac).
- [Figures](figures/): `python3 paper/figures/make_figures.py` rebuilds all four figures from collected JSON only (no models, no datasets).
- [Google Docs manuscript](https://docs.google.com/document/d/11QDjW49qwxiZFJGYxMB5_dELf5keka6Qeh3ijXpW10s): owned by constantin@encord.com, shared with constantinertel@berkeley.edu (writer). User edits there are authoritative; fetch it before changing prose. Sync state: see below.
- [Shared bibliography](references.bib) (citation-key library; new entries vpp2, disc, octo, vqbet, diffusionpolicy were checked against primary sources on 18 October 2026), [literature notes](literature_notes.md), [coverage audit](../docs/RELATED_WORK.md).
- Evidence: [STATUS.md](../STATUS.md) (newest entry), [RESULTS.md](../RESULTS.md) (top section).

## Sync state (18 October 2026)

- Before the rewrite, the Google Doc was read through the Drive connector and compared with the local Markdown. Its content matched the 11 October synchronization (differences in format only), so no user edits had to be merged.
- The results rewrite is in `raid_manuscript.md`, `raid_manuscript.tex` and the Google Doc. The Doc body was replaced in place on 18 October 2026 (same link) with the Google Docs connector and verified block by block against the Markdown (213/213 paragraphs and tables equal); see `revisions/results_v7_20261018/verification.json`. The four figure images are not in the Doc yet: placeholder lines name the PNG files in `figures/`, to be inserted by hand (Insert > Image).
- The Doc keeps no HTML source comments and no local status note; the Markdown is the place for table provenance. `revisions/results_v7_20261018/build_doc_requests.py` rebuilds the Docs requests from the Markdown.
- References are numbered by first citation. MIRA and MEM were removed from the manuscript (not cited, or not relevant to IDM adaptation); they remain in `references.bib`.

## Historical drafts and evidence

The proposal-stage manuscript of 11 October (protocol v6, no results) is kept in `revisions/results_v7_20261018/raid_manuscript_before.md` and `.tex`. The independent introduction_v1 and related_work_v1 Markdown/TeX files, raid_manuscript_v5_google_docs.md, prior protocol notes, and RAID_Report_vf.pdf are historical snapshots, not current instructions. The May PDF stays unchanged; do not merge its outcomes into the current paper.

Earlier Google Docs structures and local manuscripts are retained under `revisions/hybrid_v6_20261011/` and `revisions/wam_feedback_20261011/`. Google Docs and local copies do not synchronize automatically. Do not invent source identities or author information when adding bibliographic records.
