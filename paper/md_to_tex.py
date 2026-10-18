"""Regenerate paper/raid_manuscript.tex from paper/raid_manuscript.md (the synchronized source).

Usage (repository root):  python3 paper/md_to_tex.py
Needs pandoc on PATH. The output is a standalone pdflatex document with the numbered bibliography embedded,
as in the earlier synchronized TeX. Citations [n](url) become \\cite{refn}; HTML source comments become LaTeX
comments; the three formula blocks become display math; figures use the PDFs in paper/figures/.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parent
MD, TEX = PAPER / "raid_manuscript.md", PAPER / "raid_manuscript.tex"

DISPLAY = {
    "d = 0.5 (1 − cos(": r"""\[
d = 0.5\,\bigl(1-\cos(\varphi_{\mathrm{ref}}(o_t),\varphi_{\mathrm{ref}}(o'_t))\bigr)
  + 0.5\,\bigl(1-\cos(\varphi_{\mathrm{ref}}(o_{t+H}),\varphi_{\mathrm{ref}}(o'_{t+H}))\bigr)
  + 0.1\cdot\operatorname{mean}\bigl((q_t-q'_t)^2\bigr),
\]""",
    "direct = f_θ(s),": r"""\begin{align*}
\mathrm{direct} &= f_\theta(s),\\
\alpha_i &= \operatorname{softmax}_i\!\Bigl((W_q s)\cdot W_k\bigl[\hat\varphi_{\mathrm{ref}}(o^i_t),\hat\varphi_{\mathrm{ref}}(o^i_{t+H}),\mathrm{sim}_i\bigr]\big/\sqrt{128}\Bigr)\ \text{over valid candidates},\\
\mathrm{prior} &= \textstyle\sum_i \alpha_i a^i,\\
g &= \sigma(W_g s)\in(0,1)^{56},\\
\hat a &= g\odot\mathrm{direct} + (1-g)\odot\mathrm{prior},
\end{align*}""",
    "L_sel = KL(": r"""\[
\mathcal{L}_{\mathrm{sel}} = \mathrm{KL}(w^\ast\,\|\,\alpha),\qquad w^\ast_i\propto\exp\bigl(-\mathrm{mse}(a^i,a)\bigr)\ \text{restricted to valid candidates},
\]""",
}

UNICODE = [
    ("φ̂", r"\ensuremath{\hat{\varphi}}"), ("â", r"\ensuremath{\hat{a}}"), ("−", r"\ensuremath{-}"),
    ("φ", r"\ensuremath{\varphi}"), ("π", r"\ensuremath{\pi}"), ("α", r"\ensuremath{\alpha}"),
    ("θ", r"\ensuremath{\theta}"), ("σ", r"\ensuremath{\sigma}"), ("Σ", r"\ensuremath{\Sigma}"),
    ("∈", r"\ensuremath{\in}"), ("ℝ", r"\ensuremath{\mathbb{R}}"), ("√", r"\ensuremath{\surd}"),
    ("⊙", r"\ensuremath{\odot}"), ("∝", r"\ensuremath{\propto}"), ("‖", r"\ensuremath{\|}"),
    ("′", r"\ensuremath{'}"), ("×", r"\ensuremath{\times}"), ("·", r"\ensuremath{\cdot}"),
    ("§", r"\S{}"), ("…", r"\ldots{}"),
]
SUB = {"ₜ": "t", "ₕ": "H", "₊": "+", "ᵢ": "i", "₇": "7", "₈": "8"}
SUP = {"⁻": "-", "²": "2", "⁴": "4", "⁵": "5", "⁶": "6", "¹": "1", "ⁱ": "i"}


def split_manuscript(md: str):
    title = md.splitlines()[0].lstrip("# ").strip()
    abstract = re.search(r"## Abstract\n\n(.*?)\n\nKeywords: (.*?)\n", md, re.S)
    body = md[abstract.end():]
    main, rest = body.split("\n## References\n", 1)
    refs, appendix = rest.split("\n## A. ", 1)
    return title, abstract.group(1).strip(), abstract.group(2).strip(), main, refs, "## A. " + appendix


def to_markdown_for_pandoc(text: str) -> str:
    # source comments -> raw LaTeX comments
    text = re.sub(r"<!--\s*(.*?)\s*-->", lambda m: "```{=latex}\n% " + m.group(1).replace("\n", " ") + "\n```", text,
                  flags=re.S)
    # formula paragraphs -> display math
    paras = text.split("\n\n")
    for i, para in enumerate(paras):
        for start, latex in DISPLAY.items():
            if para.startswith(start):
                paras[i] = "```{=latex}\n" + latex + "\n```"
    text = "\n\n".join(paras)
    # figures: image + "Figure N. caption" paragraph -> one captioned figure
    text = re.sub(r"!\[Figure \d+\]\((figures/[^)]+)\.png\)\n\nFigure \d+\. ([^\n]+)",
                  lambda m: "```{=latex}\n\\begin{figure}[t]\\centering\n\\includegraphics[width=\\linewidth]{"
                            + m.group(1) + ".pdf}\n\\caption{" + escape(m.group(2)) + "}\n\\end{figure}\n```", text)
    # citations
    text = re.sub(r"\[(\d+)\]\((?:https?://[^)]+)\)", r"`\\cite{ref\1}`{=latex}", text)

    def group(m):
        ids = []
        for part in re.split(r"\s*,\s*", m.group(1)):
            if "–" in part:
                a, b = map(int, part.split("–"))
                ids += range(a, b + 1)
            else:
                ids.append(int(part))
        return "`\\cite{" + ",".join(f"ref{i}" for i in ids) + "}`{=latex}"
    text = re.sub(r"\[(\d+(?:\s*[–,]\s*\d+)*)\](?![(\-\d])", group, text)
    # headings: "## 1. Intro" -> section, "### 1.1 X" -> subsection, unnumbered extras, appendix letters
    text = re.sub(r"^## (?:\d+\. )(.*)$", r"# \1", text, flags=re.M)
    text = re.sub(r"^### (?:\d+\.\d+ )(.*)$", r"## \1", text, flags=re.M)
    text = re.sub(r"^## ([A-C])\. (.*)$", r"# \2", text, flags=re.M)
    text = re.sub(r"^## (Acknowledgments and Declarations|Reproducibility and Artifact Availability)$", r"# \1 {-}",
                  text, flags=re.M)
    return text


def escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            .replace("_", r"\_").replace("$", r"\$"))


def pandoc(md: str) -> str:
    out = subprocess.run(["pandoc", "-f", "markdown-auto_identifiers+raw_attribute", "-t", "latex", "--wrap=none",
                          "--columns=110"], input=md, capture_output=True, text=True, check=True).stdout
    return out


def unicode_to_latex(tex: str) -> str:
    tex = re.sub("[" + "".join(SUB) + "]+", lambda m: r"\textsubscript{" + "".join(SUB[c] for c in m.group(0)) + "}", tex)
    tex = re.sub("[" + "".join(SUP) + "]+", lambda m: r"\textsuperscript{" + "".join(SUP[c] for c in m.group(0)) + "}", tex)
    for a, b in UNICODE:
        tex = tex.replace(a, b)
    tex = re.sub(r"\\cite\{([^}]*)\},\\cite\{", r"\\cite{\1,", tex)
    return tex


def bibliography(refs: str) -> str:
    items = []
    for n, body in re.findall(r"^\[(\d+)\] (.*)$", refs, flags=re.M):
        parts, pos = [], 0
        for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", body):
            parts.append(escape(body[pos:m.start()]))
            parts.append(r"\href{" + m.group(2) + "}{" + escape(m.group(1)) + "}")
            pos = m.end()
        parts.append(escape(body[pos:]))
        items.append(r"\bibitem{ref" + n + "}" + "".join(parts))
    return "\\begin{thebibliography}{99}\n\n" + "\n\n".join(items) + "\n\n\\end{thebibliography}"


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,amsmath,amssymb}
\usepackage{booktabs,tabularx,array,longtable,calc,enumitem,xcolor,xurl,graphicx}
\usepackage[colorlinks=true,linkcolor=black,citecolor=blue!55!black,urlcolor=blue!55!black]{hyperref}
\setlength{\parindent}{1em}
\setlength{\parskip}{0.2em}
\setlength{\emergencystretch}{2em}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.25em,topsep=0.4em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\renewcommand{\arraystretch}{1.1}
% Generated by paper/md_to_tex.py from paper/raid_manuscript.md; edit the Markdown, then regenerate.
"""


def main():
    md = MD.read_text()
    title, abstract, keywords, main_md, refs, appendix_md = split_manuscript(md)
    body = unicode_to_latex(pandoc(to_markdown_for_pandoc(main_md)))
    appendix = unicode_to_latex(pandoc(to_markdown_for_pandoc(appendix_md)))
    abstract_tex = unicode_to_latex(pandoc(abstract))
    # tables: smaller font
    body = re.sub(r"(\\begin\{longtable\})", r"{\\small\n\1", body).replace(r"\end{longtable}", "\\end{longtable}\n}")
    appendix = re.sub(r"(\\begin\{longtable\})", r"{\\small\n\1", appendix).replace(r"\end{longtable}", "\\end{longtable}\n}")
    tex = (PREAMBLE + "\n\\hypersetup{pdftitle={" + escape(title) + "}}\n\n\\title{\\textbf{" + escape(title) + "}}\n\n"
           "\\author{Constantin Ertel \\and Phillip Kerger\\\\ University of California, Berkeley}\n\n"
           "\\date{Protocol v6, frozen final-analysis amendment 1b42ba59 \\textbar\\ 18 October 2026}\n\n"
           "\\begin{document}\n\n\\maketitle\n\n\\begin{abstract}\n\n" + abstract_tex + "\n\n\\noindent Keywords: "
           + escape(keywords) + "\n\n\\noindent\\textit{Evidence boundary:} \\texttt{held\\_out\\_claims\\_eligible = false}. "
           "The final tasks were never used for design, but no claim is made that LIBERO is unseen by the pretrained models.\n\n"
           "\\end{abstract}\n\n" + body + "\n\n" + unicode_to_latex(bibliography(refs)) + "\n\n\\appendix\n\n" + appendix
           + "\n\n\\end{document}\n")
    leftover = sorted({c for c in tex if ord(c) > 127 and c not in "—–"})
    if leftover:
        raise SystemExit(f"unmapped characters: {leftover}")
    TEX.write_text(tex)
    print(f"wrote {TEX} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
