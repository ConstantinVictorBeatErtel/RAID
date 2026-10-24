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
d = 0.5\,\bigl(1-\cos(\varphi(o_t),\varphi(o'_t))\bigr)
  + 0.5\,\bigl(1-\cos(\varphi(o_{t+H}),\varphi(o'_{t+H}))\bigr)
  + 0.1\cdot\operatorname{mean}\bigl((q_t-q'_t)^2\bigr),
\]""",
    "direct = f_θ(s), prior": r"""\[
\mathrm{direct} = f_\theta(s),\qquad \mathrm{prior} = \textstyle\sum_i \alpha_i a^i,
\]""",
    "c = [top-1 similarity": r"""\[
c = \bigl[\,\mathrm{sim}_{(1)},\ \overline{\mathrm{sim}},\ \tilde H(\alpha),\ |\mathrm{direct}-\mathrm{prior}|\,\bigr],
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
                  lambda m: "```{=latex}\n\\begin{figure}[tbp]\\centering\n\\includegraphics[width=\\linewidth]{"
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


def finish(tex: str) -> str:
    """Layout fixes: table font by column count, breakable file paths and digests."""
    def table(m):
        rows = [line for line in m.group(0).splitlines() if " & " in line]
        ncol = max((line.count(" & ") + 1 for line in rows), default=1)
        width = max((len(re.sub(r"\\[a-zA-Z]+|[{}]", "", line)) for line in rows), default=0)
        if ncol >= 9:
            size = r"\scriptsize\setlength{\tabcolsep}{3pt}"
        elif ncol >= 6 or width > 90:
            size = r"\footnotesize\setlength{\tabcolsep}{3.5pt}"
        else:
            size = r"\small"
        # short tables: an unbreakable tabular moves to the next page as a whole (longtable mis-breaks next to floats)
        body = m.group(0)
        spec = re.match(r"\\begin\{longtable\}\[\]\{(.*?)\}\n", body, re.S)
        inner = body[spec.end():body.rindex("\\end{longtable}")]
        keep = [line for line in inner.split("\n") if line.strip() not in ("\\endhead", "\\endlastfoot",
                                                                           "\\bottomrule\\noalign{}", "\\endfirsthead")]
        tab = "\\begin{tabular}{" + spec.group(1) + "}\n" + "\n".join(keep).rstrip() + "\n\\bottomrule\\noalign{}\n\\end{tabular}"
        return "{" + size + "\n\\begin{center}\n" + tab + "\n\\end{center}\n}"
    tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", table, tex, flags=re.S)

    def path(m):
        inner = m.group(1)
        if "/" in inner or (len(inner) > 24 and r"\_" in inner):
            return r"\nolinkurl{" + inner.replace(r"\_", "_") + "}"
        return m.group(0)
    tex = re.sub(r"\\texttt\{([^{}]*)\}", path, tex)
    # keep a table caption with its first rows: start a new page when less than 12 lines remain
    tex = re.sub(r"^(Table [A-Z]?\d+\. )", r"\\needspace{12\\baselineskip}\n\1", tex, flags=re.M)
    # artifact file names in running text cannot hyphenate; let that section break loosely
    tex = tex.replace("\\section*{Reproducibility and Artifact Availability}",
                      "\\section*{Reproducibility and Artifact Availability}\\sloppy")
    return re.sub(r"\b([0-9a-f]{40,})\b", r"\\nolinkurl{\1}", tex)


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
\usepackage{booktabs,tabularx,array,longtable,calc,enumitem,xcolor,xurl,graphicx,needspace}
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
    body, appendix = finish(body), finish(appendix)
    tex = (PREAMBLE + "\n\\hypersetup{pdftitle={" + escape(title) + "}}\n\n\\title{\\textbf{" + escape(title) + "}}\n\n"
           "\\author{Constantin Ertel \\and Phillip Kerger\\\\ University of California, Berkeley}\n\n"
           "\\date{Preprint draft \\textbar\\ 24 October 2026}\n\n"
           "\\begin{document}\n\n\\maketitle\n\n\\begin{abstract}\n\n" + abstract_tex + "\n\n\\noindent Keywords: "
           + escape(keywords) + "\n\n\\end{abstract}\n\n" + body + "\n\n\\fussy\n\n" + unicode_to_latex(bibliography(refs)) + "\n\n\\appendix\n\n" + appendix
           + "\n\n\\end{document}\n")
    # TeX dash ligatures render in every engine (XeTeX/Tectonic has no T1 glyph for a literal U+2013 in bold)
    tex = tex.replace("—", "---").replace("–", "--")
    leftover = sorted({c for c in tex if ord(c) > 127})
    if leftover:
        raise SystemExit(f"unmapped characters: {leftover}")
    TEX.write_text(tex)
    print(f"wrote {TEX} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
