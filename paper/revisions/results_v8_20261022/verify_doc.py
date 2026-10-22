"""Compare a Google Doc readback (read_doc JSON) with the paragraphs/tables built from the manuscript.

Usage: python3 verify_doc.py MANUSCRIPT.md DOC.json [IMAGE_BASE_URL]
Prints body end index, number of matching leading blocks, and the first mismatch (if any).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_doc_requests as b  # noqa: E402


def doc_blocks(path):
    d = json.load(open(path))["content"]
    body = d["tabs"][0]["documentTab"]["body"]["content"]
    out = []
    for e in body:
        if "paragraph" in e:
            t = ""
            for r in e["paragraph"]["elements"]:
                if "textRun" in r:
                    t += r["textRun"]["content"]
                elif "inlineObjectElement" in r:
                    t += "￼"
            out.append(("p", t.rstrip("\n")))
        elif "table" in e:
            rows = []
            for row in e["table"]["tableRows"]:
                rows.append(["".join("".join(r.get("textRun", {}).get("content", "") for r in ce["paragraph"]["elements"])
                                     for ce in c["content"] if "paragraph" in ce).rstrip("\n") for c in row["tableCells"]])
            out.append(("table", rows))
    return d["revisionId"], body[-1]["endIndex"], out


def expected(md, base):
    b.IMAGE_BASE = base
    out = []
    for blk in b.blocks(md):
        if blk[0] == "p":
            out.append(("p", b.parse_inline(blk[2])[0]))
        elif blk[0] == "img":
            out.append(("p", "￼"))
        else:
            out.append(("table", [[b.parse_inline(c)[0] for c in r] for r in blk[1]]))
    return out


if __name__ == "__main__":
    rev, end, got = doc_blocks(sys.argv[2])
    exp = expected(Path(sys.argv[1]).read_text(), sys.argv[3] if len(sys.argv) > 3 else None)
    if got and got[0] == ("p", ""):
        got = got[1:]  # section break paragraph
    n = 0
    while n < min(len(got), len(exp)) and got[n] == exp[n]:
        n += 1
    print({"revision": rev[:16], "body_end": end, "doc_blocks": len(got), "expected_blocks": len(exp), "matching_prefix": n})
    if n < len(got):
        print("first doc block not matching:", repr(got[n])[:300])
        print("expected:", repr(exp[n])[:300] if n < len(exp) else None)
