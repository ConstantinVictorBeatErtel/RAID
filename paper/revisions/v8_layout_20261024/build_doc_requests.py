"""Build Google Docs batchUpdate requests that replace the whole body with paper/raid_manuscript.md.

Usage: python3 build_doc_requests.py MANUSCRIPT.md BODY_END OUT_DIR [CHUNK_CHARS] [IMAGE_BASE_URL]
With IMAGE_BASE_URL, each figure line ![Figure N](figures/x.png) becomes an inline image fetched from
IMAGE_BASE_URL/figures/x.png (Docs needs a publicly reachable URL); without it, a text placeholder.
Writes OUT_DIR/chunk_NN.json (lists of requests) and OUT_DIR/plan.json. Positions are computed sequentially
from the state after the first request (delete [1, BODY_END-1]); chunks must be applied in order with no other
edits in between (each call is guarded by a fresh revisionId).
"""
import json
import re
import sys
from pathlib import Path

TAB = "t.0"
IMAGE_BASE = None
IMAGE_SIZE = {"width": {"magnitude": 468, "unit": "PT"}, "height": {"magnitude": 640, "unit": "PT"}}
HEADER_BG = {"color": {"rgbColor": {"red": 0.92156863, "green": 0.9411765, "blue": 0.96862745}}}


def u16(s):
    return len(s.encode("utf-16-le")) // 2


def rng(a, b):
    return {"startIndex": a, "endIndex": b, "tabId": TAB}


def parse_inline(text):
    """Return (plain, runs) with runs = [(start, end, style_dict, fields)] in UTF-16 offsets of plain."""
    out, runs = [], []
    pos = 0
    i = 0
    pat = re.compile(r"\[(\d+)\]\((https?://[^)]+)\)|\[([^\]]+)\]\((https?://[^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|(?<![\w*\\])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|\\([*_\\`\[\]])")
    for m in pat.finditer(text):
        lit = text[i:m.start()]
        out.append(lit); pos += u16(lit)
        if m.group(1):  # numbered citation: [n] with link on n
            out.append("["); pos += 1
            n = m.group(1)
            runs.append((pos, pos + u16(n), {"link": {"url": m.group(2)}}, "link"))
            out.append(n); pos += u16(n)
            out.append("]"); pos += 1
        elif m.group(3):
            inner, iruns = parse_inline(m.group(3))
            runs += [(pos + a, pos + b, s, f) for a, b, s, f in iruns]
            runs.append((pos, pos + u16(inner), {"link": {"url": m.group(4)}}, "link"))
            out.append(inner); pos += u16(inner)
        elif m.group(5):
            inner, iruns = parse_inline(m.group(5))
            runs += [(pos + a, pos + b, s, f) for a, b, s, f in iruns]
            runs.append((pos, pos + u16(inner), {"bold": True}, "bold"))
            out.append(inner); pos += u16(inner)
        elif m.group(6):
            runs.append((pos, pos + u16(m.group(6)), {"weightedFontFamily": {"fontFamily": "Courier New"}}, "weightedFontFamily"))
            out.append(m.group(6)); pos += u16(m.group(6))
        elif m.group(7):
            inner, iruns = parse_inline(m.group(7))
            runs += [(pos + a, pos + b, s, f) for a, b, s, f in iruns]
            runs.append((pos, pos + u16(inner), {"italic": True}, "italic"))
            out.append(inner); pos += u16(inner)
        else:
            out.append(m.group(8)); pos += 1
        i = m.end()
    out.append(text[i:])
    return "".join(out), runs


def blocks(md):
    """Yield ('p', style, text, list_kind) or ('table', rows) blocks from the manuscript Markdown."""
    md = re.sub(r"<!--.*?-->\n?", "", md, flags=re.S)
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("> "):
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split(" | ")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield ("table", rows); continue
        m = re.match(r"^(#{1,3}) (.*)$", line)
        if m:
            style = {1: "TITLE", 2: "HEADING_1", 3: "HEADING_2"}[len(m.group(1))]
            yield ("p", style, m.group(2), None); i += 1; continue
        m = re.match(r"^!\[(Figure \d+)\]\(([^)]+)\)$", line)
        if m:
            if IMAGE_BASE:
                yield ("img", f"{IMAGE_BASE}/{m.group(2)}")
            else:
                yield ("p", "NORMAL_TEXT", f"*[{m.group(1)} image: paper/{m.group(2)}]*", None)
            i += 1; continue
        m = re.match(r"^- (.*)$", line)
        if m:
            yield ("p", "NORMAL_TEXT", m.group(1), "bullet"); i += 1; continue
        m = re.match(r"^\d+\. (.*)$", line)
        if m:
            yield ("p", "NORMAL_TEXT", m.group(1), "number"); i += 1; continue
        # paragraph: join soft-wrapped lines; a trailing backslash is a hard break -> separate paragraphs
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "- ", "> ", "![")) \
                and not re.match(r"^\d+\. ", lines[i]):
            para.append(lines[i]); i += 1
        buf = []
        for l in para:
            if l.endswith("\\"):
                buf.append(l[:-1]); yield ("p", "NORMAL_TEXT", " ".join(buf), None); buf = []
            else:
                buf.append(l)
        if buf:
            text = " ".join(buf)
            if text.startswith(("*Working manuscript", "*Preprint")):
                yield ("p", "SUBTITLE", text.strip("*"), None)
            else:
                yield ("p", "NORMAL_TEXT", text, None)


def build(md, body_end):
    reqs = [[{"deleteContentRange": {"range": rng(1, body_end - 1)}}]]
    cur = 1
    group = []  # pending paragraphs

    def flush():
        nonlocal cur, group
        if not group:
            return
        text = "".join(p[0] + "\n" for p in group)
        r = [{"insertText": {"location": {"index": cur, "tabId": TAB}, "text": text}}]
        end = cur + u16(text)
        r.append({"updateTextStyle": {"range": rng(cur, end), "textStyle": {},
                                      "fields": "bold,italic,underline,link,weightedFontFamily,foregroundColor,baselineOffset"}})
        r.append({"updateParagraphStyle": {"range": rng(cur, end), "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                                           "fields": "namedStyleType"}})
        p0 = cur
        lists = []
        for plain, runs, style, kind in group:
            a, b = p0, p0 + u16(plain) + 1
            if style != "NORMAL_TEXT":
                r.append({"updateParagraphStyle": {"range": rng(a, b), "paragraphStyle": {"namedStyleType": style},
                                                   "fields": "namedStyleType"}})
            if kind:
                if lists and lists[-1][0] == kind and lists[-1][2] == a:
                    lists[-1][2] = b
                else:
                    lists.append([kind, a, b])
            for s, e, st, f in runs:
                r.append({"updateTextStyle": {"range": rng(a + s, a + e), "textStyle": st, "fields": f}})
            p0 = b
        for kind, a, b in lists:
            r.append({"createParagraphBullets": {"range": rng(a, b), "bulletPreset":
                      "BULLET_DISC_CIRCLE_SQUARE" if kind == "bullet" else "NUMBERED_DECIMAL_ALPHA_ROMAN"}})
        reqs.append(r)
        cur = end
        group = []

    ntables = 0
    for blk in blocks(md):
        if blk[0] == "p":
            _, style, text, kind = blk
            plain, runs = parse_inline(text)
            if plain.startswith("Keywords:"):
                runs = runs + [(0, 9, {"bold": True}, "bold")]
            group.append((plain, runs, style, kind))
            continue
        flush()
        if blk[0] == "img":
            # empty centred paragraph, then the image before its newline: the paragraph spans [cur, cur + 2)
            reqs.append([{"insertText": {"location": {"index": cur, "tabId": TAB}, "text": "\n"}},
                         {"updateParagraphStyle": {"range": rng(cur, cur + 1),
                                                   "paragraphStyle": {"namedStyleType": "NORMAL_TEXT", "alignment": "CENTER"},
                                                   "fields": "namedStyleType,alignment"}},
                         {"insertInlineImage": {"location": {"index": cur, "tabId": TAB}, "uri": blk[1],
                                                "objectSize": IMAGE_SIZE}}])
            cur += 2
            continue
        rows = blk[1]
        R, C = len(rows), len(rows[0])
        assert all(len(x) == C for x in rows), rows
        i = cur - 1  # newline of the preceding paragraph
        r = [{"insertTable": {"rows": R, "columns": C, "location": {"index": i, "tabId": TAB}}}]
        cells = [[parse_inline(c) for c in row] for row in rows]
        base = lambda rr, cc: i + 4 + rr * (2 * C + 1) + 2 * cc
        for rr in reversed(range(R)):
            for cc in reversed(range(C)):
                plain = cells[rr][cc][0]
                if plain:
                    r.append({"insertText": {"location": {"index": base(rr, cc), "tabId": TAB}, "text": plain}})
        shift = 0
        for rr in range(R):
            for cc in range(C):
                plain, runs = cells[rr][cc]
                start = base(rr, cc) + shift
                for s, e, st, f in runs:
                    r.append({"updateTextStyle": {"range": rng(start + s, start + e), "textStyle": st, "fields": f}})
                shift += u16(plain)
        r.append({"updateTableCellStyle": {"tableRange": {"tableCellLocation": {"tableStartLocation": {"index": i + 1, "tabId": TAB},
                  "rowIndex": 0, "columnIndex": 0}, "rowSpan": 1, "columnSpan": C},
                  "tableCellStyle": {"backgroundColor": HEADER_BG}, "fields": "backgroundColor"}})
        reqs.append(r)
        cur = i + 3 + R * (2 * C + 1) + shift
        ntables += 1
    flush()
    reqs.append([{"deleteContentRange": {"range": rng(cur, cur + ntables)}}] if ntables else [])
    return reqs, cur, ntables


def main():
    md = Path(sys.argv[1]).read_text()
    body_end = int(sys.argv[2])
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 45000
    global IMAGE_BASE
    IMAGE_BASE = sys.argv[5].rstrip("/") if len(sys.argv) > 5 else None
    groups, cur, ntables = build(md, body_end)
    chunks, curchunk, size = [], [], 0
    for g in groups:
        s = len(json.dumps(g, ensure_ascii=False))
        if curchunk and size + s > limit:
            chunks.append(curchunk); curchunk, size = [], 0
        curchunk += g; size += s
    if curchunk:
        chunks.append(curchunk)
    for k, c in enumerate(chunks, 1):
        (out / f"chunk_{k:02d}.json").write_text(json.dumps(c, ensure_ascii=False, separators=(",", ":")))
    plan = {"chunks": len(chunks), "requests": sum(map(len, chunks)), "final_cur": cur, "tables": ntables,
            "sizes": [len(json.dumps(c, ensure_ascii=False, separators=(",", ":"))) for c in chunks]}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(plan)


if __name__ == "__main__":
    main()
