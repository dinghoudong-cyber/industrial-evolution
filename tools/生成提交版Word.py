"""按参考策划案的字体字号，生成提交版的两份 Word 文档。

排版基准（逐项取自《【策划案】…（2026年秋季）.docx》）：
  纸张      Letter 12240 × 15840 twips
  页边距    上下 1440 twips (2.54 cm)，左右 1800 twips (3.17 cm)
  正文      宋体 / Times New Roman，10.5 磅，1.5 倍行距，首行缩进 2 字符
  标题      黑体加粗：文档标题 22 磅、一级 15 磅、二级 13 磅
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\16494\AppData\Local\Temp\ie-docx-libs")

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Pt, Twips  # noqa: E402

ROOT = Path(r"D:\industrial_evolution")
OUT = Path(r"D:\桌面\智训引擎-校赛提交")
OUT.mkdir(parents=True, exist_ok=True)

BODY_SIZE = 10.5
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def style_run(run, size: float, bold: bool = False, cn: str = "宋体", en: str = "Times New Roman") -> None:
    run.font.name = en
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def add_inline(paragraph, text: str, size: float = BODY_SIZE) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            style_run(paragraph.add_run(part[2:-2]), size, bold=True)
        elif part.startswith("`") and part.endswith("`"):
            style_run(paragraph.add_run(part[1:-1]), size - 0.5, cn="宋体", en="Consolas")
        else:
            style_run(paragraph.add_run(part), size)


def fmt(paragraph, indent: bool = True, line: float = 1.5, before: float = 0, after: float = 0,
        left: float = 0, align=None) -> None:
    pf = paragraph.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.first_line_indent = Pt(2 * BODY_SIZE) if indent else Pt(0)
    if left:
        pf.left_indent = Pt(left)
    if align is not None:
        paragraph.alignment = align


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph()
    if level == 1:
        fmt(p, indent=False, before=0, after=14, align=WD_ALIGN_PARAGRAPH.CENTER)
        style_run(p.add_run(text), 22, bold=True, cn="黑体")
    elif level == 2:
        fmt(p, indent=False, before=13, after=7)
        style_run(p.add_run(text), 15, bold=True, cn="黑体")
    else:
        fmt(p, indent=False, before=9, after=5)
        style_run(p.add_run(text), 13, bold=True, cn="黑体")


def add_body(doc: Document, text: str, bullet: bool = False) -> None:
    p = doc.add_paragraph()
    if bullet:
        fmt(p, indent=False, after=2, left=21)
        style_run(p.add_run("· "), BODY_SIZE)
        add_inline(p, text)
    else:
        fmt(p, indent=True, after=2)
        add_inline(p, text)


def add_code(doc: Document, lines: list[str]) -> None:
    for line in lines:
        p = doc.add_paragraph()
        fmt(p, indent=False, line=1.0, after=0, left=18)
        style_run(p.add_run(line if line else " "), 9, cn="宋体", en="Consolas")


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            fmt(p, indent=False, line=1.15, after=0)
            text = row[j] if j < len(row) else ""
            if i == 0:
                style_run(p.add_run(text.replace("**", "")), 9.5, bold=True, cn="黑体")
            else:
                add_inline(p, text)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_number(doc: Document) -> None:
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(p.add_run("— 第 "), 9)
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._element.addnext(fld)
    style_run(p.add_run(" 页 —"), 9)


def build(md_path: Path, out_path: Path) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Twips(12240)
    sec.page_height = Twips(15840)
    sec.top_margin = Twips(1440)
    sec.bottom_margin = Twips(1440)
    sec.left_margin = Twips(1800)
    sec.right_margin = Twips(1800)
    sec.header_distance = Twips(720)
    sec.footer_distance = Twips(720)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(BODY_SIZE)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5

    lines = md_path.read_text(encoding="utf-8").splitlines()
    add_page_number(doc)

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code(doc, block)
            continue
        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = 1 if len(m.group(1)) == 1 else 2 if len(m.group(1)) == 2 else 3
            add_heading(doc, m.group(2), level)
            i += 1
            continue
        if stripped.startswith("> "):
            add_body(doc, stripped[2:])
            i += 1
            continue
        bullet = bool(re.match(r"^[-*]\s+|^\d+[.)]\s+", stripped))
        text = re.sub(r"^[-*]\s+|^\d+[.)]\s+", "", stripped)
        add_body(doc, text, bullet=bullet)
        i += 1

    doc.save(out_path)
    print(f"{out_path.name}: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    build(
        ROOT / "submission/提交版/设计说明/设计说明.md",
        OUT / "设计说明.docx",
    )
    build(
        ROOT / "submission/提交版/技术报告/技术报告正文.md",
        OUT / "技术报告.docx",
    )
