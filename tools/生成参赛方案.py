"""把一页纸、技术报告、设计说明、运行说明合并为一份《参赛方案》PDF。

用途：PyTRiO「提交参赛方案」表单的「作品文档」只能选一个文件，本脚本把
提交包内的四份核心材料按阅读顺序合成为单个 PDF，供该字段直接上传。

实现：
  1. 两份 HTML 源稿按 section 作用域化 CSS（避免 h1/h2/table 等全局样式互相污染）
  2. 设计说明、运行说明由内置的轻量 Markdown 转换器渲染为 HTML
  3. 统一 @page 为 A4 / 13mm 边距，每节强制分页
  4. Chrome headless 打印，输出到内部目录与提交副本

依赖：Chrome、bs4。无第三方 PDF 库，因此不拼接既有 PDF，而是统一排版后一次打印。
"""

from __future__ import annotations

import html as htmllib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(r"D:\industrial_evolution")
SRC = ROOT / "submission/提交版"
OUT = Path(r"D:\桌面\智训引擎-提交")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

ONEPAGER = SRC / "答辩/答辩一页纸.html"
REPORT = SRC / "技术报告/技术报告.html"
DESIGN = SRC / "设计说明/设计说明.md"
RUNBOOK = OUT / "运行说明.md"

TARGETS = (OUT / "参赛方案.pdf", SRC / "答辩/参赛方案.pdf")

# 统一页边距：一页纸原为 11mm/12mm，报告原为 16mm/15mm，取折中 13mm。
PAGE_CSS = "@page { size: A4; margin: 12mm 13mm; }"


def scope_css(css: str, scope: str) -> str:
    """把一段 CSS 的所有选择器限制在 .scope 之内，丢弃 @page。"""
    css = re.sub(r"@page\s*\{[^}]*\}", "", css)

    def repl(match: re.Match[str]) -> str:
        selector, body = match.group(1).strip(), match.group(2)
        if not selector or selector.startswith("@"):
            return ""
        parts = []
        for item in selector.split(","):
            item = item.strip()
            if not item:
                continue
            if item in {"body", "html"}:
                parts.append(f".{scope}")
            elif item == "*":
                parts.append(f".{scope} *")
            else:
                parts.append(f".{scope} {item}")
        return ", ".join(parts) + " {" + body + "}"

    return re.sub(r"([^{}]+)\{([^{}]*)\}", repl, css)


def extract(html_path: Path, scope: str) -> str:
    """取 HTML 的 body 内容，并把其 <style> 作用域化后内联返回。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    css = "\n".join(style.string or "" for style in soup.find_all("style"))
    body = soup.body
    assert body is not None
    inner = "".join(str(child) for child in body.children)
    return f"<style>{scope_css(css, scope)}</style>\n<div class=\"{scope}\">{inner}</div>"


# ------------------------------------------------------------------ Markdown

INLINE_RULES: tuple[tuple[str, str], ...] = (
    (r"`([^`]+)`", r"<code>\1</code>"),
    (r"\*\*([^*]+)\*\*", r"<strong>\1</strong>"),
    (r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>'),
)


def inline(text: str) -> str:
    """行内标记：先转义，再恢复代码/粗体/链接。"""
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    for pattern, _ in INLINE_RULES:
        text = re.sub(pattern, stash, text)
    text = htmllib.escape(text, quote=False)
    for index in reversed(range(len(placeholders))):
        # 占位内容本身尚未转义，先 escape 再套标记，避免 <tar> 之类被当成 HTML 标签吃掉
        rendered = htmllib.escape(placeholders[index], quote=False)
        for pattern, replacement in INLINE_RULES:
            rendered = re.sub(pattern, replacement, rendered)
        text = text.replace(f"\x00{index}\x00", rendered)
    return text


def md_to_html(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if line.startswith("```"):
            index += 1
            block: list[str] = []
            while index < total and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            out.append("<pre><code>" + htmllib.escape("\n".join(block), quote=False) + "</code></pre>")
            continue

        if not line.strip():
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2).strip())}</h{level}>")
            index += 1
            continue

        if line.lstrip().startswith("|") and index + 1 < total and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[index + 1]):
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            index += 2
            rows: list[list[str]] = []
            while index < total and lines[index].lstrip().startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            out.append("<table><thead><tr>")
            out.extend(f"<th>{inline(cell)}</th>" for cell in header)
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        if line.lstrip().startswith("> "):
            block = []
            while index < total and lines[index].lstrip().startswith(">"):
                block.append(lines[index].lstrip()[1:].strip())
                index += 1
            out.append("<blockquote><p>" + inline(" ".join(block)) + "</p></blockquote>")
            continue

        list_match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if list_match:
            ordered = list_match.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < total:
                match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[index])
                if not match:
                    break
                items.append(match.group(3))
                index += 1
            out.append(f"<{tag}>")
            out.extend(f"<li>{inline(item)}</li>" for item in items)
            out.append(f"</{tag}>")
            continue

        paragraph: list[str] = []
        while index < total and lines[index].strip() and not re.match(r"^(#{1,6}\s|```|\s*\||\s*>|\s*([-*]|\d+\.)\s)", lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
        else:
            index += 1

    return "\n".join(out)


MD_CSS = """
.md { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; font-size: 10pt;
      line-height: 1.62; color: #1c2530; }
.md h1 { font-size: 19pt; color: #12385e; margin: 0 0 4mm; }
.md h2 { font-size: 12.5pt; color: #12385e; margin: 6mm 0 2.4mm;
         border-left: 3.5px solid #2f6fb5; padding-left: 3mm; }
.md h3 { font-size: 10.8pt; color: #234; margin: 4mm 0 1.6mm; }
.md p { margin: 0 0 2.2mm; text-align: justify; }
.md ul, .md ol { margin: 0 0 2.2mm; padding-left: 6mm; }
.md li { margin-bottom: 1mm; }
.md table { width: 100%; border-collapse: collapse; font-size: 8.8pt; margin: 2mm 0 3mm; }
.md th, .md td { border: .5px solid #c9d4e0; padding: 1.4mm 2mm; text-align: left; vertical-align: top; }
.md th { background: #eaf1f8; color: #12385e; font-weight: 600; }
.md code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt;
           background: #f1f4f7; padding: 0 1mm; color: #1d4d78; }
.md pre { background: #f6f8fa; border: .5px solid #d8e0e8; border-radius: 1mm;
          padding: 2mm 2.5mm; font-size: 8.4pt; line-height: 1.45; white-space: pre-wrap;
          page-break-inside: avoid; margin: 2mm 0 3mm; }
.md pre code { background: none; padding: 0; font-size: 8.4pt; }
.md blockquote { margin: 2mm 0; padding: 1.8mm 3mm; background: #f2f7fc;
                 border-left: 3px solid #2f6fb5; font-size: 9.2pt; color: #2b3b4b; }
.md blockquote p { margin: 0; }
.md strong { color: #12385e; }
.md tr { page-break-inside: avoid; }
"""

COVER_CSS = """
.cover { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #1c2530;
         min-height: 250mm; display: flex; flex-direction: column; }
.cover .bar { height: 4mm; background: linear-gradient(90deg, #12385e 0%, #2f6fb5 100%); margin-bottom: 22mm; }
.cover .team { font-size: 10pt; color: #55636f; letter-spacing: 1px; margin-bottom: 6mm; }
.cover h1 { font-size: 30pt; color: #12385e; margin: 0 0 3mm; line-height: 1.25; }
.cover .sub { font-size: 13pt; color: #2f6fb5; margin: 0 0 14mm; font-weight: 600; }
.cover .lead { font-size: 10.5pt; line-height: 1.75; color: #2b3b4b; background: #eef4fb;
               border-left: 3.5px solid #2f6fb5; padding: 3mm 4mm; margin-bottom: 12mm; }
.cover table { width: 100%; border-collapse: collapse; font-size: 9pt; }
.cover th, .cover td { border-bottom: .5px solid #cdd8e3; padding: 2.2mm 1mm; text-align: left; vertical-align: top; }
.cover th { width: 26mm; color: #12385e; font-weight: 600; }
.cover .sect { margin-top: 14mm; font-size: 8.6pt; color: #55636f; letter-spacing: 2px; }
.cover .toc { display: flex; gap: 4mm; margin-top: 3mm; }
.cover .toc div { flex: 1; border-top: 2px solid #2f6fb5; padding-top: 2mm; font-size: 9pt; color: #2b3b4b; }
.cover .toc span { display: block; font-size: 8pt; color: #8a97a3; margin-bottom: 1mm; }
.cover .foot { margin-top: auto; font-size: 8.5pt; color: #6b7783;
               border-top: 1px solid #cdd8e3; padding-top: 2.5mm; }
"""

COVER = """
<div class="cover">
  <div class="bar"></div>
  <div class="team">会赢的，对不队　·　北京印刷学院</div>
  <h1>工业模型自进化框架</h1>
  <div class="sub">参赛方案 · 技术报告与设计说明</div>
  <div class="lead">面向工业数据漂移的受控自进化 Harness：把数据校验、漂移监测、策略决策、回放训练、冠军-挑战者门控、版本管理与异常回滚组成无人干预闭环，每一轮「观察 → 诊断 → 决策 → 结果 → 判定」完整留痕、可离线复算。</div>
  <table>
    <tr><th>系统镜像</th><td><code>industrial-evolution-school-2026-11-15.tar</code>（OCI layout，542 MiB，15 层，四模式入口 <code>check</code> / <code>demo</code> / <code>csv</code> / <code>record</code>）</td></tr>
    <tr><th>策略权重</th><td>PyTRiO 平台 · <code>Qwen/Qwen3.8-27B</code> 基座，LoRA rank 32，三轮续训（平均损失 7.396 → 3.008），第三轮权重经 30 条真实请求采样验证合规 30/30</td></tr>
    <tr><th>训练记录</th><td>SwanLab 公开项目 <code>industrial_evolution</code>，三轮训练各一条 run，含损失曲线与超参</td></tr>
    <tr><th>源码仓库</th><td><code>https://github.com/dinghoudong-cyber/industrial-evolution</code>（MIT，156 个文件，包内不含任何模型权重）</td></tr>
  </table>
  <div class="sect">文 档 构 成</div>
  <div class="toc">
    <div><span>01</span>方案速览</div>
    <div><span>02</span>技术报告</div>
    <div><span>03</span>设计说明</div>
    <div><span>04</span>运行说明</div>
  </div>
  <div class="foot">文档版本 2026-11-15　|　全部实验结论均标注数据来源与评估口径，可在提交目录内按 SHA-256 校验后逐项复算。</div>
</div>
"""

DOC_CSS = """
html, body { margin: 0; padding: 0; }
section.page { page-break-before: always; }
section.page:first-of-type { page-break-before: auto; }
section.page > :last-child { margin-bottom: 0; }
/* 段末页脚在合订本中隐去：信息与封面、索引重复，且其整块高度会在页尾撑出一张空白页 */
div.s2 .foot { display: none !important; }
"""


def build_html() -> str:
    parts = [
        f"<style>{PAGE_CSS}</style>",
        f"<style>{DOC_CSS}</style>",
        f"<style>{COVER_CSS}</style>",
        f"<style>{MD_CSS}</style>",
        '<section class="page">' + COVER + "</section>",
        '<section class="page">' + extract(ONEPAGER, "s1") + "</section>",
        '<section class="page">' + extract(REPORT, "s2") + "</section>",
    ]
    design = md_to_html(DESIGN.read_text(encoding="utf-8"))
    parts.append('<section class="page"><div class="md">' + design + "</div></section>")
    runbook = md_to_html(RUNBOOK.read_text(encoding="utf-8"))
    parts.append('<section class="page"><div class="md">' + runbook + "</div></section>")

    body = "\n".join(parts)
    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>工业模型自进化框架 · 参赛方案</title></head><body>"
        + body
        + "</body></html>"
    )


def render(document: str) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "参赛方案.html"
        source.write_text(document, encoding="utf-8")
        pdf = Path(tmp) / "参赛方案.pdf"
        command = [
            str(CHROME), "--headless=new", "--disable-gpu", "--no-sandbox",
            "--disable-dev-shm-usage", f"--user-data-dir={tmp}/profile",
            "--no-pdf-header-footer", "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf}", source.as_uri(),
        ]
        proc = subprocess.run(command, capture_output=True, timeout=300)
        if not pdf.is_file() or pdf.stat().st_size < 20_000:
            raise RuntimeError(f"PDF 渲染失败: {proc.returncode} / {proc.stderr[-500:]!r}")
        for target in TARGETS:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, target)
        return TARGETS[0]


def page_count(path: Path) -> str:
    """Chrome 打印出的 Pages 树是嵌套的，根节点的 /Count 是最大值。"""
    counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", path.read_bytes())]
    return str(max(counts)) if counts else "?"


def main() -> None:
    document = build_html()
    output = render(document)
    print(f"参赛方案.pdf: {output} / {output.stat().st_size:,} bytes / {page_count(output)} 页")


if __name__ == "__main__":
    main()
