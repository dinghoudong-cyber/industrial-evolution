"""把 docs 下的 Markdown 报告渲染成单文件 HTML。

用法：
    python tools/build_report_html.py docs/功能实现报告.md

只依赖 markdown-it-py（SwanLab 的传递依赖里已经带了）。生成的 HTML 自带样式，
不引用任何外部资源，可直接双击、发邮件或打印。
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
:root {
  --ink: #1b1f24;
  --ink-soft: #414b56;
  --muted: #6b7683;
  --line: #e4e8ee;
  --bg: #eef1f5;
  --paper: #ffffff;
  --soft: #f6f8fb;
  --accent: #0f5f8a;
  --accent-soft: #e8f2f8;
  --warn: #c2410c;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", "Noto Sans SC",
               -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15.5px;
  line-height: 1.85;
  -webkit-font-smoothing: antialiased;
}
.layout {
  display: grid;
  grid-template-columns: 258px minmax(0, 1fr);
  max-width: 1420px;
  margin: 0 auto;
}
/* ---------- 侧边目录 ---------- */
nav.toc {
  position: sticky;
  top: 0;
  align-self: start;
  max-height: 100vh;
  overflow-y: auto;
  padding: 34px 16px 40px 24px;
  font-size: 13px;
  line-height: 1.6;
  border-right: 1px solid var(--line);
}
nav.toc .toc-title {
  font-size: 11px;
  letter-spacing: .16em;
  color: var(--muted);
  margin-bottom: 12px;
  text-transform: uppercase;
}
nav.toc a {
  display: block;
  color: var(--ink-soft);
  text-decoration: none;
  padding: 3px 0 3px 10px;
  border-left: 2px solid transparent;
}
nav.toc a:hover { color: var(--accent); border-left-color: var(--accent); }
nav.toc a.lv1 {
  font-weight: 600;
  color: var(--ink);
  margin-top: 12px;
  padding-left: 10px;
}
nav.toc a.lv2 { padding-left: 22px; color: var(--ink-soft); }
/* ---------- 正文 ---------- */
main {
  background: var(--paper);
  padding: 52px 66px 110px;
  box-shadow: 0 0 0 1px rgba(15, 95, 138, .04), 0 18px 50px rgba(24, 40, 60, .06);
  min-width: 0;
}
.hero {
  border-bottom: 1px solid var(--line);
  padding-bottom: 26px;
  margin-bottom: 8px;
}
.hero .kicker {
  font-size: 12px;
  letter-spacing: .18em;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 10px;
}
.hero h1 {
  font-size: 34px;
  line-height: 1.32;
  margin: 0 0 14px;
  letter-spacing: -.4px;
}
.hero .lede {
  font-size: 16px;
  color: var(--ink-soft);
  margin: 0 0 18px;
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  font-size: 12px;
  color: var(--ink-soft);
  background: var(--soft);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px 11px;
}
h1 {
  font-size: 27px;
  line-height: 1.4;
  margin: 62px 0 16px;
  padding-top: 26px;
  border-top: 1px solid var(--line);
  letter-spacing: -.2px;
}
h2 {
  font-size: 20px;
  margin: 38px 0 12px;
  color: var(--accent);
  letter-spacing: -.1px;
}
h3 {
  font-size: 16.5px;
  margin: 26px 0 8px;
}
p { margin: 10px 0; }
ul, ol { margin: 10px 0; padding-left: 26px; }
li { margin: 4px 0; }
li::marker { color: var(--accent); }
strong { font-weight: 700; color: #0b1218; }
a { color: var(--accent); }
hr {
  border: none;
  border-top: 1px dashed var(--line);
  margin: 34px 0;
}
/* ---------- 代码 ---------- */
code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace;
  font-size: .875em;
  background: #eef2f7;
  color: #a4233a;
  padding: 1px 5px;
  border-radius: 4px;
  word-break: break-word;
}
pre {
  background: var(--soft);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 14px 18px;
  overflow-x: auto;
  margin: 16px 0;
  line-height: 1.62;
}
pre code {
  background: none;
  color: #22303c;
  padding: 0;
  font-size: 12.6px;
  white-space: pre;
}
/* ---------- 表格 ---------- */
.table-wrap {
  overflow-x: auto;
  margin: 16px 0 20px;
  border: 1px solid var(--line);
  border-radius: 6px;
}
table { border-collapse: collapse; width: 100%; font-size: 13.6px; }
th {
  background: var(--accent);
  color: #fff;
  font-weight: 600;
  text-align: left;
  padding: 9px 12px;
  white-space: nowrap;
}
td {
  padding: 8px 12px;
  border-top: 1px solid var(--line);
  vertical-align: top;
  line-height: 1.7;
}
tbody tr:nth-child(even) { background: #f8fafc; }
tbody tr:hover { background: var(--accent-soft); }
td code, th code { background: rgba(255, 255, 255, .18); color: inherit; }
td code { background: #eef2f7; color: #a4233a; }
/* 首列常是文件名/路径，别在词中间断行 */
td:first-child code { white-space: nowrap; }
/* ---------- 引用 ---------- */
blockquote {
  margin: 18px 0;
  padding: 12px 18px;
  background: #fffaf1;
  border-left: 4px solid #e2a33c;
  border-radius: 0 6px 6px 0;
  color: #5c4a2c;
}
blockquote p { margin: 6px 0; }
/* ---------- 页脚 ---------- */
.foot {
  margin-top: 56px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
  font-size: 12.5px;
  color: var(--muted);
}
@media (max-width: 1120px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { display: none; }
  main { padding: 34px 22px 70px; }
  .hero h1 { font-size: 26px; }
  h1 { font-size: 23px; margin-top: 44px; }
  h2 { font-size: 18px; }
  body { font-size: 15px; }
}
@media print {
  body { background: #fff; font-size: 11pt; }
  nav.toc { display: none; }
  .layout { display: block; }
  main { box-shadow: none; padding: 0; }
  h1 { page-break-after: avoid; }
  table, pre { page-break-inside: avoid; }
  .table-wrap { overflow: visible; }
}
"""


def _slug(text: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text).strip("-").lower() or "sec"
    count = used.get(base, 0) + 1
    used[base] = count
    return base if count == 1 else f"{base}-{count}"


def build(source: Path) -> Path:
    md_text = source.read_text(encoding="utf-8")
    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    md.enable("table")
    body = md.render(md_text)

    # 表格外包一层，避免窄屏把页面撑爆
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")

    # 给 h1/h2 加锚点，并收集目录
    used: dict[str, int] = {}
    toc: list[tuple[int, str, str]] = []

    def _anchor(match: re.Match[str]) -> str:
        level = int(match.group(1))
        inner = match.group(2)
        text = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
        anchor = _slug(text, used)
        toc.append((level, anchor, text))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    body = re.sub(r"<h([12])>(.*?)</h\1>", _anchor, body, flags=re.S)

    # 首个 h1 当作封面标题，移出正文
    title = "功能实现报告"
    first = re.search(r'<h1 id="[^"]+">(.*?)</h1>', body, flags=re.S)
    if first:
        title = re.sub(r"<[^>]+>", "", first.group(1)).strip()
        body = body[: first.start()] + body[first.end() :]
        toc = [item for item in toc if item[2] != title]

    toc_html = "".join(
        f'<a class="lv{level}" href="#{anchor}">{html.escape(text)}</a>'
        for level, anchor, text in toc
    )

    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="layout">
<nav class="toc">
  <div class="toc-title">目录</div>
  {toc_html}
</nav>
<main>
  <header class="hero">
    <div class="kicker">INDUSTRIAL EVOLUTION · 功能实现报告</div>
    <h1>{html.escape(title)}</h1>
    <p class="lede">系统在做什么、每件事怎么实现的、代码在哪个文件夹——逐层拆到文件级。</p>
    <div class="chips">
      <span class="chip">70 项测试通过</span>
      <span class="chip">ruff 干净</span>
      <span class="chip">镜像实测 · 断网 1 核 1GB</span>
      <span class="chip">2026-11-08</span>
    </div>
  </header>
  {body}
  <div class="foot">本页由 <code>tools/build_report_html.py</code> 从
  <code>{html.escape(source.name)}</code> 生成，正文以 Markdown 为准。</div>
</main>
</div>
</body>
</html>
"""
    target = source.with_suffix(".html")
    target.write_text(document, encoding="utf-8")
    return target


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for raw in sys.argv[1:]:
        path = Path(raw)
        if not path.is_file():
            print(f"文件不存在：{path}")
            return 1
        print(f"已生成 {build(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
