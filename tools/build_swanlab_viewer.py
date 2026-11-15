"""把一个离线 SwanLab run 目录渲染成单文件 HTML，方便不登录看板也能翻记录。

用法：
    python tools/build_swanlab_viewer.py runs/swanlab-demo/.swanlab/run-20261109_155008-nhaechv4
    python tools/build_swanlab_viewer.py <run_dir> -o 输出路径.html

只读本地文件，不联网，不改动原目录。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

# 要看趋势的数值列（顺序即展示顺序）
CURVE_COLUMNS = [
    ("score", "窗口得分"),
    ("feature_drift_score", "特征漂移量"),
    ("performance_drop", "性能下降"),
    ("label_shift", "标签漂移"),
    ("candidate_improvement", "候选提升"),
    ("labels_used", "累计标签预算"),
    ("training_seconds_used", "累计训练秒数"),
]


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _svg_curve(rows: list[dict[str, str]], key: str, label: str) -> str:
    points: list[tuple[float, float]] = []
    for row in rows:
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        try:
            points.append((float(row.get("window") or 0), float(raw)))
        except ValueError:
            continue
    if len(points) < 2:
        return f'<div class="chart-empty">「{html.escape(label)}」没有足够的数值点</div>'

    w, h, pad_l, pad_r, pad_t, pad_b = 520, 150, 46, 14, 16, 26
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1.0
        y0 -= 1.0

    def sx(value: float) -> float:
        return pad_l + (value - x0) / max(x1 - x0, 1e-9) * (w - pad_l - pad_r)

    def sy(value: float) -> float:
        return h - pad_b - (value - y0) / (y1 - y0) * (h - pad_t - pad_b)

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points)
    )
    dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.2"/>' for x, y in points)
    grid = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + i * (h - pad_t - pad_b) / 3:.1f}" '
        f'x2="{w - pad_r}" y2="{pad_t + i * (h - pad_t - pad_b) / 3:.1f}" class="grid"/>'
        for i in range(4)
    )
    return f"""<figure class="chart">
  <figcaption>{html.escape(label)}<span>{y1:.4g} ↑ / {y0:.4g} ↓</span></figcaption>
  <svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}">
    {grid}
    <path d="{path}" class="curve"/>
    {dots}
    <text x="{pad_l - 6}" y="{sy(y1) + 4:.1f}" class="axis" text-anchor="end">{y1:.3g}</text>
    <text x="{pad_l - 6}" y="{sy(y0) + 4:.1f}" class="axis" text-anchor="end">{y0:.3g}</text>
    <text x="{pad_l}" y="{h - 8}" class="axis">w{int(x0)}</text>
    <text x="{w - pad_r}" y="{h - 8}" class="axis" text-anchor="end">w{int(x1)}</text>
  </svg>
</figure>"""


_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    out = html.escape(text)
    out = _MD_BOLD.sub(r"<strong>\1</strong>", out)
    out = _MD_CODE.sub(r"<code>\1</code>", out)
    return out


def _render_text_card(path: Path) -> str:
    body = path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    title = lines[0].lstrip("# ").strip()
    parts = [f"<h3>{_inline(title)}</h3>"]
    for line in lines[1:]:
        if line.startswith("> "):
            parts.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        else:
            parts.append(f"<p>{_inline(line)}</p>")
    verdict_class = ""
    joined = body
    for token, klass in (
        ("假设成立", "ok"),
        ("假设被否定", "no"),
        ("证据不足", "meh"),
        ("未触发", "skip"),
    ):
        if token in joined:
            verdict_class = klass
            break
    return f'<article class="card {verdict_class}">' + "\n".join(parts) + "</article>"


def _human_size(num: int) -> str:
    for unit in ("B", "KB", "MB"):
        if num < 1024 or unit == "MB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024  # type: ignore[assignment]
    return f"{num:.1f} MB"


def build(run_dir: Path, output: Path | None = None) -> Path:
    files_dir = run_dir / "files"
    text_dir = run_dir / "media" / "text"
    rows = _read_metrics(files_dir / "metrics.csv")

    summary: dict = {}
    summary_path = files_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    charts = "\n".join(_svg_curve(rows, key, label) for key, label in CURVE_COLUMNS)

    table_head = (
        "<tr><th>窗</th><th>得分</th><th>漂移</th><th>动作</th><th>状态</th><th>候选提升</th></tr>"
    )
    table_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td>"
        "<td>{}</td><td>{}</td></tr>".format(
            html.escape(row.get("window", "")),
            html.escape(row.get("score", "")[:7]),
            html.escape(row.get("drift_kind", "")),
            html.escape(row.get("action", "")),
            html.escape(row.get("status", "")),
            html.escape(row.get("candidate_improvement", "")[:7]),
        )
        for row in rows
    )

    cards = []
    for path in sorted(text_dir.glob("*.txt")) if text_dir.exists() else []:
        card = _render_text_card(path)
        if card:
            cards.append(card)

    file_rows = ""
    for path in sorted(files_dir.rglob("*")) if files_dir.exists() else []:
        if path.is_file():
            rel = path.relative_to(files_dir).as_posix()
            file_rows += (
                f"<tr><td><code>{html.escape(rel)}</code></td>"
                f"<td>{_human_size(path.stat().st_size)}</td></tr>"
            )

    summary_rows = "".join(
        f"<tr><td><code>{html.escape(str(key))}</code></td>"
        f"<td>{html.escape(json.dumps(value, ensure_ascii=False))}</td></tr>"
        for key, value in summary.items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SwanLab run 查看器 · {html.escape(run_dir.name)}</title>
<style>
  :root {{
    --bg: #f7f8fa; --panel: #fff; --ink: #1c2531; --dim: #6b7686;
    --line: #e3e7ee; --accent: #2f6feb; --ok: #1f8a51; --no: #b4322f;
    --meh: #8a6d1f; --skip: #5b6472;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }}
  header {{ padding: 34px 40px 24px; background: var(--panel);
    border-bottom: 1px solid var(--line); }}
  h1 {{ margin: 0 0 6px; font-size: 22px; letter-spacing: .3px; }}
  .sub {{ color: var(--dim); font-size: 13px; font-family: ui-monospace, Consolas, monospace; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
  .kpi {{ background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
    padding: 9px 14px; min-width: 120px; }}
  .kpi b {{ display: block; font-size: 19px; font-variant-numeric: tabular-nums; }}
  .kpi span {{ color: var(--dim); font-size: 12px; }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 72px; }}
  section {{ margin-bottom: 38px; }}
  h2 {{ font-size: 16px; margin: 0 0 14px; padding-left: 10px;
    border-left: 3px solid var(--accent); }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 14px; }}
  .chart {{ margin: 0; background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 14px 6px; }}
  figcaption {{ font-size: 13px; font-weight: 600; display: flex;
    justify-content: space-between; color: var(--ink); }}
  figcaption span {{ font-weight: 400; color: var(--dim);
    font-family: ui-monospace, Consolas, monospace; font-size: 11px; }}
  svg {{ width: 100%; height: auto; }}
  .grid {{ stroke: #eef1f6; stroke-width: 1; }}
  .curve {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
  circle {{ fill: var(--accent); }}
  .axis {{ fill: var(--dim); font-size: 10px; font-family: ui-monospace, monospace; }}
  .chart-empty {{ color: var(--dim); font-size: 13px; padding: 24px 0; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--line); border-radius: 10px; overflow: hidden; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ background: #eef2f8; font-weight: 600; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  td {{ font-variant-numeric: tabular-nums; }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: 12.5px;
    background: #eef2f7; padding: 1px 5px; border-radius: 4px; color: #a4233a; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
    gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-left-width: 4px;
    border-radius: 10px; padding: 14px 16px; font-size: 13.5px; }}
  .card.ok {{ border-left-color: var(--ok); }}
  .card.no {{ border-left-color: var(--no); }}
  .card.meh {{ border-left-color: var(--meh); }}
  .card.skip {{ border-left-color: var(--skip); }}
  .card h3 {{ margin: 0 0 8px; font-size: 13px; letter-spacing: .4px; color: var(--dim);
    font-family: ui-monospace, Consolas, monospace; }}
  .card p {{ margin: 5px 0; }}
  .card blockquote {{ margin: 6px 0 6px 2px; padding: 2px 0 2px 10px;
    border-left: 2px solid var(--line); color: var(--dim); font-size: 12.5px; }}
  .note {{ background: #fff8e6; border: 1px solid #f0dfae; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #6d5312; }}
  footer {{ color: var(--dim); font-size: 12px; text-align: center; padding-top: 20px; }}
  @media print {{ body {{ background: #fff; }} .card, .chart, table {{ break-inside: avoid; }} }}
</style>
</head>
<body>
<header>
  <h1>SwanLab run 查看器</h1>
  <div class="sub">{html.escape(str(run_dir))}</div>
  <div class="kpis">
    <div class="kpi"><b>{summary.get("windows_processed", len(rows))}</b><span>处理窗口</span></div>
    <div class="kpi"><b>{summary.get("updates_attempted", "-")}</b><span>尝试更新</span></div>
    <div class="kpi"><b>{summary.get("updates_deployed", "-")}</b><span>成功上线</span></div>
    <div class="kpi"><b>{summary.get("updates_rejected", "-")}</b><span>被否决</span></div>
    <div class="kpi"><b>{summary.get("final_score", "-")}</b><span>最终得分</span></div>
    <div class="kpi"><b>{summary.get("final_deployed_score", "-")}</b><span>上线版分数</span></div>
  </div>
</header>
<main>
  <section>
    <h2>曲线层 · {len(rows)} 个窗口，{len(CURVE_COLUMNS)} 条指标</h2>
    <div class="charts">{charts}</div>
  </section>

  <section>
    <h2>窗口明细</h2>
    <table><thead>{table_head}</thead><tbody>{table_rows}</tbody></table>
  </section>

  <section>
    <h2>判断链 · {len(cards)} 条记录</h2>
    <p class="note">绿=假设成立，红=假设被否定，黄=证据不足，灰=本窗未触发更新。
       被否定的候选同样保留——负结果不丢。</p>
    <div class="cards" style="margin-top:14px">{"".join(cards)}</div>
  </section>

  <section>
    <h2>产物层 · 随 run 上传的文件</h2>
    <table><thead><tr><th>文件</th><th>大小</th></tr></thead><tbody>{file_rows}</tbody></table>
  </section>

  <section>
    <h2>summary.json</h2>
    <table><thead><tr><th>字段</th><th>值</th></tr></thead><tbody>{summary_rows}</tbody></table>
  </section>

  <footer>由 tools/build_swanlab_viewer.py 生成，只读本地文件。</footer>
</main>
</body>
</html>
"""

    target = output or (run_dir.parent.parent / f"{run_dir.name}-查看器.html")
    target.write_text(doc, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="把离线 SwanLab run 渲染成单文件 HTML")
    parser.add_argument("run_dir", type=Path, help="形如 runs/<x>/.swanlab/run-2026... 的目录")
    parser.add_argument("-o", "--output", type=Path, help="输出 HTML 路径")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        print(f"目录不存在：{args.run_dir}")
        return 2
    target = build(args.run_dir, args.output)
    print(f"已生成：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
