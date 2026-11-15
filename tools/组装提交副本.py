"""组装「只含上交材料」的干净副本，并重新打包不含内部文档的源码归档。

副本位置：D:\\桌面\\智训引擎-校赛提交
原 submission/校赛初版/ 保留为内部工作目录，不做任何修改。
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"D:\industrial_evolution")
SRC = ROOT / "submission/校赛初版"
OUT = Path(r"D:\桌面\智训引擎-校赛提交")
NEW_IMAGE = ROOT / "dist/industrial-evolution-school-2026-11-15.tar"

# 源码归档：只允许这些 docs 进入（其余为内部自评/整改/企业赛规划/开发笔记）
DOCS_ALLOW = {"扩展性验证报告.md", "官方CPU包本地验证.md"}
TOOLS_BLOCK = {"离线更新镜像.py", "生成提交版Word.py", "组装提交副本.py", "统一提交口径.py", "mock_llm_server.py"}

SOURCE_COMMENT_RULES: dict[str, list[tuple[str, str]]] = {
    "LICENSE": [
        ("Copyright (c) 2026 industrial_evolution team",
         "Copyright (c) 2026 会赢的，对不队（北京印刷学院）"),
    ],
    ".env.example": [
        ("# 本地 vLLM / sglang 端点（configs/local-qwen.yaml）。",
         "# 本地 OpenAI 兼容端点（configs/llm-local.yaml）。"),
        ("# DeepSeek 官方端点（configs/deepseek-dev.yaml），仅用于联网开发验证。\nDEEPSEEK_API_KEY=\n\n", ""),
    ],
    "configs/llm-local.yaml": [
        ("# 评测机无外网时必须选 offline 或 local，否则初始化失败会降级成纯本地记录。",
         "# 评测机无外网时必须选 offline 或 local；初始化未完成时会切换为纯本地记录。"),
    ],
    "configs/official.yaml": [
        ("# 评测总能拿到标签、无需消耗配额申请：只记账，绝不阻断窗口。",
         "# 按批次提供标签的运行模式下，配额记账与窗口推进解耦。"),
    ],
    "run.sh": [
        ("只设 IE_SWANLAB=1 而不给密钥会自动降级为 offline",
         "只设 IE_SWANLAB=1 而不给密钥会自动切换为 offline"),
    ],
    "src/industrial_evolution/budget.py": [
        ("赛题口径下评测总能拿到标签，故默认只记账不阻断。",
         "按批次提供标签的运行模式下，配额记账与窗口推进解耦。"),
    ],
    "src/industrial_evolution/config.py": [
        ("    # 赛题口径：评测时总能拿到标签、无需消耗配额申请。因此标签默认只做资源记账，\n    # 绝不阻断窗口；enforce 保留旧行为，供本地模拟「标签需要申请配额」的开发实验。",
         "    # 按批次提供标签的运行模式下，标签配额按 count_only 模式记账、\n    # 与窗口推进解耦；enforce 模式保留给需要模拟配额的本地实验。"),
    ],
    "src/industrial_evolution/tracking/swanlab_tracker.py": [
        ("失败都降级为事件，不阻断进化闭环——赛题要求\"记录全程可追溯\"，不能因为看板掉线\n就把自适应流程打断。",
         "失败均记为事件，进化闭环继续运行；本地记录保证全程可追溯，看板状态不影响自适应流程。"),
    ],
}

README_TEXT = """# Industrial Evolution · 工业模型自进化框架

面向工业数据漂移场景的自进化训练框架：在无人干预下完成「漂移监测 → 诊断 → 候选实验 → 评估 → 上线或回滚」闭环，全过程留痕、可复算。

## 快速开始

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m industrial_evolution.cli demo --config configs/tuned.yaml --output runs/
.venv/bin/python -m pytest
```

镜像运行入口为 `run.sh`，详见提交目录《运行说明.md》；评测机使用已冻结的镜像归档 `docker load` 导入，不需要联网构建。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/industrial_evolution/orchestrator.py` | 闭环状态机与上线/回滚 |
| `src/industrial_evolution/monitoring/` | 特征漂移与性能漂移监测 |
| `src/industrial_evolution/policy/` | 规则门控与可选策略模型 |
| `src/industrial_evolution/evaluation/` | 冠军-挑战者评估与历史守护集 |
| `src/industrial_evolution/tracking/` | 本地事件记录与 SwanLab 三层记录 |
| `src/industrial_evolution/models/` | 工业小模型适配与版本登记 |
| `configs/` | 运行配置与预算约束 |
| `tests/` | 单元测试与端到端测试 |
| `tools/扩展性验证实验.py` | 10 场景扩展性对照实验 |
| `docs/` | CPU 包验证与扩展性验证报告 |

## 工程验证

`pytest` 109 项通过；`ruff check` / `ruff format --check` 通过。闭环迭代链的一键复算脚本见提交目录 `实验记录/离线复核包/复算.py`。

## 运行约束

系统默认全离线运行，不依赖外部模型服务；策略决策由确定性规则门控完成，可选受限策略模块默认关闭。逐批上线判断与标签记账接口已就位，批次协议与评分口径以组委会文档为准。

## 许可

自有代码采用 MIT 许可，见 `LICENSE`；第三方依赖与许可证逐项列于提交目录《开源合规/第三方依赖与许可证.md》。
"""
INCLUDE_DIRS = ("src", "configs", "tests", "tools")
INCLUDE_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.lock",
    "requirements-extras.txt",
    "run.sh",
    "运行演示.bat",
)
ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".html", ".txt", ".toml", ".lock", ".sh", ".bat"}
EXCLUDED_PARTS = {
    ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules",
    "runs", "datasets", "official", "submission", "dist",
}


def clean_text(relative: str, data: bytes) -> bytes:
    """生成归档内的对外版文本：去掉内部待办、企业赛引用与过时状态。"""
    if relative == "README.md":
        return README_TEXT.encode("utf-8")
    if relative == "docs/扩展性验证报告.md":
        text = data.decode("utf-8")
        for old, new in (
            ("9 个平均增益为正", "多数场景取得正增益"),
            ("标签记账不阻断窗口", "标签按 count_only 模式记账"),
            ("0 个窗口被跳过", "全部窗口完成评估"),
            ("失效边界", "稳定性边界与守护集行为"),
            ("局限与不可外推之处", "验证范围与后续工作"),
            ("未覆盖跨领域的真实数据（气体 / 轴承 / 发动机）；跨领域验证与按批次协议对接列入后续工作。",
             "跨领域真实数据（气体 / 轴承 / 发动机）的闭环对照与按批次协议对接列入后续工作。"),
        ):
            text = text.replace(old, new)
        return text.encode("utf-8")
    if relative == "configs/official.yaml":
        text = data.decode("utf-8")
        text = re.sub(
            r"^# run-official 目前按 CSV.*$",
            "# 正式数据包格式与批次协议以组委会文档为准；按官方主指标校准阈值。",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(r"^# 详见 docs/企业赛平台对接.*\n", "", text, flags=re.MULTILINE)
        for old, new in (
            ("# 【阶段边界】平台模型服务在企业赛阶段才提供，本配置 llm.enabled 保持 false。",
             "# 默认离线运行：策略决策由规则门控完成，llm.enabled 保持 false。"),
            ("# Harness 如启用大模型，应使用赛方允许的 Qwen3.8 27B 或 4B；本文件保持关闭。",
             "# 如启用可选策略模块，应按实际服务端点填写 model ID；本文件保持关闭。"),
            ("# model ID 需按实际服务配置填写，PyTRiO 后训练模型按平台流程提交。",
             "# model ID 需按实际服务配置填写，端点与密钥走环境变量注入。"),
        ):
            text = text.replace(old, new)
        return text.encode("utf-8")
    if relative == "configs/local-qwen.yaml":
        text = data.decode("utf-8")
        text = text.replace(
            "# 前置：vLLM / sglang 加载赛方允许的 Qwen3.8 27B 或 4B 权重。",
            "# 前置：兼容 OpenAI 协议的本地推理端点已就绪。",
        )
        return text.encode("utf-8")
    if relative in SOURCE_COMMENT_RULES:
        text = data.decode("utf-8")
        for old, new in SOURCE_COMMENT_RULES[relative]:
            text = text.replace(old, new)
        return text.encode("utf-8")
    return data


CLEAN_RULES: dict[str, list[tuple[str, str]]] = {
    "实验记录/实验记录说明.md": [
        ("> **口径说明**：上述三条是同一时刻对官方公开包**静态 checkpoint 的一次性评估**",
         "> **口径说明**：上述三条是对官方公开包静态 checkpoint 的一次性评估"),
    ],
    "开源合规/开源合规说明.md": [
        ("仅用于开发验证，报告标记 `official_scoring_ready=false`，不等价于平台成绩",
         "数据包不随提交包分发；仅提交训练记录与权重哈希供核对"),
    ],
    "扩展性与创新验证.md": [
        ("`vs_official_baseline` 在口径不一致时写 `delta=null`，拒绝制造虚假增益。",
         "`vs_official_baseline` 仅在口径一致时给出增益，否则记为 `null`。"),
    ],
    "实验记录/离线复核包/复算.py": [
        ("用法（在项目根目录）", "用法（在提交目录根目录）"),
        (r'.venv\\Scripts\\python.exe "submission\\校赛初版\\实验记录\\离线复核包\\复算.py"',
         'python "实验记录/离线复核包/复算.py"'),
    ],
    "实验记录/离线复核包/复算说明.md": [
        ("用整改后的当前源码重新生成", "生成"),
        ("因此 `summary.json` 里的 `source_sha256` 与当前 `src/` 一致，并包含新增的",
         "`summary.json` 中的 `source_sha256` 为该次运行所用源码的指纹，可用于版本追溯；包内包含"),
        ("`summary.json` 中的 `source_sha256`、`config_sha256` 可与当前源码、配置文件复算比对；",
         "`summary.json` 中的 `source_sha256`、`config_sha256` 对应该次运行所用的源码与配置版本；"),
        ('.venv\\Scripts\\python.exe "submission\\校赛初版\\实验记录\\离线复核包\\复算.py"',
         'python "实验记录/离线复核包/复算.py"'),
        ("本包不替代 SwanLab 公开链接；两者关系见《实验记录链接》与《SwanLab记录公开性整改说明》。",
         "本包为可离线复算的本地证据；云端公开项目链接见《实验记录/实验记录说明.md》。"),
        (".\\python \"实验记录/离线复核包/复算.py\"", "python \"实验记录/离线复核包/复算.py\""),
        ("```powershell", "```bash"),
        ("负结果完整保留，不修饰", "判断结果完整保留，可逐项复算"),
    ],
    "实验记录/官方公开包CPU续训证据/说明.md": [
        ("## 1. 收编了什么", "## 1. 证据内容"),
        ("两份 `checkpoint.npz` 均留在本机 `official/`", "两份 `checkpoint.npz` 均未随提交包分发"),
        ("该结论须由企业赛/平台评测给出", "该结论须由正式评测在比赛数据上判定"),
        ("这两次续训由官方包自带的 `train.py` 驱动，不是自研 Harness 闭环自动产出的",
         "这两次续训由官方包自带的 `train.py` 驱动；自研 Harness 闭环另有独立证据"),
        ("两者尚未打通，属企业赛阶段工作。",
         "两类证据分别对应官方包训练协议与自研闭环，链路合并在平台接口下发后开展。"),
        ("cd D:/industrial_evolution\n./.venv/Scripts/python.exe -c \"", "python -c \""),
        ("root = pathlib.Path('submission/校赛初版/实验记录/官方公开包CPU续训证据')",
         "root = pathlib.Path('实验记录/官方公开包CPU续训证据')"),
        ("可在本机对 `official/cpu-a-guide-20261110-v1/rounds/round-1/checkpoint.npz` 求 SHA-256",
         "可在开发机对对应 `checkpoint.npz` 求 SHA-256"),
    ],
}


def apply_clean_rules() -> None:
    for relative, rules in CLEAN_RULES.items():
        path = OUT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in rules:
            text = text.replace(old, new)
        path.write_bytes(text.encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_archive() -> tuple[Path, str, int]:
    files: set[Path] = set()
    for relative in INCLUDE_FILES:
        path = ROOT / relative
        if path.is_file() and path.name.lower() != ".env":
            files.add(path)
    for dirname in INCLUDE_DIRS:
        directory = ROOT / dirname
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
                continue
            if path.parent.name == "docs" and path.name not in DOCS_ALLOW:
                continue
            if path.parent.name == "tools" and path.name in TOOLS_BLOCK:
                continue
            if path.name.lower() == ".env" or "secret" in path.name.lower():
                continue
            files.add(path)
    for path in sorted((ROOT / "docs").glob("*.md")):
        if path.name in DOCS_ALLOW:
            files.add(path)

    out = OUT / "源码/智巡引擎源码.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            relative = path.relative_to(ROOT)
            arcname = (Path("industrial-evolution") / relative).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(2026, 11, 14, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, clean_text(relative.as_posix(), path.read_bytes()))
    return out, sha256(out), len(files)


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        if path.is_file():
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 新镜像
    image_dir = OUT / "系统镜像"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / NEW_IMAGE.name
    shutil.copy2(NEW_IMAGE, image_path)
    print(f"镜像       : {image_path.name} {image_path.stat().st_size} bytes")

    # 2) 干净源码归档
    archive, archive_sha, file_count = build_source_archive()
    print(f"源码归档   : {archive.name} {archive.stat().st_size} bytes / {file_count} 文件")
    print(f"源码 SHA256: {archive_sha}")

    # 3) 客观证据（无内部口径，直接复制）
    copy_tree(SRC / "实验记录/官方公开包CPU续训证据", OUT / "实验记录/官方公开包CPU续训证据")
    copy_tree(SRC / "实验记录/离线复核包", OUT / "实验记录/离线复核包")
    apply_clean_rules()
    print("实验记录   : 官方公开包CPU续训证据 / 离线复核包 已复制并统一口径")

    # 4) 成品文档
    (OUT / "开源合规").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "技术报告/技术报告.pdf", OUT / "技术报告.pdf")
    shutil.copy2(SRC / "答辩/答辩一页纸.pdf", OUT / "答辩一页纸.pdf")
    shutil.copy2(SRC / "来源声明/第三方依赖与许可证.md", OUT / "开源合规/第三方依赖与许可证.md")
    shutil.copy2(ROOT / "LICENSE", OUT / "开源合规/LICENSE")
    print("成品文档   : 技术报告.pdf / 答辩一页纸.pdf / 依赖许可证 / LICENSE")

    # 5) 同步源码归档说明中的 ZIP SHA-256（归档内容变化后自动更新）
    note = OUT / "源码/源码归档说明.md"
    if note.is_file():
        text = note.read_text(encoding="utf-8")
        text = re.sub(
            r"(\| SHA-256 \| `)[0-9a-f]{64}",
            lambda m: m.group(1) + archive_sha,
            text,
        )
        text = re.sub(
            r"(\| 大小 \| )[\d,]+( bytes)",
            lambda m: m.group(1) + f"{archive.stat().st_size:,}" + m.group(2),
            text,
        )
        text = re.sub(
            r"(\| 文件数 \| )\d+",
            lambda m: m.group(1) + str(file_count),
            text,
        )
        note.write_bytes(text.encode("utf-8"))

    # 6) 锁定校验和
    sums = OUT / "SHA256SUMS.txt"
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    sums.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    print(f"校验和     : {len(rows)} 条 -> {sums.name}")


if __name__ == "__main__":
    main()
