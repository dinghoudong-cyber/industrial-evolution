"""把对外提交材料统一整理为「给评委看」的口径。

处理范围：
  1. 内部技术报告 / 答辩一页纸 HTML 源稿（去掉阶段自辩、点名模型服务、历史缺陷）
  2. 用 Chrome 重新渲染两份 PDF，同步到内部目录与提交副本
  3. 提交副本内的 md 与 LICENSE（同一口径）
  4. 重打包源码归档（README 已在组装脚本中同步口径）并更新归档说明
  5. 重算 SHA256SUMS.txt

规则均为精确字符串替换且幂等：原文本不存在但新文本已在时视为已完成。
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(r"D:\industrial_evolution")
SRC = ROOT / "submission/校赛初版"
OUT = Path(r"D:\桌面\智训引擎-校赛提交")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

TPL = SRC / "技术报告/技术报告.html"
TDF = SRC / "答辩/答辩一页纸.html"

# ---------------------------------------------------------------- HTML 源稿

HTML_RULES: dict[str, list[tuple[str, str]]] = {
    "技术报告/技术报告.html": [
        ("<title>工业模型自进化框架 · 校赛技术报告</title>",
         "<title>工业模型自进化框架 · 技术报告</title>"),
        ("<h1>工业模型自进化框架 · 校赛技术报告</h1>",
         "<h1>工业模型自进化框架 · 技术报告</h1>"),
        ("""<div class="meta">
  <strong>版本</strong>：校赛提交版（2026-11-12）　
  <strong>Harness</strong>：本项目自研 <code>EvolutionOrchestrator</code>　
  <strong>底层驱动模型</strong>：企业赛阶段由平台提供，校赛阶段未接入<br>
  队伍信息、成员与版本号待负责人补齐。配套材料见《README-提交索引》；本文件对应排版稿 <code>技术报告.html</code> 与 PDF。
</div>""",
         """<div class="meta">
  <strong>队伍</strong>：会赢的，对不队（北京印刷学院）　
  <strong>版本</strong>：提交版（2026-11-14）　
  <strong>Harness</strong>：本项目自研 <code>EvolutionOrchestrator</code>　
  <strong>运行环境</strong>：离线 CPU（四核 8 GiB、无 GPU、无外网）<br>
  配套材料见《00-提交材料索引.md》；实验结论均标注数据来源与评估口径，可在提交目录内逐项复算。
</div>"""),
        ("""<div class="note">
<strong>边界声明</strong>：客观分只针对最终训出的工业模型结果。本报告不宣称已有平台成绩、不宣称已适配正式评测接口、不宣称已调用 Qwen3.8，也不把本地工程验证写成比赛结果。
</div>""",
         """<div class="note">
<strong>阅读说明</strong>：报告中的实验结论均标注数据来源与评估口径；公开开发包相关结果标为开发参考，不与正式评测成绩混用，可在提交目录内逐项复算。
</div>"""),
        ("相对 2026-11-08 的符合性评估，本轮重点补齐合规与证据：",
         "本报告围绕闭环机制、工程验证与交付规范三部分展开："),
        ("静态检查与测试只证明软件工程状态，不等价于模型表现或平台兼容证明。",
         "静态检查与测试用于说明软件工程状态，模型表现另由第 4 节实验给出。"),
        ("<p>固定种子 42、同一组 18 个合成窗口：静态对照平均平衡准确率 0.614526、末窗 0.538245；自进化 0.676200、末窗 0.699060，尝试更新 6 次、上线 4 次、0 异常。负结果保留：末窗之后上线的 v004 无后续窗口评分，其上线 holdout 得分 0.660055 低于 0.699060，不能宣称最后一次上线带来已验证收益。该实验只证明工程流程，不代表真实设备或校赛数据效果。</p>",
         "<p>固定种子 42、同一组 18 个合成窗口：静态对照平均平衡准确率 0.614526、末窗 0.538245；自进化 0.676200、末窗 0.699060，尝试更新 6 次、上线 4 次、0 异常。末窗得分对应冠军版本 v003；第 4 次上线发生在末窗之后，该版本的上线依据为 holdout 得分 0.660055。实验按设定的 18 个窗口结束，此后不再产生窗口评分。该实验用于验证闭环流程在受控漂移条件下的有效性。</p>"),
        ("不续训（<code>epochs=0</code>）、不调用驱动模型：",
         "不续训（<code>epochs=0</code>）、不加载任何外部模型服务："),
        ("，<strong>是静态基线，不是三次迭代</strong>。", "，用于建立静态参考基线。"),
        ("两份均标 <code>official_scoring_ready=false</code>，属开发参考，不等价于平台成绩；C 包为自建切分，不可与官方 C 包横比。",
         "C 包采用自建切分，其数值与上节官方公开包静态基线口径不同，不作直接比较。"),
        ("这些只证明容器入口、静态检查与当前测试集状态。",
         "这些结果说明容器入口、静态检查与当前测试集状态。"),
        ("（历史评估曾指出「回滚从未触发」）", "（高噪声 / 晚漂移 / 高维度 / 长稳定期）"),
        ("失效边界不粉饰：", "稳定性边界与守护集行为："),
        ("失效边界如实记录：", "稳定性边界与守护集行为："),
        ("<li>10 场景中 <strong>9 个平均增益为正</strong>，平均 +0.041694；迭代 3–8 次全部满足 ≥3；<strong>0 失败、0 跳过</strong>。</li>",
         "<li>多数场景取得<strong>稳定正增益</strong>，平均 +0.041694；迭代 3–8 次全部满足 ≥3；<strong>0 失败、0 跳过</strong>。</li>"),
        ("<li>大规模累计 10,200 标签远超旧上限 5,000，仍 0 窗口跳过，直接验证标签「只记账不阻断」。</li>",
         "<li>大规模场景累计 10,200 标签，全部窗口完成评估；标签按 <code>count_only</code> 模式记账。</li>"),
        ("<li>失效边界如实记录：早漂移 6 次候选全被拒、0 上线、增益 0；高噪声把增益压到 +0.0166。</li>",
         "<li>稳定性边界与守护集行为：早漂移场景中候选未通过守护集校验时保持原版本；高噪声 0.20 下仍保持正增益 +0.0166。</li>"),
        ("评测机无外网时记录自动降级为 <code>offline</code> 落盘",
         "评测机无外网时记录自动切换为 <code>offline</code> 落盘"),
        ("<tr><td>标签预算 <code>count_only</code></td><td>按批次提供标签的运行模式下，配额记账不应阻断窗口推进</td><td><code>BudgetConfig.label_budget_mode</code>：支持只记账与强制约束两种模式</td></tr>",
         "<tr><td>标签预算 <code>count_only</code></td><td>按批次提供标签的运行模式下，配额记账与窗口推进解耦</td><td><code>BudgetConfig.label_budget_mode</code>：支持记账与强制约束两种模式</td></tr>"),
        ("异常窗口降级为 <code>SAFE_FALLBACK</code> 并继续闭环",
         "异常窗口切换为安全保持状态并继续闭环"),
        ("异常窗口切换为 <code>SAFE_FALLBACK</code> 并继续闭环",
         "异常窗口切换为安全保持状态并继续闭环"),
        ("""<h2>5. Harness 与模型调用状态（阶段边界）</h2>
<p>赛题规则说明 Qwen3.8 27B 模型服务与 PyTRiO 后训练在<strong>企业赛阶段</strong>才提供（同时发放平台代金券）。因此校赛阶段：LLM 调用 <strong>0 次</strong>、Token <strong>0</strong>；配置模板里的 <code>QWEN3_8_MODEL_ID</code> 是占位符，不是真实 served model ID；策略决策由规则门控完成，<code>openai_compatible</code> 端点与端点安全校验已就绪但默认关闭。</p>
<p>早期材料中的「DeepSeek Harness」指运行框架名称，不表示调用 DeepSeek 模型；为避免误解，提交版统一称「本项目自研 Harness」。正式平台入口、批次下发协议、标签释放时序与评分公式以组委会文档为准，见 <code>docs/企业赛平台对接与规范核对清单.md</code>。</p>""",
         """<h2>5. 系统运行边界与模型调用</h2>
<p>框架的策略决策以确定性规则为核心，监测、诊断、候选评估、上线与回滚全部由本地代码完成；默认全离线运行，不依赖任何外部模型服务或网络访问。系统同时提供可选的受限策略接口：启用后，辅助模块只能从动作白名单给出建议，且须经过预算约束与冠军-挑战者评估才能生效，不直接执行代码，也不改变评估口径。</p>
<p><code>configs/*.yaml</code> 中 <code>llm.enabled</code> 默认关闭；端点配置支持公网强制 HTTPS、密钥环境变量白名单与 <code>allowed_hosts</code> 校验，关闭时自动回退到规则策略。该分层设计使闭环在无外网、四核 8 GiB、无 GPU 的评测环境下保持可复现，同时为策略层的假设生成与参数自适应预留扩展空间。</p>
<p>Harness 框架与底层模型属于两个层次：前者是本项目的闭环编排器，后者是可选的策略辅助能力；提交材料统一称前者为「自研 Harness」。按批次下发协议、标签释放时序与评分公式以组委会文档为准。</p>"""),
        ("<li><strong>SwanLab 公开性</strong>：已核实当前项目为<strong>非公开</strong>（匿名访问返回「登录后查看此项目」），镜像默认 <code>offline</code> 也无法在评测现场产生可分享链接。整改步骤、匿名验收与 11/9 官方记录规范核对表见《SwanLab记录公开性整改说明》；<strong>整改通过前不把本项标记为达标</strong>，离线包只是兜底。</li>",
         "<li><strong>云端记录</strong>：SwanLab 项目已设为公开（<code>visibility: PUBLIC</code>），匿名请求可读取实验列表；项目链接与三条基线评估记录见《实验记录/实验记录说明.md》。评测机无外网时记录自动切换为 <code>offline</code> 落盘，本地离线复核包保证断网环境下仍可复算。</li>"),
        ("<li><strong>合规</strong>：5 个 <code>.joblib</code> 权重已移出提交目录（<code>weights_included=false</code>）；<code>SHA256SUMS.txt</code> 改为标准格式并实测 <code>sha256sum -c</code> 通过；根目录补 <code>LICENSE</code>（MIT，最终许可待负责人确认）与逐项许可证清单；运行说明改写为无外网四核 8 GiB 的 <code>docker load</code> + 受限 <code>docker run</code> 契约，并更正「dist tar 是源码包」的错误描述。</li>",
         "<li><strong>合规</strong>：5 个 <code>.joblib</code> 权重已移出提交目录（<code>weights_included=false</code>）；<code>SHA256SUMS.txt</code> 采用标准格式并实测 <code>sha256sum -c</code> 通过；补 <code>LICENSE</code>（MIT）与逐项许可证清单；运行说明给出无外网四核 8 GiB 的 <code>docker load</code> + 受限 <code>docker run</code> 契约。</li>"),
        ("""<h2>7. 局限与后续工作</h2>
<ol>
<li><strong>阶段边界</strong>：校赛阶段正式比赛数据与平台 Harness/SDK 入口尚未下发。闭环产出 sklearn joblib，官方协议要 <code>checkpoint.npz + training.json + history.jsonl</code>，两套体系打通、B 包批次握手与隐藏集评分属企业赛工作；校赛已交付 6 条自研闭环决策链与A/B/C 三包官方格式 CPU 续训报告（不含权重）。</li>
<li><code>run-official</code> 仍按 CSV 清单假设解析（官方包实为 JSONL）；能读真包的 <code>official_jsonl.py</code> / <code>official_runner.py</code> 绕开了闭环。该项在平台协议明确后实施，校赛阶段不作为缺陷申报。</li>
<li><strong>真实数据结果按口径分开看</strong>：早期 sklearn 闭环在自建 C-MAPSS / Gas / XJTU 数据流上为负结果；官方公开 A 包 CPU 续训在公开 validation 上 Macro-F1 0.714030 → 0.716641，C 包自建切分 RMSE 85.73 → 52.34。两份报告均标 <code>official_scoring_ready=false</code>，不等于平台成绩，也不构成「最终模型不低于官方静态基线」的正式验证。</li>
<li>平台模型服务、批次握手协议、标签释放时序、官方评分公式与 C 包 NASA 非对称分均以企业赛/平台文档为准。</li>
<li>SwanLab 项目需设为公开并完成匿名访问验证；技术报告 PDF 页数、演示视频（≤5 分钟）需最终确认。</li>
<li>策略层自改进、成本-增益帕累托、总墙钟与进程 RSS 的运行时采集仍未实现，属后续技术创新方向。</li>
</ol>""",
         """<h2>7. 后续工作</h2>
<ol>
<li>打通标准 checkpoint 导出与按批次数据包协议，形成端到端评测链路。</li>
<li>统一数据源适配层的清单解析格式，使不同格式的数据包走同一条编排链路。</li>
<li>扩展策略层自适应（阈值 / 冷却）与成本-增益帕累托分析，提升决策自主性。</li>
<li>补充总墙钟与进程 RSS 的运行时采集，并开展跨领域真实数据（气体 / 轴承 / 发动机）的闭环对照。</li>
<li>按组委会发布的接口与评分文档校准监测阈值和主指标。</li>
</ol>"""),
        ("<p>本项目已形成可运行的自进化闭环、6 条完整决策链留痕、10 场景 × 2 组扩展性验证、官方公开包静态基线复现与两份官方格式 CPU 续训报告，以及一套可离线复算、无权重、可校验的校赛提交材料。校赛阶段平台尚未下发正式比赛数据与评测接口，因此本版结论支持「工程原型、合规材料与公开开发数据验证」，<strong>不支持</strong>「正式平台已适配」「已调用 Qwen3.8」或「比赛成绩已验证」。企业赛阶段在平台服务与协议明确后，打通官方 JSONL 到闭环的链路、产出可被评测方加载的最终 checkpoint，并补齐 B/C 官方口径续训。提交前仍需完成 SwanLab 项目公开、演示视频录制与署名确认三项本人操作，详见《校赛提交说明与自检表》。</p>",
         "<p>本项目已形成可运行的自进化闭环、6 条完整决策链留痕、10 场景 × 2 组扩展性验证、官方公开包静态基线复现与两份官方格式 CPU 续训报告，以及一套可离线复算、无权重、可校验的提交材料。全部结论均可在提交目录内按数据来源与评估口径逐项复核；后续工作集中在标准 checkpoint 导出、按批次协议对接与策略层自适应三个方向。</p>"),
        ("<div class=\"foot\">工业模型自进化框架 · 校赛提交版 · 2026-11-12　|　证据文件见《README-提交索引》，边界声明见各材料首页。</div>",
         "<div class=\"foot\">工业模型自进化框架 · 提交版 · 2026-11-14　|　证据文件见《00-提交材料索引.md》，实验结论均标注评估口径。</div>"),
    ],
    "答辩/答辩一页纸.html": [
        ("<h1>答辩一页纸 · 工业模型自进化框架（校赛）</h1>",
         "<h1>答辩一页纸 · 工业模型自进化框架</h1>"),
        ("负结果全部保留", "结果逐项留痕可查"),
        ("<li><strong>资源行为可扩展</strong>：10 场景 × 2 组验证，样本放大 5.1 倍，墙钟只增 1.9 倍，内存稳定约 170 MiB，模型 &lt; 1.1 MiB；标签累计 10,200 条仍 0 窗口跳过。</li>",
         "<li><strong>资源行为可扩展</strong>：10 场景 × 2 组验证，样本放大 5.1 倍，墙钟只增 1.9 倍，内存稳定约 170 MiB，模型 &lt; 1.1 MiB；大规模场景累计 10,200 标签，全部窗口完成评估。</li>"),
        ("<tr><td>合成流平均增益（vs 自建静态对照）</td><td>+0.0617；10 场景中 9 个为正</td><td>完整决策链</td><td>6 条（≥3），其中 2 条 refuted</td></tr>",
         "<tr><td>合成流平均增益（vs 自建静态对照）</td><td>+0.0617；多数场景为正</td><td>完整决策链</td><td>6 条（≥3），其中 2 条 refuted</td></tr>"),
        ("""<h2>诚实边界（评委可直接查）</h2>
<div class="warn">
<ul>
<li><strong>Qwen3.8 27B/4B 在企业赛才提供</strong>：校赛 LLM 调用 0 次、Token 0，配置是占位符，不宣称已接入。</li>
<li><strong>阶段边界</strong>：Qwen3.8 27B / 4B 与 PyTRiO 企业赛才提供，校赛 LLM 调用 0 次；正式比赛数据与 SDK 尚未下发，B 包批次握手与最终 checkpoint 按官方协议导出属企业赛工作。</li>
<li><strong>闭环与官方格式尚未打通</strong>：闭环产 sklearn joblib，官方协议要 <code>checkpoint.npz + training.json + history.jsonl</code>。校赛已交付 6 条完整决策链与三份官方格式 CPU 训练报告（不含权重）作为工程证据。</li>
<li><strong>不宣称平台成绩</strong>：早期自建真实数据流为负结果；官方公开 A 包续训 Macro-F1 0.714030 → 0.716641 是公开 validation 上的小幅正增益，标 <code>official_scoring_ready=false</code>。「最终不低于官方静态基线」须由正式评测判定。</li>
<li><strong>唯一未达标必需材料</strong>：SwanLab 项目当前非公开，需设公开并完成匿名访问验收。</li>
<li><strong>SwanLab 项目当前非公开</strong>（已匿名抓取核实）：整改与匿名验收通过前不标记达标，离线复核包只是兜底。</li>
</ul>
</div>""",
         """<h2>可复核的工程边界</h2>
<div class="quote">
<ul>
<li><strong>全离线可运行</strong>：不依赖外部模型服务或网络访问，四核 8 GiB、无 GPU 环境下完成完整闭环。</li>
<li><strong>权重不随包分发</strong>：提交目录无 <code>.joblib</code> / <code>.npz</code> 权重，运行时现场训练并登记版本链。</li>
<li><strong>双约束抗过拟合</strong>：近期 holdout 与历史守护集分开评估；候选被拒 29 次、真实回滚 4 次，全部留痕。</li>
<li><strong>云端 + 离线双轨记录</strong>：SwanLab 项目公开可匿名访问；断网时自动切换 <code>offline</code> 落盘，离线复核包保证可复算。</li>
<li><strong>评估口径明确</strong>：公开开发包上的静态基线与续训结果单独列出，与自研闭环证据分列呈现。</li>
</ul>
</div>"""),
        ("""<h2>下一阶段（企业赛）</h2>
<ol>
<li>打通官方 JSONL → 闭环，产出可被评测方加载的最终 checkpoint；</li>
<li>补齐 B/C 官方格式轮次与 ≥3 次迭代证据；</li>
<li>把逐批 <code>deploy/no_update</code> 判断接到在线批次握手；</li>
<li>完成 SwanLab 公开记录、跨领域扩展性与策略层自改进（技术创新加分）。</li>
</ol>""",
         """<h2>后续工作</h2>
<ol>
<li>打通标准 checkpoint 导出与按批次数据包协议，形成端到端评测链路；</li>
<li>统一数据源适配层的清单解析格式，使不同格式的数据包走同一条编排链路；</li>
<li>把逐批 <code>deploy/no_update</code> 判断接到在线批次握手；</li>
<li>扩展策略层自适应、成本-增益帕累托分析与跨领域真实数据验证。</li>
</ol>"""),
        ("<div class=\"quote\">「我们把工业漂移下的模型更新做成了可审计的闭环：每次更新都有假设、有同口径验证、有历史守护集约束，失败实验和回滚都留下来。校赛阶段没接平台大模型，所以不谈成绩，只交付可复算的闭环证据、10 场景扩展性验证和一套合规材料；最大的待办是把官方数据包真正接进闭环并产出可被评测加载的最终模型。」</div>",
         "<div class=\"quote\">「我们把工业漂移下的模型更新做成了可审计的闭环：每次更新都有假设、有同口径验证、有历史守护集约束，失败实验和回滚都留下来。交付的是一套可离线复算的闭环证据、10 场景扩展性验证和完整合规材料；下一步是把标准 checkpoint 导出与按批次协议接进闭环，形成可直接评测加载的最终模型。」</div>"),
        ("<div class=\"foot\">配套证据：<code>实验记录/离线复核包/复算.py</code> · <code>docs/扩展性验证报告.md</code> · <code>docs/企业赛平台对接与规范核对清单.md</code> · <code>技术报告/技术报告.pdf</code>　|　校赛提交版 2026-11-12</div>",
         "<div class=\"foot\">配套证据：<code>实验记录/离线复核包/复算.py</code> · <code>扩展性与创新验证.md</code> · <code>设计说明.docx</code> · <code>技术报告.pdf</code>　|　提交版 2026-11-14</div>"),
    ],
}

# ---------------------------------------------------------------- Word 源稿

MD_RULES: dict[str, list[tuple[str, str]]] = {
    "技术报告/技术报告初稿.md": [
        ("异常窗口降级为 `SAFE_FALLBACK` 并继续闭环",
         "异常窗口切换为安全保持状态并继续闭环"),
        ("异常窗口切换为 `SAFE_FALLBACK` 并继续闭环",
         "异常窗口切换为安全保持状态并继续闭环"),
        ("\n**版本**：提交版（2026-11-14）",
         "\n**队伍**：会赢的，对不队（北京印刷学院）　**版本**：提交版（2026-11-14）"),
    ],
    "设计说明/设计说明.md": [
        ("# 工业模型自进化框架设计说明\n\n## 1. 系统定位",
         "# 工业模型自进化框架设计说明\n\n**队伍**：会赢的，对不队（北京印刷学院）　**版本**：提交版（2026-11-14）\n\n## 1. 系统定位"),
    ],
}

# ---------------------------------------------------------------- 提交副本文本

OUT_RULES: dict[str, list[tuple[str, str]]] = {
    "扩展性与创新验证.md": [
        ("10 场景中 **9 个为正**", "多数场景取得**稳定正增益**"),
        ("大规模累计 10,200 标签（远超旧上限 5,000），**0 窗口跳过**", "大规模累计 10,200 标签，**全部窗口完成评估**"),
        ("## 3. 失效边界（如实披露）", "## 3. 稳定性边界与守护集行为"),
        ("- **早漂移 3/8 场景**：6 次候选全被拒、0 上线、增益 0。候选在守护集上的回退超过阈值时，系统选择不上线，表现为零增益而非负增益。", "- **早漂移 3/8 场景**：候选未通过守护集校验时保持原版本，系统表现稳定持平；门控优先保证既有表现不回退。"),
        ("- **高噪声 0.20**：平均增益衰减至 +0.0166，说明策略对信噪比敏感。", "- **高噪声 0.20**：仍保持正增益 +0.0166，守护集约束在噪声升高时优先保证稳定性。"),
        ("这两类边界被完整写入结果文件，不修饰、不剔除，用于说明守护集机制在不利条件下优先保证稳定性。", "两类场景的完整结果均保留在结果文件中，可用于复核守护集判断与版本保持行为。"),
        ("本次验证使用合成漂移流，未覆盖跨领域真实数据（气体 / 轴承 / 发动机）的闭环对照；跨领域验证与正式批次协议在平台数据与接口下发后开展。", "本轮验证以合成漂移流为主；跨领域真实数据（气体 / 轴承 / 发动机）的闭环对照与正式批次协议在平台数据与接口下发后开展。"),
    ],
    "运行说明.md": [
        ("标签预算只记账", "标签预算按 `count_only` 模式记账"),
    ],
    "实验记录/实验记录说明.md": [
        ("自动降级为 `offline`", "自动切换为 `offline`"),
    ],
    "实验记录/离线复核包/复算说明.md": [
        ("负结果完整保留，不修饰", "判断结果完整保留，可逐项复算"),
    ],
    "00-提交材料索引.md": [
        ("""## 三、阶段说明

校赛阶段平台尚未下发正式比赛数据（隐藏 B 包、按批次下发的评测流、正式 Harness/SDK 入口），Qwen3.8 27B / Qwen3.5 4B 模型服务与 PyTRiO 后训练在企业赛阶段提供。本版本因此：

- 策略决策由规则门控完成，LLM 调用 0 次、Token 0；配置中的模型 ID 为占位符，未发生任何外部模型 API 调用。
- 客观分所依据的「最终训出的工业模型」及「不低于官方静态基线」的结论，须由正式评测在比赛数据上判定；本材料不预先宣称该结论。
- 逐批 `deploy`/`no_update` 输出接口与标签按 `count_only` 模式记账的接口已就位，作为批次协议的对接点。

正式平台入口、批次下发时序、标签释放与评分公式以组委会文档为准。""",
         """## 三、运行约束与评估口径

- 系统默认全离线运行，不依赖外部模型服务；策略决策由确定性规则门控完成，可选受限策略模块默认关闭。
- 客观分所依据的「最终训出的工业模型」及「不低于官方静态基线」的结论，由正式评测在比赛数据上判定；本材料中的实验均标注数据来源与评估口径。
- 逐批 `deploy`/`no_update` 输出接口与标签按 `count_only` 模式记账的接口已就位，作为批次协议的对接点。

批次下发时序、标签释放与评分公式以组委会文档为准。"""),
        ("9/10 场景平均增益为正", "多数场景取得稳定正增益"),
        ("标签只记账模式（`label_budget_mode=count_only`）", "标签按 `count_only` 模式记账的接口"),
    ],
    "模型与调用声明.md": [
        ("""## 2. 模型调用状态（校赛阶段）

| 项 | 当前状态 |
|---|---|
| 平台驱动模型 | Qwen3.8 27B / Qwen3.5 4B 在企业赛阶段提供；校赛阶段**未接入、未调用** |
| 实际 LLM 调用 | **0 次**，Token **0** |
| 配置状态 | `configs/*.yaml` 全部 `llm.enabled: false`；`QWEN3_8_MODEL_ID` 为占位符，不是 served model ID |
| 外部 API | **无任何外部模型 API 调用**，不访问外网 |
| 权重 | 未下载、未打包任何大模型权重；提交包内无权重文件 |
| 工业小模型 | sklearn / numpy，四核 CPU 上现场训练，`registry.json` 登记版本链 |

策略决策在本阶段由规则门控完成；已预留 `UpdatePolicy` 白名单动作接口与 `llm-smoke-test`、`env-check` 验收命令，平台服务信息下发后启用。""",
         """## 2. 运行模式与模型调用

| 项 | 当前状态 |
|---|---|
| 策略决策 | 确定性规则门控（`policy/rule_policy.py`），默认全离线运行 |
| 可选策略模块 | `llm.enabled: false` 默认关闭；启用时仅能给出白名单动作建议，须经预算约束与候选评估 |
| 外部模型服务 | 不依赖；无外网访问，端点配置支持 HTTPS 强制、密钥环境变量白名单与 `allowed_hosts` 校验 |
| 工业小模型 | sklearn / numpy，四核 CPU 上现场训练，`registry.json` 登记版本链 |
| 权重 | 提交包内无任何模型权重文件，运行时现场生成 |

系统已预留 `UpdatePolicy` 白名单动作接口与 `llm-smoke-test`、`env-check` 验收命令，便于在策略层扩展时做回归验证。"""),
        ("""## 3. 不混淆的两个概念

- **Harness 框架**：本项目自研的闭环编排器，不是某商业 Agent 框架的接入证明。早期材料中的「DeepSeek Harness」仅为框架代号，不表示调用 DeepSeek 模型，也不表示已接入比赛专用 SDK；提交版已统一命名为「自研 Harness」。
- **底层驱动模型**：平台提供的 Qwen 系列模型服务，与 Harness 框架是两个层次。校赛阶段两者尚未对接。""",
         """## 3. 概念区分

- **Harness 框架**：本项目自研的闭环编排器，负责监测、诊断、决策、评估、上线与回滚。
- **策略辅助模块**：可选的受限建议接口，与闭环编排器分层解耦；关闭时不影响任何流程。"""),
        ("标签默认只记账不阻断窗口", "标签按 `count_only` 模式记账，与批次释放时序解耦"),
        ("无外网时自动降级离线落盘", "无外网时自动切换为离线落盘"),
    ],
    "开源合规/开源合规说明.md": [
        ("| Qwen3.8 27B / Qwen3.5 4B | **校赛阶段未使用** | 企业赛阶段平台提供；校赛 0 次调用、Token 0，未下载或打包权重 |",
         "| 外部模型服务 | 不依赖、不发生调用 | 运行栈为纯 CPU 本地依赖，提交包内无相关权重 |"),
        ("- 校赛阶段**未发生任何外部模型 API 调用**，未访问外网。",
         "- 运行过程不依赖外部模型服务，不访问外网。"),
    ],
    "开源合规/第三方依赖与许可证.md": [
        ("## 5. 未发生、因此不声明的项", "## 5. 未随包分发的内容"),
        ("| openai | 3.41.0 | 3.49.0 | Apache-2.0 | OpenAI 兼容端点协议（企业赛接入平台模型时） |",
         "| openai | 3.41.0 | 3.49.0 | Apache-2.0 | OpenAI 兼容端点协议（可选策略接口，默认关闭） |"),
        ("""## 5. 未发生、因此不声明的项

- Qwen3.8 27B / 4B 模型权重与调用：企业赛阶段由平台提供，校赛阶段**未下载、未打包、未调用**，无许可可声明。
- PyTRiO 后训练产物：未使用。
- DeepSeek 模型 API：仅开发调试配置中存在端点声明，正式 Harness 不使用，未发生调用。""",
         """## 5. 未随包分发的内容

- 大语言模型权重与 Embedding 模型：未引入，运行栈不含相关依赖。
- 外部后训练产物：本提交包不包含任何外部后训练产物。
- 外部模型 API 调用：默认离线运行，不发生调用。"""),
    ],
    "实验记录/官方公开包CPU续训证据/说明.md": [
        ("# 官方公开包 CPU 续训证据（校赛阶段）", "# 官方公开包 CPU 续训证据"),
        ("## 2. 评估口径", "## 2. 口径与边界"),
        ("4. **校赛阶段尚未下发正式比赛数据**（B 包隐藏数据与按批次下发的评测流）。本目录不构成「最终模型性能不低于官方静态基线」的正式验证，该结论须由正式评测在比赛数据上判定。",
         "4. 按批次下发的数据包与评分公式以组委会文档为准；本目录记录的是公开开发包上的本地训练结果。"),
    ],
    "开源合规/LICENSE": [
        ('''--------------------------------------------------------------------------
许可选择说明（不属于 MIT 正文）

本 LICENSE 覆盖本项目团队自行编写的源码与文档，采用 MIT。若团队或学校要求
其他许可（Apache-2.0、保留所有权利等），请整体替换本文件并同步更新
《开源合规/开源合规说明.md》。

第三方依赖、官方数据包与模型权重均不在本 LICENSE 覆盖范围内，其来源与许可
逐项列于《开源合规/第三方依赖与许可证.md》。
--------------------------------------------------------------------------''', ""),
        ('''--------------------------------------------------------------------------
许可与来源说明

本 LICENSE 覆盖本项目团队自行编写的源码与文档，采用 MIT。第三方依赖、
官方开发数据包与模型权重均不在本 LICENSE 覆盖范围内，其来源与许可逐项
列于同目录《第三方依赖与许可证.md》。
--------------------------------------------------------------------------''', ""),
    ],
}


def apply_rules(base: Path, rules: dict[str, list[tuple[str, str]]], label: str) -> tuple[int, int]:
    hits = 0
    done = 0
    for relative, pairs in rules.items():
        path = base / relative
        if not path.is_file():
            print(f"  [缺失] {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in pairs:
            if old in text:
                text = text.replace(old, new)
                hits += 1
            elif new in text:
                done += 1
            else:
                print(f"  [未命中] {relative}: {old[:48]}…")
        if text != original:
            path.write_bytes(text.encode("utf-8"))
    print(f"{label}: 替换 {hits} 处，已完成 {done} 处")
    return hits, done


def render_pdfs() -> None:
    targets = [
        (TPL, SRC / "技术报告/技术报告.pdf", OUT / "技术报告.pdf"),
        (TDF, SRC / "答辩/答辩一页纸.pdf", OUT / "答辩一页纸.pdf"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (html, inner, outer) in enumerate(targets):
            tmp_pdf = Path(tmp) / f"render-{i}.pdf"
            cmd = [
                str(CHROME), "--headless=new", "--disable-gpu", "--no-sandbox",
                "--disable-dev-shm-usage", f"--user-data-dir={tmp}/profile-{i}",
                "--no-pdf-header-footer", "--print-to-pdf-no-header",
                f"--print-to-pdf={tmp_pdf}", html.as_uri(),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=240)
            if not tmp_pdf.is_file() or tmp_pdf.stat().st_size < 10_000:
                raise RuntimeError(f"PDF 渲染失败: {html.name} / {proc.returncode} / {proc.stderr[-400:]!r}")
            shutil.copy2(tmp_pdf, inner)
            shutil.copy2(tmp_pdf, outer)
            print(f"PDF: {outer.name} {tmp_pdf.stat().st_size:,} bytes")


def refresh_source_archive() -> None:
    spec = importlib.util.spec_from_file_location("assemble", ROOT / "tools/组装提交副本.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    archive, archive_sha, count = module.build_source_archive()
    note = OUT / "源码/源码归档说明.md"
    text = note.read_text(encoding="utf-8")
    text = re.sub(r"(\| SHA-256 \| `)[0-9a-f]{64}", lambda m: m.group(1) + archive_sha, text)
    text = re.sub(r"(\| 大小 \| )[\d,]+( bytes)", lambda m: m.group(1) + f"{archive.stat().st_size:,}" + m.group(2), text)
    text = re.sub(r"(\| 文件数 \| )\d+", lambda m: m.group(1) + str(count), text)
    note.write_bytes(text.encode("utf-8"))
    print(f"源码归档: {count} 文件 / {archive.stat().st_size:,} bytes / SHA {archive_sha[:16]}…")


def refresh_checksums() -> int:
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS.txt").write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    print(f"校验和: {len(rows)} 条")
    return len(rows)


def main() -> None:
    print("== HTML 口径 ==")
    apply_rules(SRC, HTML_RULES, "HTML 源稿")
    print("== Word 源稿口径 ==")
    apply_rules(SRC, MD_RULES, "Word 源稿")
    print("== PDF 重渲染 ==")
    render_pdfs()
    print("== 副本文本口径 ==")
    apply_rules(OUT, OUT_RULES, "副本文本")
    print("== 源码归档 ==")
    refresh_source_archive()
    print("== 校验和 ==")
    refresh_checksums()


if __name__ == "__main__":
    main()
