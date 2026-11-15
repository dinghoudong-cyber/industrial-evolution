# 校赛提交材料索引

> **版本状态**：校赛提交版（2026-11-14 定稿）。**先看同目录《校赛提交说明与自检表.md》**，那里给出七类必需材料自检、提交前 3 件本人操作与阶段边界判定。
> **阶段前提**：校赛阶段平台尚未下发正式比赛数据（B 包隐藏数据、按批次下发的评测流、正式 Harness/SDK 入口），Qwen3.8 27B / 4B 与 PyTRiO 后训练到企业赛阶段才提供。
> 本目录只汇总可复核的工程事实；未达标、未录制、平台未开放的内容明确标出，不代表已通过校赛或正式评测。**客观分只针对最终训出的工业模型结果，本目录不宣称任何平台成绩。**

## 一、七类必需材料核对

| 提交项 | 当前状态 | 文件 / 说明 |
|---|---|---|
| ① 系统镜像 | ✅ **齐** | `dist/industrial-evolution-school-2026-11-15.tar`（542 MiB，OCI layout）；`镜像/镜像与启动说明.md` 给出无外网四核 8 GiB 的 `docker load` + 受限 `docker run` 契约。本地已实测 `--network none --memory 1g --cpus 1 --read-only` 跑满 18 窗口、0 失败 |
| ② Harness 及模型配置 | ✅ **齐** | `configs/`、`src/industrial_evolution/orchestrator.py`；`模型调用信息/模型与调用声明.md` 明确：Qwen3.8 27B/4B 在企业赛阶段才提供，校赛 LLM 调用 0 次、Token 0，配置里是占位符 |
| ③ 源代码、依赖与复现说明 | ✅ **齐** | `源码/industrial-evolution-school-initial-source.zip`（含根目录 `LICENSE`）；`requirements.lock` 精确锁定；运行说明写明「无外网不能 `docker build`，只能 `docker load`」；`SHA256SUMS.txt` 标准格式，实测 `sha256sum -c` 全通过 |
| ④ SwanLab 实验记录 | ⚠️ **唯一未达标项（已核实）** | 云端项目为**非公开**，匿名抓取返回「需要登录并确认访问权限」。整改步骤与匿名验收见 `实验记录/SwanLab记录公开性整改说明.md`；`实验记录/离线复核包/` 是**兜底证据，不替代公开要求**。**整改通过前不得标记达标** |
| ⑤ 设计论证 | ✅ **齐** | `设计说明/设计说明.md`、`技术报告/技术报告初稿.md`、排版版 `技术报告.html`、**`技术报告.pdf`（4 页，≤10 页）**；`docs/扩展性验证报告.md` |
| ⑥ 运行说明 | ✅ **齐** | `镜像/镜像与启动说明.md`、根目录 `run.sh`、`README.md` |
| ⑦ 开源合规展示 | ✅ **齐（署名待确认）** | 根目录 `LICENSE`（MIT）；`来源声明/来源与许可声明.md`、`第三方依赖与许可证.md`（逐项从包元数据读取）。5 个 `.joblib` 权重已移出提交目录，`registry.json` 记 `weights_included=false` |

### 补充材料

| 文件 | 用途 |
|---|---|
| `校赛提交说明与自检表.md` | **提交门面**：材料自检、3 件本人操作、阶段边界重判、一键验收 |
| `实验记录/官方公开包CPU续训证据/` | A/B/C 三包各一份官方格式 CPU 续训报告（**不含权重**），含《说明.md》与核对命令 |
| `实验记录/离线复核包/` | 6 条完整决策链、18 行逐批判断、`复算.py` 一键复算（不含权重） |
| `答辩/答辩一页纸.md` / `.html` / **`.pdf`（1 页）** | 客观前 20 名线下答辩 |
| `docs/扩展性验证报告.md` | 10 场景 × 2 组，扩展性验证与技术创新加分 |
| `docs/企业赛平台对接与规范核对清单.md` | 11/9 记录规范、代金券、served model ID、批次协议、4B 取舍 |
| `演示/演示视频脚本.md` | 4 分 30 秒分镜（**视频尚未录制**） |

## 二、方案与阶段边界

- 项目源码目录：`D:\industrial_evolution`；容器入口 `run.sh`；Python CLI `python -m industrial_evolution.cli`。
- Harness：**本项目自研 `EvolutionOrchestrator`**。早期材料的「DeepSeek Harness」是框架名称，不表示调用 DeepSeek 模型，也不表示已接入某商业产品；提交版已统一命名，避免评委误判违规外联。
- 底层驱动模型：赛题规则说明 Qwen3.8 27B / Qwen3.5 4B 与 PyTRiO 后训练**在企业赛阶段**才提供。校赛阶段未下载权重、未打包权重、未发生调用；模型调用记录 0 次 / Token 0。
- 平台正式 Harness/SDK 入口、批次下发协议、标签释放时序与评分公式以组委会文档为准。校赛阶段这些数据与接口尚未下发，**相应适配属企业赛工作，不作为校赛缺陷申报**；逐批 `deploy`/`no_update` 判断接口与 `label_budget_mode=count_only` 已就位作为对接点。

## 三、可复核的工程事实

- `pytest` **114 项通过**；`ruff check` / `ruff format --check` 通过。
- 合成漂移流工程对照：平均平衡准确率 0.614526（自建静态对照）→ 0.676200；6 次候选、4 次上线、2 次被否定的假设。
- **闭环迭代证据**：`实验记录/离线复核包/` 的 158 条事件、18 次读历史、6 条完整决策链（窗口 7/9/11/13/15/17），逐窗口 `decisions.jsonl` 上线/不上线判断——**满足「≥3 次迭代」门槛**。
- **CPU 训练证据**：`实验记录/官方公开包CPU续训证据/` 三份报告均为 `device: cpu`、请求 1 线程，分别以 A/B/C 官方包 checkpoint 续训 2 个 epoch：Macro-F1 **0.714030 → 0.716641**（0.043 秒）、RMSE **681.159833 → 656.351785** 分钟（0.040 秒）、RMSE **52.868573 → 52.740936** cycles（0.242 秒）。按赛题要求，`checkpoint.npz` 权重未随包分发。
- **扩展性验证**（`docs/扩展性验证报告.md`）：10 场景 × 2 组，多数场景取得稳定正增益（平均 +0.041694）；样本放大 5.1 倍墙钟仅增 1.9 倍；大规模场景累计 10,200 标签，全部窗口完成评估；真实触发回滚 4 次；全部 ≥3 次迭代、0 失败。
- 镜像在 `--network none --memory 1g --cpus 1 --read-only` 下跑满 18 窗口、0 失败，`final_score` 与宿主机逐位一致。
- A/B/C 三条云端记录是**静态 baseline**，不是三次迭代；`rounds/round-1/` 的官方格式续训报告现已收编进提交目录。

> **口径说明**：早期 sklearn 闭环在自建 C-MAPSS / Gas / XJTU 数据流上为负结果；官方公开 A 包续训在公开 validation 上有小幅正增益。两者都标 `official_scoring_ready=false`，**均不等价于平台评测，也不构成「最终模型不低于官方静态基线」的正式验证**——该结论须由正式评测判定。校赛无正式比赛数据，本目录不伪造该结论。

## 四、本地复核命令

```powershell
cd D:\industrial_evolution
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe "submission\校赛初版\实验记录\离线复核包\复算.py"
.\.venv\Scripts\python.exe "tools\扩展性验证实验.py"     # 重跑扩展性矩阵
```

```bash
cd submission/校赛初版 && sha256sum -c SHA256SUMS.txt
```

## 五、提交前待办（按优先级）

**必须本人完成（我代不了）**

1. **SwanLab 项目设为公开** + 未登录浏览器匿名访问截图 + 闭环迭代 run 同步上云并回填链接（唯一未达标的必需材料）。
2. **录制演示视频**，放入 `演示/`，≤5 分钟。
3. **`LICENSE` 署名与许可已确认**：会赢的，对不队（北京印刷学院），MIT；改动后需重跑校验和锁定。

**可选**：平台侧用同版本 Docker 实测 `docker load -i` 导入 OCI 归档并留存输出；向组委会索取官方 A/B/C 数据包与 checkpoint 的正式使用条款原文。

**企业赛阶段（非校赛缺陷）**：平台 served model ID、代金券、API Key 安全提交、PyTRiO 后训练、4B 选型对照、B 包批次协议、官方 JSONL 包接入闭环与最终 checkpoint 导出——见 `docs/企业赛平台对接与规范核对清单.md`。
