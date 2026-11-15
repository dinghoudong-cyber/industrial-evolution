# 工业模型自进化框架

面向工业漂移数据流的可复现持续学习框架。系统在无人干预下自动完成：

```text
监测 → 诊断 → 更新决策 → 回放训练 → 候选评估 → 上线或拒绝 → 回滚
```

核心是一套自研的五步闭环调度程序（`EvolutionOrchestrator`）：数据按批次进入，程序自行判断分布是否漂移、是否需要训练候选模型、候选是否优于当前模型，并完成版本登记与回滚；每一步的判断依据都写入可复算的实验记录。全流程默认离线运行，不依赖外部模型服务。

## 主要能力

- PSI、KS、Wasserstein 及缺失值、非有限值等数据质量监测；
- 滚动性能下降、Page-Hinkley 与多类别标签分布监测；
- 训练回放与不可训练历史守护集严格隔离，防标签泄漏；
- Champion-Challenger 安全上线、模型注册、版本保存与自动回滚；
- 标签、训练时长、更新次数等资源记账，异常窗口安全降级；
- JSONL 事件、CSV 指标、JSON 汇总，以及可选 SwanLab 镜像记录；
- 逐批次「上线 / 不上线」判断落盘 `decisions.jsonl`；
- 主指标可配置（Macro-F1 / negative_rmse / balanced_accuracy）与同口径基线比较；
- 合成漂移演示、CSV 顺序流、公开数据集适配器与容器运行入口。

## 运行方式

### 本地环境

```powershell
cd D:\industrial_evolution
.\.venv\Scripts\python.exe -m industrial_evolution.cli demo --config configs\tuned.yaml
```

也可双击 `运行演示.bat`。

### 重新安装开发环境

```powershell
D:\Python313\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

### 容器

镜像内置统一入口 `run.sh`，评测方无需关心内部调用：

```bash
docker build -t industrial-evolution:0.1.0 .
docker run --rm -v "$PWD/out:/app/runs" industrial-evolution:0.1.0 check   # 环境自检
docker run --rm -v "$PWD/out:/app/runs" industrial-evolution:0.1.0 demo    # 完整闭环演示
docker run --rm -v "$PWD/out:/app/runs" \
  -e IE_CSV=/data/stream.csv -e IE_TARGET=label -e IE_INITIAL_SIZE=1000 \
  industrial-evolution:0.1.0 csv
```

| 变量 | 含义 |
|---|---|
| `IE_CONFIG` | 配置文件路径，默认 `/app/configs/tuned.yaml` |
| `IE_RUN_ROOT` | 输出根目录，默认 `/app/runs` |
| `IE_CSV`、`IE_TARGET`、`IE_INITIAL_SIZE`、`IE_WINDOW_SIZE` | `csv` 模式的数据文件与切窗参数 |
| `IE_SWANLAB` | 设为 `1` 开启 SwanLab 镜像记录 |
| `IE_SWANLAB_MODE` | `online` / `local` / `offline` / `disabled` |
| `SWANLAB_API_KEY` | 云端记录密钥；只开记录不给密钥会自动降级为 `offline` |

`check` 打印 Python 版本、依赖可用性、端点连通性与输出目录可写性，加 `--strict` 可作为容器健康检查。`record` 会真写一条标量与一条文本，用于确认记录通道可用。

## 使用自己的 CSV

CSV 必须保持真实时间顺序，当前版本只接受数值特征：

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli run-csv `
  --csv D:\数据\train.csv `
  --target label `
  --initial-size 1000 `
  --window-size 200 `
  --drop-column timestamp
```

## 公开数据包接入

数据包结构按「命令行参数 > 包内清单（manifest）> 常见命名推断」三级解析；任何一级都无法确定时启动即报错，并列出已尝试的文件名与缺失列：

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli run-official `
  --pack D:\数据包 --print-spec
.\.venv\Scripts\python.exe -m industrial_evolution.cli run-official `
  --pack D:\数据包 --config configs\official.yaml
```

公开包同时提供 `official-evaluate`（静态基线与 CPU 续训对照）和 `official-predict`（按参考协议从 stdin 读 JSONL、向 stdout 输出预测）两个子命令。

## 策略模型接入（可选）

闭环默认由确定性规则门控完成全部决策，`llm.enabled` 为 `false`，不产生任何外部调用。

如需让模型参与策略判断，系统只提供一种接口形态：`openai_compatible`，即由**本机或内网的 OpenAI 兼容端点**（vLLM / sglang 等推理服务）承载模型。密钥只通过环境变量注入，不写入配置、源码或实验产物。

```powershell
# 用最小模拟端点先把决策链路跑通
.\.venv\Scripts\python.exe tools\mock_llm_server.py --port 8123
.\.venv\Scripts\python.exe -m industrial_evolution.cli llm-smoke-test `
  --config configs\llm-local.yaml --base-url http://127.0.0.1:8123/v1
```

端点安全边界由 `LLMConfig` 在启动时校验：`base_url` 不得带账号、查询串或片段；公网地址必须 https，本机与内网可用 http；`api_key_env` 必须是大写且以 `API_KEY` / `TOKEN` 结尾；可用 `allowed_hosts` 再加一层白名单。

运行时容错：端点不支持 `response_format` 时自动降级为提示词约束并重试；端点不可用、超时、低置信度或返回非法 JSON 时回退规则策略，闭环不中断，事件流记录 `llm_fallback` 及原因。

## 配置文件

| 文件 | 用途 |
|---|---|
| `configs/tuned.yaml` | 合成流推荐演示配置，离线 |
| `configs/default.yaml` | 较保守的通用起点，离线 |
| `configs/static.yaml` | 禁止更新的静态对照组，离线 |
| `configs/llm-local.yaml` | 可选策略模型接入示例，指向本机 OpenAI 兼容端点 |
| `configs/official.yaml` | 公开数据包运行配置，主指标与静态基线对齐 |
| `configs/cmapss.yaml` | C-MAPSS FD001~FD004 的 RUL 回归配置 |
| `configs/gas.yaml` | UCI Gas 批次漂移分类配置 |
| `configs/xjtu.yaml` | XJTU-SY 单轴承全寿命离线 RUL 回归配置 |

## 公开工业数据集适配器

- **C-MAPSS**：以每台 engine 的最大 cycle 构造 piecewise-linear RUL 标签，按 engine 内时间顺序添加滑动统计特征；
- **Gas**：解析 UCI libsvm 文件，按 batch1→batch10 保留数据顺序；
- **XJTU-SY**：每次只处理一个指定轴承，把每分钟振动 CSV 聚合为双通道统计量。

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli run-cmapss `
  --data-dir datasets\cmapss --fd FD001 --config configs\cmapss.yaml
.\.venv\Scripts\python.exe -m industrial_evolution.cli run-gas `
  --data-dir datasets\gas --config configs\gas.yaml
```

三个适配器均实现统一的 `StreamSource` 协议，可进入同一个进化闭环。XJTU-SY 依赖完整寿命序列，仅适用于全寿命离线实验。

## 实验记录

SwanLab 只做镜像，本地 `runs/<run_id>/` 永远是主记录；上传失败不打断闭环。

记录分三层：**曲线**（每窗口得分、漂移与预算，`step` 为窗口序号）、**判断链**（观察→诊断→决策→预期→结果→判定，被否定的假设同样保留）、**产物**（`summary.json`、`metrics.csv`、`events.jsonl`、`decisions.jsonl` 与模型注册表，第三方无需接触本地目录即可复算）。无凭证时自动降级为 `offline` 落盘。

```text
runs\run-日期时间-编号\
├── events.jsonl       # 状态迁移、诊断、决策、训练、上线、回滚
├── decisions.jsonl    # 逐批次上线 / 不上线判断
├── metrics.csv        # 每个窗口的指标、动作与累计成本
├── summary.json       # 整次运行摘要
└── models\            # 模型版本与注册表
```

## 对照结果

固定随机种子 `42`，在相同 18 个合成窗口上：

| 配置 | 平均平衡准确率 | 最后窗口 | 更新尝试 | 上线 | 异常 |
|---|---:|---:|---:|---:|---:|
| 静态基线 | 0.614526 | 0.538245 | 0 | 0 | 0 |
| 自进化（无泄漏守护集） | 0.676200 | 0.699060 | 6 | 4 | 0 |

该结果用于验证工程闭环可复现，不代表正式评测成绩。相同种子复跑后关键指标、动作次数与模型版本一致。

## 项目结构

```text
src/industrial_evolution/
├── data/           # 合成流、CSV 流与公开数据包接入
├── monitoring/     # 特征、性能与标签漂移
├── memory/         # 历史回放缓冲区
├── models/         # 模型适配器与版本注册
├── policy/         # 规则门控与混合策略
├── llm/            # 受控策略客户端协议、结构与 OpenAI 兼容适配器
├── training/       # 候选训练
├── evaluation/     # Champion-Challenger 评估
├── tracking/       # 本地记录与 SwanLab 镜像
├── orchestrator.py # 端到端状态机与假设链记录
└── cli.py          # 演示、数据流、评估与自检命令
```

## 开发验证

以下命令请在项目根目录运行。Ruff 默认只检查本项目维护的 `src/` 和 `tests/`，不扫描随官方包提供的参考实现或提交材料生成脚本：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

当前 109 项测试全部通过，覆盖漂移检测、数据质量告警、预算与标签去重、训练与守护集隔离、策略冷却、候选评估、状态回滚、记录器降级、端到端运行、跨平台 CLI 编码、可复现基准，以及可选策略端点的安全校验与容错。

## 后续工作

- 按正式接口规范统一数据包格式与在线批次握手；
- 打通闭环产物与官方 checkpoint 清单格式的自动导出；
- 策略层自适应（阈值与冷却期自整定）、成本-增益帕累托分析；
- 运行时总墙钟与进程内存采集，纳入实验记录。
