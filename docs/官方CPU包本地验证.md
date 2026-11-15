# 官方 CPU A/B/C 包：公开数据本地验证

本入口提供**本地公开数据验证**，不是比赛平台 SDK，也不代表隐藏测试或正式评测通过。只处理命令行显式指定的公开 JSONL，不递归查找、不读取隐藏文件。默认不启用 SwanLab、不联网；只有显式开启 `--swanlab --swanlab-mode online` 时才尝试把实验记录镜像到云端。

## 竞赛使用边界与已验证基线

组委会说明：A/B/C 公开数据仅用于开发参考；正式比赛数据由平台在测评时提供，平台执行最终评分。因此下列结果仅用于确认包内 checkpoint 与公开 validation 的本地兼容性，不代表比赛表现，也不能作为平台漂移阶段得分。

| Profile | 公开 validation 静态基线 | 样本数 | 本地运行记录 |
|---|---:|---:|---|
| A | Macro-F1 0.714030；Accuracy 0.878617 | 1,244 | `run-20261112-135939-198e08` |
| B | RMSE 681.159833 分钟；MAE 563.105841 分钟 | 1,679 | `run-20261112-135941-2827d3` |
| C | RMSE 52.868573 cycles；MAE 38.899830 cycles | 23,049 | `run-20261112-135944-71fdd2` |

以上 run 在 `runs/official/` 下，各自包含摘要与哈希。这些运行用于复现官方公开包的静态基线，验证本地读取与评估链路；按批次下发的数据包与评分公式以组委会文档为准。

主编排器的本地闭环会在 `events.jsonl` 记录 `experiment_history_read` 和 `hypothesis_recorded`：前者是策略决策前读取的最近历史摘要，后者记录当前窗口的观察、决策、验证结果和判定。记录摘要不包含原始样本或标签。可用这些本地记录演示开发闭环，但不能冒称平台隐藏数据实验。

## SwanLab 开发实验记录

`official-evaluate` 默认只生成本地记录。要把公开包评估镜像到 SwanLab，先确认依赖/凭证和云端通道可写：

```powershell
.\\.venv\\Scripts\\python.exe -m industrial_evolution.cli swanlab-check --online
```

之后在 A/B/C 的 `official-evaluate` 命令中加上配置和开关，例如：

```powershell
--config configs\\tuned.yaml --swanlab --swanlab-mode online
```

每次运行会先完整写入本地 `runs/official/`，再镜像指标、事件和摘要；CLI 会打印云端看板 URL。若没有凭证或网络不可用，SwanLab 会降级为 offline，本地 run 仍保留，但不能把离线目录说成云端链接。只做离线演练时显式使用 `--swanlab-mode offline`。API Key 只通过 SwanLab 登录或环境变量注入，不能放进命令文本、配置或实验产物。

## A：Gas 分类

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli official-evaluate `
  --pack-root official\cpu-a-guide-20261110-v1 --profile A `
  --train official\cpu-a-guide-20261110-v1\public\train.jsonl `
  --validation official\cpu-a-guide-20261110-v1\public\validation.jsonl
```

报告 Macro-F1（主指标）与 Accuracy，类别顺序使用 checkpoint 元数据。

## B：轴承分钟级 RUL

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli official-evaluate `
  --pack-root official\cpu-b-rul-20261110-v1 --profile B `
  --train official\cpu-b-rul-20261110-v1\public\train.jsonl `
  --validation official\cpu-b-rul-20261110-v1\public\validation.jsonl
```

报告分钟单位 RMSE/MAE。续训（仅使用公开 train，validation 只评估）还必须显式提供标签来源声明：

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli official-evaluate `
  --pack-root official\cpu-b-rul-20261110-v1 --profile B `
  --train official\cpu-b-rul-20261110-v1\public\train.jsonl `
  --validation official\cpu-b-rul-20261110-v1\public\validation.jsonl `
  --epochs 5 --label-provenance D:\数据\B标签来源说明.json
```

声明文件格式为 `{"status":"draft|approved","method":"标签计算依据"}`。`approved` 是声明值，不由本地程序或项目验证为组委会批准。

> 说明：上述约束由本项目 CLI 入口施加。官方包自带的 `baselines/train.py` 仅对历史三分类 B 配置文件要求该声明；本包 B 为 RUL 回归任务，标签按 `(N − i) × 1 分钟` 确定性计算，可直接用 `--from-baseline checkpoint.npz` 续训，无需额外声明。

## C：C-MAPSS cycles RUL

```powershell
.\.venv\Scripts\python.exe -m industrial_evolution.cli official-evaluate `
  --pack-root official\cpu-c-guide-20261110-v1 --profile C `
  --train official\cpu-c-guide-20261110-v1\public\train.jsonl `
  --validation official\cpu-c-guide-20261110-v1\public\validation.jsonl
```

报告 cycles 单位 RMSE/MAE。包文档提到 NASA 非对称分数，但未给出此入口可验证的精确公式和聚合实现，因此本地暂不自行猜算；该项应以正式评测 SDK 为准。C 包的 `label_status`/训练报告状态会保存在记录中。

## 续训与静态基线保护

`--epochs 0`（默认）只评估原始 `checkpoint.npz`。设置正整数后，候选从包内 checkpoint warm-start，保留权重、Adam、标准化和目标缩放；仅用公开 train 更新，validation 不参与训练。候选只有在公开 validation 的主指标不低于原始 checkpoint 时才会写入**新的 run 目录**；不覆盖随包基线。这个验证门槛用于公开验证集上的本地对照。

各数据集读入时会检查 JSONL 字段、唯一 ID、有限数值、任务和特征维数/路径；A 按批次、B 按轴承、C 按子集与发动机实体检查 train/validation 隔离。`id`、`meta`、`answer` 和目标派生信息不会作为输入特征。

每次运行记录于 `runs/official/` 下新建目录，包含 `events.jsonl`、`metrics.csv`、`summary.json`，通过摘要保存公开输入文件与基线 checkpoint 的 SHA-256、指标、时间与决策。该处是本地观测，不等同于赛事平台对训练额度、标签和 Token 的核算。默认不启用 SwanLab、LLM 或模型服务；需要实验看板时按上一节显式开启 SwanLab。

## 本地参考预测协议

项目提供包内参考脚本所用的 stdin/stdout JSONL 形态，严格只加载 checkpoint 和 request，不读标签或训练：

```powershell
Get-Content .\requests.jsonl | .\.venv\Scripts\python.exe -m industrial_evolution.cli official-predict `
  --pack-root official\cpu-a-guide-20261110-v1 --profile A
```

请求仅允许 `task`、`inputs`；每条输出一个 JSON 行。它只是包内参考协议的本地实现，比赛评测平台的 SDK 入口、生命周期、标签释放时序、超时和资源计量仍需以组委会/SDK 文档核实，不能据此宣称已经适配正式提交接口。
