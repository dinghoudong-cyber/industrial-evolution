# PyTRiO Sampler 微调工作区

企业赛测试入口开放后，提交表单要求「持久化 Sampler 微调权重」。这个目录放的是产出该权重
所需的全部代码与数据，**与校赛提交包完全隔离**（校赛包不声明任何 LLM 调用）。

## 一、表单逐项对照（当前状态）

| 表单字段 | 要求 | 状态 | 值 / 取法 |
|---|---|---|---|
| 模型类型 | 持久化 Sampler 微调权重（下拉固定） | — | 不用填 |
| **完整 Sampler 权重路径** | `save_weights_for_sampler()` 返回的 `trio://` | **已有** | 见下方「当前产物」 |
| **训练记录链接** | 本次大模型微调的实验记录 | **已有** | SwanLab run 链接，见「当前产物」 |
| 已选择的 Harness 镜像 | `8c6c68e9…`（平台预置，灰色） | 平台给 | 不用填 |
| **源码仓库链接** | 公网 HTTPS 仓库 | **已有** | https://github.com/dinghoudong-cyber/industrial-evolution |
| **PyTRiO API Key** | 须有权访问所填权重或基础模型 | **已有** | 账号内生成，仅走环境变量，不落盘 |
| 作品文档 | 上传文件 | **已有** | `技术报告.pdf`、`设计说明.docx` |

> 「已选择的 Harness 镜像」是平台预置的官方 Harness，**不是**本校赛提交的
> `industrial-evolution:school-2026-11-15`。两者不是同一个东西，不要试图替换。

## 二、当前产物

**填表直接看《表单填写清单.md》**，那里是照抄用的完整版。摘要：

| 项 | 值 |
|---|---|
| 基座模型 | `Qwen/Qwen3.8-27B` |
| 提交用 Sampler（第 3 轮） | `trio://run_n7w9j6kns8cp/sampler_weights/sampler-r3` |
| 训练记录 | https://swanlab.cn/@1649435028/industrial_evolution/runs/wjh74ixw |
| 源码仓库 | https://github.com/dinghoudong-cyber/industrial-evolution |
| 权重大小 | 1,155,193,338 bytes |

三轮训练与验证结果：

| 轮次 | 平均 loss | 输出合规率 | 与规则一致率 |
|---|---|---|---|
| 第 1 轮 | 7.3962 | 30/30 = 100% | 30/30 = 100% |
| 第 2 轮 | 3.6631 | 29/30 = 96.7% | 29/30 = 96.7% |
| 第 3 轮 | 3.0076 | 30/30 = 100% | 30/30 = 100% |

每轮结束都会写 `outputs/<时间戳>-sampler.json`，含路径、断点、步数、loss 摘要与
SwanLab 链接；验证明细写 `outputs/verify-sampler-r*.json`。

**路径必须原样复制。** 表单明确写了「不要自行拼接」，手改的路径在平台校验不过。

## 三、能不能填这个字段：产物类型对照

表单底部原话：「Train checkpoint、临时采样权重及工业模型产物路径不能替代。」

| 已有产物 | 类型 | 能否填 |
|---|---|---|
| `save_state()` 产出的权重（如 `weights/round1`） | Train | 否，只能续训、不能推理 |
| `save_weights_and_get_sampling_client()` 的临时权重 | 临时 | 否 |
| 官方 A/B/C 包的 `checkpoint.npz` | 工业模型产物 | 否 |
| sklearn 训练出的 `.pkl` | 工业模型产物 | 否 |
| `save_weights_for_sampler()` 的 `sampler_weights/*` | **Sampler** | **是，唯一可用** |

校赛那套 A/B/C CPU 续训证据在这里**完全用不上**——那是小模型，不是驱动 Harness 的 LLM。

## 四、架构上为什么能对得上

平台的 Sampler 是「驱动 Harness 的大语言模型」。本项目 Harness 的策略层
`HybridPolicy` 本来就把决策委托给 LLM：

```
LLMDecisionRequest（15 个字段）
  → system prompt + JSON
  → 模型输出 {"action","confidence","reason"}
  → 校验后决定 no_update / retrain_with_replay
```

微调任务就是让模型学会这个输出。`build_sft_data.py` 从 `events.jsonl` 的
`drift_diagnosis` + `update_decision` 事件对里，把 15 个字段原样还原成训练输入，
把规则策略的决策作为监督目标（**规则策略蒸馏**）。

置信度说明：规则策略不产出置信度（事件里是 null），脚本用信号强度做单调映射，
函数写在 `build_sft_data.py::_confidence_for()` 里，可逐项复核，不是凭空标注。

## 五、多轮续训怎么跑

平台的 `save_state()` 存 Train 断点，`create_training_client_from_state()` 从断点接续，
`save_weights_for_sampler()` 才转成可提交的 Sampler 权重。三轮串起来的实际命令：

```bash
# 环境（独立 venv，会带入 transformers / modelscope，不要装进交付镜像的环境）
python -m venv .venv-pytrio && .venv-pytrio/Scripts/activate
pip install pytrio swanlab

# 数据（当前 1381 条：1118 no_update / 263 retrain_with_replay）
python pytrio-train/build_sft_data.py

# 查基座模型
set PYTRIO_API_KEY=你的key
python pytrio-train/smoke_train.py --list-models

# 第 1 轮：从基座起步，存断点 + 存 Sampler
python pytrio-train/smoke_train.py --base-model Qwen/Qwen3.8-27B \
  --data pytrio-train/sft_data.jsonl --steps 100 --batch-size 2 \
  --save-state round1 --weight-name sampler-r1 --offset 0 --swanlab

# 第 2 轮：接第 1 轮断点，offset 换样本
python pytrio-train/smoke_train.py --resume 'trio://.../weights/round1' \
  --data pytrio-train/sft_data.jsonl --steps 100 --batch-size 2 \
  --save-state round2 --weight-name sampler-r2 --offset 200 --swanlab

# 第 3 轮：接第 2 轮断点
python pytrio-train/smoke_train.py --resume 'trio://.../weights/round2' \
  --data pytrio-train/sft_data.jsonl --steps 100 --batch-size 2 \
  --save-state round3 --weight-name sampler-r3 --offset 400 --swanlab
```

参数要点：

- `--resume` 之后 **`--rank` 与基座由断点决定**，不需要再传 `--base-model`。
- `--seed` + `--offset` 决定每轮喂哪批样本。固定种子保证可复现，offset 让三轮不重复。
- `--no-save` 只训练不存 Sampler 权重，用于测速探测，避免生成 1.15GB 权重占额度。
- `--swanlab` 上报到已公开的 `industrial_evolution` 项目；新建项目务必先改公开，
  否则链接对外打不开。`SWANLAB_RUN_NAME` 可指定 run 名。

## 六、实测数据

| 项 | 值 |
|---|---|
| 单步耗时 | 5.8–6.0 s/步（batch 2，27B，每轮 100 步约 10 分钟） |
| 6 步探测 | loss 58.2425 → 10.8164 |
| 第 1 轮 100 步 | loss 58.2425 → 0.0514，平均 7.3962 |
| 第 2 轮 100 步 | loss 1.2466 → 5.9478，平均 3.6631 |
| 第 3 轮 100 步 | loss 0.4428 → 6.7761，平均 3.0076 |
| token 长度 | 247–430（平均约 330），1381 条 |

loss 有波动（部分 step 到 0.03、部分到 7），是样本难度差异导致的正常现象：
`no_update` 类样本目标串短且重复，容易压到极低；`retrain_with_replay` 类样本的
`reason` 字段更长，loss 更高。**只看末步 loss 会误判**，要看多步平均。

## 六之二、权重可用性验证（verify_sampler.py）

只有一个 `trio://` 路径不够——表单要求 API Key「须有权访问并**调用**所填权重」。
`verify_sampler.py` 用 Harness 实际会发的 prompt 采样，按线上 schema 解析，输出两个指标：

- **输出合规率**：能否解析成合规 JSON（三字段、action 合法、confidence 在 0–1、reason 1–120 字）
- **与规则一致率**：`action` 是否等于规则策略的决策

验证中如实记录的两个现象：

1. **结束符污染**：原始 tokenizer 采样返回文本尾部带 `<|im_end|>`，直接 `json.loads` 报
   `Extra data`。走 OpenAI 兼容 HTTP 接口不会出现，脚本按线上口径剥掉后再解析。
2. **`reason` 偶发超长**：第 2 轮出现 1 条 122 字（上限 120）。训练数据最长恰好 120、
   无超限样本，属模型自身轻微越界；线上 Harness 会判不合法并回落规则策略。

两个指标都不等于「效果提升」。微调目标是让模型**稳定复现**规则策略的输出格式与决策，
不是超越规则——对外表述必须区分「格式对齐」与「能力增强」。

## 七、数据现状与风险

- 样本 **1381 条**，动作分布 `no_update 1118 / retrain_with_replay 263`，类别不平衡。
- 数据来自合成漂移流的闭环记录，**不含真实工业产线数据**，泛化性有限。
- 训练输入是规则策略的蒸馏结果，不是人工标注的「最优策略」，模型学到的是
  「复现规则」，不是「超越规则」。对外表述要写清这一点。
- 1.15GB Sampler 权重是 LoRA 合并后的产物，每存一次占一份平台存储。

## 八、与校赛材料的关系

这个目录**不会**进入 `D:\桌面\智训引擎-校赛提交\`，校赛包里也不声明任何 LLM 调用。
企业赛材料是否公开这部分工作，按赛题阶段要求再定。
