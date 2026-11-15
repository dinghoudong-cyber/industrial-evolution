# PyTRiO Sampler 微调工作区

平台的企业赛测试入口开放后，提交表单要求「持久化 Sampler 微调权重」。这个目录放的是
产出该权重所需的全部代码与数据，**与校赛提交包完全隔离**（校赛包不含任何 LLM 调用声明）。

## 一、表单逐项对照

| 表单字段 | 要求 | 当前状态 | 怎么拿到 |
|---|---|---|---|
| 模型类型 | 持久化 Sampler 微调权重（下拉，已固定） | — | 不用填 |
| **完整 Sampler 权重路径** | `save_weights_for_sampler()` 返回的 `trio://` 路径 | **缺** | `smoke_train.py` 跑通后控制台直接打印 |
| **训练记录链接** | 本次大模型微调的 HTTPS 实验记录 | **缺**（现有 SwanLab 是校赛闭环的，不是本次微调） | `smoke_train.py --swanlab` 会上报新 run 并给链接 |
| 已选择的 Harness 镜像 | `8c6c68e9…`（平台预置，灰色不可改） | 平台给 | 不用填 |
| **源码仓库链接** | 公网 HTTPS 仓库 | **缺**，但仓库已建好 | 推到 GitHub / Gitee 后取 URL |
| **PyTRiO API Key** | 须有权访问并调用所填权重或基础模型 | **缺** | 你的平台账号生成 |
| 作品文档 | 上传文件 | **已有** | `D:\桌面\智训引擎-校赛提交\技术报告.pdf`、`设计说明.docx` |

> 「已选择的 Harness 镜像」是平台预置的官方 Harness，**不是**我们校赛提交的
> `industrial-evolution:school-2026-11-15`。两者不是同一个东西，不要试图替换。

## 二、三条硬阻塞（按顺序解决）

1. **额度为 0 就跑不了。** 之前截图显示 `可用 ¥0.00`、`权重 0`。没有额度时
   `create_lora_training_client` 会直接失败。先确认额度，再谈其他。
2. **API Key。** 账号内生成，填两处：本地环境变量 `PYTRIO_API_KEY` + 表单里的 API Key 字段。
3. **公网仓库。** 本地 `git` 仓库已初始化并提交（143 文件、约 1.8MB，已排除权重、
   官方数据包与镜像归档），只差推送到远端。

## 三、不能用来填这个字段的东西（最容易踩的坑）

表单底部原话：「Train checkpoint、临时采样权重及工业模型产物路径不能替代。」

| 已有产物 | 类型 | 能否填 |
|---|---|---|
| `save_state()` 产出的 `Train` 权重 | Train | 否，只能续训、不能推理 |
| `save_weights_and_get_sampling_client()` 的临时权重 | 临时 | 否 |
| 官方 A/B/C 包的 `checkpoint.npz` | 工业模型产物 | 否 |
| 我们 sklearn 训练出的 `.pkl` | 工业模型产物 | 否 |
| `save_weights_for_sampler()` 的 `Sampler` 权重 | **Sampler** | **是，唯一可用** |

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

所以微调任务就是让模型学会这个输出。`build_sft_data.py` 从
`events.jsonl` 的 `drift_diagnosis` + `update_decision` 事件对里，把 15 个字段
原样还原成训练输入，把规则策略的决策作为监督目标（**规则策略蒸馏**）。

置信度说明：规则策略不产出置信度（事件里是 null），脚本用信号强度做单调映射，
函数写在 `build_sft_data.py::_confidence_for()` 里，可逐项复核，不是凭空标注。

## 五、操作步骤

```bash
# 0) 装 SDK（建议独立虚拟环境，它会带入 transformers / modelscope）
python -m venv .venv-pytrio && .venv-pytrio/Scripts/activate
pip install pytrio

# 1) 生成训练数据（当前已生成 1381 条：1118 no_update / 263 retrain）
python build_sft_data.py

# 2) 查平台支持的基座模型
set PYTRIO_API_KEY=你的key
python smoke_train.py --list-models

# 3) 冒烟：1 步，只验证链路
python smoke_train.py --base-model <模型ID> --steps 1

# 4) 正式小规模微调（先少步数观察 loss 是否下降）
python smoke_train.py --base-model <模型ID> --steps 20 --data sft_data.jsonl --swanlab
```

冒烟通过后，控制台会打印：

```
====================================================================
提交表单「完整 Sampler 权重路径」填这一串（原样复制，不要改写）：
trio://...
====================================================================
```

同时写入 `outputs/<时间戳>-sampler.json`，含路径、基座模型、步数、SwanLab 链接，方便回填表单。

**路径必须原样复制。** 表单明确写了「不要自行拼接」，手改的路径在平台上校验不过。

## 六、数据现状与风险

- 样本 **1381 条**，动作分布 `no_update 1118 / retrain_with_replay 263`，类别不平衡，
  正式训练时建议按类加权或对少数类过采样。
- 数据来自合成漂移流的闭环记录，**不含真实工业产线数据**，泛化性有限。
- 若平台按 token 计费，1381 条全量训练成本要先估一遍，先用 `--steps 20` 小规模验证。

## 七、与校赛材料的关系

这个目录**不会**进入 `D:\桌面\智训引擎-校赛提交\`，校赛包里也不声明任何 LLM 调用。
企业赛阶段把权重跑出来、链路验证完，再决定是否在提交材料里公开这部分工作。
