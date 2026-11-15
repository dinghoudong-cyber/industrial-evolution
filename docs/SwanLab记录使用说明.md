# SwanLab 记录使用说明

SwanLab 在本项目里是**镜像**，不是主记录。本地 `runs/<run_id>/` 永远是完整的，
SwanLab 只负责把同一份事实搬到看板上，方便多轮运行横向对比、也让评审能直接看到
"每轮为什么改、改完有没有用"。上传失败不会打断进化闭环。

## 0. 三种用法，选一种

| 场景 | 模式 | 需要什么 | 你看到什么 |
| --- | --- | --- | --- |
| 联网开发、要云端看板 | `online` | 先 `swanlab login`，或设 `SWANLAB_API_KEY` | 云端项目页，可分享链接 |
| 本机看板、不想出网 | `local` | 无 | 本地 web 看板 |
| 评测机 / 容器，全程离线 | `offline` | 无 | `runs/.swanlab/` 下待补传的 run 目录 |
| 只想关掉 | `disabled` | 无 | 只有本地 `runs/` |

`online` 模式下如果没有凭证，程序会**自动降级成 offline** 并打印原因，不会卡在
"请输入 API Key"的交互菜单上——实测 SwanLab 在 Git Bash / ConPTY 下 `isatty()`
返回真，会阻塞等待输入，无人值守跑会被挂死。

## 1. 三步开起来

```bash
# ① 装依赖（本地开发）
pip install -e ".[swanlab]"

# ② 登录一次（云端模式才需要；offline/local 跳过这步）
swanlab login
# 或者不想交互：export SWANLAB_API_KEY=<你的 key>

# ③ 确认记录通道真的能写
python -m industrial_evolution swanlab-check            # 离线探针，不联网
python -m industrial_evolution swanlab-check --online    # 真连云端，验证凭证
```

`swanlab-check` 会真写一条标量和一条文本，返回 JSON：

```json
{
  "installed": true,
  "version": "0.10.4",
  "probe": "ok",
  "probe_mode": "offline",
  "offline_files": [ "...media/text/000-xxxx.txt" ]
}
```

`probe` 不是 `ok` 就先解决它，别急着跑训练——否则你会得到一堆没有云端记录的 run。

## 2. 跑一轮，记录同时开启

三种方式任选，优先级从高到低：

```bash
# ① 命令行开关（最省事，不改配置）
python -m industrial_evolution demo --swanlab --swanlab-mode offline

# ② 环境变量（Docker / CI 里用）
IE_SWANLAB=1 IE_SWANLAB_MODE=online python -m industrial_evolution demo

# ③ 改配置文件 configs/*.yaml
tracking:
  enable_swanlab: true
  swanlab_project: "industrial-evolution"
  swanlab_mode: "online"
```

公开 A/B/C 包的 `official-evaluate` 也支持同一记录通道；给命令追加 `--config configs\\tuned.yaml --swanlab --swanlab-mode online`，LocalTracker 仍是完整主记录，SwanLab 镜像指标、事件和 run 产物。`official-predict` 为保持 JSONL stdout 纯净不创建实验记录。云端 URL 只有实际在线成功时才会打印；自动降级 offline 时只有待同步文件，不是可分享链接。

跑起来后会打印一行确认：

```text
[swanlab] 看板：https://swanlab.cn/@you/industrial-evolution/runs/xxxx
[swanlab] 记录模式 offline，落盘目录：runs/.swanlab
```

## 3. 到底记了什么

不是只记 loss 曲线。分三层，对应评审关心的三件事：

### ① 曲线层 —— 每窗口一行，`step` = 窗口序号

窗口得分、漂移类型、特征漂移量、性能下降、标签漂移、动作、状态、候选提升、
预算消耗（更新次数 / 训练秒数 / 标签数 / Token 数）、LLM 调用次数与延迟。

因为 `step` 用窗口序号，多轮运行可以在同一坐标系里直接叠加对比。

状态类字段（动作、状态、漂移类型、决策来源）在曲线上是**数值码**，不是字符串。
原因很实际：SwanLab 0.10 收到字符串标量会抛
`TypeError: Unsupported scalar string value`，并且在内部吞掉——列会静默变空，
你只会以为"这一列没数据"。所以曲线里统一上报 `<字段>_code`，可读的名字保留在
下面的判断链文本里。

| 列 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `action_code` | no_update | retrain_with_replay | rollback | error | | | |
| `status_code` | no_update | deployed | rejected | rolled_back | budget_exhausted | data_quality_failed | failed |
| `drift_kind_code` | none | covariate | label | concept_suspected | mixed | | |
| `decision_source_code` | system | rule | llm | hybrid | | | |

`drift_kind_code` 的 `-1` 表示未知。查具体某一轮到底发生了什么，看文本列，别猜码。

### ② 文本层 —— 人可以直接读的判断链

每个窗口一条，固定六段：

```markdown
### 窗口 7 · gradual

**观察** 得分 0.6307（基线 0.7341，Δ -0.1034）｜漂移类型 concept_suspected
｜特征漂移 0.7747｜性能下降 0.0720｜标签漂移 0.0608｜可执行信号 是

**诊断** 滚动性能下降 0.0720

**决策** `retrain_with_replay`（来源 rule，模型 —，置信 —）
> 滚动性能下降 0.0720

**预期** 候选在本窗 holdout 上提升 ≥ 0.0080，且历史守护集回退 ≤ 0.1000，则上线

**结果** deployed｜holdout 提升 0.0387｜历史回退 0.0582｜上线版本 v001

**判定** ✅ 假设成立 —— 候选在 holdout 上提升 +0.0387，历史回退在阈值内，已上线并作为新基线
```

判定有四档，**否定的结论也保留**：

| 判定 | 含义 |
| --- | --- |
| ✅ 假设成立 | 重训挽回漂移，已上线并成为新基线 |
| ❌ 假设被否定 | 候选无提升，或历史回退超阈值，旧模型保留 |
| ➖ 证据不足 | 窗口停在失败/预算耗尽，验证环节没走完 |
| ⏭️ 未触发更新 | 无有效漂移信号，决策本身不需要验证 |

另外，关键事件单独成列：初始模型就绪、候选评估、模型上线、模型回滚、窗口失败、
LLM 决策、LLM 降级、最终部署模型得分归因。

### ③ 产物层 —— run 结束后上传可复算的证据

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 整轮汇总：最终分、最终部署模型分、动作分布、预算、状态 |
| `metrics.csv` | 每窗口逐行指标，和看板曲线同源 |
| `events.jsonl` | 全量事件流（含每个状态迁移），审计用 |
| `models/registry.json` | 版本登记表：来源窗口、分数、训练耗时 |

模型权重 `.joblib` 不上传（体积大且是二进制），单文件超过
`swanlab_artifact_max_mb`（默认 64MB）的会被跳过并记一条说明。

## 4. 看板怎么读

云端：登录 [swanlab.cn](https://swanlab.cn) → 项目 `industrial-evolution` → 选一次 run。
本地：`swanlab watch runs/.swanlab` 起本地服务，浏览器打开它给的地址。

### 不联网也能翻：单文件查看器

`offline` 模式落盘的 run 目录可以直接渲染成一张 HTML，双击就能看，不需要起服务、
不需要登录：

```bash
python tools/build_swanlab_viewer.py runs/<run_id>/.swanlab/run-2026xxxx_xxxxxx-xxxxxx
# → runs/<run_id>/run-2026....-查看器.html
```

页面内容与看板一一对应：顶部 KPI（窗口数 / 尝试 / 上线 / 否决 / 最终分）、
七条窗口曲线、窗口明细表、全部判断链卡片（绿=成立、红=否定、黄=证据不足、
灰=未触发）、以及随 run 上传的产物清单和 `summary.json` 全字段。
答辩现场没有网络时，用它代替云端看板。

建议的读法：

1. 先看 `score` 曲线和 `action` 标签列，判断"什么时候动了、动完有没有回来"；
2. 再看 `判断链/窗口结论` 文本列，按 step 逐条读——这是给答辩讲的东西；
3. 最后看 run 文件区，`summary.json` 里 `final_deployed_score` 是最终上线模型在
   未见 holdout 上的分数，`final_score` 是最后窗口的全量得分，两者不是一回事。

> 易错点：`final_score` 记的是**打分时的冠军**（比如 v003），最后一个窗口之后才上线的
> v004 不会被任何窗口评过分。所以最终模型的能力要看 `final_deployed_score`
> 与 `final_deployed_guard_score`，来源写在 `final_deployed_score_source` 里。

## 5. 在 Docker 里跑

镜像里 `run.sh` 已经接好环境变量：

```bash
docker run --rm \
  -e IE_SWANLAB=1 \
  -e IE_SWANLAB_MODE=offline \
  -v "$PWD/runs:/app/runs" \
  industrial-evolution demo
```

要有云端记录就把 key 传进去（不要写进镜像）：

```bash
docker run --rm \
  -e IE_SWANLAB=1 \
  -e IE_SWANLAB_MODE=online \
  -e SWANLAB_API_KEY="$SWANLAB_API_KEY" \
  -v "$PWD/runs:/app/runs" \
  industrial-evolution demo
```

容器里注意两点：

- `log_dir` 默认落在 `$IE_RUN_ROOT/.swanlab`，必须和 `runs` 一起挂出来，否则容器一停
  离线记录就没了；
- 只读根文件系统（`--read-only`）下，`runs` 必须是可写挂载，否则 SwanLab 写不进去
  会降级，虽然不影响进化流程，但记录就丢了。

离线记录事后补传：

```bash
swanlab sync runs/.swanlab/<run-xxxx>
```

`record` 模式也能直接查：

```bash
docker run --rm -e IE_SWANLAB=1 industrial-evolution record
```

## 6. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 打印"未检测到 SwanLab 凭证，已降级为 offline" | 没 login 也没设 key | `swanlab login` 或设 `SWANLAB_API_KEY` |
| 打印"初始化失败，本次仅写本地记录" | 装了包但 init 抛错（网络/配置） | 看括号里的异常；先跑 `swanlab-check` |
| `events.jsonl` 里有 `tracker_mirror_failure` | 某次上传失败，主流程未受影响 | 连续 5 次才停用镜像；偶发可忽略 |
| 有 `tracker_mirror_disabled` | 连续失败达阈值 | 看板侧问题解决后重启运行即可 |
| 看板上没曲线 | `enable_swanlab` 没开 | 加 `--swanlab` |
| 曲线有列但全是空的 | 往 `log_metrics` 塞了字符串，被 SwanLab 内部吞掉 | 已用码表规避；新增字段请先看上一节 |
| 曲线有列但全是空的 | 往 `log_metrics` 塞了字符串，被 SwanLab 内部吞掉 | 已用码表规避；新增字段请先看上一节 |
| 看板上指标缺列 | 该字段是字符串且不在码表里（`reason`、`phase`、`llm_model`） | 正常：字符串只进文本层，曲线只收数值与码 |

## 7. 和赛题评分点的对应

赛题要求"SwanLab 必须全程留痕"，落到可核查的证据是：

- **全程**：`events.jsonl` 记录每一次状态迁移（monitor→diagnose→decide→train→
  evaluate→deploy/reject/rollback），没有窗口是空白的；
- **留痕**：曲线 + 文本 + 产物三层同源，`metrics.csv` 与看板曲线来自同一份数据；
- **可复算**：`summary.json` + `registry.json` 能让第三方在不接触你本地目录的情况下
  复核最终分数与版本演进；
- **负结果**：被拒绝的候选、被否定的假设同样入库，不是只留成功的那些。
