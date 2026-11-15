# Ruff 与 Docker 使用说明

## 1. Ruff 是什么

Ruff 是 Python 静态检查和格式检查工具。它不运行训练程序，而是在提交代码前检查：

- 未定义名称和漏导入；
- 未使用的导入；
- 导入顺序；
- 常见 Python 错误写法；
- 代码格式是否统一。

本项目将 Ruff 安装在项目虚拟环境中，不进行全局安装。下载缓存放在 D 盘：

```powershell
cd D:\industrial_evolution
$env:PIP_CACHE_DIR = "D:\pip-cache"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

检查命令：

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
```

自动修复前应先查看结果。只修复确定安全的问题：

```powershell
.\.venv\Scripts\python.exe -m ruff check --fix src tests
```

## 2. Docker 是什么

Docker 把 Python、依赖、项目源码、配置和启动命令封装成镜像，使开发机和评测机使用一致环境。它主要解决“本机能运行，换一台机器不能运行”的问题。赛题要求提交系统镜像，因此 Docker 是正式交付载体。

Docker 不是模型算法，也不会自动提高准确率。

## 3. Windows 11 家庭版安装 Docker Desktop

家庭版没有 Hyper-V，只能用 **WSL 2 后端**。以下步骤在 Windows 11 家庭版 26200 + Docker Desktop 4.98 上实测通过。

### 3.1 检查虚拟化

打开任务管理器 → 性能 → CPU，确认“虚拟化”为“已启用”。如果未启用，需要在 BIOS/UEFI 中开启 Intel VT-x 或 AMD-V。

系统若已在运行“内存完整性”（VBS），则虚拟化已被 hypervisor 接管，无需再进 BIOS。

### 3.2 安装 WSL2

**先关掉系统代理再执行**。若开着系统代理，`wsl --install` 会因商店接口被拦而报 `已禁止 (403)`。

管理员 PowerShell：

```powershell
wsl --install --no-distribution
```

`--no-distribution` 表示不安装 Ubuntu，Docker Desktop 会自行创建专用发行版，可省约 2 GB。

若长时间卡在 0.1% 或报 403，改走手工路径（不依赖商店）：

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

重启后安装官方独立包（不经过商店）：

```
https://github.com/microsoft/WSL/releases/latest   # 取 wsl.<版本>.x64.msi
```

校验通过后再安装，验证：

```powershell
wsl --version
```

看到版本号与内核版本即为成功。

### 3.3 安装 Docker Desktop

从官网下载安装器（家庭版不支持 Hyper-V，安装界面必须选 **WSL 2**）：

```
https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
```

安装类型选 **Per-user installation**，勾选 WSL 2 后端，安装完成后重启。

注意：国内直连 Docker Hub 会超时（`registry-1.docker.io` 无法连接），首次拉基础镜像前必须准备镜像加速或代理，否则 `docker build` 会卡在 `FROM`。

### 3.4 验证安装

```powershell
docker version
docker info --format "{{.OSType}}/{{.Architecture}} {{.ServerVersion}}"
```

`linux/amd64` 说明引擎正常。

## 4. 构建和运行项目镜像

### 4.1 准备基础镜像

国内网络下 Docker Hub 不可直连。用镜像站拉取后打上本地标签，Dockerfile 无需改动：

```powershell
docker pull docker.m.daocloud.io/library/python:3.13.7-slim-bookworm
docker tag docker.m.daocloud.io/library/python:3.13.7-slim-bookworm python:3.13.7-slim-bookworm
```

### 4.2 构建

```powershell
cd D:\industrial_evolution
docker build -t industrial-evolution:0.1 .
```

国内构建时 pip 直连 pypi.org 会很慢（实测单次请求 15 秒）。用构建参数换源，**不改变锁定版本**——`requirements.lock` 是精确 pin 的，换源只换“从哪里取同一个版本”：

```powershell
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t industrial-evolution:0.1 .
```

实测耗时 35 秒，镜像 546 MB。

### 4.3 运行

镜像默认使用 `configs/tuned.yaml`，无参数即跑完整闭环：

```powershell
docker run --rm -v "D:\industrial_evolution\runs:/app/runs" industrial-evolution:0.1
```

模拟评测环境（断网、限资源）：

```powershell
docker run --rm --network none --memory 8g --cpus 4 -v "D:\industrial_evolution\runs:/app/runs" industrial-evolution:0.1
```

可用子命令：

| 命令 | 用途 |
| --- | --- |
| `demo` | 合成漂移数据完整闭环（默认） |
| `check` | 环境自检：依赖、端点连通性、输出目录可写 |
| `csv` | 在给定 CSV 上运行，需 `IE_CSV` 与 `IE_TARGET` |
| `smoke` | 对策略模型端点做一次真实调用 |
| `record` | 记录通道自检：SwanLab 包、凭证、真正写一条标量与文本 |

切换配置：`-e IE_CONFIG=/app/configs/static.yaml`。

镜像默认 `IE_SWANLAB=1`、`IE_SWANLAB_MODE=offline`：默认就在留痕，但落盘不上传，
无外网的评测机也能完整记录，事后 `swanlab sync` 补传。要直连云端看板，运行时注入
`-e IE_SWANLAB_MODE=online -e SWANLAB_API_KEY=...`；密钥不写进镜像。

**Windows 上执行注意**：Git Bash 会把 `-e VAR=/app/...` 里的值当路径改写，需加 `export MSYS_NO_PATHCONV=1`；PowerShell 无此问题。

开发用 DeepSeek 密钥只能在运行时注入，不能写进 Dockerfile、配置或镜像：

```powershell
docker run --rm `
  -e DEEPSEEK_API_KEY `
  -v "D:\industrial_evolution\runs:/app/runs" `
  industrial-evolution:0.1 `
  demo --config /app/configs/deepseek-dev.yaml
```

正式 Harness 使用赛方建议的 Qwen3.8 27B 或 4B；如需后训练，可基于 Qwen 使用 PyTRiO，并按平台安全流程提交模型 ID 与 API Key。密钥不写进 Dockerfile、配置或镜像。DeepSeek 仅开发调试，绝不作为最终 Harness。

### 4.4 导出离线交付包

评测机通常不能联网拉镜像，需导出 tar：

```powershell
docker save industrial-evolution:0.1 -o dist\industrial-evolution-0.1.tar
```

540 MB，可再压缩。评测方导入：

```powershell
docker load -i industrial-evolution-0.1.tar
```

## 5. 当前验证状态

2026-11-08 在本机完成首次实际构建与运行验证：

| 项目 | 结果 |
| --- | --- |
| Docker Desktop | 4.98.0，引擎 29.9.2，linux/amd64 |
| 构建 | 成功，35 秒，镜像 546 MB |
| 断网运行 | `--network none` 下 18 窗口完成，0 失败 |
| 极限资源 | `--memory 1g --cpus 1 --read-only` 下仍完整跑完，7 秒 |
| 结果一致性 | 容器 `final_score` = 0.69906、`v004` 上线、holdout 0.660055，与宿主机运行逐位一致 |
| 离线导出 | `docker save` 产出 540 MB tar |
| 环境自检 | `check` 正确报告 openai 3.46.0 / swanlab 0.10.4 可用、输出目录可写；断网时端点解析失败被如实记录，不误判为可用 |

`check` 在断网环境下的输出示例：

```
"endpoint_reachable": "gaierror: [Errno -3] Temporary failure in name resolution",
"run_root_writable": true
```

这是**期望行为**：断网评测时端点不可达属于正常，策略会自动退回规则通道；自检只是把事实记录下来。
