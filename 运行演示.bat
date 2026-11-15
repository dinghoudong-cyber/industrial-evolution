@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 尚未找到项目虚拟环境：%~dp0.venv
    echo 请先按照 README.md 安装依赖。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m industrial_evolution.cli demo --config "configs\tuned.yaml"
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo 演示运行完成，结果位于：%~dp0runs
) else (
    echo 演示运行失败，退出码：%EXIT_CODE%
)
pause
exit /b %EXIT_CODE%
