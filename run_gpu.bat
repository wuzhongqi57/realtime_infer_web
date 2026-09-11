@echo off
rem GPU 启动变体：等价于 `run.bat --device cuda`
call "%~dp0run.bat" --device cuda %*
