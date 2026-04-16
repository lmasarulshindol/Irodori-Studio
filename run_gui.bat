@echo off
chcp 65001 >nul
set "Path=%USERPROFILE%\.local\bin;%Path%"
cd /d "%~dp0"
py -3 main.py
if errorlevel 1 pause
