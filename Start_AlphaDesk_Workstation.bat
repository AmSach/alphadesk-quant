@echo off
title AlphaDesk Quant - 24/7 Autonomous Quantitative Workstation
echo =========================================================================
echo  ALPHADESK QUANT - INSTITUTIONAL REINFORCEMENT LEARNING WORKSTATION
echo =========================================================================
echo Starting C++20 Accelerated Engine & 24/7 Autonomous Trading Service...
echo.

cd /d "E:\Alphadesk Quant"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

start "" python server.py 8000
timeout /t 2 /nobreak > nul
start "" python gui.py

echo.
echo Workstation launched:
echo  - Native Desktop UI: Running with System Tray 24/7 Persistence
echo  - Web Workstation: http://localhost:8000 (LAN: http://10.10.33.206:8000)
echo.
