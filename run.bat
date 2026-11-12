@echo off
title CLASH IQ — Unified Simultaneous AI Engine
color 0B
cd /d "%~dp0"

echo ==============================================================================
echo                CLASH IQ: UNIFIED SIMULTANEOUS AI ENGINE
echo   [Threads: Capture (60Hz) + Perception (20Hz) + Invariants + F12 Killswitch]
echo ==============================================================================
echo.
echo Launching simultaneous live engine...
echo Press F12 at any moment to trigger IMMEDIATE HARD KILLSWITCH.
echo.

python -u main.py

pause
