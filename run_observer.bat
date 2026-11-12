@echo off
title CR_BOT OBSERVER PIPELINE
cd /d "%~dp0"
echo ===========================================================================
echo Starting CR_BOT Pure Observer Pipeline
echo (Press F12 anytime to HALT)
echo ===========================================================================
python -u src\observer\run_observer.py
pause
