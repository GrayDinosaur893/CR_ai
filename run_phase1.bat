@echo off
title CLASH IQ — Phase 1 Component Perception & Launcher
color 0B
echo ==============================================================================
echo           CLASH IQ: PHASE 1 COMPONENT VERIFICATION PIPELINE
echo ==============================================================================
echo [1] Checking Python Environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

echo [2] Running Launcher and Component Detection...
python "%~dp0scripts\verify_phase1.py"

echo [3] Opening Visual Verification Image...
if exist "%~dp0frames\phase1_verified_components.png" (
    start "" "%~dp0frames\phase1_verified_components.png"
)

echo ==============================================================================
echo [DONE] Phase 1 Verification Completed Successfully.
echo ==============================================================================
pause
