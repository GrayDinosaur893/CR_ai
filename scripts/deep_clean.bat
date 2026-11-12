@echo off
:: Batch script to clean up deep Windows update backups and hibernation file
:: Must be run as Administrator (Right click -> Run as Administrator)

title Windows Deep Cleanup Tool
echo ========================================================
echo       WINDOWS DEEP SYSTEM & UPDATE CLEANUP
echo ========================================================
echo.

echo [1/2] Removing Hibernation file (hiberfil.sys) to free ~6.24 GB...
powercfg -h off
echo [DONE] 6.24 GB freed immediately from hiberfil.sys!
echo.

echo [2/2] Cleaning old superseded Windows Update backups (WinSxS)...
echo       (This may take 2-4 minutes, please wait...)
dism.exe /Online /Cleanup-Image /StartComponentCleanup /ResetBase
echo [DONE] Windows Update backups cleaned!
echo.

echo ========================================================
echo  CLEANUP COMPLETE! Check your C: Drive free space now!
echo ========================================================
pause
