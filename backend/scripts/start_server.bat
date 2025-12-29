@echo off
cd /d "%~dp0.."
echo Starting REQUAGNIZE Backend Server...
echo Working Directory: %CD%
C:\Users\mu\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
