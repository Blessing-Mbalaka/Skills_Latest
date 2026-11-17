@echo off
cd /d "%~dp0"
echo Starting CurriculumMatch at http://127.0.0.1:5000/new-analysis
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
pause
