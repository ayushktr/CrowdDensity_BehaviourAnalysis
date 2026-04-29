@echo off
echo ===================================================
echo     Crowd Emotion Analyzer - Setup and Launch
echo ===================================================

set PYTHON_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not exist %PYTHON_EXE% (
    echo Python 3.11 was not found at %PYTHON_EXE%. 
    echo Please ensure Python is installed and try again.
    pause
    exit /b
)

echo Installing required dependencies...
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install opencv-python deepface tf-keras numpy

echo.
echo Starting the application...
echo.
%PYTHON_EXE% emotion_analyzer.py

echo.
echo Application closed. Detailed report should be generated in analysis_report.md
pause
