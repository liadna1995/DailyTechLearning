@echo off
cd /d "%~dp0"
echo Starting Daily Tech...
py -m streamlit run app.py
pause

