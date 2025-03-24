@echo off
start cmd /k "cd %~dp0 && python app.py"
start cmd /k "cd %~dp0 && npm start" 