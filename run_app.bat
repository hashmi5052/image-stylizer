@echo off
cd /d E:\image-stylizer
call .\venv\Scripts\activate
start http://127.0.0.1:7860
python main.py
pause
