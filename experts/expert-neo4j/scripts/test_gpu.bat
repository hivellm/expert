@echo off
cd /d F:\Node\hivellm\expert
call venv_windows\Scripts\activate.bat
cd experts\expert-neo4j
python test_gpu.py

