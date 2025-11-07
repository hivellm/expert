# Script to run qualitative analysis with proper Python environment
$pythonExe = "F:\Node\hivellm\expert\cli\venv_windows\Scripts\python.exe"
Set-Location F:\Node\hivellm\expert\experts\expert-neo4j\tests
& $pythonExe qualitative_analysis.py 2>&1 | Tee-Object -FilePath qualitative_analysis_log.txt

