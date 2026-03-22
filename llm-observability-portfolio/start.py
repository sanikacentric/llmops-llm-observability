import subprocess
import sys

print("\n  Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)

print("\n  Starting LLM Observability Dashboard...")
subprocess.run([sys.executable, "run_dashboard.py"])
