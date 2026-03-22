#!/usr/bin/env bash
set -e

echo ""
echo "  Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "  Starting LLM Observability Dashboard..."
python run_dashboard.py
