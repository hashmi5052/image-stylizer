#!/bin/bash
cd /mnt/e/image-stylizer  # For WSL or adjust path for Linux/macOS
source venv/bin/activate
xdg-open http://127.0.0.1:7861 &> /dev/null  # Use 'open' on macOS
python main.py
