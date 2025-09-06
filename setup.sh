#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install black isort flake8 mypy pytest

echo "Virtual environment setup complete!"
echo "Activate with: source venv/bin/activate"
