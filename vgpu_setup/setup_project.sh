#!/bin/bash

git clone https://github.com/deflagg/ai_workbench.git
git pull
cd ai_workbench/experiments/aaaa
python3 -m venv venv
source venv/bin/activate
chmod +x wikitext2.py
pip install -r requirements.txt 
