#!/bin/bash

git clone https://github.com/deflagg/ai_workbench.git && cd ai_workbench/experiments/aaaa && python3 -m venv venv && source venv/bin/activate && chmod +x wikitext2.py && pip install -r requirements.txt

git pull && python wikitext2_enhanced.py
git pull && python llm_ngpu.py
git pull && python mymodel.py

# Run inference
python llm_ngpu.py --mode inference --model_path checkpoints/checkpoint_epoch_1.pt

wandb artifact put --name checkpoint_epoch_2 --type model /ai_workbench/experiments/aaaa/checkpoints/checkpoint_epoch_2.pt


git clone https://github.com/deflagg/ai_workbench.git
cd ai_workbench/experiments/aaaa
python3 -m venv venv
source venv/bin/activate
chmod +x wikitext2.py
pip install -r requirements.txt 
# Add .env file to root of project
cat > .env

git reset --hard HEAD
git pull

