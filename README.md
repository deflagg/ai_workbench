This repo is where I try and test new AI tooling, patterns, startegies, and other stuff.  It's a sandbox.  

Currently working with a decoder only model to understand the practical implementation of all aspects.  It can be found in the source, experiments/aaaa/llm_ngpu.py.  The model does appear to learn and also starts to be coherent where fragments of sequences make some sense. 

Type: GPT-style, decoder-only Transformer (causal LM).  
Embeddings: Token embedding + learned positional embedding.  
Blocks: Pre-LayerNorm → masked self-attention → residual; then Pre-LayerNorm → FFN (ReLU, Dropout) → residual. Repeated num_layers times.  
Masking: Causal mask (no peeking ahead) + key-padding mask for batch padding.  
Head: Final LayerNorm → linear layer to vocab logits.  
Data: WikiText-103, concatenated; sliding-window chunks with stride. Each sample prepends [BOS].  
Batching: Inputs padded with [PAD]; targets padded with -100 (ignored by loss).  
Training: Adam + OneCycleLR; CrossEntropyLoss(ignore_index=-100); early stopping; checkpoints; optional multi-GPU (DataParallel); W&B logging.  
Inference: Top-k/top-p sampling with temperature (stochastic); simple generate loop.  
CLI: train or inference modes; configurable model sizes, LR, batch, seq length, stride, checkpoints.

There are a few bugs which will express themselves in the context is pushed too hard but it's find for normal use and it's a work in progress. The BOS and POS tokens are now missing so there's that. In addition, more optimizations can be made but I'm ok with it for now.  Below are a couple images of the training run's in Weights & Biases.    

<img width="753" height="518" alt="image" src="https://github.com/user-attachments/assets/44723e9a-ca74-4391-96e6-02ee1c2c3abf" />

<img width="80%" height="80%" alt="image" src="https://github.com/user-attachments/assets/1050ed3d-4f69-4d0e-a9c5-d52adaf7da14" />


This repo is only for learning. You will see unnecessary calls to llms and logical designs that don't make sense.

Save imports
pip freeze > requirements.txt

Create a .env file in the root of the folder to setup environment variables

  \# LangChain / LangGraph  
  LANGCHAIN_API_KEY = ""

  \# LangChain Tracing (requires an account. Search for LangChain Tracing)  
  LANGCHAIN_TRACING_V2 = "false"

  \# OpenAI  
  OPENAI_API_KEY = ""

  \# TAVILY AI Search  
  TAVILY_API_KEY = ""
  
  \# Google Search  
  GOOGLE_CSE_ID = ""  
  GOOGLE_API_KEY = ""
  
  \# Davinci Resolve (variables can be found in the Davinci Resolve documentation)  
  RESOLVE_SCRIPT_API = ""  
  RESOLVE_SCRIPT_LIB=""  
  PYTHONPATH=""
  
  \# Github  
  GITHUB_ACCESS_TOKEN = ""  
