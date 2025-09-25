# AI Workbench

This repository is a sandbox for experimenting with new AI tooling, patterns, strategies, and other concepts.

## Disclaimer

This repository is only for learning purposes. You will see unnecessary calls to LLMs and logical designs that don't make sense.

## Current Project

Currently, I'm working on a decoder-only model to understand the practical implementation of all aspects. The model can be found in `experiments/aaaa/llm_ngpu.py`. It appears to learn and generate coherent fragments of sequences.

## Model Architecture

- **Type**: GPT-style, decoder-only Transformer (causal language model).
- **Embeddings**: Token embedding + learned positional embedding.
- **Blocks**: Pre-LayerNorm → masked self-attention → residual; then Pre-LayerNorm → FFN (ReLU, Dropout) → residual. Repeated `num_layers` times.
- **Masking**: Causal mask (no peeking ahead) + key-padding mask for batch padding.
- **Head**: Final LayerNorm → linear layer to vocabulary logits.
- **Data**: WikiText-103, concatenated; sliding-window chunks with stride. Each sample prepends [BOS].
- **Batching**: Inputs padded with [PAD]; targets padded with -100 (ignored by loss).
- **Training**: Adam + OneCycleLR; CrossEntropyLoss(ignore_index=-100); early stopping; checkpoints; optional multi-GPU (DataParallel); W&B logging.
- **Inference**: Top-k/top-p sampling with temperature (stochastic); simple generate loop.
- **CLI**: Train or inference modes; configurable model sizes, LR, batch, sequence length, stride, checkpoints.

## Known Issues

There are a few bugs that manifest when the context is pushed too hard, but it's fine for normal use. This is a work in progress. The BOS and EOS tokens are currently missing. Additionally, more optimizations can be made, but I'm okay with it for now.

Below are a couple of images from the training runs in Weights & Biases. I manually adjust the hyperparameters to understand how each affects the graphs and to identify what's most important in training runs.

<img width="753" height="518" alt="image" src="https://github.com/user-attachments/assets/44723e9a-ca74-4391-96e6-02ee1c2c3abf" />

<img width="80%" height="80%" alt="image" src="https://github.com/user-attachments/assets/1050ed3d-4f69-4d0e-a9c5-d52adaf7da14" />

## Setup

### Save Imports

```bash
pip freeze > requirements.txt
```

### Environment Variables

Create a `.env` file in the root of the folder to set up environment variables:

```env
# LangChain / LangGraph
LANGCHAIN_API_KEY = ""

# LangChain Tracing (requires an account. Search for LangChain Tracing)
LANGCHAIN_TRACING_V2 = "false"

# OpenAI
OPENAI_API_KEY = ""

# TAVILY AI Search
TAVILY_API_KEY = ""

# Google Search
GOOGLE_CSE_ID = ""
GOOGLE_API_KEY = ""

# Davinci Resolve (variables can be found in the Davinci Resolve documentation)
RESOLVE_SCRIPT_API = ""
RESOLVE_SCRIPT_LIB=""
PYTHONPATH=""

# Github
GITHUB_ACCESS_TOKEN = ""
```  
