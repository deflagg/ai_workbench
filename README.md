This repo is where I try and test new AI tooling, patterns, startegies, and other stuff.  It's a sandbox.  

Currently working with a decoder only model to understand the practical implementation of all aspects.  It can be found in the source, experiments/aaaa/llm_ngpu.py.  The model does appear to learn and also starts to be coherent where fragments of sequences make some sense. There is a couple bugs and more optimizations can be made but I'm ok with it for now.  Below are a couple images of the training run's in Weights & Biases.

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
