# WiC-LLM

Experiments on the **Word-in-Context (WiC)** task with open-weight large language models (LLMs), using both **Ollama** and **Hugging Face Transformers** frameworks.

## 📖 Overview

The WiC task evaluates whether a given word (lemma) has the **same or different meaning** across two sentences. This project explores how modern LLMs handle semantic disambiguation in context.

We provide:

- Prompt construction tailored for semantic judgments.

- Inference pipelines for both Ollama and Transformers.

- Support for deterministic generation via **random seeds**.

- A simple evaluation workflow for comparing model predictions with gold-standard labels.

## ⚙️ Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/bradmhauer/WiC-LLM.git
cd WiC-LLM
pip install -r requirements.txt
```

### Requirements

- Python 3.9+

- [Ollama](https://ollama.com/) (for local inference with supported models)

- [Transformers](https://huggingface.co/docs/transformers) (for Hugging Face models)

- pandas

## 🚀 Usage

### 1. Run WiC with Transformers

```python
from wic_transformers import llm_for_wic

lemma = "bank"
s1 = "Under the bridge on the bank of the river."
s2 = "I have to go to the bank to deposit some cash."
s3 = "Is the bank still open this late?."

print(llm_for_wic(lemma, s1, s2, model="Qwen/Qwen3-1.7B", seed=9999)) # Returns False
print(llm_for_wic(lemma, s2, s3, model="Qwen/Qwen3-1.7B", seed=9999)) # Returns True
```

### 2. Run WiC with Ollama

```python
from wic_ollama import llm_for_wic

lemma = "bank"
s1 = "Under the bridge on the bank of the river."
s2 = "I have to go to the bank to deposit some cash."
s3 = "What do I need to open an account at the bank?."

print(llm_for_wic(lemma, s1, s2, model="qwen3:4b", seed=9999))
print(llm_for_wic(lemma, s2, s3, model="qwen3:4b", seed=9999))

```

## 🎯 Example Task

For lemma = **bank**:

- Sentence 1: *Under the bridge on the bank of the river.*

- Sentence 2: *I have to go to the bank to deposit some cash.*  
  ➡️ Prediction: **different**

## 🔑 Features

- 🔄 **Cross-framework support** (Ollama + Transformers)

- 🎲 **Reproducible runs** with seed control

- 🧪 **Evaluation-ready** functions for batch experiments

- 📝 **Lightweight, flexible prompts** (with optional `/no_think` directive for Qwen3 Hybrid thinking models)
