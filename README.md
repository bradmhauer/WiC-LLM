# WiC-LLM

Experiments on the **Word-in-Context (WiC)** task with open-weight large language models (LLMs), using **Hugging Face Transformers** and **Ollama** frameworks.


## 📖 Overview

The WiC task evaluates whether a given word (technically, lemma) has the **same or different meaning** across two sentences. This project explores how modern LLMs handle semantic disambiguation in context.

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

- pandas

- tqdm

- [Transformers](https://huggingface.co/docs/transformers) (for Hugging Face models)

- Optional: [Ollama](https://ollama.com/) (for local inference with supported models)


## 🎯 Example Task

For lemma = **bank**:

- Sentence 1: *Under the bridge on the bank of the river.*
- Sentence 2: *I have to go to the bank to deposit some cash.*  
  ➡️ Prediction: **different**


## 🔑 Features

- 🔄 **Cross-framework support** (Transformers + Ollama)

- 🎲 **Reproducible runs** with seed control

- 🧪 **Dataset processing** functions for experiments

- 📝 **Lightweight, flexible prompts** (with optional `/no_think` directive for Qwen3 Hybrid thinking models)


## 🗃️  Data

We suggest using the original dataset for this task, as provided by the following NAACL 2019 paper:

Pilehvar, Mohammad Taher, and Jose Camacho-Collados.
"WiC: the Word-in-Context Dataset for Evaluating Context-
Sensitive Meaning Representations." In Proceedings of
NAACL-HLT, pp. 1267-1273. 2019.

In particular, per the paper, the dataset can be obtained [here](https://pilehvar.github.io/wic/).

Once downloaded, place WiC_dataset.zip in the "data" directory, and unzip it.


## 🚀 Usage

These examples are all intended to be run in the "scripts" sub-directory. So:
```bash
cd scripts/
```

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

print(llm_for_wic(lemma, s1, s2, model="qwen3:1.7b-fp16", seed=9999)) # Returns False
print(llm_for_wic(lemma, s2, s3, model="qwen3:1.7b-fp16", seed=9999)) # Returns True
```

### 3. Full experiments on the original WiC dataset
```python
python wic_llm.py --framework transformers --model Qwen/Qwen3-1.7B
python wic_llm.py --framework ollama --model qwen3:1.7b-fp16

```

Both commands will deploy the specified model, via the specified framework, to solve the development set. It will then evaluate the output and place detailed experiment information in results/results_owic_dev.tsv.

