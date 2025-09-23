# Valohai-NVIDIA-BioNeMo+NIMs

This repository demonstrates an end-to-end protein workflow that integrates **NVIDIA BioNeMo Framework**, **NIMs (NVIDIA Inference Microservices)**, and **Valohai MLOps** for seamless model training, inference, evaluation, and deployment.

---

## Overview

This project shows how to:

* Preprocess and load biological datasets for BioNeMo models
* Run **ESM2 inference** for protein property prediction
* Generate novel protein sequences with **ProtGPT2**
* Evaluate generated sequences with custom metrics
* Visualize protein similarity embeddings
* Deploy models into a **NIM container** for production inference

---

## Project Structure

* **prepare-data.py** → Preprocess and prepare datasets for BioNeMo models
* **predict-properties.py** → Run property prediction using BioNeMo’s pretrained ESM2 models
* **generate-proteins.py** → Generate protein sequences with ProtGPT2
* **evaluate-generated.py** → Evaluate quality and diversity of generated proteins
* **protein-similarity-visualization.py** → Visualize similarity between embeddings
* **convert-model.py** → Convert Hugging Face models into NIM-compatible format (safetensors layout)
* **valohai.yaml** → Defines all pipeline steps for Valohai execution

---

## Pipeline Steps

<img width="1188" height="549" alt="image" src="https://github.com/user-attachments/assets/fadfb182-da16-4026-bfe6-88a6e83f3e96" />


The Valohai pipeline automates the complete workflow:

### 1. **Load Data**

* Uses BioNeMo framework container (`nvcr.io/nvidia/clara/bionemo-framework:1.3`)
* Converts UniRef50 dataset to FASTA
* Splits the dataset for training, validation and testing

---

### 2. **Predict Protein Properties**

* Runs ESM2 property prediction with BioNeMo framework (nightly image)
* Configurable parameters:

  * `num_gpus` (default: 1)
  * `precision` (`fp16`, `fp32`)
  * `micro-batch-size`

---

### 3. **Generate Protein Sequences**

* Runs **ProtGPT2** sequence generation on CUDA runtime
* Parameters for controlling generation:

  * `max_length`
  * `top_k`
  * `repetition_penalty`
  * `num_return_sequences`
  * `eos_token_id`

---

### 4. **Evaluate Generated Sequences**

* Evaluates generated sequences for validity and diversity
* Inputs sequences from the previous step

---

### 5. **Visualize Protein Similarity**

* Embedding-based similarity visualization
* Parameters:

  * `query_idx` → query sequence index
  * `topk` → number of similar sequences retrieved

---

### 6. **Deploy to NIM**

Prerequisites: 
You have to setup a gpu instance, and initially setup the restart_nim.sh

example of the one used for this project
```
#!/bin/bash
set -e

CONTAINER_NAME="protgpt2-nim"

echo "[INFO] Stopping old container (if any)..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "[INFO] Starting new ProtGPT2 NIM..."
docker run -d --rm --name=protgpt2-nim \
  --runtime=nvidia --gpus all \
  -p 8000:8000 \
  -v /home/ec2-user/models/protgpt2:/models/protgpt2 \
  -e NIM_MODEL_NAME=/models/protgpt2 \
  nvcr.io/nim/nvidia/llm-nim:1.13.0

echo "[INFO] Waiting for NIM to become healthy..."
for i in {1..150}; do
  if curl -fsS http://localhost:8000/v1/metadata >/dev/null; then
    echo "[INFO] NIM is healthy!"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "[ERROR] Container exited unexpectedly"
    docker logs --tail=200 $CONTAINER_NAME
    exit 1
```

* Converts models into **NIM safetensors layout**
* Copies them securely to a remote **NIM host (eg.EC2 GPU instance)**
* Restarts the NIM container to load the new model
* Requires SSH key as Valohai input

Parameters:

* `host_name` → target instance host

---

## Running the Pipeline

To execute the full workflow:

```bash
vh pipeline run bionemo_end_to_end
```

This will:

1. Load and preprocess the dataset
2. Predict protein properties with BioNeMo
3. Generate new protein sequences
4. Evaluate sequence quality
5. Visualize similarities
6. Deploy models into NIM for production use (Requires user approval)

---

## Dependencies

* NVIDIA BioNeMo Framework
* NVIDIA CUDA 11.8 + PyTorch 2.7.1
* Valohai MLOps platform
* Hugging Face Transformers (for ProtGPT2 conversion)
* Python packages: `torch`, `pandas`, `scikit-learn`, `matplotlib`, `valohai-utils`

Install Python dependencies locally with:

```bash
pip install -r requirements.txt
```

---

## Model Conversion for LLM-NIM

NIMs require models in a specific safetensors layout:

```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
```

The step **deploy-NIM** automatically handles this conversion and deployment.

---

## Notes

* You must provide your **SSH private key** as a Valohai input for deployment.
* For NVIDIA NGC images, set up authentication in Valohai project registry:

  * **image pattern**: `nvcr.io/*`
  * **username**: `$oauthtoken`
  * **password**: your `NGC_API_KEY`

---

## Acknowledgments

* **NVIDIA BioNeMo** – Foundation models for proteins, chemistry, and biology
* **NVIDIA NIMs** – Optimized inference microservices for deploying AI models
* **Valohai** – End-to-end machine learning automation platform

---
