# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLVR (Reinforcement Learning with Verifiable Rewards) project for fine-tuning LLMs using Unsloth and GRPO (Group Relative Policy Optimization). Based on Matthew Berman's tutorial using OpenAI's GPT-OSS model to learn the 2048 game.

## Environment Setup

**Conda environment:** `RL`
**Python version:** 3.11
**GPU:** NVIDIA GB10 (DGX Spark) with CUDA 13.0

### Activate Environment
```bash
conda activate RL
```

### Key Dependencies
- PyTorch 2.10.0+cu130
- Triton 3.6.0
- Transformers 4.56.2
- TRL 0.22.2 (for GRPO training)
- Unsloth 2026.1.4
- PEFT, BitsAndBytes, Accelerate

## Common Commands

### Start Jupyter
```bash
conda activate RL && jupyter lab
```

### Verify GPU Access
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir=outputs
```

## Architecture

### GRPO Training Pipeline
1. **Model Loading**: Load GPT-OSS-20B with 4-bit quantization via Unsloth
2. **LoRA Configuration**: Apply Low-Rank Adaptation (r=4) to reduce trainable parameters to ~0.01%
3. **Reward Functions**: Three signals - syntax validation, anti-cheating, game success
4. **Training Loop**: GRPO optimizer with temperature sampling for diverse generations

### Key Configuration
```python
# Model settings
max_seq_length = 768
load_in_4bit = True
lora_rank = 4

# GRPO training
learning_rate = 5e-5
per_device_train_batch_size = 1
num_generations = 2
max_steps = 1000
```

## Resources

- [2048 Notebook](https://colab.research.google.com/github/openai/gpt-oss/blob/main/examples/reinforcement-fine-tuning.ipynb)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Docs](https://docs.unsloth.ai)
