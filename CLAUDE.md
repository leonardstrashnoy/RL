# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLVR (Reinforcement Learning with Verifiable Rewards) project for fine-tuning LLMs using Unsloth and GRPO (Group Relative Policy Optimization). Based on Matthew Berman's tutorial using OpenAI's GPT-OSS model to learn the 2048 game.

## Environment Setup

**Machine:** DGX Spark (spark-a005)
**Architecture:** ARM64 (aarch64) - always use ARM64-compatible packages
**Conda environment:** `RL`
**Python version:** 3.11
**GPU:** NVIDIA GB10 (DGX Spark) with CUDA 13.0
**Permissions:** No sudo access - use conda/pip for all installations

### Activate Environment
```bash
conda activate RL
```

### Environment Notes
- Always activate the `RL` conda environment before running any Python scripts
- GPU availability can be verified with the command in "Verify GPU Access" below
- When installing new packages, use `pip install` or `conda install` (no sudo needed)
- Training outputs are saved to `./outputs/` directory
- Model checkpoints and adapters are saved incrementally during training

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

## Project Structure

```
outputs/                 # Training outputs and logs
├── checkpoint-*/        # Model checkpoints saved during training
└── runs/               # TensorBoard logs

*.ipynb                 # Jupyter notebooks for training experiments
*.py                    # Python scripts (training, evaluation)
lora_model/            # Saved LoRA adapter weights
```

## Resources

- [2048 Notebook](https://colab.research.google.com/github/openai/gpt-oss/blob/main/examples/reinforcement-fine-tuning.ipynb)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Docs](https://docs.unsloth.ai)

## Working with This Project

### Before Training
1. Ensure you're in the `RL` conda environment
2. Verify GPU is accessible with the command above
3. Check available disk space - training generates large checkpoint files

### During Training
- Monitor progress with TensorBoard (see Common Commands)
- Training can take several hours depending on max_steps
- Checkpoints are saved periodically - don't interrupt mid-checkpoint

### After Training
- Evaluate models using evaluation scripts
- LoRA adapters can be merged with base model or used directly for inference
- Commit trained adapters and outputs to git if performance is good
