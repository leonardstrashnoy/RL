# RLVR - Reinforcement Learning with Verifiable Rewards

Fine-tune LLMs using GRPO (Group Relative Policy Optimization) to learn the 2048 game. Based on [Matthew Berman's tutorial](https://www.youtube.com/watch?v=9t-BAjzBWj8) using NVIDIA and Unsloth.

## Overview

This project trains OpenAI's GPT-OSS-20B model to autonomously generate Python strategies for playing the 2048 game. The model learns through reinforcement learning by receiving rewards based on:

- **Syntax validity** (+1.0): Code must be valid Python
- **No cheating** (-20.0 penalty): No external library imports allowed
- **Game success** (+20.0): Successfully reaching 2048 tile

## Requirements

- NVIDIA GPU with CUDA support (tested on GB10/DGX Spark with CUDA 13.0)
- Miniconda or Anaconda
- ~15GB VRAM (with 4-bit quantization)

## Installation

### Quick Setup

```bash
conda create -n RL python=3.11 -y
conda activate RL

# PyTorch with CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Core dependencies
pip install transformers==4.56.2 datasets accelerate peft bitsandbytes

# Unsloth and TRL
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth" --no-deps trl==0.22.2

# Jupyter and monitoring
pip install jupyter ipykernel tensorboard
python -m ipykernel install --user --name=RL --display-name="Python (RL)"
```

Or run the setup script:
```bash
chmod +x setup.sh && ./setup.sh
```

## Usage

### Start Training

```bash
conda activate RL
jupyter lab
```

Open `gpt_oss_20B_RL_2048_Game.ipynb` and run the cells.

### Monitor Training

```bash
tensorboard --logdir=outputs
```

### Verify Installation

```bash
python -c "import unsloth; import torch; print(f'Unsloth: {unsloth.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Configuration

Key training parameters in the notebook:

```python
# Model
max_seq_length = 768
load_in_4bit = True
lora_rank = 4

# GRPO Training
learning_rate = 5e-5
per_device_train_batch_size = 1
num_generations = 2
max_steps = 1000
save_steps = 100
```

## Project Structure

```
RL/
├── README.md                        # This file
├── CLAUDE.md                        # Claude Code guidance
├── gpt_oss_20B_RL_2048_Game.ipynb   # Main training notebook
├── setup.sh                         # Installation script
├── outputs/                         # Training checkpoints (created during training)
└── cmds                             # Resource links
```

## Resources

- [OpenAI GPT-OSS Notebook](https://colab.research.google.com/github/openai/gpt-oss/blob/main/examples/reinforcement-fine-tuning.ipynb)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [NVIDIA Blog: Fine-Tuning with Unsloth](https://blogs.nvidia.com/blog/rtx-ai-garage-fine-tuning-unsloth-dgx-spark/)

## License

This project uses open-source tools. See individual repositories for licensing.
