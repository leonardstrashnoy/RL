.PHONY: setup jupyter tensorboard test clean help

help:
	@echo "RLVR Project Commands"
	@echo ""
	@echo "  make setup       - Create conda environment and install dependencies"
	@echo "  make jupyter     - Start Jupyter Lab"
	@echo "  make tensorboard - Start TensorBoard to monitor training"
	@echo "  make test        - Verify installation"
	@echo "  make clean       - Remove cache and output files"
	@echo ""

setup:
	./setup.sh

jupyter:
	jupyter lab

tensorboard:
	tensorboard --logdir=outputs

test:
	python test_setup.py

clean:
	rm -rf outputs/
	rm -rf unsloth_compiled_cache/
	rm -rf __pycache__/
	rm -rf .ipynb_checkpoints/
	find . -type f -name "*.pyc" -delete
