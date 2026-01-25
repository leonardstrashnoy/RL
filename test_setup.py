#!/usr/bin/env python3
"""Verify RLVR environment setup."""

import sys

def check_import(module_name, min_version=None):
    """Check if a module can be imported and optionally verify version."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"  {module_name}: FAILED - {e}")
        return False

def main():
    print("=" * 50)
    print("RLVR Environment Verification")
    print("=" * 50)

    all_passed = True

    # Check Python version
    print(f"\nPython: {sys.version}")

    # Check core packages
    print("\nCore Packages:")
    packages = ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]
    for pkg in packages:
        if not check_import(pkg):
            all_passed = False

    # Check Unsloth (must be imported first)
    print("\nUnsloth:")
    if not check_import("unsloth"):
        all_passed = False

    # Check CUDA
    print("\nGPU Status:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  WARNING: No GPU detected - training will be slow")
    except Exception as e:
        print(f"  ERROR checking GPU: {e}")
        all_passed = False

    # Check Jupyter
    print("\nJupyter:")
    check_import("jupyter")
    check_import("ipykernel")

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("All checks passed! Ready to train.")
    else:
        print("Some checks failed. Review errors above.")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
