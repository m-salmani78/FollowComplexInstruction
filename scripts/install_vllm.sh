#!/bin/bash

echo "Installing vLLM for efficient LLM inference..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing vLLM with CUDA support..."
    
    # First, uninstall existing PyTorch to avoid conflicts
    echo "Uninstalling existing PyTorch packages..."
    pip uninstall -y torch torchvision torchaudio
    
    # Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install vLLM
    echo "Installing vLLM..."
    pip install vllm
else
    echo "No CUDA detected, installing CPU-only version..."
    pip install vllm-cpu-only
fi

# Install additional dependencies
pip install --upgrade transformers
pip install --upgrade accelerate

echo "vLLM installation completed!"
echo ""
echo "Usage example:"
echo "python vllm_inference.py --data_path=data.jsonl --res_path=results.jsonl --model_path=google/gemma-3-4b-it"