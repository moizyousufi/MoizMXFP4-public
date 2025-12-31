#!/bin/bash
# Installation script for MoizMXFP4 on Blackwell with QuTLASS

set -e

echo "=========================================="
echo "Installing MoizMXFP4 for Blackwell"
echo "=========================================="

# Check for CMake (required for QuTLASS build)
if ! command -v cmake &> /dev/null; then
    echo "❌ ERROR: CMake not found!"
    echo ""
    echo "QuTLASS requires CMake to build CUDA kernels."
    echo "Install with:"
    echo "  conda install cmake ninja"
    echo ""
    exit 1
fi

echo "✓ CMake found: $(cmake --version | head -n1)"
echo ""

# Step 1: Install base package (torch + triton)
echo ""
echo "Step 1/4: Installing MoizMXFP4 base..."
pip install -e .

# Step 2: Install QuTLASS (needs torch from step 1)
echo ""
echo "Step 2/4: Installing QuTLASS for native FP4..."

# Detect PyTorch CUDA version to avoid mismatch with system CUDA
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "PyTorch CUDA version: $TORCH_CUDA_VERSION"

# Point to conda's CUDA installation (matches PyTorch version)
if [ -d "$CONDA_PREFIX/pkgs/cuda-toolkit" ]; then
    export CUDA_HOME="$CONDA_PREFIX/pkgs/cuda-toolkit"
elif [ -d "$CONDA_PREFIX" ]; then
    # Search for CUDA in conda environment
    CUDA_PKG=$(find "$CONDA_PREFIX" -type d -path "*/pkgs/cuda-toolkit*" -o -path "*/pkgs/cuda_*" | head -n1)
    if [ -n "$CUDA_PKG" ]; then
        export CUDA_HOME="$CUDA_PKG"
    fi
fi

# If no conda CUDA found, use system CUDA (may cause version mismatch)
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "⚠ Warning: Using system CUDA (may mismatch PyTorch CUDA $TORCH_CUDA_VERSION)"
    export CUDA_HOME="/usr/local/cuda"
fi

echo "Using CUDA from: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Install moiz-qutlass (fork with critical bug fixes)
# Bug fix: Scale tensor dimension correction - fixes 32x memory over-allocation
# Uses CUTLASS backend for 2-3x speedup on MXFP4
# See: https://github.com/moizyousufi/moiz-qutlass-public/blob/main/BUGFIX.md
pip install --no-build-isolation git+https://github.com/moizyousufi/moiz-qutlass-public.git

# Step 3: Fix cuDNN conflicts (if any)
echo ""
echo "Step 3/4: Checking for cuDNN conflicts..."

# Check if pip installed conflicting cuDNN
if pip list 2>/dev/null | grep -q nvidia-cudnn-cu12; then
    echo "⚠️  Found conflicting pip cuDNN package (overrides conda cuDNN 9.15)"
    echo "   Removing nvidia-cudnn-cu12..."
    pip uninstall nvidia-cudnn-cu12 nvidia-cudnn-frontend -y 2>/dev/null || true
    echo "✅ Removed conflicting packages"
fi

# Ensure conda's cuDNN is prioritized
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify cuDNN version
CUDNN_VERSION=$(python -c "import torch; v=torch.backends.cudnn.version(); print(f'{v//1000}.{(v%1000)//100}.{v%100}')" 2>/dev/null || echo "unknown")
echo "PyTorch detected cuDNN: $CUDNN_VERSION"
echo "✅ cuDNN configured"

# Step 4: Verify installation
echo ""
echo "Step 4/4: Verifying installation..."
python -c "
import qutlass
import torch
print('✅ QuTLASS imported successfully')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'   GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})')
"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "QuTLASS CUTLASS backend provides 2-3x speedup on Blackwell."
echo ""
echo "To make cuDNN fix permanent (survives conda deactivate):"
echo "  mkdir -p \$CONDA_PREFIX/etc/conda/activate.d"
echo "  cat > \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'"
echo "  #!/bin/sh"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  EOF"
echo "  chmod +x \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo ""
echo "Next steps:"
echo "  - Test quantization: python test_correct_api.py"
echo "  - Run benchmarks: python benchmarks/benchmark_blackwell.py"
