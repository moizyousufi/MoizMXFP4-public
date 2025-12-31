# MoizMXFP4 Installation Guide

## Quick Install

### Non-Blackwell GPUs (Triton backend)
```bash
pip install git+https://github.com/moizyousufi/MoizMXFP4.git
```

### Blackwell GPUs (QuTLASS + Triton)

**Option A: Using pip extras (requires CMake + CUDA)**
```bash
# Prerequisites: Install CMake and CUDA toolkit first
conda install cmake cuda-toolkit -c conda-forge

# Install with Blackwell acceleration
pip install "moiz-mxfp4[blackwell] @ git+https://github.com/moizyousufi/MoizMXFP4.git"
```

**Option B: Using install script (recommended, handles everything)**
```bash
git clone https://github.com/moizyousufi/MoizMXFP4.git
cd MoizMXFP4
./install_blackwell.sh
```

## What Each Method Does

### pip install (base)
- Installs: torch, triton, numpy
- Backend: Triton only
- Works on: All GPUs
- Performance: Good on Blackwell, slow on others

### pip install [blackwell]
- Installs: base + moiz-qutlass
- Backend: QuTLASS + Triton
- Works on: B200/B300 (SM100)
- Performance: 2-3x speedup
- Requirements:
  - CMake 3.x+
  - CUDA toolkit (matching PyTorch)
  - Blackwell GPU during installation

### install_blackwell.sh (recommended)
- Installs: Everything from [blackwell] extra
- Plus: Automatic CUDA environment setup
- Plus: cuDNN conflict resolution
- Plus: Installation verification
- Best for: First-time Blackwell setup

## Why install_blackwell.sh Still Exists

The shell script handles edge cases that pip can't:
1. **CUDA environment detection** - Finds conda/system CUDA and sets CUDA_HOME
2. **cuDNN conflict resolution** - Removes conflicting pip cudnn packages
3. **Build flag configuration** - Sets --no-build-isolation for qutlass
4. **Installation verification** - Tests import and GPU detection

**When to use it:**
- First time installing on Blackwell
- Encountering CUDA/cuDNN errors with pip
- Need reproducible environment setup

**When you can skip it:**
- Already have CMake and CUDA configured
- Installing in a known-good environment
- Using pre-built wheels (future)

## Future: Deprecating install_blackwell.sh

To fully deprecate the script, we would need:

1. **Pre-built wheels for qutlass**
   - Build wheels for each CUDA version (12.4, 12.6, 12.8, 13.0)
   - Publish to PyPI or GitHub releases
   - Requires CI/CD with Blackwell GPU runners

2. **Better pip integration**
   - Add setup.py with custom install hooks
   - Detect and validate CUDA environment
   - Auto-resolve cuDNN conflicts

3. **Conda package**
   - Publish to conda-forge
   - Let conda handle CUDA dependencies
   - Best option for production use

## Recommended Installation Flow

**For users:**
```bash
# Use the script - it just works
./install_blackwell.sh
```

**For developers/integrators:**
```bash
# Add to your environment.yml or requirements.txt
dependencies:
  - pip:
    - moiz-mxfp4[blackwell] @ git+https://github.com/moizyousufi/MoizMXFP4.git
```

**For Docker/CI:**
```dockerfile
# Install build deps first
RUN conda install cmake cuda-toolkit -c conda-forge

# Then install package
RUN pip install --no-build-isolation \
    "moiz-mxfp4[blackwell] @ git+https://github.com/moizyousufi/MoizMXFP4.git"
```
