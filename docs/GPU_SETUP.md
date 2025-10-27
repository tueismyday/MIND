# GPU Setup Guide for MIND System

This guide explains how to configure the MIND system to run the embedding and reranker models on the same GPU as your vLLM server.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Understanding GPU Memory Allocation](#understanding-gpu-memory-allocation)
4. [Configuration Options](#configuration-options)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

## Overview

The MIND system uses three main GPU-intensive components:

1. **vLLM Server**: Large language model (Qwen3-30B) for text generation
2. **Embedding Model**: Qwen3-Embedding-0.6B (~1.2GB) for semantic search
3. **Reranker Model**: Qwen3-Reranker-0.6B (~1.2GB) for relevance scoring

**Total GPU memory needed**: ~2.5-3GB for embedding + reranker (with buffer)

## Quick Start

### Step 1: Verify GPU Setup

Run the verification script to check if your GPU can support all models:

```bash
./scripts/launch_with_gpu.sh verify
```

Or directly:

```bash
python scripts/verify_gpu_setup.py
```

This script will:
- Check CUDA availability
- Show GPU memory status
- Test loading embedding and reranker models
- Provide configuration recommendations

### Step 2: Get Recommended Configuration

```bash
./scripts/launch_with_gpu.sh
```

This will show the recommended vLLM launch command based on your GPU memory.

### Step 3: Launch vLLM with Optimal Settings

**Option A: Use the launch script**
```bash
./scripts/launch_with_gpu.sh vllm
```

**Option B: Manual launch with recommended settings**

For a GPU with 24GB+ memory:
```bash
vllm serve cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.70 \
    --max-num-seqs 1 \
    --cpu-offload-gb 32
```

**Key change**: Reduced `--gpu-memory-utilization` from `0.90` to `0.70` to leave ~30% (7-8GB) for embedding/reranker models.

### Step 4: Launch Your Application

The embedding and reranker models will automatically detect available GPU memory and load on GPU if sufficient space is available. Otherwise, they'll fall back to CPU.

```bash
python main.py  # or your application entry point
```

## Understanding GPU Memory Allocation

### Your Current Setup (NOT WORKING)
```
┌─────────────────────────────────────┐
│  vLLM: 90% GPU                      │  ← vLLM using 90%
├─────────────────────────────────────┤
│  Free: 10%                          │  ← Only 10% free
│  Embedding + Reranker need ~12-15%  │  ← Not enough space!
└─────────────────────────────────────┘
Result: Out of memory error ❌
```

### Recommended Setup (WORKING)
```
┌─────────────────────────────────────┐
│  vLLM: 70% GPU                      │  ← vLLM using 70%
├─────────────────────────────────────┤
│  Free: 30% (~7-8GB)                 │  ← Enough space!
│  Embedding: ~5%                     │
│  Reranker: ~5%                      │
│  Buffer: ~20%                       │
└─────────────────────────────────────┘
Result: All models on GPU ✅
```

### GPU Memory Recommendations by Size

| GPU Memory | vLLM Utilization | Free for Models | Suitable? |
|------------|------------------|-----------------|-----------|
| 8GB        | 0.50 (50%)       | ~4GB            | Tight fit |
| 12GB       | 0.60 (60%)       | ~5GB            | Workable  |
| 16GB       | 0.65 (65%)       | ~6GB            | Good      |
| 24GB       | 0.70-0.75        | ~6-8GB          | Excellent |
| 40GB+      | 0.75-0.80        | ~8-10GB         | Ideal     |

## Configuration Options

### Automatic Device Selection (Default)

The system automatically selects the appropriate device based on available GPU memory:

**File**: `config/settings.py`

```python
# Automatic configuration (default)
EMBEDDING_DEVICE, RERANKER_DEVICE, DEVICE_INFO = get_device_config()
```

This function:
1. Checks if CUDA is available
2. Measures free GPU memory
3. Selects GPU if ≥2.5GB free, otherwise CPU
4. Prints device info on startup

### Manual Device Override

You can manually override device selection:

**File**: `config/settings.py`

```python
# Manual override - force GPU
EMBEDDING_DEVICE = "cuda"  # or "cuda:0" for specific GPU
RERANKER_DEVICE = "cuda"

# Manual override - force CPU
EMBEDDING_DEVICE = "cpu"
RERANKER_DEVICE = "cpu"

# Mixed configuration (if you have multiple GPUs)
EMBEDDING_DEVICE = "cuda:0"  # Use first GPU
RERANKER_DEVICE = "cuda:1"   # Use second GPU
```

**Note**: When manually setting to "cuda", the system will still fall back to CPU if GPU memory is insufficient.

## Troubleshooting

### Problem: "CUDA out of memory" error

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions**:

1. **Reduce vLLM memory usage** (recommended):
   ```bash
   vllm serve ... --gpu-memory-utilization 0.70  # or lower
   ```

2. **Check current GPU usage**:
   ```bash
   nvidia-smi
   ```

3. **Use CPU for embedding/reranker** (temporary workaround):
   Edit `config/settings.py`:
   ```python
   EMBEDDING_DEVICE = "cpu"
   RERANKER_DEVICE = "cpu"
   ```

4. **Clear GPU memory** (if vLLM is not running):
   ```bash
   # Kill any processes using GPU
   pkill -f vllm

   # Wait a moment, then restart
   ```

### Problem: Models loading on CPU despite GPU availability

**Symptoms**:
```
[WARNING] Insufficient GPU memory (X GB free, need ~2.5GB)
[WARNING] Falling back to CPU for embedding and reranker models
```

**Solutions**:

1. **Check if vLLM is using too much memory**:
   ```bash
   nvidia-smi  # Look at memory usage
   ```

2. **Reduce vLLM utilization**:
   ```bash
   vllm serve ... --gpu-memory-utilization 0.65  # Lower value
   ```

3. **Start embedding/reranker BEFORE vLLM**:
   - Start your application first
   - Then start vLLM
   - This allows embedding/reranker to claim memory first

### Problem: Slow performance on CPU

**Symptoms**:
- Embedding/reranker running on CPU
- Searches taking longer than expected

**Solutions**:

1. **Free up GPU memory** for models:
   - Reduce vLLM GPU utilization
   - Use smaller context lengths in vLLM
   - Reduce `--max-num-seqs` in vLLM

2. **Use multiple GPUs** if available:
   ```python
   # config/settings.py
   EMBEDDING_DEVICE = "cuda:1"  # Use second GPU
   RERANKER_DEVICE = "cuda:1"
   ```

   Then launch vLLM on first GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 vllm serve ...
   ```

### Problem: "No module named 'torch'" or CUDA not available

**Solutions**:

1. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Verify installation**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print GPU name
   ```

## Advanced Configuration

### Multiple GPU Setup

If you have multiple GPUs, you can distribute the workload:

**Option 1: vLLM on GPU 0, Embedding/Reranker on GPU 1**

```python
# config/settings.py
EMBEDDING_DEVICE = "cuda:1"
RERANKER_DEVICE = "cuda:1"
```

```bash
# Launch vLLM on GPU 0 only
CUDA_VISIBLE_DEVICES=0 vllm serve cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
    --max-model-len 14000 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 \
    --cpu-offload-gb 32
```

**Option 2: Split models across different GPUs**

```python
# config/settings.py
EMBEDDING_DEVICE = "cuda:0"
RERANKER_DEVICE = "cuda:1"
```

### Monitoring GPU Usage

**Real-time monitoring**:
```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run your application
python main.py
```

**Check memory allocation in Python**:
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
```

### Fine-tuning Memory Usage

**Reduce embedding model memory footprint**:

```python
# core/embeddings.py (add to model_kwargs)
model_kwargs = {
    'device': device,
    'model_kwargs': {
        'torch_dtype': torch.float16  # Use half precision
    }
}
```

**Reduce reranker batch size**:

```python
# tools/hybrid_search.py (in _calculate_cross_encoder_scores)
raw_scores = self.cross_encoder.predict(query_doc_pairs, batch_size=1)
# Reduce batch_size if memory is tight
```

## Performance Comparison

| Configuration | Embedding Speed | Reranker Speed | Trade-offs |
|---------------|----------------|----------------|------------|
| Both on GPU   | ~10ms/query    | ~15ms/doc      | Best performance, needs GPU memory |
| Both on CPU   | ~50ms/query    | ~80ms/doc      | Slower, but always works |
| Embedding GPU, Reranker CPU | ~10ms/query | ~80ms/doc | Good compromise |

## Best Practices

1. **Always run verification first**:
   ```bash
   ./scripts/launch_with_gpu.sh verify
   ```

2. **Start with conservative vLLM settings**:
   - Begin with `--gpu-memory-utilization 0.70`
   - Increase gradually if needed

3. **Monitor memory during operation**:
   - Use `nvidia-smi` to watch GPU memory
   - Look for memory spikes during searches

4. **Test your configuration**:
   - Run a few queries
   - Check response times
   - Verify models are on expected devices

5. **Keep fallback to CPU**:
   - Don't disable CPU fallback
   - It ensures system reliability

## Summary

The key to running all models on the same GPU is **proper memory allocation**:

1. ✅ **Reduce vLLM GPU utilization** from 0.90 to 0.70-0.75
2. ✅ **Let automatic device selection work** (default config)
3. ✅ **Verify setup before production use** (run verification script)
4. ✅ **Monitor GPU memory** during operation
5. ✅ **Keep CPU fallback enabled** for reliability

For most setups with 24GB+ GPU memory, using `--gpu-memory-utilization 0.70` for vLLM will allow all models to run on GPU successfully.
