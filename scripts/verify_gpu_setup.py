#!/usr/bin/env python3
"""
GPU Setup Verification Script for MIND System

This script verifies that the embedding and reranker models can be loaded on GPU
and provides recommendations for vLLM memory configuration.

Usage:
    python scripts/verify_gpu_setup.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_cuda_availability():
    """Check if CUDA is available and print GPU information."""
    print("=" * 80)
    print("CUDA AVAILABILITY CHECK")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available!")
        print("Possible reasons:")
        print("  - No NVIDIA GPU detected")
        print("  - CUDA drivers not installed")
        print("  - PyTorch not installed with CUDA support")
        return False

    print(f"[SUCCESS] CUDA is available!")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total_mem - reserved

        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {total_mem:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Free: {free:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print()

    return True

def test_embedding_loading():
    """Test loading the embedding model."""
    print("=" * 80)
    print("EMBEDDING MODEL LOADING TEST")
    print("=" * 80)

    try:
        from core.embeddings import get_embeddings
        from config.settings import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME

        print(f"Model: {EMBEDDING_MODEL_NAME}")
        print(f"Target Device: {EMBEDDING_DEVICE}")
        print()

        embeddings = get_embeddings()

        if embeddings:
            print("[SUCCESS] Embedding model loaded successfully!")

            # Test encoding
            test_text = "This is a test sentence for embedding."
            print(f"\nTesting encoding: '{test_text}'")
            # Note: HuggingFaceEmbeddings doesn't expose the model directly,
            # so we test it through the embed_query method
            result = embeddings.embed_query(test_text)
            print(f"[SUCCESS] Encoding test passed! Vector dimension: {len(result)}")

            return True
        else:
            print("[ERROR] Failed to load embedding model")
            return False

    except Exception as e:
        print(f"[ERROR] Exception during embedding loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_reranker_loading():
    """Test loading the reranker model."""
    print("=" * 80)
    print("RERANKER MODEL LOADING TEST")
    print("=" * 80)

    try:
        from sentence_transformers import CrossEncoder
        from config.settings import RERANKER_DEVICE, RERANKER_MODEL_NAME
        import torch

        print(f"Model: {RERANKER_MODEL_NAME}")
        print(f"Target Device: {RERANKER_DEVICE}")
        print()

        # Load model with device handling
        device = RERANKER_DEVICE
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available, using CPU")
            device = "cpu"

        if device.startswith("cuda"):
            gpu_idx = 0 if device == "cuda" else int(device.split(":")[1])
            props = torch.cuda.get_device_properties(gpu_idx)
            free_mem = (props.total_memory - torch.cuda.memory_reserved(gpu_idx)) / (1024**3)
            print(f"[INFO] GPU free memory before loading: {free_mem:.2f}GB")

        cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=device)
        print(f"[SUCCESS] Reranker loaded successfully on {device}!")

        # Test prediction
        query = "What is the patient's blood pressure?"
        document = "The patient's blood pressure is 120/80 mmHg."
        print(f"\nTesting reranking...")
        print(f"Query: '{query}'")
        print(f"Document: '{document}'")

        score = cross_encoder.predict([(query, document)])
        print(f"[SUCCESS] Reranking test passed! Relevance score: {score[0]:.4f}")

        if device.startswith("cuda"):
            gpu_idx = 0 if device == "cuda" else int(device.split(":")[1])
            allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            print(f"[INFO] GPU memory allocated: {allocated:.2f}GB")

        return True

    except Exception as e:
        print(f"[ERROR] Exception during reranker loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_memory_for_models():
    """Check if there's enough GPU memory for all models."""
    print("=" * 80)
    print("GPU MEMORY ANALYSIS")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, models will run on CPU")
        return

    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    free = total_mem - reserved

    print(f"Total GPU Memory: {total_mem:.2f} GB")
    print(f"Currently Reserved: {reserved:.2f} GB")
    print(f"Currently Free: {free:.2f} GB")
    print()

    # Estimate memory requirements
    embedding_mem = 1.2  # ~0.6B parameters * 2 bytes (float16) ≈ 1.2GB
    reranker_mem = 1.2   # ~0.6B parameters * 2 bytes (float16) ≈ 1.2GB
    total_needed = embedding_mem + reranker_mem + 0.5  # +0.5GB buffer

    print(f"Estimated Memory Requirements:")
    print(f"  Embedding Model: ~{embedding_mem:.1f} GB")
    print(f"  Reranker Model: ~{reranker_mem:.1f} GB")
    print(f"  Buffer: ~0.5 GB")
    print(f"  Total Needed: ~{total_needed:.1f} GB")
    print()

    if free >= total_needed:
        print(f"[SUCCESS] Sufficient GPU memory available ({free:.2f} GB >= {total_needed:.1f} GB)")
    else:
        print(f"[WARNING] Insufficient GPU memory ({free:.2f} GB < {total_needed:.1f} GB)")
        print()
        print("RECOMMENDATIONS:")
        print("1. Reduce vLLM GPU memory utilization:")
        print(f"   Current recommended: --gpu-memory-utilization 0.70 to 0.75")
        print(f"   This will leave ~{total_mem * 0.25:.1f}-{total_mem * 0.30:.1f} GB for embedding/reranker")
        print()
        print("2. Alternative: Use separate GPUs if available:")
        print("   - vLLM on cuda:0")
        print("   - Embedding/Reranker on cuda:1")
        print()
        print("3. Keep models on CPU (slower but reliable):")
        print("   - Models will automatically fall back to CPU if GPU memory is insufficient")

def print_vllm_recommendations():
    """Print recommendations for vLLM configuration."""
    print("=" * 80)
    print("vLLM CONFIGURATION RECOMMENDATIONS")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, vLLM recommendations skipped")
        return

    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024**3)

    print(f"For GPU with {total_mem:.2f} GB total memory:")
    print()
    print("RECOMMENDED vLLM LAUNCH COMMAND:")
    print("-" * 80)

    # Calculate recommended utilization
    models_mem = 2.9  # embedding + reranker + buffer
    recommended_util = min(0.75, (total_mem - models_mem) / total_mem)

    print(f"vllm serve cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \\")
    print(f"    --max-model-len 14000 \\")
    print(f"    --gpu-memory-utilization {recommended_util:.2f} \\  # Reduced from 0.90 to leave room")
    print(f"    --max-num-seqs 1 \\")
    print(f"    --cpu-offload-gb 32")
    print()
    print(f"This configuration leaves ~{total_mem * (1 - recommended_util):.1f} GB for embedding/reranker models")
    print()

    print("ALTERNATIVE CONFIGURATIONS:")
    print("-" * 80)
    print()
    print("Option 1: More aggressive GPU sharing (if sufficient VRAM)")
    print(f"    --gpu-memory-utilization 0.70  # Leaves ~{total_mem * 0.30:.1f} GB")
    print()
    print("Option 2: Use multiple GPUs (if available)")
    print("    vllm serve ... --tensor-parallel-size 1 --gpu-memory-utilization 0.90")
    print("    Set in config/settings.py: EMBEDDING_DEVICE='cuda:1', RERANKER_DEVICE='cuda:1'")
    print()
    print("Option 3: Keep models on CPU (simplest, slightly slower)")
    print("    vllm serve ... --gpu-memory-utilization 0.90")
    print("    Models will automatically use CPU if GPU memory insufficient")

def main():
    """Main verification routine."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MIND SYSTEM - GPU SETUP VERIFICATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    results = {
        'cuda': False,
        'embedding': False,
        'reranker': False
    }

    # Check CUDA
    results['cuda'] = check_cuda_availability()
    print()

    # Check memory
    check_gpu_memory_for_models()
    print()

    # Test embedding
    results['embedding'] = test_embedding_loading()
    print()

    # Test reranker
    results['reranker'] = test_reranker_loading()
    print()

    # Print vLLM recommendations
    print_vllm_recommendations()
    print()

    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    print(f"CUDA Available: {'✓ PASS' if results['cuda'] else '✗ FAIL'}")
    print(f"Embedding Model: {'✓ PASS' if results['embedding'] else '✗ FAIL'}")
    print(f"Reranker Model: {'✓ PASS' if results['reranker'] else '✗ FAIL'}")
    print()

    if all(results.values()):
        print("[SUCCESS] All checks passed! Your GPU setup is ready.")
        return 0
    else:
        print("[WARNING] Some checks failed. Review the output above for details.")
        print("The system will still work, but may fall back to CPU for some models.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
