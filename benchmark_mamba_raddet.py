import torch
import torch.nn as nn
import time
import numpy as np
from models.mamba_raddet import RadarMamba_fpn

try:
    from fvcore.nn import FlopCountAnalysis
    fvcore_available = True
except ImportError:
    print("Warning: fvcore not available. GFLOPs calculation will be skipped.")
    fvcore_available = False

def benchmark_model(model, input_shape=(1, 2, 64, 256, 256), device='cuda', warmup_runs=10, test_runs=100):
    """
    Benchmark the model for latency, FPS, and GPU memory usage

    Args:
        model: PyTorch model to benchmark
        input_shape: Shape of input tensor (B, C, D, H, W)
        device: Device to run on ('cuda' or 'cpu')
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations

    Returns:
        dict: Dictionary containing latency, fps, and gpu_memory metrics
    """
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize before measuring
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure latency
    print(f"Running {test_runs} test iterations...")
    latencies = []

    with torch.no_grad():
        for _ in range(test_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    fps = 1000.0 / avg_latency

    # Measure GPU memory
    gpu_memory_mb = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    results = {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'fps': fps,
        'gpu_memory_mb': gpu_memory_mb
    }

    return results

def print_results(results, params_millions=0, gflops=0):
    """Print benchmark results in a formatted table"""
    print("\n" + "="*60)
    print("Model Benchmark Results")
    print("="*60)
    if params_millions > 0:
        print(f"Parameters:      {params_millions:.2f} M")
    if gflops > 0:
        print(f"GFLOPs:          {gflops:.2f}")
    print(f"\nLatency (ms):")
    print(f"  Average:       {results['avg_latency_ms']:.2f} ms")
    print(f"  Std Dev:       {results['std_latency_ms']:.2f} ms")
    print(f"  Min:           {results['min_latency_ms']:.2f} ms")
    print(f"  Max:           {results['max_latency_ms']:.2f} ms")
    print(f"\nFPS:             {results['fps']:.2f}")
    print(f"\nGPU Memory:      {results['gpu_memory_mb']:.2f} MB")
    print("="*60)

if __name__ == '__main__':
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. GPU benchmarks require CUDA.")

    # Initialize model
    print("\nInitializing RadarMamba_fpn model...")
    model = RadarMamba_fpn(depth=[2, 2, 2], embed_dim=[48, 96, 192])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_millions = total_params / 1e6
    print(f"Total Parameters: {total_params_millions:.2f}M")

    # Calculate GFLOPs
    gflops = 0
    if fvcore_available:
        print("\nCalculating GFLOPs...")
        input_shape = (1, 2, 64, 256, 256)
        dummy_input = torch.randn(input_shape).to(device)
        model_for_flops = model.to(device)
        try:
            flop_counter = FlopCountAnalysis(model_for_flops, dummy_input)
            gflops = flop_counter.total() / 1e9
            print(f"GFLOPs: {gflops:.2f}")
        except Exception as e:
            print(f"GFLOPs calculation failed: {e}")
            # Try alternative method without fvcore
            print("Trying alternative GFLOPs calculation...")
            try:
                from thop import profile
                dummy_input_cpu = torch.randn(input_shape)
                model_cpu = RadarMamba_fpn(depth=[2, 2, 2], embed_dim=[48, 96, 192])
                flops, params = profile(model_cpu, inputs=(dummy_input_cpu,), verbose=False)
                gflops = flops / 1e9
                print(f"GFLOPs (thop): {gflops:.2f}")
            except Exception as e2:
                print(f"Alternative calculation also failed: {e2}")
                gflops = 0
    else:
        print("\nGFLOPs calculation skipped (fvcore not installed)")


    # Run benchmark
    input_shape = (1, 2, 64, 256, 256)
    print(f"Input shape: {input_shape}")

    results = benchmark_model(
        model=model,
        input_shape=input_shape,
        device=device,
        warmup_runs=10,
        test_runs=100
    )

    # Print results
    print_results(results, params_millions=total_params_millions, gflops=gflops)

    # Save results to file
    output_file = "benchmark_results_mamba_raddet.txt"
    with open(output_file, 'w') as f:
        f.write("Model Benchmark Results - RadarMamba_fpn\n")
        f.write("="*60 + "\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Total Parameters: {total_params_millions:.2f}M\n")
        if gflops > 0:
            f.write(f"GFLOPs: {gflops:.2f}\n")
        f.write(f"\nLatency (ms):\n")
        f.write(f"  Average:       {results['avg_latency_ms']:.2f} ms\n")
        f.write(f"  Std Dev:       {results['std_latency_ms']:.2f} ms\n")
        f.write(f"  Min:           {results['min_latency_ms']:.2f} ms\n")
        f.write(f"  Max:           {results['max_latency_ms']:.2f} ms\n")
        f.write(f"\nFPS:             {results['fps']:.2f}\n")
        f.write(f"\nGPU Memory:      {results['gpu_memory_mb']:.2f} MB\n")
        f.write("="*60 + "\n")

    print(f"\nResults saved to {output_file}")
