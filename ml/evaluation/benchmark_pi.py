"""
Benchmark TFLite model on Raspberry Pi 5.
Measures latency, throughput, and stability metrics.
"""

import argparse
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List


def benchmark_tflite_model(
    model_path: str,
    input_shape: tuple = (320, 320, 3),
    num_runs: int = 1000,
    warmup_runs: int = 10,
    num_threads: int = 4
) -> Dict:
    """
    Benchmark TFLite model inference.
    
    Args:
        model_path: Path to TFLite model
        input_shape: Input shape (height, width, channels)
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        num_threads: Number of threads for inference
    
    Returns:
        Dictionary with benchmark metrics
    """
    # Load model
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare dummy input
    if input_details[0]['dtype'] == np.uint8:
        dummy_input = np.random.randint(0, 255, size=(1, *input_shape), dtype=np.uint8)
    else:
        dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
    
    # Warmup
    print(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    print(f"Running {num_runs} inference runs...")
    latencies = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        # Get outputs (to ensure full inference)
        _ = interpreter.get_tensor(output_details[0]['index'])
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_runs} runs")
    
    # Compute metrics
    latencies = np.array(latencies)
    
    metrics = {
        'latency_ms': {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        },
        'fps': {
            'mean': float(1000.0 / np.mean(latencies)),
            'min': float(1000.0 / np.max(latencies)),
            'max': float(1000.0 / np.min(latencies)),
        },
        'stability': {
            'cv': float(np.std(latencies) / np.mean(latencies)),  # Coefficient of variation
            'jitter_p95': float(np.percentile(np.abs(np.diff(latencies)), 95)),
        },
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TFLite model on Raspberry Pi 5"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to TFLite model"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[320, 320],
        help="Input size [height, width]"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1000,
        help="Number of inference runs"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for metrics"
    )
    
    args = parser.parse_args()
    
    input_shape = (*args.input_size, 3)
    
    print("="*50)
    print("RASPBERRY PI 5 BENCHMARK")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Input size: {input_shape}")
    print(f"Threads: {args.num_threads}")
    print(f"Runs: {args.num_runs}")
    print("="*50)
    
    metrics = benchmark_tflite_model(
        model_path=args.model,
        input_shape=input_shape,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        num_threads=args.num_threads,
    )
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"\nLatency (ms):")
    print(f"  Mean: {metrics['latency_ms']['mean']:.2f}")
    print(f"  Std:  {metrics['latency_ms']['std']:.2f}")
    print(f"  P50:  {metrics['latency_ms']['p50']:.2f}")
    print(f"  P95:  {metrics['latency_ms']['p95']:.2f}")
    print(f"  P99:  {metrics['latency_ms']['p99']:.2f}")
    print(f"  Min:  {metrics['latency_ms']['min']:.2f}")
    print(f"  Max:  {metrics['latency_ms']['max']:.2f}")
    
    print(f"\nFPS:")
    print(f"  Mean: {metrics['fps']['mean']:.2f}")
    print(f"  Min:  {metrics['fps']['min']:.2f}")
    print(f"  Max:  {metrics['fps']['max']:.2f}")
    
    print(f"\nStability:")
    print(f"  CV (Coefficient of Variation): {metrics['stability']['cv']:.4f}")
    print(f"  Jitter P95: {metrics['stability']['jitter_p95']:.2f} ms")
    
    # Check against target
    target_latency_ms = 50.0
    mean_latency = metrics['latency_ms']['mean']
    p95_latency = metrics['latency_ms']['p95']
    
    print(f"\nTarget Latency: {target_latency_ms} ms")
    if mean_latency <= target_latency_ms:
        print(f"✓ Mean latency ({mean_latency:.2f} ms) meets target")
    else:
        print(f"✗ Mean latency ({mean_latency:.2f} ms) exceeds target")
    
    if p95_latency <= target_latency_ms * 1.5:
        print(f"✓ P95 latency ({p95_latency:.2f} ms) is acceptable")
    else:
        print(f"✗ P95 latency ({p95_latency:.2f} ms) is too high")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
