import cv2
import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # FIXED: Import 'Config' instead of 'InspectionConfig'
    from main import InferenceEngine, Config
except ImportError as e:
    print(f"ERROR: Could not import main.py: {e}")
    print("Make sure this script is in the same directory as main.py")
    sys.exit(1)


def benchmark_engine(engine_type: str, num_iterations: int = 100) -> dict:
    """Benchmark a specific inference engine"""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING {engine_type.upper()} ENGINE")
    print(f"{'='*70}")
    
    try:
        # Initialize engine
        # FIXED: Removed 'labels_path' argument (InferenceEngine handles it internally now)
        engine = InferenceEngine(engine_type=engine_type)
        
        # Create dummy test image (224x224 RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warmup runs
        print("Warming up engine...")
        for _ in range(10):
            engine.predict(test_image)
        
        # Actual benchmark
        print(f"Running {num_iterations} iterations...")
        inference_times = []
        
        for i in range(num_iterations):
            start = time.perf_counter()
            # FIXED: predict() now returns only a dictionary (probs), not a tuple
            _ = engine.predict(test_image)
            end = time.perf_counter()
            
            inference_time_ms = (end - start) * 1000
            inference_times.append(inference_time_ms)
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        fps = 1000 / avg_time
        
        results = {
            'engine': engine_type,
            'avg_latency_ms': avg_time,
            'min_latency_ms': np.min(inference_times),
            'max_latency_ms': np.max(inference_times),
            'std_latency_ms': np.std(inference_times),
            'fps': fps
        }
        
        print(f"\n✅ {engine_type.upper()} Benchmark Complete")
        print(f"  Average Latency: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error benchmarking {engine_type}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_comparison_table(tflite_results: dict, openvino_results: dict):
    """Generate formatted comparison table"""
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    if not tflite_results or not openvino_results:
        print("ERROR: Missing benchmark results")
        return
    
    # Calculate speedup
    speedup = tflite_results['avg_latency_ms'] / openvino_results['avg_latency_ms']
    fps_improvement = (openvino_results['fps'] / tflite_results['fps'] - 1) * 100
    
    print(f"\n{'Metric':<30} {'TFLite':<15} {'OpenVINO':<15} {'Improvement'}")
    print("-" * 70)
    print(f"{'Average Latency (ms)':<30} {tflite_results['avg_latency_ms']:>14.2f} {openvino_results['avg_latency_ms']:>14.2f} {speedup:>9.2f}x")
    print(f"{'FPS':<30} {tflite_results['fps']:>14.2f} {openvino_results['fps']:>14.2f} {fps_improvement:>8.1f}%")
    print("-" * 70)
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"  • OpenVINO is {speedup:.2f}x FASTER than TFLite")
    print(f"  • FPS improved by {fps_improvement:.1f}%")


def main():
    print("\n" + "="*70)
    print("READ AUTOMATION - PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Run TFLite benchmark
    tflite_results = benchmark_engine("tflite")
    
    # Run OpenVINO benchmark
    openvino_results = benchmark_engine("openvino")
    
    # Generate comparison
    if tflite_results and openvino_results:
        generate_comparison_table(tflite_results, openvino_results)

if __name__ == "__main__":
    main()