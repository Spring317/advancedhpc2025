# GPU Financial Analytics - Troubleshooting Guide

## 🔧 Common Issues and Solutions

---

## 1. Environment Setup Issues

### Issue: CUDA Not Available
```
RuntimeError: CUDA is not available
```

**Solutions:**

**For Google Colab:**
```python
# Change runtime to GPU
# 1. Click "Runtime" → "Change runtime type"
# 2. Select "T4 GPU" or "GPU" from Hardware accelerator
# 3. Click "Save"
# 4. Restart runtime
```

**Verify GPU:**
```python
import cupy as cp
print(f"CUDA Available: {cp.cuda.is_available()}")

# If False, check:
!nvidia-smi  # Should show GPU details
```

**For Local Jupyter:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install CUDA toolkit if missing
# Visit: https://developer.nvidia.com/cuda-downloads
```

---

### Issue: Numba CUDA Not Working
```
ImportError: cannot import name 'cuda' from 'numba'
```

**Solution:**
```python
# Reinstall numba with CUDA support
!pip uninstall numba -y
!pip install numba --upgrade

# Enable PYNVJITLINK
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = True

# Verify
from numba import cuda
print(f"CUDA Available in Numba: {cuda.is_available()}")
```

---

### Issue: CuPy Installation Fails
```
ModuleNotFoundError: No module named 'cupy'
```

**Solution:**
```python
# For CUDA 11.x
!pip install cupy-cuda11x

# For CUDA 12.x
!pip install cupy-cuda12x

# For Colab (auto-detect)
!pip install cupy

# Verify installation
import cupy as cp
print(cp.__version__)
```

---

## 2. Data Loading Issues

### Issue: Fixture Files Not Found
```
FileNotFoundError: Test fixture file not found: cpu_test_fixtures.json
```

**Solutions:**

**For Google Colab:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload files to:
# /content/drive/MyDrive/HPC_Project_2025/Test_Fixtures/

# Or upload directly
from google.colab import files
uploaded = files.upload()  # Select cpu_test_fixtures.json
```

**For Local Jupyter:**
```python
# Ensure files are in same directory as notebook
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Copy fixtures to notebook directory
# cp path/to/cpu_test_fixtures.json .
# cp path/to/cpu_benchmarks_1M.json .
```

---

### Issue: JSON Decode Error
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution:**
```python
# Verify file content
with open('cpu_test_fixtures.json', 'r') as f:
    content = f.read()
    print(f"First 100 chars: {content[:100]}")

# File should start with '{' or '['
# If not, file may be corrupted - re-download/re-upload
```

---

## 3. Memory Issues

### Issue: Out of GPU Memory
```
cupy.cuda.memory.OutOfMemoryError: out of memory
```

**Solutions:**

**1. Clear GPU Memory:**
```python
import cupy as cp

# Clear memory pool
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

# Or restart runtime (Colab)
# Runtime → Restart runtime
```

**2. Reduce Data Size:**
```python
# Use smaller benchmark dataset
benchmark = GPUBenchmark(cpu_benchmarks, n_elements=100000)  # Instead of 1M
```

**3. Process in Batches:**
```python
# For large datasets, process segments in batches
def process_batches(data, batch_size=10000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        result = process_batch(batch)
        results.append(result)
    return cp.concatenate(results)
```

---

### Issue: Memory Leak
```
# Memory keeps growing across multiple runs
```

**Solution:**
```python
# Explicitly delete large arrays
del large_array
cp.get_default_memory_pool().free_all_blocks()

# Use context manager pattern
def compute_with_cleanup():
    try:
        data = cp.zeros(1000000)
        result = process(data)
        return result
    finally:
        del data
        cp.get_default_memory_pool().free_all_blocks()
```

---

## 4. Performance Issues

### Issue: Slow GPU Performance
```
# GPU slower than expected or slower than CPU
```

**Diagnostics:**
```python
# 1. Check GPU utilization
!nvidia-smi

# 2. Verify data is on GPU
import cupy as cp
data = cp.array([1, 2, 3])
print(f"On GPU: {isinstance(data, cp.ndarray)}")

# 3. Profile execution
import time
start = time.perf_counter()
result = gpu_function(data)
cp.cuda.Stream.null.synchronize()  # Important!
end = time.perf_counter()
print(f"Time: {(end-start)*1000:.2f} ms")
```

**Solutions:**

**1. Ensure GPU Synchronization:**
```python
# Always sync after GPU operations for timing
result = gpu_function(data)
cp.cuda.Stream.null.synchronize()  # Add this!
```

**2. Minimize CPU↔GPU Transfers:**
```python
# BAD: Multiple transfers
for i in range(100):
    cpu_data = np.array([i])
    gpu_data = cp.array(cpu_data)  # Transfer!
    result = process(gpu_data)
    cpu_result = result.get()  # Transfer!

# GOOD: One transfer
cpu_data = np.arange(100)
gpu_data = cp.array(cpu_data)  # One transfer
result = process(gpu_data)
cpu_result = result.get()  # One transfer
```

**3. Use Larger Batch Sizes:**
```python
# Too small - overhead dominates
benchmark = GPUBenchmark(cpu_benchmarks, n_elements=100)

# Better - amortize overhead
benchmark = GPUBenchmark(cpu_benchmarks, n_elements=1000000)
```

---

### Issue: No Speedup Over CPU
```
# GPU speedup is close to 1× instead of 100×
```

**Diagnostics:**
```python
# Check if actually using GPU
print(f"CuPy backend: {cp.cuda.is_available()}")

# Verify data type
print(f"Data type: {data.dtype}")  # Should be float32, not float64

# Check array device
print(f"Device: {data.device}")
```

**Solutions:**

**1. Use float32:**
```python
# float64 is slower on most GPUs
data = cp.array(values, dtype=cp.float32)  # Not float64
```

**2. Ensure GPU Arrays:**
```python
# If data is numpy, convert to cupy
if isinstance(data, np.ndarray):
    data = cp.array(data)
```

**3. Increase Problem Size:**
```python
# GPU shows benefits at larger scales
n_elements = 1000000  # Not 1000
```

---

## 5. Numerical Accuracy Issues

### Issue: Validation Fails
```
AssertionError: Arrays not equal
Max absolute error: 0.001
```

**Solutions:**

**1. Adjust Tolerance:**
```python
# If errors are small but consistent
tolerance = 1e-4  # Instead of 1e-5

np.testing.assert_allclose(gpu_result, cpu_result, 
                          rtol=tolerance, atol=tolerance)
```

**2. Check for NaN/Inf:**
```python
# Diagnose numerical issues
print(f"NaN count: {cp.isnan(result).sum()}")
print(f"Inf count: {cp.isinf(result).sum()}")

# Add guards
result = cp.where(cp.isnan(result), 0, result)
result = cp.where(cp.isinf(result), 0, result)
```

**3. Use Stable Algorithms:**
```python
# For variance, avoid subtraction of large numbers
# BAD:
variance = mean_sq - mean**2

# GOOD:
variance = cp.maximum(mean_sq - mean**2, 0)  # Guard against negative
```

---

### Issue: Log of Negative Number
```
RuntimeWarning: invalid value encountered in log
```

**Solution:**
```python
# Ensure positive values before log
returns = (prices[1:] - prices[:-1]) / prices[:-1]
log_returns = cp.log(1.0 + returns)  # 1.0 + ensures positive

# Or add epsilon
log_returns = cp.log(cp.maximum(1.0 + returns, 1e-10))
```

---

## 6. Test Execution Issues

### Issue: Tests Skip or Fail
```
s  # Test skipped
F  # Test failed
```

**Solutions:**

**1. Check Implementation:**
```python
# Ensure function is implemented
def my_function(data):
    raise NotImplementedError()  # Remove this!
    # Add actual implementation
```

**2. Verify Data Types:**
```python
# Ensure correct input types
flags = cp.array(flags, dtype=cp.int32)  # Not float
values = cp.array(values, dtype=cp.float32)
```

**3. Check Array Shapes:**
```python
# Debug shape mismatches
print(f"GPU result shape: {gpu_result.shape}")
print(f"CPU result shape: {len(cpu_result)}")

# Convert CPU list to array
cpu_result = np.array(cpu_result)
```

---

### Issue: Import Errors in Tests
```
NameError: name 'GPUFinancialPrimitives' is not defined
```

**Solution:**
```python
# Run cells in order!
# 1. Imports and setup
# 2. GPU Primitives definition
# 3. GPU Metrics definition
# 4. Test classes
# 5. Run tests

# Or check class name spelling
GPUFinancialPrimitives  # Capital letters matter!
```

---

## 7. Benchmark Issues

### Issue: Benchmark Crashes
```
RuntimeError: CUDA error during benchmark
```

**Solution:**
```python
# Add error handling
try:
    result = gpu_function(data)
except Exception as e:
    print(f"Error: {e}")
    # Clear GPU state
    cp.get_default_memory_pool().free_all_blocks()
    # Reduce data size
    data = data[:len(data)//2]
```

---

### Issue: Inconsistent Timing
```
# Results vary wildly between runs
```

**Solution:**
```python
# Increase burn-in and runs
benchmark.burnin_duration = 1.0  # Instead of 0.333
benchmark.n_runs = 100  # Instead of 50

# Always synchronize
cp.cuda.Stream.null.synchronize()
```

---

## 8. Visualization Issues

### Issue: Plots Don't Show
```
# No output after plt.show()
```

**Solution:**
```python
# For Jupyter/Colab, use inline backend
%matplotlib inline
import matplotlib.pyplot as plt

# Then create plots
plt.figure()
plt.plot(data)
plt.show()
```

---

### Issue: "No Data to Visualize"
```
No performance data to visualize
```

**Solution:**
```python
# Ensure benchmarks ran successfully
print(f"Benchmark results: {len(benchmark_results)}")
print(f"Successful: {sum(1 for r in benchmark_results.values() if r.get('status') == 'success')}")

# Check for errors
for name, result in benchmark_results.items():
    if result.get('status') == 'error':
        print(f"Error in {name}: {result.get('message')}")
```

---

## 9. Advanced Debugging

### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for CuPy
import os
os.environ['CUPY_DUMP_CUDA_SOURCE_ON_ERROR'] = '1'
```

### Profile GPU Performance
```python
from cupyx import profiler

@profiler.time_range()
def my_function(data):
    return process(data)

# Run with profiling
with profiler.time_range("benchmark", color_id=0):
    result = my_function(data)

# Print profile
print(profiler.get_elapsed_time())
```

### Check CUDA Kernel Compilation
```python
from numba import cuda

# Compile kernel explicitly
@cuda.jit
def my_kernel(data):
    idx = cuda.grid(1)
    if idx < len(data):
        data[idx] *= 2

# Check compilation
print(my_kernel.inspect_types())
```

---

## 10. Quick Fixes Checklist

Before asking for help, verify:

- [ ] GPU runtime enabled (Colab)
- [ ] CUDA available (`cp.cuda.is_available()`)
- [ ] Test fixtures loaded successfully
- [ ] All cells run in order
- [ ] No `NotImplementedError` in code
- [ ] Data types correct (float32, int32)
- [ ] Arrays on GPU (cp.ndarray, not np.ndarray)
- [ ] GPU synchronized for timing
- [ ] Memory cleared if needed
- [ ] Latest version of libraries installed

---

## 🆘 Still Having Issues?

### Systematic Debugging Approach

1. **Isolate the Problem:**
```python
# Test each component separately
flags = cp.array([1, 0, 0, 1], dtype=cp.int32)
seg_ids = GPUFinancialPrimitives.exclusive_scan(flags)
print(seg_ids)  # Should be [0, 0, 0, 1]
```

2. **Compare with Simple Case:**
```python
# Use minimal test case
simple_values = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
simple_seg_ids = cp.array([0, 0, 1, 1], dtype=cp.int32)
result = GPUFinancialPrimitives.segmented_scan_sum(simple_values, simple_seg_ids)
print(result)  # Should be [1.0, 3.0, 3.0, 7.0]
```

3. **Check Intermediate Results:**
```python
# Print shapes and values
print(f"Input shape: {data.shape}")
print(f"Input type: {data.dtype}")
print(f"First 10 values: {data[:10]}")
print(f"On GPU: {isinstance(data, cp.ndarray)}")
```

4. **Restart Fresh:**
```python
# Sometimes state gets corrupted
# Runtime → Restart runtime (Colab)
# Kernel → Restart (Jupyter)
# Then run all cells from top
```

---

**Most issues are resolved by:**
1. Ensuring GPU runtime is enabled
2. Running cells in correct order
3. Loading fixture files properly
4. Using correct data types (float32)
5. GPU synchronization for timing

**Good luck! 🚀**
