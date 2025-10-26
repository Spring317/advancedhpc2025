# GPU Financial Analytics - Quick Reference Card

## 🔧 GPU Primitives API

### Segment ID Generation
```python
seg_ids = GPUFinancialPrimitives.exclusive_scan(flags)
```
**Input**: Binary flags array [1,0,0,1,0,0,1]  
**Output**: Segment IDs [0,0,0,1,1,1,2]

---

### Segmented Scan Operations

#### Sum
```python
cumulative = GPUFinancialPrimitives.segmented_scan_sum(values, seg_ids)
```
**Example**:
- Values: [1, 2, 3, 4, 5, 6]
- Seg_ids: [0, 0, 0, 1, 1, 1]
- Result: [1, 3, 6, 4, 9, 15]

#### Maximum
```python
running_max = GPUFinancialPrimitives.segmented_scan_max(values, seg_ids)
```
**Example**:
- Values: [5, 2, 8, 3, 9, 4]
- Seg_ids: [0, 0, 0, 1, 1, 1]
- Result: [5, 5, 8, 3, 9, 9]

---

### Segmented Reduce Operations

#### Sum per Segment
```python
totals = GPUFinancialPrimitives.segmented_reduce_sum(values, seg_ids)
```
**Example**:
- Values: [1, 2, 3, 4, 5, 6]
- Seg_ids: [0, 0, 0, 1, 1, 1]
- Result: [6, 15]  # One value per segment

#### Max per Segment
```python
maximums = GPUFinancialPrimitives.segmented_reduce_max(values, seg_ids)
```

#### Min per Segment
```python
minimums = GPUFinancialPrimitives.segmented_reduce_min(values, seg_ids)
```

---

## 💰 Financial Metrics API

### Cumulative Returns
```python
returns = GPUFinancialMetrics.cumulative_returns(prices, seg_ids)
```
**Formula**: R = exp(∑ ln(1 + r)) - 1  
**Use Case**: Total return over time periods

---

### Simple Moving Average
```python
sma = GPUFinancialMetrics.simple_moving_average(prices, seg_ids, window=5)
```
**Parameters**:
- `prices`: Price time series
- `seg_ids`: Segment identifiers
- `window`: Moving average window (default: 5)

**Example**: 5-day moving average of stock prices

---

### Rolling Standard Deviation
```python
volatility = GPUFinancialMetrics.rolling_std(prices, seg_ids, window=20)
```
**Parameters**:
- `window`: Rolling window size (default: 20)

**Use Case**: Historical volatility calculation

---

### Maximum Drawdown
```python
drawdown = GPUFinancialMetrics.max_drawdown(prices, seg_ids)
```
**Formula**: DD = (Current - Peak) / Peak  
**Use Case**: Risk assessment, downside analysis

---

### Portfolio Value
```python
portfolio_val = GPUFinancialMetrics.portfolio_value(holdings, prices)
```
**Input Shapes**:
- `holdings`: [M securities × N days]
- `prices`: [M securities × N days]

**Output**: [N days] total portfolio value

---

### High-Water Mark
```python
hwm = GPUFinancialMetrics.high_water_mark(portfolio_values, seg_ids)
```
**Use Case**: Performance fee calculation basis

---

## 🧪 Testing & Validation

### Run Unit Tests
```python
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored', '-vv'], exit=False)
```

### Numerical Validation
```python
validator = NumericalValidator(test_suite)
validation_results = validator.run()
```

### Performance Benchmarking
```python
benchmark = GPUBenchmark(cpu_benchmarks, n_elements=1000000)
benchmark_results = benchmark.run()
```

---

## 📊 Analysis & Reporting

### Generate Performance Report
```python
analyzer = PerformanceAnalyzer(benchmark_results, validation_results)
analyzer.generate_report()
```

### Create Visualizations
```python
visualize_performance(benchmark_results)
```

---

## 🎯 Expected Performance

| Metric | Target Speedup | Notes |
|--------|---------------|-------|
| Exclusive Scan | 100-150× | Parallel prefix sum |
| Segmented Scan | 100-200× | Depends on segment size |
| Segmented Reduce | 150-300× | Efficient reduction |
| Cumulative Returns | 80-150× | Log-space computation |
| Moving Average | 100-200× | Windowed operations |
| Rolling Std | 50-100× | Dual scan overhead |
| Max Drawdown | 100-200× | Running maximum |
| Portfolio Value | 200-500× | Element-wise ops |

*Note: Google Colab targets may be 2× higher*

---

## 🔍 Quick Diagnostics

### Check GPU Availability
```python
import cupy as cp
print(f"CUDA Available: {cp.cuda.is_available()}")
print(f"Device: {cp.cuda.Device()}")
```

### Memory Usage
```python
mempool = cp.get_default_memory_pool()
print(f"GPU Memory: {mempool.used_bytes() / 1e9:.2f} GB")
```

### Verify Test Fixtures
```python
print(f"Test categories: {list(test_suite.get('tests', {}).keys())}")
print(f"Benchmark categories: {list(cpu_benchmarks.get('cpu_benchmarks').keys())}")
```

---

## ⚠️ Common Pitfalls

1. **Segment Boundaries**: Always ensure operations respect segment IDs
2. **Data Types**: Use `float32` for GPU efficiency
3. **Memory Transfer**: Minimize CPU↔GPU transfers
4. **Numerical Stability**: Use log-space for products
5. **Window Edges**: Handle partial windows at boundaries

---

## 💡 Optimization Tips

1. **Use CuPy Built-ins**: `cumsum`, `maximum.accumulate` are highly optimized
2. **Batch Operations**: Process multiple segments together
3. **Avoid Loops**: Use vectorized operations where possible
4. **Segment Masking**: `cp.where()` for conditional operations
5. **Memory Reuse**: Allocate once, reuse buffers

---

## 📖 Formula Reference

### Cumulative Returns
```
R_t = exp(∑[i=1 to t] ln(1 + r_i)) - 1
where r_i = (P_i - P_{i-1}) / P_{i-1}
```

### Moving Average
```
SMA_t = (∑[i=t-w+1 to t] P_i) / w
```

### Rolling Standard Deviation
```
σ_t = sqrt(E[P²] - E[P]²)
where E[P²] = (∑ P_i²) / w
      E[P] = (∑ P_i) / w
```

### Maximum Drawdown
```
DD_t = (P_t - max[i≤t](P_i)) / max[i≤t](P_i)
```

### Portfolio Value
```
V_t = ∑[k=1 to M] h_{k,t} × P_{k,t}
```

---

## 🚀 Quick Start Checklist

- [ ] Install dependencies (`numba`, `cupy`)
- [ ] Verify CUDA availability
- [ ] Load test fixtures
- [ ] Run environment detection cell
- [ ] Implement GPU primitives
- [ ] Implement financial metrics
- [ ] Run unit tests
- [ ] Execute benchmarks
- [ ] Generate validation report
- [ ] Create visualizations
- [ ] Review performance analysis

---

**Happy GPU Programming! 🎉**
