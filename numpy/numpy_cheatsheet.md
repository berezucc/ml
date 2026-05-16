# NumPy Study Cheat Sheet
---

## 1. Array Creation & Properties (5-10 min)

### Core Functions
```python
np.array([...])           # From list/tuple
np.zeros(shape)           # All zeros
np.ones(shape)            # All ones
np.arange(start, stop, step)  # Like range()
np.linspace(start, stop, n)   # n evenly spaced points
np.full(shape, value)     # Fill with constant
np.eye(n)                 # Identity matrix
np.random.rand(shape)     # Uniform [0, 1)
np.random.randn(shape)    # Standard normal
```

### Properties (know these cold)
```python
arr.shape          # Tuple of dimensions
arr.dtype          # Data type
arr.ndim           # Number of dimensions
arr.size           # Total elements
arr.T              # Transpose
len(arr)           # Length of first axis
```

**Key insight**: NumPy arrays are fixed-size, typed, and contiguous in memory (fast).

---

## 2. Indexing & Slicing (10-15 min)

### Single/Multi-dimensional
```python
arr[0]             # First element
arr[-1]            # Last element
arr[1:3]           # Slice: indices 1, 2 (not 3)
arr[1:3, 2:5]      # 2D slice
arr[arr > 5]       # Boolean mask (returns flattened view)
arr[[0, 2, 4]]     # Fancy indexing: specific indices
```

### Important
- **Slicing returns a view, not a copy** (modifying affects original)
- **Boolean indexing returns a copy** (can safely modify)
- Use `arr.copy()` if you need an independent copy

**Speed tip**: Boolean indexing is fast; prefer it over loops.

---

## 3. Shape Manipulation (10 min)

```python
arr.reshape(new_shape)    # Reshape without copying
arr.flatten()             # 1D copy
arr.ravel()               # 1D view
np.concatenate([a, b])    # Join along axis (default 0)
np.stack([a, b])          # Join as new axis
np.vstack([a, b])         # Vertical stack (rows)
np.hstack([a, b])         # Horizontal stack (columns)
arr.squeeze()             # Remove axes of size 1
arr.expand_dims(arr, 0)   # Add axis of size 1
```

**Key**: Know the difference between `reshape` (view) and `reshape().copy()` (explicit copy).

---

## 4. Arithmetic & Broadcasting (10-15 min)

### Element-wise operations
```python
a + b, a - b, a * b, a / b, a ** 2, np.sqrt(a), np.exp(a)
```

### Broadcasting Rule (critical)
Two arrays broadcast if dimensions are compatible **from right to left**:
- Dimensions match, OR
- One dimension is 1 (expands to match), OR
- One array has fewer dimensions (pad left with 1s)

**Example**:
```python
(3, 4) + (4,)         → (3, 4) + (1, 4) → (3, 4)  ✓
(3, 1) + (1, 4)       → (3, 4)  ✓
(3, 4) + (5, 4)       → ERROR (leftmost dims 3 ≠ 5)
```

**Speed tip**: Broadcasting is zero-copy; use it instead of loops.

---

## 5. Aggregation Functions (5-10 min)

```python
np.sum(arr, axis=...)           # Sum, optionally along axis
np.mean(arr, axis=...)          # Mean
np.std(arr, axis=...)           # Standard deviation
np.min(arr), np.max(arr)        # Min/max
np.argmin(arr), np.argmax(arr)  # Index of min/max
np.cumsum(arr)                  # Cumulative sum
np.cumprod(arr)                 # Cumulative product
np.sort(arr)                    # Sorted copy
np.argsort(arr)                 # Indices that would sort
```

**Axis parameter**: 
- `axis=0` → along rows (reduce rows)
- `axis=1` → along columns (reduce columns)
- `axis=None` → flatten and reduce (default)

**Keep returns**: Most return a **copy**, not a view.

---

## 6. Linear Algebra (10-15 min)

```python
np.dot(a, b)          # Matrix multiply (or @ operator)
a @ b                 # Matrix multiply (same as np.dot)
np.linalg.norm(a)     # Frobenius norm
np.linalg.inv(a)      # Inverse (square matrix only)
np.linalg.eig(a)      # Eigenvalues & eigenvectors
np.linalg.svd(a)      # Singular value decomposition
np.trace(a)           # Sum of diagonal
np.diag(a)            # Extract diagonal
```

**Critical**: `@` is matrix multiply; `*` is element-wise.

---

## 7. Logical Operations & Comparisons (5 min)

```python
a > 5                 # Element-wise comparison → boolean array
a == b                # Equality check
np.logical_and(a, b)  # Element-wise AND
np.logical_or(a, b)   # Element-wise OR
np.logical_not(a)     # Element-wise NOT
np.any(arr)           # Any True?
np.all(arr)           # All True?
np.where(condition, if_true, if_false)  # Conditional select
```

---

## 8. Advanced Indexing Patterns (10 min)

### Combining operations
```python
# Filter rows by condition
arr[arr[:, 0] > 5]    # Rows where first column > 5

# Multi-condition
arr[(arr[:, 0] > 5) & (arr[:, 1] < 3)]

# Using np.where
np.where(arr > 5, arr, 0)  # Replace values

# Using np.isin
np.isin(arr, [1, 3, 5])  # Boolean mask for membership

# Using np.nonzero / np.argwhere
np.nonzero(arr > 5)  # Indices where condition is True
```

---

## 9. Common Patterns You'll See (15 min)

### Sorting by column
```python
arr[np.argsort(arr[:, 2])]  # Sort by column 2
```

### Unique values
```python
np.unique(arr)              # Sorted unique values
np.unique(arr, return_counts=True)  # With counts
```

### Append/insert
```python
np.append(arr, values, axis=...)
np.insert(arr, index, values, axis=...)
```

### Delete rows/columns
```python
np.delete(arr, indices, axis=...)
```

### Repeat/tile
```python
np.repeat(arr, 3, axis=0)  # Repeat each row 3 times
np.tile(arr, (2, 3))       # Tile array 2×3 times
```

### Set operations (for 1D arrays)
```python
np.intersect1d(a, b)       # Common elements
np.union1d(a, b)           # All unique elements
np.setdiff1d(a, b)         # In a, not in b
```

---

## 10. Dtypes & Performance (5 min)

### Common dtypes
```python
np.int32, np.int64         # Signed integers
np.float32, np.float64     # Floating point
np.bool_                   # Boolean
np.complex128              # Complex numbers
```

### Type conversion
```python
arr.astype(np.float32)     # Convert dtype (returns copy)
arr.dtype                  # Check current dtype
```

**Speed note**: 
- `float32` is 2× faster than `float64` on some hardware
- Integer operations are faster than float
- Minimize type conversions in loops

---

## 11. Quick Mental Model

| Task | Function |
|------|----------|
| Create | `np.array()`, `zeros()`, `ones()`, `arange()`, `linspace()` |
| Reshape | `reshape()`, `flatten()`, `squeeze()`, `expand_dims()` |
| Slice | `arr[i:j, k:l]`, `arr[mask]`, `arr[[list]]` |
| Math | `+`, `-`, `*`, `/`, `**`, broadcasting |
| Reduce | `sum()`, `mean()`, `min()`, `max()`, `std()` |
| Join | `concatenate()`, `stack()`, `vstack()`, `hstack()` |
| Sort/Find | `sort()`, `argsort()`, `argmax()`, `nonzero()` |
| Logical | `>`, `<`, `==`, `&`, `\|`, `where()` |
| Linear Alg | `@`, `dot()`, `linalg.*` |
| Unique | `unique()`, `isin()`, `intersect1d()` |

---

## 13. Debugging Checklist (when stuck)

1. **Check shape**: `print(arr.shape, arr.dtype)` — 90% of bugs are shape mismatches
2. **Test broadcasting**: Does your operation broadcast as expected?
3. **Check axis**: Are you aggregating along the right axis?
4. **Verify dtype**: Did a type conversion happen unexpectedly?
5. **Try small example**: Reduce to 2×3 array, verify logic works, then scale up

---

## 14. NumPy Idioms to Master

```python
# Efficient: vectorized
arr[arr > 5] = 0           # Boolean indexing assignment

# Slow: loop (never do this in NumPy)
for i in range(len(arr)):
    if arr[i] > 5:
        arr[i] = 0

# Efficient: broadcasting
(arr - arr.mean(axis=0)) / arr.std(axis=0)  # Normalize columns

# Efficient: use math functions
np.exp(arr)                # Better than [math.exp(x) for x in arr]

# Efficient: cumulative operations
np.cumsum(arr)             # Better than [sum(arr[:i]) for i in range(len(arr))]
```

---

## 15. Reference Cheat (keep this handy)

```python
# Shape ops
reshape, flatten, ravel, squeeze, expand_dims, transpose (.T)

# Join ops
concatenate, stack, vstack, hstack

# Aggregations (all take axis= param)
sum, mean, std, min, max, argmin, argmax, cumsum

# Indexing (all are fast)
Boolean indexing, fancy indexing, slicing, np.where

# Logical
any, all, logical_and/or/not

# Sorting
sort, argsort, lexsort (multi-key)

# Unique
unique, isin, intersect1d, union1d, setdiff1d

# Linear algebra
@, dot, norm, inv, eig, svd

# Type
astype, dtype check

# Random
rand, randn, seed
```

