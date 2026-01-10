# 02 - sr-vectors: SIMD Vector Operations

**Phase:** 2  
**Duration:** 1-2 weeks  
**Priority:** High (foundation for embedding and retrieval)  
**Dependencies:** 01-project-setup  

## Objective

Implement SIMD-optimized vector distance functions and an optional in-memory HNSW index. These primitives replace the pure Python implementations in:

- `modules/storage/retrieval/cache.py` (cosine similarity for cache lookups)
- `modules/storage/chunking/semantic.py` (cosine similarity for semantic chunking)
- `modules/storage/vectors/` (if bypassing pgvector for edge deployments)

## Deliverables

- [ ] SIMD cosine similarity (single pair and batch)
- [ ] SIMD Euclidean distance (single pair and batch)
- [ ] SIMD dot product (single pair and batch)
- [ ] Vector normalization
- [ ] HNSW index implementation (optional, feature-gated)
- [ ] Benchmarks comparing to NumPy and pure Python
- [ ] PyO3 bindings with NumPy zero-copy interop

---

## 1. Performance Targets

| Operation | Baseline (Python) | NumPy | Target (Rust SIMD) | Improvement |
| --------- | ----------------- | ----- | ------------------ | ----------- |
| Cosine similarity (1 pair, 384-dim) | 15μs | 2μs | 0.2μs | 75x / 10x |
| Cosine similarity batch (1K×1K, 384-dim) | 15s | 1.5s | 0.15s | 100x / 10x |
| Euclidean distance (1 pair, 384-dim) | 12μs | 1.5μs | 0.15μs | 80x / 10x |
| Normalize (1 vector, 384-dim) | 8μs | 1μs | 0.1μs | 80x / 10x |
| HNSW search (10K vectors, top-10) | N/A | N/A | 0.5ms | Enables local index |

---

## 2. Crate Structure

```Text
crates/sr-vectors/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── distance.rs         # Distance functions
│   ├── normalize.rs        # Normalization
│   ├── batch.rs            # Batch operations
│   ├── simd/
│   │   ├── mod.rs          # SIMD dispatch
│   │   ├── avx2.rs         # AVX2 implementation
│   │   ├── sse.rs          # SSE4.1 fallback
│   │   └── scalar.rs       # Scalar fallback
│   └── hnsw/
│       ├── mod.rs          # HNSW public API
│       ├── graph.rs        # Graph structure
│       ├── search.rs       # Search algorithm
│       └── builder.rs      # Index construction
└── benches/
    └── vector_bench.rs
```

---

## 3. Core Distance Functions

### 3.1 SIMD Abstraction

Use the `wide` crate for portable SIMD that works across x86_64 (SSE, AVX2) and ARM (NEON):

```rust
// src/simd/mod.rs

use wide::f32x8;

/// SIMD-accelerated dot product for f32 slices.
/// Handles arbitrary lengths with scalar tail handling.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector length mismatch");
    
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;
    
    let mut sum = f32x8::ZERO;
    
    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        sum += va * vb;
    }
    
    // Horizontal sum of SIMD register
    let mut result: f32 = sum.reduce_add();
    
    // Handle remainder
    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }
    
    result
}

/// SIMD-accelerated squared L2 norm.
#[inline]
pub fn squared_norm_f32(v: &[f32]) -> f32 {
    dot_product_f32(v, v)
}

/// SIMD-accelerated L2 norm.
#[inline]
pub fn norm_f32(v: &[f32]) -> f32 {
    squared_norm_f32(v).sqrt()
}
```

### 3.2 Distance Functions

```rust
// src/distance.rs

use crate::simd::{dot_product_f32, norm_f32};

/// Compute cosine similarity between two vectors.
/// 
/// Returns a value in [-1, 1] where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");
    
    let dot = dot_product_f32(a, b);
    let norm_a = norm_f32(a);
    let norm_b = norm_f32(b);
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Compute cosine distance (1 - cosine_similarity).
/// Returns a value in [0, 2] where 0 means identical.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute Euclidean (L2) distance between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_distance(a, b).sqrt()
}

/// Compute squared Euclidean distance (avoids sqrt for comparisons).
#[inline]
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");
    
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;
    
    let mut sum = wide::f32x8::ZERO;
    
    for i in 0..chunks {
        let offset = i * 8;
        let va = wide::f32x8::from(&a[offset..offset + 8]);
        let vb = wide::f32x8::from(&b[offset..offset + 8]);
        let diff = va - vb;
        sum += diff * diff;
    }
    
    let mut result: f32 = sum.reduce_add();
    
    let tail_start = chunks * 8;
    for i in 0..remainder {
        let diff = a[tail_start + i] - b[tail_start + i];
        result += diff * diff;
    }
    
    result
}

/// Compute dot product (inner product) of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    dot_product_f32(a, b)
}

/// Distance metric enum for generic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    SquaredEuclidean,
}

impl DistanceMetric {
    /// Compute distance between two vectors using this metric.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b), // Negate for "distance"
            DistanceMetric::SquaredEuclidean => squared_euclidean_distance(a, b),
        }
    }
}
```

### 3.3 Normalization

```rust
// src/normalize.rs

use crate::simd::norm_f32;

/// Normalize a vector to unit length in-place.
/// Returns the original norm.
pub fn normalize_inplace(v: &mut [f32]) -> f32 {
    let norm = norm_f32(v);
    if norm == 0.0 {
        return 0.0;
    }
    
    let inv_norm = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
    
    norm
}

/// Normalize a vector to unit length, returning a new vector.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = norm_f32(v);
    if norm == 0.0 {
        return vec![0.0; v.len()];
    }
    
    let inv_norm = 1.0 / norm;
    v.iter().map(|x| x * inv_norm).collect()
}

/// Normalize a batch of vectors in-place.
pub fn normalize_batch_inplace(vectors: &mut [Vec<f32>]) {
    for v in vectors.iter_mut() {
        normalize_inplace(v);
    }
}
```

---

## 4. Batch Operations

```rust
// src/batch.rs

use crate::distance::{cosine_similarity, euclidean_distance, dot_product, DistanceMetric};
use rayon::prelude::*;

/// Compute pairwise cosine similarity matrix between two sets of vectors.
/// 
/// Returns a matrix where result[i][j] = cosine_similarity(queries[i], documents[j]).
pub fn cosine_similarity_matrix(
    queries: &[&[f32]],
    documents: &[&[f32]],
) -> Vec<Vec<f32>> {
    queries
        .par_iter()
        .map(|q| {
            documents
                .iter()
                .map(|d| cosine_similarity(q, d))
                .collect()
        })
        .collect()
}

/// Compute top-k most similar documents for each query.
/// 
/// Returns indices and scores for each query.
pub fn top_k_similar(
    queries: &[&[f32]],
    documents: &[&[f32]],
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<(usize, f32)>> {
    queries
        .par_iter()
        .map(|q| {
            let mut scores: Vec<(usize, f32)> = documents
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let score = match metric {
                        DistanceMetric::Cosine => cosine_similarity(q, d),
                        DistanceMetric::DotProduct => dot_product(q, d),
                        _ => -metric.compute(q, d), // Negate distance for similarity
                    };
                    (i, score)
                })
                .collect();
            
            // Partial sort for top-k
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(k);
            scores
        })
        .collect()
}

/// Compute all pairwise distances within a single set (for clustering, etc.).
pub fn pairwise_distances(
    vectors: &[&[f32]],
    metric: DistanceMetric,
) -> Vec<Vec<f32>> {
    let n = vectors.len();
    
    (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        metric.compute(vectors[i], vectors[j])
                    }
                })
                .collect()
        })
        .collect()
}
```

---

## 5. HNSW Index (Feature-Gated)

The Hierarchical Navigable Small World (HNSW) algorithm provides approximate nearest neighbor search with O(log n) query time.

### 5.1 Feature Flag

```toml
# In Cargo.toml
[features]
default = []
hnsw = ["parking_lot", "rand"]
```

### 5.2 HNSW Types

```rust
// src/hnsw/mod.rs

#[cfg(feature = "hnsw")]
mod graph;
#[cfg(feature = "hnsw")]
mod search;
#[cfg(feature = "hnsw")]
mod builder;

#[cfg(feature = "hnsw")]
pub use graph::HnswGraph;
#[cfg(feature = "hnsw")]
pub use builder::HnswBuilder;
#[cfg(feature = "hnsw")]
pub use search::SearchResult;

use crate::distance::DistanceMetric;

/// Configuration for HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node at layer 0.
    pub m: usize,
    /// Maximum number of connections per node at higher layers.
    pub m0: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            metric: DistanceMetric::Cosine,
            seed: None,
        }
    }
}
```

### 5.3 HNSW Graph Structure

```rust
// src/hnsw/graph.rs

use parking_lot::RwLock;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::distance::DistanceMetric;
use super::HnswConfig;

/// A node in the HNSW graph.
struct Node {
    /// Vector data.
    vector: Vec<f32>,
    /// Connections at each layer. connections[layer] = vec of neighbor indices.
    connections: Vec<Vec<usize>>,
    /// Maximum layer this node exists on.
    max_layer: usize,
}

/// HNSW graph for approximate nearest neighbor search.
pub struct HnswGraph {
    /// Configuration.
    config: HnswConfig,
    /// All nodes in the graph.
    nodes: RwLock<Vec<Node>>,
    /// Entry point (node index with highest layer).
    entry_point: RwLock<Option<usize>>,
    /// Maximum layer in the graph.
    max_layer: RwLock<usize>,
    /// Vector dimension.
    dimension: usize,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(Vec::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            dimension,
        }
    }
    
    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }
    
    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a vector by index.
    pub fn get(&self, idx: usize) -> Option<Vec<f32>> {
        self.nodes.read().get(idx).map(|n| n.vector.clone())
    }
    
    /// Insert a vector into the index.
    pub fn insert(&self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dimension, "Dimension mismatch");
        
        let mut rng = self.config.seed
            .map(|s| rand::rngs::StdRng::seed_from_u64(s))
            .unwrap_or_else(rand::rngs::StdRng::from_entropy);
        
        // Calculate random level for this node
        let level = self.random_level(&mut rng);
        
        let node = Node {
            vector,
            connections: vec![Vec::new(); level + 1],
            max_layer: level,
        };
        
        let node_idx = {
            let mut nodes = self.nodes.write();
            let idx = nodes.len();
            nodes.push(node);
            idx
        };
        
        // Connect to graph
        self.connect_node(node_idx, level);
        
        node_idx
    }
    
    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        // Implementation in search.rs
        super::search::search_layer(self, query, k, ef)
    }
    
    fn random_level(&self, rng: &mut impl rand::Rng) -> usize {
        let ml = 1.0 / (self.config.m as f64).ln();
        let r: f64 = rng.gen();
        (-r.ln() * ml).floor() as usize
    }
    
    fn connect_node(&self, node_idx: usize, level: usize) {
        // Update entry point if needed
        {
            let mut max_layer = self.max_layer.write();
            let mut entry_point = self.entry_point.write();
            
            if level > *max_layer || entry_point.is_none() {
                *max_layer = level;
                *entry_point = Some(node_idx);
            }
        }
        
        // Connect at each layer (simplified - full implementation in builder.rs)
    }
}
```

### 5.4 HNSW Search

```rust
// src/hnsw/search.rs

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

use super::HnswGraph;

/// Search result with index and distance.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub index: usize,
    pub distance: f32,
}

/// Internal candidate for priority queue (min-heap by distance).
#[derive(Clone)]
struct Candidate {
    index: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Search the HNSW graph for k nearest neighbors.
pub fn search_layer(
    graph: &HnswGraph,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Vec<(usize, f32)> {
    let entry_point = match *graph.entry_point.read() {
        Some(ep) => ep,
        None => return Vec::new(),
    };
    
    let nodes = graph.nodes.read();
    let max_layer = *graph.max_layer.read();
    
    // Compute distance to entry point
    let mut current = entry_point;
    let mut current_dist = graph.config.metric.compute(query, &nodes[current].vector);
    
    // Traverse from top layer to layer 1
    for layer in (1..=max_layer).rev() {
        loop {
            let mut changed = false;
            
            if layer < nodes[current].connections.len() {
                for &neighbor in &nodes[current].connections[layer] {
                    let dist = graph.config.metric.compute(query, &nodes[neighbor].vector);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            
            if !changed {
                break;
            }
        }
    }
    
    // Search layer 0 with ef candidates
    let mut candidates = BinaryHeap::new();
    let mut visited = HashSet::new();
    let mut results = BinaryHeap::new();
    
    candidates.push(Candidate { index: current, distance: current_dist });
    visited.insert(current);
    results.push(Candidate { index: current, distance: -current_dist }); // Max-heap for results
    
    while let Some(candidate) = candidates.pop() {
        // Check if we can stop early
        if let Some(worst) = results.peek() {
            if candidate.distance > -worst.distance && results.len() >= ef {
                break;
            }
        }
        
        // Explore neighbors at layer 0
        if !nodes[candidate.index].connections.is_empty() {
            for &neighbor in &nodes[candidate.index].connections[0] {
                if visited.insert(neighbor) {
                    let dist = graph.config.metric.compute(query, &nodes[neighbor].vector);
                    
                    let dominated = results.len() >= ef && {
                        if let Some(worst) = results.peek() {
                            dist >= -worst.distance
                        } else {
                            false
                        }
                    };
                    
                    if !dominated {
                        candidates.push(Candidate { index: neighbor, distance: dist });
                        results.push(Candidate { index: neighbor, distance: -dist });
                        
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }
    
    // Extract top-k results
    let mut output: Vec<(usize, f32)> = results
        .into_iter()
        .map(|c| (c.index, -c.distance))
        .collect();
    
    output.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    output.truncate(k);
    output
}
```

---

## 6. PyO3 Bindings

```rust
// In crates/sr-python/src/vectors.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use sr_vectors::{distance, batch, normalize};

#[pyfunction]
#[pyo3(name = "cosine_similarity")]
pub fn py_cosine_similarity(
    a: PyReadonlyArray1<'_, f32>,
    b: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let a = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = b.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}", a.len(), b.len()
        )));
    }
    
    Ok(distance::cosine_similarity(a, b))
}

#[pyfunction]
#[pyo3(name = "euclidean_distance")]
pub fn py_euclidean_distance(
    a: PyReadonlyArray1<'_, f32>,
    b: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let a = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = b.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}", a.len(), b.len()
        )));
    }
    
    Ok(distance::euclidean_distance(a, b))
}

#[pyfunction]
#[pyo3(name = "dot_product")]
pub fn py_dot_product(
    a: PyReadonlyArray1<'_, f32>,
    b: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let a = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = b.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}", a.len(), b.len()
        )));
    }
    
    Ok(distance::dot_product(a, b))
}

#[pyfunction]
#[pyo3(name = "normalize")]
pub fn py_normalize<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let v = v.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = normalize::normalize(v);
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(name = "cosine_similarity_batch")]
pub fn py_cosine_similarity_batch<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let queries = queries.as_array();
    let documents = documents.as_array();
    
    let q_refs: Vec<&[f32]> = queries.rows().into_iter()
        .map(|r| r.as_slice().unwrap())
        .collect();
    let d_refs: Vec<&[f32]> = documents.rows().into_iter()
        .map(|r| r.as_slice().unwrap())
        .collect();
    
    let result = batch::cosine_similarity_matrix(&q_refs, &d_refs);
    
    let flat: Vec<f32> = result.into_iter().flatten().collect();
    let shape = (q_refs.len(), d_refs.len());
    
    Ok(PyArray2::from_vec2_bound(py, &result).unwrap())
}

/// Register vector functions in the module.
pub fn register_vectors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosine_similarity_batch, m)?)?;
    Ok(())
}
```

---

## 7. Benchmarks

```rust
// benches/vector_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use sr_vectors::distance::{cosine_similarity, euclidean_distance, dot_product};
use sr_vectors::batch::cosine_similarity_matrix;

fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn bench_single_operations(c: &mut Criterion) {
    let dims = [128, 384, 768, 1536];
    
    let mut group = c.benchmark_group("single_pair");
    
    for dim in dims {
        let a = random_vector(dim);
        let b = random_vector(dim);
        
        group.throughput(Throughput::Elements(dim as u64));
        
        group.bench_function(BenchmarkId::new("cosine", dim), |bencher| {
            bencher.iter(|| cosine_similarity(&a, &b))
        });
        
        group.bench_function(BenchmarkId::new("euclidean", dim), |bencher| {
            bencher.iter(|| euclidean_distance(&a, &b))
        });
        
        group.bench_function(BenchmarkId::new("dot_product", dim), |bencher| {
            bencher.iter(|| dot_product(&a, &b))
        });
    }
    
    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let dim = 384;
    let sizes = [10, 100, 1000];
    
    let mut group = c.benchmark_group("batch_cosine");
    
    for size in sizes {
        let queries: Vec<Vec<f32>> = (0..size).map(|_| random_vector(dim)).collect();
        let documents: Vec<Vec<f32>> = (0..size).map(|_| random_vector(dim)).collect();
        
        let q_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let d_refs: Vec<&[f32]> = documents.iter().map(|v| v.as_slice()).collect();
        
        group.throughput(Throughput::Elements((size * size) as u64));
        
        group.bench_function(BenchmarkId::new("matrix", size), |bencher| {
            bencher.iter(|| cosine_similarity_matrix(&q_refs, &d_refs))
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_single_operations, bench_batch_operations);
criterion_main!(benches);
```

---

## 8. Integration Points in ATLAS

### 8.1 Replace `modules/storage/retrieval/cache.py`

Current:

```python
def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

After:

```python
from neuralcore.vectors import cosine_similarity

def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
    return cosine_similarity(a, b)
```

### 8.2 Replace `modules/storage/chunking/semantic.py`

Current:

```python
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
```

After:

```python
from neuralcore.vectors import cosine_similarity
# Function now imported, remove local definition
```

---

## 9. Acceptance Criteria

- [ ] All distance functions match NumPy results within float tolerance
- [ ] Benchmark shows ≥10x improvement over pure Python
- [ ] Benchmark shows ≥2x improvement over NumPy for large batches
- [ ] HNSW search returns correct top-k for test datasets
- [ ] PyO3 bindings work with both lists and NumPy arrays
- [ ] No memory leaks (test with valgrind/miri)
- [ ] Thread-safe for parallel usage

---

## 10. Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(cosine_similarity(&v, &v), 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_relative_eq!(cosine_similarity(&a, &b), 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert_relative_eq!(cosine_similarity(&a, &b), -1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_relative_eq!(euclidean_distance(&a, &b), 1.0, epsilon = 1e-6);
    }
}
```

### Property-Based Tests (Python)

```python
from hypothesis import given, strategies as st
import numpy as np
from neuralcore.vectors import cosine_similarity

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=10, max_size=100))
def test_cosine_self_similarity(v):
    """Cosine similarity of a vector with itself should be 1 (if non-zero)."""
    v = np.array(v, dtype=np.float32)
    if np.linalg.norm(v) > 0:
        result = cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-5
```

---

## 11. Open Questions

1. **f32 vs f64:** Use f32 for speed (matches most embedding models) or support both?
   - Recommendation: f32 primary, f64 optional

2. **HNSW persistence:** Add save/load to disk?
   - Recommendation: Yes, using memory-mapped files

3. **GPU acceleration:** Support CUDA for very large batches?
   - Recommendation: Defer to phase 3 (embedding inference handles GPU)
