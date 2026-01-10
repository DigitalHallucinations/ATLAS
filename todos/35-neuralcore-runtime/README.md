# Neuralcore Runtime

High-performance Rust runtime for ATLAS providing native acceleration of CPU-bound operations that are bottlenecked by Python's GIL and interpreter overhead.

## Overview

**Package Name:** `neuralcore-runtime`  
**Location:** `/neuralcore-runtime/` (monorepo)  
**Python Package:** `neuralcore`  
**License:** Same as ATLAS  

## Motivation

ATLAS is a Python-first application with excellent developer ergonomics, but specific subsystems are performance-constrained:

1. **Embedding Inference** - `sentence-transformers` runs PyTorch under the hood with significant Python overhead. Converting tensors to Python lists for storage is expensive.

2. **Vector Math** - Pure Python cosine/euclidean distance calculations in `modules/storage/retrieval/cache.py` and `modules/storage/chunking/semantic.py` lack SIMD optimization.

3. **Message Bus (NCB)** - 2,251 lines of async Python with `heapq`-based priority queues. Lock contention on `asyncio.Condition` and serialization overhead limit throughput.

4. **Serialization** - msgpack/JSON encoding for every message, even for in-process routing.

5. **Redis/Kafka Bridging** - Python async I/O adds latency at scale.

## Architecture

```Text
┌─────────────────────────────────────────────────────────────────────┐
│                         ATLAS (Python)                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐  │
│  │ Orchestration│ │   Plugins   │ │  Security   │ │   Config     │  │
│  │   & Routing  │ │   & Hooks   │ │   & RBAC    │ │  & Hot Reload│  │
│  └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ └──────┬───────┘  │
│         │                │               │               │          │
│         └────────────────┴───────────────┴───────────────┘          │
│                                  │                                   │
│                                  ▼                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    neuralcore (PyO3 bindings)                 │  │
│  │  from neuralcore import EmbeddingEngine, VectorOps, MessageBus│  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ FFI (PyO3)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    neuralcore-runtime (Rust)                        │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐  │
│  │  sr-core    │ │sr-embedding │ │ sr-vectors  │ │ sr-messaging │  │
│  │             │ │             │ │             │ │              │  │
│  │ • Types     │ │ • ONNX RT   │ │ • SIMD dist │ │ • Lock-free  │  │
│  │ • Errors    │ │ • Quantize  │ │ • Normalize │ │   queues     │  │
│  │ • Config    │ │ • Batch     │ │ • HNSW idx  │ │ • Persistence│  │
│  │             │ │             │ │             │ │ • Redis/Kafka│  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      sr-python (PyO3)                         │  │
│  │  #[pymodule] neuralcore - exports all functionality           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Crate Structure

```Text
neuralcore-runtime/
├── Cargo.toml                      # Workspace root
├── pyproject.toml                  # maturin build config
├── README.md
├── LICENSE
├── .cargo/
│   └── config.toml                 # Build optimizations
│
├── crates/
│   ├── sr-core/                    # Shared types, errors, config
│   ├── sr-embedding/               # ONNX Runtime inference
│   ├── sr-vectors/                 # SIMD vector operations + HNSW
│   ├── sr-messaging/               # Message bus core + bridges
│   └── sr-python/                  # PyO3 bindings
│
├── python/
│   └── neuralcore/                 # Python fallback + type stubs
│       ├── __init__.py
│       ├── embedding.py
│       ├── vectors.py
│       ├── messaging.py
│       └── py.typed
│
├── models/                         # Pre-converted ONNX models
│   └── README.md
│
├── benches/                        # Criterion benchmarks
│   ├── embedding_bench.rs
│   ├── vector_bench.rs
│   └── messaging_bench.rs
│
└── tests/
    ├── rust/                       # Rust unit/integration tests
    └── python/                     # Python integration tests
```

## Phasing

| Phase | Focus | Duration | Deliverables |
| ----- | ----- | -------- | ------------ |
| **1** | Project Setup | 1 week | Cargo workspace, CI, PyO3 skeleton |
| **2** | sr-vectors | 1-2 weeks | SIMD distance functions, benchmarks |
| **3** | sr-embedding | 2-3 weeks | ONNX inference, model loading, batching |
| **4** | sr-messaging (core) | 2-3 weeks | Lock-free queues, persistence |
| **5** | sr-messaging (bridges) | 1-2 weeks | Redis/Kafka integration |
| **6** | Integration | 1-2 weeks | Wire into ATLAS, migration path |

**Total estimated:** 8-13 weeks

## Key Design Decisions

### 1. Pure Python Fallback

Every native function has a pure Python fallback in `python/neuralcore/`. The package works without Rust compilation, just slower.

```python
# python/neuralcore/vectors.py
try:
    from neuralcore._native import cosine_similarity_batch
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if NATIVE_AVAILABLE:
        return cosine_similarity_batch([a], [b])[0]
    # Pure Python fallback
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

### 2. Zero-Copy Where Possible

Use `numpy` interop via PyO3 to avoid copying large arrays:

```rust
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn cosine_similarity_batch<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    Ok(vectors::cosine_similarity(a, b))
}
```

### 3. Async Bridge

The Rust message bus exposes both sync and async Python APIs:

```python
# Sync (for simple cases)
bus.publish("channel", message)

# Async (for high-throughput)
await bus.publish_async("channel", message)
```

### 4. Configuration via Python

All configuration stays in Python (`config.yaml`, `ConfigManager`). Rust receives config at initialization, not runtime.

## Dependencies (Rust)

### Core

- `thiserror` - Error handling
- `serde` / `serde_json` - Serialization
- `tracing` - Structured logging

### Embedding

- `ort` (2.0+) - ONNX Runtime bindings
- `ndarray` - N-dimensional arrays
- `rayon` - Parallel iteration
- `half` - FP16 support

### Vectors

- `simdeez` or `wide` - Portable SIMD
- `parking_lot` - Fast mutexes (for HNSW)
- `rand` - Random number generation (HNSW construction)

### Messaging

- `crossbeam` - Lock-free data structures
- `rkyv` - Zero-copy deserialization
- `rusqlite` - SQLite persistence
- `redis` - Redis client
- `rdkafka` - Kafka client (librdkafka bindings)
- `tokio` - Async runtime

### Python Bindings

- `pyo3` (0.21+) - Python bindings
- `numpy` (0.21+) - NumPy interop
- `pyo3-asyncio` - Async bridge

## Performance Targets

| Operation | Current (Python) | Target (Rust) | Improvement |
| --------- | ---------------- | ------------- | ----------- |
| Cosine similarity (1K vectors) | 1ms | 0.05ms | 20x |
| Batch embedding (100 texts) | 500ms | 50ms | 10x |
| Message publish (in-process) | 100μs | 5μs | 20x |
| Message serialize (msgpack) | 50μs | 2μs | 25x |
| Redis publish + ack | 200μs | 80μs | 2.5x |

## Risk Mitigation

| Risk | Mitigation |
| ---- | ---------- |
| PyO3 learning curve | Start with sr-vectors (simplest) |
| ONNX model compatibility | Test with sentence-transformers models first |
| Build complexity | Pre-built wheels for Linux/macOS/Windows |
| Contributor friction | Clear docs, Rust not required for Python-only work |
| GTK thread safety | All Rust calls are GIL-releasing, no GTK interaction |

## Related Documentation

- [Architecture Overview](../../docs/architecture-overview.md)
- [Polyglot Strategy](../../docs/architecture/polyglot-strategy.md)
- [Messaging Runbook](../../docs/ops/messaging.md)
- [RAG Developer Guide](../../docs/developer/rag-integration.md)

## Todo Files

1. [01-project-setup.md](01-project-setup.md) - Cargo workspace, CI, PyO3 skeleton
2. [02-sr-vectors.md](02-sr-vectors.md) - SIMD distance functions, HNSW index
3. [03-sr-embedding.md](03-sr-embedding.md) - ONNX Runtime inference
4. [04-sr-messaging.md](04-sr-messaging.md) - Lock-free queues, persistence
5. [05-sr-messaging-bridge.md](05-sr-messaging-bridge.md) - Redis/Kafka in Rust
6. [06-integration.md](06-integration.md) - Wiring into ATLAS
