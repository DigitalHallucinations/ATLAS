# 01 - Project Setup

**Phase:** 1  
**Duration:** 1 week  
**Priority:** Critical (blocks all other phases)  
**Dependencies:** None  

## Objective

Establish the Cargo workspace, PyO3 bindings skeleton, build system (maturin), CI/CD pipeline, and developer documentation. This phase produces a working but minimal Python package that can be imported.

## Deliverables

- [ ] Cargo workspace with all crate stubs
- [ ] PyO3 module that exports a version string
- [ ] maturin build configuration
- [ ] GitHub Actions CI for Rust tests + wheel builds
- [ ] Developer setup documentation
- [ ] Pre-commit hooks for Rust formatting/linting

---

## 1. Directory Structure Creation

Create the following structure under `/neuralcore-runtime/`:

```Text
neuralcore-runtime/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── LICENSE
├── rust-toolchain.toml
├── .cargo/
│   └── config.toml
├── crates/
│   ├── sr-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs
│   │       └── config.rs
│   ├── sr-embedding/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── sr-vectors/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── sr-messaging/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   └── sr-python/
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
├── python/
│   └── neuralcore/
│       ├── __init__.py
│       ├── py.typed
│       └── _fallback.py
├── benches/
│   └── .gitkeep
└── tests/
    ├── rust/
    │   └── .gitkeep
    └── python/
        ├── __init__.py
        └── test_import.py
```

---

## 2. Workspace Cargo.toml

```toml
# neuralcore-runtime/Cargo.toml

[workspace]
resolver = "2"
members = [
    "crates/sr-core",
    "crates/sr-embedding",
    "crates/sr-vectors",
    "crates/sr-messaging",
    "crates/sr-python",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/DigitalHallucinations/ATLAS"
authors = ["Jeremy Shows <jeremyshws@gmail.com>"]

[workspace.dependencies]
# Shared dependencies - pin versions for consistency
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }

# Internal crates
sr-core = { path = "crates/sr-core" }
sr-embedding = { path = "crates/sr-embedding" }
sr-vectors = { path = "crates/sr-vectors" }
sr-messaging = { path = "crates/sr-messaging" }

[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3

[profile.release-debug]
inherits = "release"
debug = true
```

---

## 3. Rust Toolchain

```toml
# neuralcore-runtime/rust-toolchain.toml

[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu"]
```

---

## 4. Cargo Config (Build Optimizations)

```toml
# neuralcore-runtime/.cargo/config.toml

[build]
# Enable all CPU features for local builds
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
# Use mold linker if available (much faster linking)
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=mold"]

[env]
# ONNX Runtime library path (set during CI/build)
# ORT_LIB_LOCATION = { value = "/usr/local/lib", force = false }
```

---

## 5. Individual Crate Cargo.toml Files

### sr-core

```toml
# crates/sr-core/Cargo.toml

[package]
name = "sr-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
```

### sr-vectors

```toml
# crates/sr-vectors/Cargo.toml

[package]
name = "sr-vectors"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
sr-core.workspace = true
thiserror.workspace = true

# SIMD - use wide for portable SIMD
wide = "0.7"

# Optional: for HNSW index
parking_lot = "0.12"
rand = "0.8"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "vector_bench"
harness = false
```

### sr-embedding

```toml
# crates/sr-embedding/Cargo.toml

[package]
name = "sr-embedding"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
sr-core.workspace = true
sr-vectors.workspace = true
thiserror.workspace = true
tracing.workspace = true

# ONNX Runtime
ort = { version = "2.0", default-features = false, features = ["ndarray"] }
ndarray = "0.15"
rayon = "1.8"
half = "2.3"  # FP16 support

[dev-dependencies]
criterion = "0.5"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }

[[bench]]
name = "embedding_bench"
harness = false
```

### sr-messaging

```toml
# crates/sr-messaging/Cargo.toml

[package]
name = "sr-messaging"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
sr-core.workspace = true
thiserror.workspace = true
tracing.workspace = true
serde.workspace = true

# Lock-free data structures
crossbeam = "0.8"
parking_lot = "0.12"

# Serialization
rkyv = { version = "0.7", features = ["validation"] }

# Persistence
rusqlite = { version = "0.31", features = ["bundled"] }

# Async runtime
tokio = { version = "1", features = ["full"] }

# Optional: Redis
redis = { version = "0.25", features = ["tokio-comp"], optional = true }

# Optional: Kafka
rdkafka = { version = "0.36", features = ["tokio"], optional = true }

[features]
default = []
redis = ["dep:redis"]
kafka = ["dep:rdkafka"]
full = ["redis", "kafka"]

[dev-dependencies]
criterion = "0.5"
tokio-test = "0.4"

[[bench]]
name = "messaging_bench"
harness = false
```

### sr-python (PyO3 bindings)

```toml
# crates/sr-python/Cargo.toml

[package]
name = "sr-python"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "neuralcore"
crate-type = ["cdylib"]

[dependencies]
sr-core.workspace = true
sr-embedding.workspace = true
sr-vectors.workspace = true
sr-messaging.workspace = true

pyo3 = { version = "0.21", features = ["extension-module", "abi3-py310"] }
numpy = "0.21"
pyo3-asyncio = { version = "0.21", features = ["tokio-runtime"] }

[features]
default = []
redis = ["sr-messaging/redis"]
kafka = ["sr-messaging/kafka"]
full = ["redis", "kafka"]
```

---

## 6. Initial Rust Source Files

### sr-core/src/lib.rs

```rust
//! Core types, errors, and configuration for neuralcore-runtime.

pub mod error;
pub mod config;

pub use error::{Error, Result};
pub use config::RuntimeConfig;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

### sr-core/src/error.rs

```rust
//! Error types for neuralcore-runtime.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Vector operation error: {0}")]
    Vector(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Messaging error: {0}")]
    Messaging(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### sr-core/src/config.rs

```rust
//! Runtime configuration.

use serde::{Deserialize, Serialize};

/// Runtime configuration passed from Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Number of threads for parallel operations
    pub num_threads: usize,
    
    /// Enable verbose logging
    pub verbose: bool,
    
    /// Path to ONNX models directory
    pub models_path: Option<String>,
    
    /// SQLite database path for message persistence
    pub persistence_path: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            verbose: false,
            models_path: None,
            persistence_path: None,
        }
    }
}
```

### sr-vectors/src/lib.rs

```rust
//! SIMD-optimized vector operations.

pub fn version() -> &'static str {
    sr_core::VERSION
}

// Placeholder - implemented in 02-sr-vectors.md
pub fn cosine_similarity(_a: &[f32], _b: &[f32]) -> f32 {
    todo!("Implemented in phase 2")
}
```

### sr-embedding/src/lib.rs

```rust
//! ONNX Runtime embedding inference.

pub fn version() -> &'static str {
    sr_core::VERSION
}

// Placeholder - implemented in 03-sr-embedding.md
```

### sr-messaging/src/lib.rs

```rust
//! High-performance message bus.

pub fn version() -> &'static str {
    sr_core::VERSION
}

// Placeholder - implemented in 04-sr-messaging.md
```

### sr-python/src/lib.rs

```rust
//! PyO3 bindings for neuralcore-runtime.

use pyo3::prelude::*;

/// neuralcore - High-performance runtime for ATLAS
#[pymodule]
fn neuralcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", sr_core::VERSION)?;
    m.add("__doc__", "High-performance Rust runtime for ATLAS")?;
    
    // Submodules will be added in later phases
    // m.add_submodule(vectors_module)?;
    // m.add_submodule(embedding_module)?;
    // m.add_submodule(messaging_module)?;
    
    Ok(())
}
```

---

## 7. pyproject.toml (maturin)

```toml
# neuralcore-runtime/pyproject.toml

[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "neuralcore"
version = "0.1.0"
description = "High-performance Rust runtime for ATLAS"
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "Jeremy Shows", email = "jeremyshws@gmail.com" }]
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["atlas", "embedding", "vector", "messaging", "rust"]

dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "hypothesis>=6.0",
    "mypy>=1.0",
]

[project.urls]
Homepage = "https://github.com/DigitalHallucinations/ATLAS"
Repository = "https://github.com/DigitalHallucinations/ATLAS"
Documentation = "https://github.com/DigitalHallucinations/ATLAS/tree/main/docs"

[tool.maturin]
# Python source directory
python-source = "python"
# Build only sr-python crate
manifest-path = "crates/sr-python/Cargo.toml"
# Module name matches crate lib name
module-name = "neuralcore._native"
# Features to enable
features = ["pyo3/extension-module"]
# Strip debug symbols in release
strip = true
# Build for stable ABI (Python 3.10+)
# python-packages = ["neuralcore"]

[tool.pytest.ini_options]
testpaths = ["tests/python"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

---

## 8. Python Package Files

### python/neuralcore/**init**.py

```python
"""neuralcore - High-performance Rust runtime for ATLAS.

This package provides native acceleration for:
- Vector operations (SIMD-optimized distance functions)
- Embedding inference (ONNX Runtime)
- Message bus (lock-free queues, Redis/Kafka bridges)

If the native extension is not available, pure Python fallbacks are used.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# Try to import native module
try:
    from neuralcore._native import __version__ as _native_version
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    _native_version = None

# Package version
__version__ = _native_version or "0.1.0.dev0"

# Re-export submodules
from neuralcore import vectors
from neuralcore import embedding
from neuralcore import messaging

__all__ = [
    "__version__",
    "NATIVE_AVAILABLE",
    "vectors",
    "embedding", 
    "messaging",
]


def runtime_info() -> dict:
    """Return information about the runtime environment."""
    return {
        "version": __version__,
        "native_available": NATIVE_AVAILABLE,
        "python_version": sys.version,
        "platform": sys.platform,
    }
```

### python/neuralcore/vectors.py

```python
"""Vector operations with optional native acceleration."""

from __future__ import annotations

from typing import List, Sequence

# Try native implementation
try:
    from neuralcore._native import vectors as _native_vectors
    NATIVE_AVAILABLE = True
except ImportError:
    _native_vectors = None
    NATIVE_AVAILABLE = False


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Cosine similarity in range [-1, 1].
        
    Raises:
        ValueError: If vectors have different lengths.
    """
    if _native_vectors is not None:
        return _native_vectors.cosine_similarity(a, b)
    
    # Pure Python fallback
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Euclidean distance (L2 norm of difference).
    """
    if _native_vectors is not None:
        return _native_vectors.euclidean_distance(a, b)
    
    # Pure Python fallback
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product of two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Dot product (inner product).
    """
    if _native_vectors is not None:
        return _native_vectors.dot_product(a, b)
    
    # Pure Python fallback
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    
    return sum(x * y for x, y in zip(a, b))


def normalize(v: Sequence[float]) -> List[float]:
    """Normalize a vector to unit length.
    
    Args:
        v: Input vector.
        
    Returns:
        Normalized vector with L2 norm of 1.
    """
    if _native_vectors is not None:
        return _native_vectors.normalize(v)
    
    # Pure Python fallback
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0:
        return [0.0] * len(v)
    return [x / norm for x in v]


# Batch operations
def cosine_similarity_batch(
    queries: Sequence[Sequence[float]],
    documents: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Compute cosine similarity matrix between query and document vectors.
    
    Args:
        queries: List of query vectors.
        documents: List of document vectors.
        
    Returns:
        Matrix where result[i][j] is similarity between queries[i] and documents[j].
    """
    if _native_vectors is not None:
        return _native_vectors.cosine_similarity_batch(queries, documents)
    
    # Pure Python fallback
    return [
        [cosine_similarity(q, d) for d in documents]
        for q in queries
    ]
```

### python/neuralcore/embedding.py

```python
"""Embedding inference with optional native acceleration."""

from __future__ import annotations

from typing import List, Optional, Sequence

# Placeholder - implemented in phase 3
NATIVE_AVAILABLE = False

__all__ = ["NATIVE_AVAILABLE"]
```

### python/neuralcore/messaging.py

```python
"""Message bus with optional native acceleration."""

from __future__ import annotations

# Placeholder - implemented in phase 4
NATIVE_AVAILABLE = False

__all__ = ["NATIVE_AVAILABLE"]
```

### python/neuralcore/py.typed

```Text
# Marker file for PEP 561
# This package supports type checking
```

---

## 9. Test Files

### tests/python/test_import.py

```python
"""Basic import and availability tests."""

import pytest


def test_import_neuralcore():
    """Test that neuralcore can be imported."""
    import neuralcore
    assert hasattr(neuralcore, "__version__")
    assert hasattr(neuralcore, "NATIVE_AVAILABLE")


def test_runtime_info():
    """Test runtime_info function."""
    from neuralcore import runtime_info
    
    info = runtime_info()
    assert "version" in info
    assert "native_available" in info
    assert "python_version" in info
    assert "platform" in info


def test_vectors_fallback():
    """Test that vector operations work in fallback mode."""
    from neuralcore.vectors import cosine_similarity, euclidean_distance
    
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [1.0, 0.0, 0.0]
    
    # Orthogonal vectors
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    
    # Identical vectors
    assert cosine_similarity(a, c) == pytest.approx(1.0)
    
    # Euclidean distance
    assert euclidean_distance(a, b) == pytest.approx(2 ** 0.5)


def test_vectors_batch():
    """Test batch vector operations."""
    from neuralcore.vectors import cosine_similarity_batch
    
    queries = [[1.0, 0.0], [0.0, 1.0]]
    docs = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    
    result = cosine_similarity_batch(queries, docs)
    
    assert len(result) == 2
    assert len(result[0]) == 3
    assert result[0][0] == pytest.approx(1.0)  # q0 vs d0
    assert result[1][1] == pytest.approx(1.0)  # q1 vs d1
```

---

## 10. GitHub Actions CI

### .github/workflows/neuralcore-ci.yml

```yaml
name: Neuralcore CI

on:
  push:
    branches: [main]
    paths:
      - 'neuralcore-runtime/**'
      - '.github/workflows/neuralcore-ci.yml'
  pull_request:
    branches: [main]
    paths:
      - 'neuralcore-runtime/**'

env:
  CARGO_TERM_COLOR: always

jobs:
  rust-checks:
    name: Rust Checks
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: neuralcore-runtime

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2
        with:
          workspaces: neuralcore-runtime

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Run tests
        run: cargo test --all-features

  python-tests:
    name: Python Tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: neuralcore-runtime

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Install maturin
        run: pip install maturin pytest pytest-asyncio

      - name: Build and install
        run: maturin develop

      - name: Run Python tests
        run: pytest tests/python -v

  build-wheels:
    name: Build Wheels
    runs-on: ${{ matrix.os }}
    needs: [rust-checks, python-tests]
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          working-directory: neuralcore-runtime
          args: --release --out dist
          manylinux: auto

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: neuralcore-runtime/dist/*.whl
```

---

## 11. Developer Documentation

### neuralcore-runtime/README.md

```markdown
# neuralcore-runtime

High-performance Rust runtime for ATLAS providing native acceleration of:

- **Vector Operations** - SIMD-optimized cosine similarity, Euclidean distance, dot product
- **Embedding Inference** - ONNX Runtime for local embedding models
- **Message Bus** - Lock-free priority queues with Redis/Kafka bridges

## Quick Start

### Using Pre-built Wheels (Recommended)

```bash
pip install neuralcore
```

### Building from Source

Requires Rust 1.75+ and maturin:

```bash
cd neuralcore-runtime
pip install maturin
maturin develop  # Development build
maturin build --release  # Release build
```

## Usage

```python
import neuralcore

# Check if native acceleration is available
print(f"Native: {neuralcore.NATIVE_AVAILABLE}")

# Vector operations (uses SIMD when native available)
from neuralcore.vectors import cosine_similarity
similarity = cosine_similarity([1.0, 0.0], [0.707, 0.707])

# Embedding inference (requires ONNX model)
from neuralcore.embedding import EmbeddingEngine
engine = EmbeddingEngine("models/all-MiniLM-L6-v2.onnx")
embeddings = engine.embed_batch(["Hello world", "Test text"])

# Message bus
from neuralcore.messaging import MessageBus
bus = MessageBus(persistence_path="messages.db")
await bus.publish("channel.name", {"data": "value"})
```

## Development

```bash
# Run Rust tests
cargo test --all-features

# Run Python tests
maturin develop
pytest tests/python

# Format code
cargo fmt
```

## License

MIT - Same as ATLAS

---

## 12. Acceptance Criteria

- [ ] `cargo build --all-features` succeeds with no errors
- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `cargo fmt --all -- --check` passes
- [ ] `maturin develop` installs the package successfully
- [ ] `import neuralcore` works in Python
- [ ] `neuralcore.__version__` returns the version string
- [ ] `neuralcore.NATIVE_AVAILABLE` returns `True`
- [ ] `pytest tests/python` passes all tests
- [ ] GitHub Actions CI is green

---

## 13. Open Questions

1. **ONNX Runtime linking:** Bundle `libonnxruntime` or require system install?
   - Recommendation: Bundle via `ort` crate's download feature for simplicity

2. **librdkafka linking:** Static or dynamic?
   - Recommendation: Static (`rdkafka` crate default) for portability

3. **SIMD target features:** Build for baseline x86_64 or require AVX2?
   - Recommendation: Runtime detection with `wide` crate, AVX2 when available

4. **Python minimum version:** 3.10 or 3.11?
   - Decision: 3.10 (matches `abi3-py310` for wider compatibility)
