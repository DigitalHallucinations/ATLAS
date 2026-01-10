# 03 - sr-embedding: ONNX Runtime Inference

**Phase:** 3  
**Duration:** 2-3 weeks  
**Priority:** High (core value proposition for local inference)  
**Dependencies:** 01-project-setup, 02-sr-vectors  

## Objective

Implement high-performance embedding inference using ONNX Runtime in Rust, replacing the `sentence-transformers` (PyTorch) backend for local embedding generation. This eliminates Python overhead, enables better batching, and supports quantization (INT8/FP16) for faster inference.

## Deliverables

- [ ] ONNX model loading and session management
- [ ] Tokenization (using `tokenizers` crate from HuggingFace)
- [ ] Single text embedding
- [ ] Batch embedding with automatic chunking
- [ ] FP16 and INT8 quantization support
- [ ] Model warm-up and caching
- [ ] Memory-efficient streaming for large batches
- [ ] GPU support (CUDA/TensorRT execution providers)
- [ ] PyO3 bindings with async support
- [ ] Model conversion utilities (PyTorch → ONNX)

---

## 1. Performance Targets

| Operation | Current (sentence-transformers) | Target (ONNX Rust) | Improvement |
| --------- | ------------------------------- | ------------------- | ----------- |
| Single embedding (384-dim) | 15ms | 2ms | 7.5x |
| Batch 100 texts (384-dim) | 500ms | 50ms | 10x |
| Batch 1000 texts (384-dim) | 5s | 400ms | 12.5x |
| Memory per model | ~500MB (PyTorch) | ~150MB (ONNX) | 3x |
| Cold start (model load) | 3s | 0.5s | 6x |

With INT8 quantization:

| Operation | FP32 | INT8 | Improvement |
| --------- | ---- | ---- | ----------- |
| Single embedding | 2ms | 0.5ms | 4x |
| Batch 100 texts | 50ms | 15ms | 3.3x |
| Model size | 90MB | 25MB | 3.6x |

---

## 2. Crate Structure

```Text
crates/sr-embedding/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── engine.rs           # EmbeddingEngine main struct
│   ├── session.rs          # ONNX session management
│   ├── tokenizer.rs        # HuggingFace tokenizers integration
│   ├── batch.rs            # Batch processing logic
│   ├── quantize.rs         # Quantization utilities
│   ├── pooling.rs          # Mean/CLS pooling strategies
│   ├── normalize.rs        # L2 normalization
│   ├── providers.rs        # Execution providers (CPU, CUDA, TensorRT)
│   ├── models/
│   │   ├── mod.rs          # Model registry
│   │   ├── config.rs       # Model configuration
│   │   └── convert.rs      # PyTorch → ONNX conversion
│   └── error.rs            # Error types
├── models/                  # Pre-converted models (gitignored, downloaded)
│   └── README.md
└── benches/
    └── embedding_bench.rs
```

---

## 3. Supported Models

### 3.1 Initial Model Support

| Model | Dimensions | Size (FP32) | Size (INT8) | Use Case |
| ----- | ---------- | ----------- | ----------- | -------- |
| all-MiniLM-L6-v2 | 384 | 90MB | 25MB | Fast, general purpose |
| all-mpnet-base-v2 | 768 | 420MB | 110MB | High quality |
| bge-small-en-v1.5 | 384 | 130MB | 35MB | BGE family, good quality |
| bge-base-en-v1.5 | 768 | 440MB | 115MB | BGE family, better quality |
| e5-small-v2 | 384 | 130MB | 35MB | Microsoft E5 family |
| nomic-embed-text-v1.5 | 768 | 550MB | 140MB | Long context (8192 tokens) |

### 3.2 Model Configuration

```rust
// src/models/config.rs

use serde::{Deserialize, Serialize};

/// Configuration for an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "all-MiniLM-L6-v2").
    pub name: String,
    
    /// Output embedding dimension.
    pub dimension: usize,
    
    /// Maximum input tokens.
    pub max_tokens: usize,
    
    /// Pooling strategy.
    pub pooling: PoolingStrategy,
    
    /// Whether to normalize output embeddings.
    pub normalize: bool,
    
    /// Input names for ONNX model.
    pub input_names: Vec<String>,
    
    /// Output name for ONNX model.
    pub output_name: String,
    
    /// Whether model uses asymmetric embeddings (query vs document prefixes).
    pub asymmetric: bool,
    
    /// Query prefix for asymmetric models.
    pub query_prefix: Option<String>,
    
    /// Document prefix for asymmetric models.
    pub document_prefix: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use [CLS] token embedding.
    Cls,
    /// Mean of all token embeddings.
    Mean,
    /// Mean of token embeddings weighted by attention mask.
    MeanWithAttention,
    /// Max pooling across tokens.
    Max,
    /// Last token embedding (for causal models).
    LastToken,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Pre-defined model configurations.
pub fn get_model_config(name: &str) -> Option<ModelConfig> {
    match name {
        "all-MiniLM-L6-v2" => Some(ModelConfig {
            name: "all-MiniLM-L6-v2".into(),
            dimension: 384,
            max_tokens: 256,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            input_names: vec!["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
            output_name: "last_hidden_state".into(),
            asymmetric: false,
            query_prefix: None,
            document_prefix: None,
        }),
        "bge-small-en-v1.5" => Some(ModelConfig {
            name: "bge-small-en-v1.5".into(),
            dimension: 384,
            max_tokens: 512,
            pooling: PoolingStrategy::Cls,
            normalize: true,
            input_names: vec!["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
            output_name: "last_hidden_state".into(),
            asymmetric: true,
            query_prefix: Some("Represent this sentence for searching relevant passages: ".into()),
            document_prefix: None,
        }),
        // Add more models...
        _ => None,
    }
}
```

---

## 4. Core Implementation

### 4.1 Embedding Engine

```rust
// src/engine.rs

use std::path::Path;
use std::sync::Arc;

use crate::error::{EmbeddingError, Result};
use crate::models::config::{ModelConfig, PoolingStrategy};
use crate::session::OnnxSession;
use crate::tokenizer::Tokenizer;
use crate::batch::BatchProcessor;
use crate::pooling::pool_embeddings;
use crate::normalize::l2_normalize;

/// High-performance embedding engine using ONNX Runtime.
pub struct EmbeddingEngine {
    /// ONNX inference session.
    session: Arc<OnnxSession>,
    
    /// Tokenizer.
    tokenizer: Arc<Tokenizer>,
    
    /// Model configuration.
    config: ModelConfig,
    
    /// Batch processor for efficient batching.
    batch_processor: BatchProcessor,
    
    /// Whether engine is initialized.
    initialized: bool,
}

impl EmbeddingEngine {
    /// Create a new embedding engine.
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<Self> {
        let session = OnnxSession::new(model_path.as_ref())?;
        let tokenizer = Tokenizer::new(tokenizer_path.as_ref())?;
        
        Ok(Self {
            session: Arc::new(session),
            tokenizer: Arc::new(tokenizer),
            config,
            batch_processor: BatchProcessor::new(32), // Default batch size
            initialized: true,
        })
    }
    
    /// Create engine from model name (downloads if needed).
    pub async fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = crate::models::config::get_model_config(model_name)
            .ok_or_else(|| EmbeddingError::UnsupportedModel(model_name.into()))?;
        
        let model_dir = crate::models::ensure_model(model_name).await?;
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        Self::new(model_path, tokenizer_path, config)
    }
    
    /// Get embedding dimension.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    /// Get maximum tokens.
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }
    
    /// Embed a single text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_with_type(text, EmbeddingType::Document)
    }
    
    /// Embed a single text with type (query or document).
    pub fn embed_with_type(&self, text: &str, embed_type: EmbeddingType) -> Result<Vec<f32>> {
        let prepared = self.prepare_text(text, embed_type);
        let encoding = self.tokenizer.encode(&prepared, self.config.max_tokens)?;
        
        let outputs = self.session.run(&[
            ("input_ids", &encoding.input_ids),
            ("attention_mask", &encoding.attention_mask),
            ("token_type_ids", &encoding.token_type_ids),
        ])?;
        
        let hidden_state = outputs.get(&self.config.output_name)
            .ok_or_else(|| EmbeddingError::MissingOutput(self.config.output_name.clone()))?;
        
        let mut embedding = pool_embeddings(
            hidden_state,
            &encoding.attention_mask,
            self.config.pooling,
        )?;
        
        if self.config.normalize {
            l2_normalize(&mut embedding);
        }
        
        Ok(embedding)
    }
    
    /// Embed a batch of texts.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_type(texts, EmbeddingType::Document)
    }
    
    /// Embed a batch of texts with type.
    pub fn embed_batch_with_type(
        &self,
        texts: &[&str],
        embed_type: EmbeddingType,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Prepare texts
        let prepared: Vec<String> = texts
            .iter()
            .map(|t| self.prepare_text(t, embed_type))
            .collect();
        
        // Process in batches
        let mut all_embeddings = Vec::with_capacity(texts.len());
        
        for batch in self.batch_processor.chunks(&prepared) {
            let encodings = self.tokenizer.encode_batch(&batch, self.config.max_tokens)?;
            
            let outputs = self.session.run_batch(&[
                ("input_ids", &encodings.input_ids),
                ("attention_mask", &encodings.attention_mask),
                ("token_type_ids", &encodings.token_type_ids),
            ])?;
            
            let hidden_states = outputs.get(&self.config.output_name)
                .ok_or_else(|| EmbeddingError::MissingOutput(self.config.output_name.clone()))?;
            
            for (i, attention_mask) in encodings.attention_masks.iter().enumerate() {
                let hidden_state = &hidden_states[i];
                
                let mut embedding = pool_embeddings(
                    hidden_state,
                    attention_mask,
                    self.config.pooling,
                )?;
                
                if self.config.normalize {
                    l2_normalize(&mut embedding);
                }
                
                all_embeddings.push(embedding);
            }
        }
        
        Ok(all_embeddings)
    }
    
    /// Prepare text by adding prefixes for asymmetric models.
    fn prepare_text(&self, text: &str, embed_type: EmbeddingType) -> String {
        if !self.config.asymmetric {
            return text.to_string();
        }
        
        match embed_type {
            EmbeddingType::Query => {
                if let Some(prefix) = &self.config.query_prefix {
                    format!("{}{}", prefix, text)
                } else {
                    text.to_string()
                }
            }
            EmbeddingType::Document => {
                if let Some(prefix) = &self.config.document_prefix {
                    format!("{}{}", prefix, text)
                } else {
                    text.to_string()
                }
            }
        }
    }
}

/// Type of embedding (affects prefix for asymmetric models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingType {
    Query,
    Document,
}
```

### 4.2 ONNX Session Management

```rust
// src/session.rs

use std::path::Path;
use std::sync::Arc;

use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder,
    Value, inputs, CUDAExecutionProvider, TensorRTExecutionProvider,
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

use crate::error::{EmbeddingError, Result};
use crate::providers::ExecutionProviderConfig;

/// ONNX Runtime session wrapper.
pub struct OnnxSession {
    session: Session,
    environment: Arc<Environment>,
}

impl OnnxSession {
    /// Create a new ONNX session from model path.
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::with_config(model_path, ExecutionProviderConfig::default())
    }
    
    /// Create a new ONNX session with custom execution provider config.
    pub fn with_config(
        model_path: &Path,
        provider_config: ExecutionProviderConfig,
    ) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("neuralcore_embedding")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()
                .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?
        );
        
        let mut builder = Session::builder()
            .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?
            .with_intra_threads(provider_config.intra_threads)
            .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?;
        
        // Configure execution providers
        builder = match provider_config.provider {
            ExecutionProviderType::Cpu => builder,
            ExecutionProviderType::Cuda { device_id } => {
                builder.with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                ])
                .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?
            }
            ExecutionProviderType::TensorRt { device_id } => {
                builder.with_execution_providers([
                    TensorRTExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                ])
                .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?
            }
        };
        
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?;
        
        Ok(Self { session, environment })
    }
    
    /// Run inference on a single input.
    pub fn run(
        &self,
        inputs: &[(&str, &[i64])],
    ) -> Result<std::collections::HashMap<String, Array2<f32>>> {
        // Prepare input tensors
        let input_values: Vec<_> = inputs
            .iter()
            .map(|(name, data)| {
                let array = Array1::from_vec(data.to_vec());
                let value = Value::from_array(array.insert_axis(ndarray::Axis(0)))
                    .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?;
                Ok((*name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        
        let outputs = self.session
            .run(ort::inputs![input_values]?)
            .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?;
        
        // Extract outputs
        let mut result = std::collections::HashMap::new();
        for (name, value) in outputs.iter() {
            let tensor = value
                .extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::OnnxRuntime(e.to_string()))?;
            let view = tensor.view();
            let array = view.to_owned().into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| EmbeddingError::Shape(e.to_string()))?;
            result.insert(name.to_string(), array);
        }
        
        Ok(result)
    }
    
    /// Run inference on a batch.
    pub fn run_batch(
        &self,
        inputs: &[(&str, &Array2<i64>)],
    ) -> Result<std::collections::HashMap<String, Array3<f32>>> {
        // Similar to run() but handles batched inputs
        // Returns 3D array: [batch, sequence, hidden]
        todo!("Implement batched inference")
    }
}

/// Execution provider type.
#[derive(Debug, Clone)]
pub enum ExecutionProviderType {
    Cpu,
    Cuda { device_id: i32 },
    TensorRt { device_id: i32 },
}
```

### 4.3 Tokenizer Integration

```rust
// src/tokenizer.rs

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{EmbeddingError, Result};

/// Token encoding result.
#[derive(Debug, Clone)]
pub struct Encoding {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

/// Batch encoding result.
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    pub input_ids: ndarray::Array2<i64>,
    pub attention_mask: ndarray::Array2<i64>,
    pub token_type_ids: ndarray::Array2<i64>,
    pub attention_masks: Vec<Vec<i64>>, // For per-sequence pooling
}

/// HuggingFace tokenizer wrapper.
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from file.
    pub fn new(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;
        
        Ok(Self { inner })
    }
    
    /// Encode a single text.
    pub fn encode(&self, text: &str, max_length: usize) -> Result<Encoding> {
        let encoding = self.inner
            .encode(text, true)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;
        
        // Truncate if needed
        let len = encoding.len().min(max_length);
        
        let input_ids: Vec<i64> = encoding.get_ids()[..len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        
        let attention_mask: Vec<i64> = encoding.get_attention_mask()[..len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        
        let token_type_ids: Vec<i64> = encoding.get_type_ids()[..len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        
        Ok(Encoding {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
    
    /// Encode a batch of texts with padding.
    pub fn encode_batch(&self, texts: &[&str], max_length: usize) -> Result<BatchEncoding> {
        let encodings: Vec<Encoding> = texts
            .iter()
            .map(|t| self.encode(t, max_length))
            .collect::<Result<Vec<_>>>()?;
        
        if encodings.is_empty() {
            return Ok(BatchEncoding {
                input_ids: ndarray::Array2::zeros((0, 0)),
                attention_mask: ndarray::Array2::zeros((0, 0)),
                token_type_ids: ndarray::Array2::zeros((0, 0)),
                attention_masks: Vec::new(),
            });
        }
        
        // Find max length in batch
        let batch_max_len = encodings.iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0);
        
        let batch_size = encodings.len();
        
        // Pad and create arrays
        let mut input_ids = ndarray::Array2::zeros((batch_size, batch_max_len));
        let mut attention_mask = ndarray::Array2::zeros((batch_size, batch_max_len));
        let mut token_type_ids = ndarray::Array2::zeros((batch_size, batch_max_len));
        let mut attention_masks = Vec::with_capacity(batch_size);
        
        for (i, enc) in encodings.iter().enumerate() {
            let len = enc.input_ids.len();
            
            for (j, &id) in enc.input_ids.iter().enumerate() {
                input_ids[[i, j]] = id;
            }
            
            for (j, &mask) in enc.attention_mask.iter().enumerate() {
                attention_mask[[i, j]] = mask;
            }
            
            for (j, &tid) in enc.token_type_ids.iter().enumerate() {
                token_type_ids[[i, j]] = tid;
            }
            
            attention_masks.push(enc.attention_mask.clone());
        }
        
        Ok(BatchEncoding {
            input_ids,
            attention_mask,
            token_type_ids,
            attention_masks,
        })
    }
}
```

### 4.4 Pooling Strategies

```rust
// src/pooling.rs

use ndarray::{Array1, ArrayView2};

use crate::error::Result;
use crate::models::config::PoolingStrategy;

/// Pool token embeddings into a single vector.
pub fn pool_embeddings(
    hidden_state: &ArrayView2<f32>,  // [sequence_len, hidden_dim]
    attention_mask: &[i64],
    strategy: PoolingStrategy,
) -> Result<Vec<f32>> {
    match strategy {
        PoolingStrategy::Cls => {
            // First token embedding
            Ok(hidden_state.row(0).to_vec())
        }
        
        PoolingStrategy::Mean => {
            // Simple mean of all tokens
            let sum: Array1<f32> = hidden_state.sum_axis(ndarray::Axis(0));
            let count = hidden_state.nrows() as f32;
            Ok(sum.mapv(|x| x / count).to_vec())
        }
        
        PoolingStrategy::MeanWithAttention => {
            // Weighted mean using attention mask
            let hidden_dim = hidden_state.ncols();
            let mut sum = vec![0.0f32; hidden_dim];
            let mut total_weight = 0.0f32;
            
            for (i, &mask) in attention_mask.iter().enumerate() {
                if mask > 0 && i < hidden_state.nrows() {
                    let weight = mask as f32;
                    total_weight += weight;
                    
                    for (j, &val) in hidden_state.row(i).iter().enumerate() {
                        sum[j] += val * weight;
                    }
                }
            }
            
            if total_weight > 0.0 {
                for val in &mut sum {
                    *val /= total_weight;
                }
            }
            
            Ok(sum)
        }
        
        PoolingStrategy::Max => {
            // Max pooling across sequence dimension
            let hidden_dim = hidden_state.ncols();
            let mut result = vec![f32::NEG_INFINITY; hidden_dim];
            
            for i in 0..hidden_state.nrows() {
                if attention_mask.get(i).copied().unwrap_or(0) > 0 {
                    for (j, &val) in hidden_state.row(i).iter().enumerate() {
                        result[j] = result[j].max(val);
                    }
                }
            }
            
            // Handle edge case of all masked
            for val in &mut result {
                if val.is_infinite() {
                    *val = 0.0;
                }
            }
            
            Ok(result)
        }
        
        PoolingStrategy::LastToken => {
            // Last non-padded token
            let last_idx = attention_mask.iter()
                .rposition(|&m| m > 0)
                .unwrap_or(0);
            
            Ok(hidden_state.row(last_idx).to_vec())
        }
    }
}
```

### 4.5 Quantization

```rust
// src/quantize.rs

use std::path::Path;

use crate::error::Result;

/// Quantization precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationPrecision {
    /// Full precision (FP32).
    Fp32,
    /// Half precision (FP16).
    Fp16,
    /// 8-bit integer (INT8).
    Int8,
    /// Dynamic quantization (INT8 weights, FP32 activations).
    DynamicInt8,
}

/// Quantize an ONNX model.
pub fn quantize_model(
    input_path: &Path,
    output_path: &Path,
    precision: QuantizationPrecision,
) -> Result<()> {
    // This is typically done offline, but we can use onnxruntime's quantization
    // For runtime, we just load the pre-quantized model
    
    match precision {
        QuantizationPrecision::Fp32 => {
            // No quantization needed, just copy
            std::fs::copy(input_path, output_path)?;
        }
        QuantizationPrecision::Fp16 => {
            // Use onnxruntime's FP16 conversion
            // This is typically done via Python tooling
            unimplemented!("FP16 conversion should be done offline with onnx tools")
        }
        QuantizationPrecision::Int8 | QuantizationPrecision::DynamicInt8 => {
            // Use onnxruntime's INT8 quantization
            unimplemented!("INT8 quantization should be done offline with onnx tools")
        }
    }
    
    Ok(())
}

/// Check if a model is quantized.
pub fn detect_quantization(model_path: &Path) -> Result<QuantizationPrecision> {
    // Parse ONNX model to detect quantization
    // For now, use naming convention
    let name = model_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    
    if name.contains("int8") || name.contains("INT8") {
        Ok(QuantizationPrecision::Int8)
    } else if name.contains("fp16") || name.contains("FP16") {
        Ok(QuantizationPrecision::Fp16)
    } else {
        Ok(QuantizationPrecision::Fp32)
    }
}
```

---

## 5. PyO3 Bindings

```rust
// In crates/sr-python/src/embedding.rs

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyArray2};

use sr_embedding::{EmbeddingEngine, EmbeddingType, ModelConfig};

/// Python wrapper for EmbeddingEngine.
#[pyclass(name = "EmbeddingEngine")]
pub struct PyEmbeddingEngine {
    inner: EmbeddingEngine,
}

#[pymethods]
impl PyEmbeddingEngine {
    /// Create a new embedding engine from model files.
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, config=None))]
    fn new(
        model_path: &str,
        tokenizer_path: &str,
        config: Option<PyModelConfig>,
    ) -> PyResult<Self> {
        let config = config
            .map(|c| c.into())
            .unwrap_or_else(|| sr_embedding::models::config::get_model_config("all-MiniLM-L6-v2")
                .unwrap());
        
        let engine = EmbeddingEngine::new(model_path, tokenizer_path, config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self { inner: engine })
    }
    
    /// Load a pre-trained model by name.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, model_name: &str) -> PyResult<Self> {
        // Use pyo3-asyncio for async loading
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let engine = EmbeddingEngine::from_pretrained(model_name)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner: engine })
        })
    }
    
    /// Get embedding dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    
    /// Get maximum tokens.
    #[getter]
    fn max_tokens(&self) -> usize {
        self.inner.max_tokens()
    }
    
    /// Embed a single text.
    fn embed<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let embedding = self.inner.embed(text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(PyArray1::from_vec_bound(py, embedding))
    }
    
    /// Embed a single text as a query (for asymmetric models).
    fn embed_query<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let embedding = self.inner.embed_with_type(text, EmbeddingType::Query)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(PyArray1::from_vec_bound(py, embedding))
    }
    
    /// Embed a batch of texts.
    fn embed_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<&str>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let embeddings = self.inner.embed_batch(&texts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        if embeddings.is_empty() {
            return Ok(PyArray2::zeros_bound(py, (0, self.inner.dimension()), false));
        }
        
        let dim = embeddings[0].len();
        let flat: Vec<f32> = embeddings.into_iter().flatten().collect();
        let array = ndarray::Array2::from_shape_vec((texts.len(), dim), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyArray2::from_owned_array_bound(py, array))
    }
}

/// Python model configuration.
#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct PyModelConfig {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    dimension: usize,
    #[pyo3(get, set)]
    max_tokens: usize,
    #[pyo3(get, set)]
    normalize: bool,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    fn new(name: String, dimension: usize, max_tokens: usize, normalize: bool) -> Self {
        Self { name, dimension, max_tokens, normalize }
    }
}

impl From<PyModelConfig> for ModelConfig {
    fn from(py: PyModelConfig) -> Self {
        ModelConfig {
            name: py.name,
            dimension: py.dimension,
            max_tokens: py.max_tokens,
            normalize: py.normalize,
            pooling: sr_embedding::models::config::PoolingStrategy::Mean,
            input_names: vec!["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
            output_name: "last_hidden_state".into(),
            asymmetric: false,
            query_prefix: None,
            document_prefix: None,
        }
    }
}

/// Register embedding functions in the module.
pub fn register_embedding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbeddingEngine>()?;
    m.add_class::<PyModelConfig>()?;
    Ok(())
}
```

---

## 6. Model Conversion Utility

Script for converting sentence-transformers models to ONNX:

```python
#!/usr/bin/env python3
"""Convert sentence-transformers models to ONNX format."""

import argparse
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def convert_model(model_name: str, output_dir: Path, opset: int = 14) -> None:
    """Convert a sentence-transformers model to ONNX."""
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Get the transformer component
    transformer = model[0].auto_model
    
    # Create dummy input
    dummy_input = tokenizer(
        "This is a sample sentence.",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    
    torch.onnx.export(
        transformer,
        (
            dummy_input["input_ids"],
            dummy_input["attention_mask"],
            dummy_input.get("token_type_ids", torch.zeros_like(dummy_input["input_ids"])),
        ),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    
    print(f"Model exported to: {onnx_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully!")
    
    # Optionally quantize
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    int8_path = output_dir / "model_int8.onnx"
    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 model exported to: {int8_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("--output", "-o", type=Path, default=Path("./models"))
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    
    convert_model(args.model_name, args.output / args.model_name.replace("/", "_"), args.opset)
```

---

## 7. Integration with ATLAS

### 7.1 Replace HuggingFaceEmbeddingProvider

```python
# modules/storage/embeddings/neuralcore_provider.py

from __future__ import annotations

import asyncio
from typing import Optional, Sequence, List

from .base import (
    EmbeddingProvider,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingInputType,
    EmbeddingProviderError,
)

try:
    from neuralcore.embedding import EmbeddingEngine
    NEURALCORE_AVAILABLE = True
except ImportError:
    NEURALCORE_AVAILABLE = False


class NeuralcoreEmbeddingProvider(EmbeddingProvider):
    """High-performance embedding provider using neuralcore (Rust ONNX)."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        if not NEURALCORE_AVAILABLE:
            raise EmbeddingProviderError(
                "neuralcore not available. Install with: pip install neuralcore"
            )
        
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._engine: Optional[EmbeddingEngine] = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "neuralcore"
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def dimension(self) -> int:
        return self._engine.dimension if self._engine else 384
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        
        # Load model in thread pool
        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None,
            lambda: EmbeddingEngine.from_pretrained(self._model_name)
        )
        self._initialized = True
    
    async def embed_text(
        self,
        text: str,
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> EmbeddingResult:
        if not self._initialized:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        
        if input_type == EmbeddingInputType.QUERY:
            embedding = await loop.run_in_executor(
                None, self._engine.embed_query, text
            )
        else:
            embedding = await loop.run_in_executor(
                None, self._engine.embed, text
            )
        
        return EmbeddingResult(
            embedding=embedding.tolist(),
            text=text,
            model=self._model_name,
        )
    
    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> BatchEmbeddingResult:
        if not self._initialized:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._engine.embed_batch, list(texts)
        )
        
        results = [
            EmbeddingResult(
                embedding=emb.tolist(),
                text=texts[i],
                model=self._model_name,
            )
            for i, emb in enumerate(embeddings)
        ]
        
        return BatchEmbeddingResult(
            embeddings=results,
            model=self._model_name,
        )
```

---

## 8. Benchmarks

```rust
// benches/embedding_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use sr_embedding::EmbeddingEngine;

fn bench_single_embedding(c: &mut Criterion) {
    let engine = EmbeddingEngine::from_pretrained("all-MiniLM-L6-v2")
        .expect("Failed to load model");
    
    let texts = [
        "Short text",
        "This is a medium length sentence for testing embedding performance.",
        "This is a much longer text that contains multiple sentences. It is designed to test how the embedding model handles longer inputs. The performance should scale with input length due to transformer self-attention complexity.",
    ];
    
    let mut group = c.benchmark_group("single_embedding");
    
    for text in texts {
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("embed", text.len()), |b| {
            b.iter(|| engine.embed(text).unwrap())
        });
    }
    
    group.finish();
}

fn bench_batch_embedding(c: &mut Criterion) {
    let engine = EmbeddingEngine::from_pretrained("all-MiniLM-L6-v2")
        .expect("Failed to load model");
    
    let base_text = "This is a sample sentence for batch embedding benchmarking.";
    
    let mut group = c.benchmark_group("batch_embedding");
    
    for batch_size in [1, 10, 32, 64, 128] {
        let texts: Vec<&str> = (0..batch_size).map(|_| base_text).collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_function(BenchmarkId::new("embed_batch", batch_size), |b| {
            b.iter(|| engine.embed_batch(&texts).unwrap())
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_single_embedding, bench_batch_embedding);
criterion_main!(benches);
```

---

## 9. Acceptance Criteria

- [ ] Can load all-MiniLM-L6-v2 and produce embeddings matching sentence-transformers within tolerance (cosine similarity > 0.99)
- [ ] Batch embedding is ≥5x faster than sentence-transformers for 100+ texts
- [ ] Memory usage is ≤50% of sentence-transformers for same model
- [ ] Supports FP32, FP16, and INT8 models
- [ ] GPU inference works with CUDA execution provider
- [ ] Model download/caching works correctly
- [ ] PyO3 bindings integrate seamlessly with existing ATLAS code
- [ ] No memory leaks in batch processing

---

## 10. Open Questions

1. **Model hosting:** Download from HuggingFace Hub or self-host pre-converted models?
   - Recommendation: Both - HF Hub for convenience, allow self-hosted for enterprise

2. **GPU memory management:** How to handle OOM for large batches?
   - Recommendation: Auto-reduce batch size on OOM, configurable memory limit

3. **Streaming inference:** Support for very large batches that don't fit in memory?
   - Recommendation: Implement generator-based batch processing in phase 3

4. **Multi-GPU:** Support for data parallel across multiple GPUs?
   - Recommendation: Defer - single GPU sufficient for embedding workloads
