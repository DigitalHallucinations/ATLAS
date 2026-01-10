# 05 - sr-messaging-bridge: Redis & Kafka Integration

**Phase:** 5  
**Duration:** 2-3 weeks  
**Priority:** Medium (required for distributed deployment)  
**Dependencies:** 04-sr-messaging  

## Objective

Implement Redis and Kafka bridges in Rust for high-performance external communication. These bridges handle the serialization, protocol negotiation, and network I/O that currently bottleneck the Python NCB implementation when interfacing with external message brokers.

## Deliverables

- [ ] Redis pub/sub bridge (tokio-based async)
- [ ] Redis streams support for persistence
- [ ] Kafka producer/consumer bridge
- [ ] Batching and compression for network efficiency
- [ ] Automatic reconnection and failover
- [ ] Schema registry integration (optional)
- [ ] Bridge configuration management
- [ ] PyO3 bindings for bridge control

---

## 1. Performance Targets

| Operation | Python (redis-py/confluent-kafka) | Rust Target | Improvement |
|-----------|-----------------------------------|-------------|-------------|
| Redis PUBLISH | 500μs | 50μs | 10x |
| Redis batch (100 msgs) | 20ms | 1ms | 20x |
| Kafka produce (single) | 2ms | 200μs | 10x |
| Kafka produce (batch) | 50ms | 5ms | 10x |
| Serialization + send | 1ms | 50μs | 20x |
| Reconnection time | 5s | 500ms | 10x |

**Throughput targets:**

| Metric | Python | Rust Target |
|--------|--------|-------------|
| Redis messages/sec | 5K | 100K |
| Kafka messages/sec | 10K | 200K |
| Concurrent connections | 10 | 100 |

---

## 2. Crate Structure

```Text
crates/sr-messaging-bridge/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── bridge.rs           # Bridge trait and registry
│   ├── config.rs           # Bridge configuration
│   ├── redis/
│   │   ├── mod.rs          # Redis module
│   │   ├── client.rs       # Redis client wrapper
│   │   ├── pubsub.rs       # Pub/sub implementation
│   │   ├── streams.rs      # Redis streams
│   │   └── pool.rs         # Connection pooling
│   ├── kafka/
│   │   ├── mod.rs          # Kafka module
│   │   ├── producer.rs     # Kafka producer
│   │   ├── consumer.rs     # Kafka consumer
│   │   ├── batch.rs        # Batching logic
│   │   └── schema.rs       # Schema registry (optional)
│   ├── routing.rs          # Message routing to bridges
│   ├── health.rs           # Health checking
│   └── metrics.rs          # Bridge-specific metrics
└── tests/
    ├── redis_tests.rs
    └── kafka_tests.rs
```

---

## 3. Cargo.toml

```toml
[package]
name = "sr-messaging-bridge"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
# Core messaging
sr-messaging = { path = "../sr-messaging" }
sr-core = { path = "../sr-core" }

# Async runtime
tokio = { version = "1.35", features = ["full"] }

# Redis
redis = { version = "0.24", features = ["tokio-comp", "connection-manager", "cluster"] }
deadpool-redis = "0.14"

# Kafka
rdkafka = { version = "0.36", features = ["tokio", "cmake-build", "ssl", "gssapi"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rkyv = { version = "0.7", features = ["validation"] }

# Compression
lz4 = "1.24"
zstd = "0.13"

# Utilities
thiserror = "1.0"
tracing = "0.1"
async-trait = "0.1"
parking_lot = "0.12"
dashmap = "5.5"

[dev-dependencies]
tokio-test = "0.4"
testcontainers = "0.15"
```

---

## 4. Bridge Trait

```rust
// src/bridge.rs

use async_trait::async_trait;
use sr_messaging::{Message, MessageId};
use crate::error::Result;

/// Common bridge configuration.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Bridge name for identification.
    pub name: String,
    
    /// Channels to bridge (empty = all).
    pub channels: Vec<String>,
    
    /// Enable compression.
    pub compress: bool,
    
    /// Compression algorithm.
    pub compression: CompressionType,
    
    /// Batch size for network sends.
    pub batch_size: usize,
    
    /// Batch timeout (microseconds).
    pub batch_timeout_us: u64,
    
    /// Reconnect interval (milliseconds).
    pub reconnect_interval_ms: u64,
    
    /// Maximum reconnection attempts.
    pub max_reconnect_attempts: u32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            channels: Vec::new(),
            compress: true,
            compression: CompressionType::Lz4,
            batch_size: 100,
            batch_timeout_us: 1000,
            reconnect_interval_ms: 1000,
            max_reconnect_attempts: 10,
        }
    }
}

/// Compression types.
#[derive(Debug, Clone, Copy, Default)]
pub enum CompressionType {
    None,
    #[default]
    Lz4,
    Zstd,
}

/// Bridge status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeStatus {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

/// External message bridge trait.
#[async_trait]
pub trait Bridge: Send + Sync {
    /// Get bridge name.
    fn name(&self) -> &str;
    
    /// Get current status.
    fn status(&self) -> BridgeStatus;
    
    /// Connect to external system.
    async fn connect(&self) -> Result<()>;
    
    /// Disconnect.
    async fn disconnect(&self) -> Result<()>;
    
    /// Publish a message to external system.
    async fn publish(&self, message: &Message) -> Result<()>;
    
    /// Subscribe to messages from external system.
    async fn subscribe(
        &self,
        channel: &str,
        callback: Box<dyn Fn(Message) + Send + Sync>,
    ) -> Result<()>;
    
    /// Health check.
    async fn health_check(&self) -> Result<bool>;
    
    /// Get metrics.
    fn metrics(&self) -> BridgeMetrics;
}

/// Bridge metrics.
#[derive(Debug, Clone, Default)]
pub struct BridgeMetrics {
    pub published: u64,
    pub received: u64,
    pub errors: u64,
    pub reconnects: u64,
    pub avg_latency_us: f64,
    pub batch_count: u64,
}

/// Bridge registry for managing multiple bridges.
pub struct BridgeRegistry {
    bridges: dashmap::DashMap<String, std::sync::Arc<dyn Bridge>>,
}

impl BridgeRegistry {
    pub fn new() -> Self {
        Self {
            bridges: dashmap::DashMap::new(),
        }
    }
    
    pub fn register(&self, bridge: std::sync::Arc<dyn Bridge>) {
        self.bridges.insert(bridge.name().to_string(), bridge);
    }
    
    pub fn get(&self, name: &str) -> Option<std::sync::Arc<dyn Bridge>> {
        self.bridges.get(name).map(|b| b.clone())
    }
    
    pub fn all(&self) -> Vec<std::sync::Arc<dyn Bridge>> {
        self.bridges.iter().map(|r| r.value().clone()).collect()
    }
}
```

---

## 5. Redis Bridge

### 5.1 Client Wrapper

```rust
// src/redis/client.rs

use std::sync::Arc;
use std::time::Duration;

use redis::aio::MultiplexedConnection;
use redis::{Client, AsyncCommands, RedisError};
use tokio::sync::RwLock;

use crate::error::{Error, Result};

/// Redis client configuration.
#[derive(Debug, Clone)]
pub struct RedisConfig {
    /// Redis URL (redis://host:port/db).
    pub url: String,
    
    /// Connection timeout (milliseconds).
    pub connect_timeout_ms: u64,
    
    /// Command timeout (milliseconds).
    pub command_timeout_ms: u64,
    
    /// Enable TLS.
    pub tls: bool,
    
    /// Password (optional).
    pub password: Option<String>,
    
    /// Database number.
    pub database: u8,
    
    /// Cluster mode.
    pub cluster: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            connect_timeout_ms: 5000,
            command_timeout_ms: 1000,
            tls: false,
            password: None,
            database: 0,
            cluster: false,
        }
    }
}

/// Redis client wrapper with connection management.
pub struct RedisClient {
    config: RedisConfig,
    connection: RwLock<Option<MultiplexedConnection>>,
    client: Client,
}

impl RedisClient {
    /// Create a new Redis client.
    pub fn new(config: RedisConfig) -> Result<Self> {
        let client = Client::open(config.url.as_str())
            .map_err(|e| Error::Connection(e.to_string()))?;
        
        Ok(Self {
            config,
            connection: RwLock::new(None),
            client,
        })
    }
    
    /// Connect to Redis.
    pub async fn connect(&self) -> Result<()> {
        let conn = tokio::time::timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            self.client.get_multiplexed_async_connection(),
        )
        .await
        .map_err(|_| Error::Connection("Connection timeout".to_string()))?
        .map_err(|e| Error::Connection(e.to_string()))?;
        
        let mut guard = self.connection.write().await;
        *guard = Some(conn);
        
        Ok(())
    }
    
    /// Get the connection.
    pub async fn get_connection(&self) -> Result<MultiplexedConnection> {
        let guard = self.connection.read().await;
        guard.clone().ok_or_else(|| Error::Connection("Not connected".to_string()))
    }
    
    /// Publish a message.
    pub async fn publish(&self, channel: &str, message: &[u8]) -> Result<()> {
        let mut conn = self.get_connection().await?;
        
        tokio::time::timeout(
            Duration::from_millis(self.config.command_timeout_ms),
            conn.publish::<_, _, ()>(channel, message),
        )
        .await
        .map_err(|_| Error::Connection("Command timeout".to_string()))?
        .map_err(|e| Error::Connection(e.to_string()))?;
        
        Ok(())
    }
    
    /// Publish multiple messages (pipeline).
    pub async fn publish_batch(
        &self,
        messages: &[(String, Vec<u8>)],
    ) -> Result<()> {
        let mut conn = self.get_connection().await?;
        
        // Use Redis pipeline for batching
        let mut pipe = redis::pipe();
        for (channel, data) in messages {
            pipe.publish::<_, _>(channel, data.as_slice());
        }
        
        tokio::time::timeout(
            Duration::from_millis(self.config.command_timeout_ms * messages.len() as u64),
            pipe.query_async::<_, ()>(&mut conn),
        )
        .await
        .map_err(|_| Error::Connection("Batch timeout".to_string()))?
        .map_err(|e| Error::Connection(e.to_string()))?;
        
        Ok(())
    }
    
    /// Subscribe to channels.
    pub async fn subscribe(
        &self,
        channels: &[String],
        callback: impl Fn(String, Vec<u8>) + Send + 'static,
    ) -> Result<()> {
        let mut pubsub = self.client.get_async_pubsub()
            .await
            .map_err(|e| Error::Connection(e.to_string()))?;
        
        for channel in channels {
            pubsub.subscribe(channel)
                .await
                .map_err(|e| Error::Connection(e.to_string()))?;
        }
        
        // Spawn message receiver
        tokio::spawn(async move {
            loop {
                let msg = pubsub.on_message().next().await;
                if let Some(msg) = msg {
                    let channel: String = msg.get_channel_name().to_string();
                    let payload: Vec<u8> = msg.get_payload().unwrap_or_default();
                    callback(channel, payload);
                }
            }
        });
        
        Ok(())
    }
    
    /// Ping for health check.
    pub async fn ping(&self) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        
        let result: redis::RedisResult<String> = tokio::time::timeout(
            Duration::from_millis(self.config.command_timeout_ms),
            redis::cmd("PING").query_async(&mut conn),
        )
        .await
        .map_err(|_| Error::Connection("Ping timeout".to_string()))?;
        
        Ok(result.map(|s| s == "PONG").unwrap_or(false))
    }
}
```

### 5.2 Redis Pub/Sub Bridge

```rust
// src/redis/pubsub.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::mpsc;

use sr_messaging::{Message, MessageId, Payload};
use sr_messaging::serde::{MessageSerializer, create_serializer, SerializerType};

use crate::bridge::{Bridge, BridgeConfig, BridgeMetrics, BridgeStatus, CompressionType};
use crate::error::{Error, Result};
use crate::redis::client::{RedisClient, RedisConfig};

/// Redis pub/sub bridge.
pub struct RedisPubSubBridge {
    name: String,
    config: BridgeConfig,
    redis_config: RedisConfig,
    client: Arc<RedisClient>,
    serializer: Box<dyn MessageSerializer>,
    status: RwLock<BridgeStatus>,
    metrics: BridgeMetricsInner,
    batch_tx: Option<mpsc::Sender<(String, Vec<u8>)>>,
}

struct BridgeMetricsInner {
    published: AtomicU64,
    received: AtomicU64,
    errors: AtomicU64,
    reconnects: AtomicU64,
    latency_sum_us: AtomicU64,
    batch_count: AtomicU64,
}

impl RedisPubSubBridge {
    /// Create a new Redis pub/sub bridge.
    pub fn new(
        name: impl Into<String>,
        bridge_config: BridgeConfig,
        redis_config: RedisConfig,
    ) -> Result<Self> {
        let client = Arc::new(RedisClient::new(redis_config.clone())?);
        
        Ok(Self {
            name: name.into(),
            config: bridge_config,
            redis_config,
            client,
            serializer: create_serializer(SerializerType::Rkyv),
            status: RwLock::new(BridgeStatus::Disconnected),
            metrics: BridgeMetricsInner {
                published: AtomicU64::new(0),
                received: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                reconnects: AtomicU64::new(0),
                latency_sum_us: AtomicU64::new(0),
                batch_count: AtomicU64::new(0),
            },
            batch_tx: None,
        })
    }
    
    /// Compress data if enabled.
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        if !self.config.compress {
            return data.to_vec();
        }
        
        match self.config.compression {
            CompressionType::None => data.to_vec(),
            CompressionType::Lz4 => lz4::block::compress(data, None, false).unwrap_or_else(|_| data.to_vec()),
            CompressionType::Zstd => zstd::encode_all(data, 3).unwrap_or_else(|_| data.to_vec()),
        }
    }
    
    /// Decompress data.
    fn decompress(&self, data: &[u8]) -> Vec<u8> {
        if !self.config.compress {
            return data.to_vec();
        }
        
        match self.config.compression {
            CompressionType::None => data.to_vec(),
            CompressionType::Lz4 => lz4::block::decompress(data, None).unwrap_or_else(|_| data.to_vec()),
            CompressionType::Zstd => zstd::decode_all(data).unwrap_or_else(|_| data.to_vec()),
        }
    }
    
    /// Start batch sender.
    fn start_batch_sender(&mut self) {
        let (tx, mut rx) = mpsc::channel::<(String, Vec<u8>)>(10_000);
        
        let client = Arc::clone(&self.client);
        let batch_size = self.config.batch_size;
        let batch_timeout = std::time::Duration::from_micros(self.config.batch_timeout_us);
        let metrics = Arc::new(self.metrics.clone());
        
        tokio::spawn(async move {
            let mut batch: Vec<(String, Vec<u8>)> = Vec::with_capacity(batch_size);
            let mut last_flush = Instant::now();
            
            loop {
                // Try to receive with timeout
                match tokio::time::timeout(batch_timeout, rx.recv()).await {
                    Ok(Some((channel, data))) => {
                        batch.push((channel, data));
                        
                        // Flush if batch is full
                        if batch.len() >= batch_size {
                            if let Err(e) = client.publish_batch(&batch).await {
                                tracing::error!("Batch publish failed: {}", e);
                                metrics.errors.fetch_add(1, Ordering::Relaxed);
                            } else {
                                metrics.batch_count.fetch_add(1, Ordering::Relaxed);
                            }
                            batch.clear();
                            last_flush = Instant::now();
                        }
                    }
                    Ok(None) => break, // Channel closed
                    Err(_) => {
                        // Timeout - flush if we have pending messages
                        if !batch.is_empty() && last_flush.elapsed() >= batch_timeout {
                            if let Err(e) = client.publish_batch(&batch).await {
                                tracing::error!("Batch publish failed: {}", e);
                                metrics.errors.fetch_add(1, Ordering::Relaxed);
                            } else {
                                metrics.batch_count.fetch_add(1, Ordering::Relaxed);
                            }
                            batch.clear();
                            last_flush = Instant::now();
                        }
                    }
                }
            }
        });
        
        self.batch_tx = Some(tx);
    }
}

#[async_trait]
impl Bridge for RedisPubSubBridge {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn status(&self) -> BridgeStatus {
        *self.status.read()
    }
    
    async fn connect(&self) -> Result<()> {
        *self.status.write() = BridgeStatus::Connecting;
        
        let mut attempts = 0;
        loop {
            match self.client.connect().await {
                Ok(()) => {
                    *self.status.write() = BridgeStatus::Connected;
                    return Ok(());
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.max_reconnect_attempts {
                        *self.status.write() = BridgeStatus::Failed;
                        return Err(e);
                    }
                    
                    self.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(std::time::Duration::from_millis(
                        self.config.reconnect_interval_ms,
                    ))
                    .await;
                }
            }
        }
    }
    
    async fn disconnect(&self) -> Result<()> {
        *self.status.write() = BridgeStatus::Disconnected;
        Ok(())
    }
    
    async fn publish(&self, message: &Message) -> Result<()> {
        let start = Instant::now();
        
        // Serialize
        let data = self.serializer.serialize(message)?;
        
        // Compress
        let compressed = self.compress(&data);
        
        // Send via batch channel or direct
        if let Some(tx) = &self.batch_tx {
            tx.send((message.channel.clone(), compressed))
                .await
                .map_err(|_| Error::Connection("Batch channel closed".to_string()))?;
        } else {
            self.client.publish(&message.channel, &compressed).await?;
        }
        
        // Record metrics
        let elapsed = start.elapsed();
        self.metrics.published.fetch_add(1, Ordering::Relaxed);
        self.metrics.latency_sum_us.fetch_add(
            elapsed.as_micros() as u64,
            Ordering::Relaxed,
        );
        
        Ok(())
    }
    
    async fn subscribe(
        &self,
        channel: &str,
        callback: Box<dyn Fn(Message) + Send + Sync>,
    ) -> Result<()> {
        let serializer = create_serializer(SerializerType::Rkyv);
        let compress = self.config.compress;
        let compression = self.config.compression;
        let metrics_received = self.metrics.received.clone();
        
        self.client.subscribe(&[channel.to_string()], move |_ch, data| {
            // Decompress
            let decompressed = if compress {
                match compression {
                    CompressionType::None => data,
                    CompressionType::Lz4 => lz4::block::decompress(&data, None).unwrap_or(data),
                    CompressionType::Zstd => zstd::decode_all(data.as_slice()).unwrap_or(data),
                }
            } else {
                data
            };
            
            // Deserialize
            if let Ok(message) = serializer.deserialize(&decompressed) {
                metrics_received.fetch_add(1, Ordering::Relaxed);
                callback(message);
            }
        }).await?;
        
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        self.client.ping().await
    }
    
    fn metrics(&self) -> BridgeMetrics {
        let published = self.metrics.published.load(Ordering::Relaxed);
        
        BridgeMetrics {
            published,
            received: self.metrics.received.load(Ordering::Relaxed),
            errors: self.metrics.errors.load(Ordering::Relaxed),
            reconnects: self.metrics.reconnects.load(Ordering::Relaxed),
            avg_latency_us: if published > 0 {
                self.metrics.latency_sum_us.load(Ordering::Relaxed) as f64 / published as f64
            } else {
                0.0
            },
            batch_count: self.metrics.batch_count.load(Ordering::Relaxed),
        }
    }
}
```

---

## 6. Kafka Bridge

### 6.1 Kafka Producer

```rust
// src/kafka/producer.rs

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord, Producer};
use rdkafka::util::Timeout;
use parking_lot::RwLock;

use sr_messaging::Message;
use sr_messaging::serde::{MessageSerializer, create_serializer, SerializerType};

use crate::bridge::{CompressionType};
use crate::error::{Error, Result};

/// Kafka producer configuration.
#[derive(Debug, Clone)]
pub struct KafkaProducerConfig {
    /// Bootstrap servers.
    pub bootstrap_servers: String,
    
    /// Client ID.
    pub client_id: String,
    
    /// Acks required (0, 1, all).
    pub acks: String,
    
    /// Compression type.
    pub compression: String,
    
    /// Batch size in bytes.
    pub batch_size: usize,
    
    /// Linger time in milliseconds.
    pub linger_ms: u64,
    
    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,
    
    /// Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL).
    pub security_protocol: String,
    
    /// SASL mechanism (PLAIN, SCRAM-SHA-256, etc.).
    pub sasl_mechanism: Option<String>,
    
    /// SASL username.
    pub sasl_username: Option<String>,
    
    /// SASL password.
    pub sasl_password: Option<String>,
    
    /// Additional configuration.
    pub additional_config: HashMap<String, String>,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        Self {
            bootstrap_servers: "localhost:9092".to_string(),
            client_id: "atlas-producer".to_string(),
            acks: "1".to_string(),
            compression: "lz4".to_string(),
            batch_size: 65536,
            linger_ms: 5,
            request_timeout_ms: 30000,
            security_protocol: "PLAINTEXT".to_string(),
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            additional_config: HashMap::new(),
        }
    }
}

/// High-performance Kafka producer.
pub struct KafkaProducerBridge {
    producer: FutureProducer,
    serializer: Box<dyn MessageSerializer>,
    topic_prefix: String,
    metrics: ProducerMetrics,
}

struct ProducerMetrics {
    produced: AtomicU64,
    errors: AtomicU64,
    latency_sum_us: AtomicU64,
}

impl KafkaProducerBridge {
    /// Create a new Kafka producer.
    pub fn new(config: KafkaProducerConfig, topic_prefix: String) -> Result<Self> {
        let mut client_config = ClientConfig::new();
        
        client_config
            .set("bootstrap.servers", &config.bootstrap_servers)
            .set("client.id", &config.client_id)
            .set("acks", &config.acks)
            .set("compression.type", &config.compression)
            .set("batch.size", config.batch_size.to_string())
            .set("linger.ms", config.linger_ms.to_string())
            .set("request.timeout.ms", config.request_timeout_ms.to_string())
            .set("security.protocol", &config.security_protocol);
        
        if let Some(mechanism) = &config.sasl_mechanism {
            client_config.set("sasl.mechanism", mechanism);
        }
        if let Some(username) = &config.sasl_username {
            client_config.set("sasl.username", username);
        }
        if let Some(password) = &config.sasl_password {
            client_config.set("sasl.password", password);
        }
        
        for (key, value) in &config.additional_config {
            client_config.set(key, value);
        }
        
        let producer: FutureProducer = client_config
            .create()
            .map_err(|e| Error::Connection(e.to_string()))?;
        
        Ok(Self {
            producer,
            serializer: create_serializer(SerializerType::Rkyv),
            topic_prefix,
            metrics: ProducerMetrics {
                produced: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                latency_sum_us: AtomicU64::new(0),
            },
        })
    }
    
    /// Map channel to Kafka topic.
    fn channel_to_topic(&self, channel: &str) -> String {
        if self.topic_prefix.is_empty() {
            channel.to_string()
        } else {
            format!("{}.{}", self.topic_prefix, channel)
        }
    }
    
    /// Produce a message.
    pub async fn produce(&self, message: &Message) -> Result<()> {
        let start = Instant::now();
        
        // Serialize
        let payload = self.serializer.serialize(message)?;
        let topic = self.channel_to_topic(&message.channel);
        
        // Create record
        let record = FutureRecord::to(&topic)
            .key(&message.id.0)
            .payload(&payload);
        
        // Send with timeout
        let delivery_status = self.producer
            .send(record, Timeout::After(Duration::from_secs(5)))
            .await;
        
        match delivery_status {
            Ok(_) => {
                let elapsed = start.elapsed();
                self.metrics.produced.fetch_add(1, Ordering::Relaxed);
                self.metrics.latency_sum_us.fetch_add(
                    elapsed.as_micros() as u64,
                    Ordering::Relaxed,
                );
                Ok(())
            }
            Err((e, _)) => {
                self.metrics.errors.fetch_add(1, Ordering::Relaxed);
                Err(Error::Connection(e.to_string()))
            }
        }
    }
    
    /// Produce a batch of messages.
    pub async fn produce_batch(&self, messages: &[Message]) -> Result<Vec<Result<()>>> {
        let mut futures = Vec::with_capacity(messages.len());
        
        for message in messages {
            let payload = self.serializer.serialize(message)?;
            let topic = self.channel_to_topic(&message.channel);
            
            let record = FutureRecord::to(&topic)
                .key(&message.id.0)
                .payload(&payload);
            
            let future = self.producer.send(record, Timeout::After(Duration::from_secs(5)));
            futures.push(future);
        }
        
        // Await all futures
        let results: Vec<Result<()>> = futures::future::join_all(futures)
            .await
            .into_iter()
            .map(|r| match r {
                Ok(_) => {
                    self.metrics.produced.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                Err((e, _)) => {
                    self.metrics.errors.fetch_add(1, Ordering::Relaxed);
                    Err(Error::Connection(e.to_string()))
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Flush pending messages.
    pub fn flush(&self, timeout: Duration) -> Result<()> {
        self.producer.flush(Timeout::After(timeout));
        Ok(())
    }
    
    /// Get metrics.
    pub fn metrics(&self) -> (u64, u64, f64) {
        let produced = self.metrics.produced.load(Ordering::Relaxed);
        let errors = self.metrics.errors.load(Ordering::Relaxed);
        let avg_latency = if produced > 0 {
            self.metrics.latency_sum_us.load(Ordering::Relaxed) as f64 / produced as f64
        } else {
            0.0
        };
        (produced, errors, avg_latency)
    }
}
```

### 6.2 Kafka Consumer

```rust
// src/kafka/consumer.rs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::Message as KafkaMessage;
use tokio::sync::broadcast;

use sr_messaging::Message;
use sr_messaging::serde::{MessageSerializer, create_serializer, SerializerType};

use crate::error::{Error, Result};

/// Kafka consumer configuration.
#[derive(Debug, Clone)]
pub struct KafkaConsumerConfig {
    /// Bootstrap servers.
    pub bootstrap_servers: String,
    
    /// Consumer group ID.
    pub group_id: String,
    
    /// Auto offset reset (earliest, latest).
    pub auto_offset_reset: String,
    
    /// Enable auto commit.
    pub enable_auto_commit: bool,
    
    /// Session timeout in milliseconds.
    pub session_timeout_ms: u64,
    
    /// Security settings (same as producer).
    pub security_protocol: String,
    pub sasl_mechanism: Option<String>,
    pub sasl_username: Option<String>,
    pub sasl_password: Option<String>,
    
    /// Additional configuration.
    pub additional_config: HashMap<String, String>,
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        Self {
            bootstrap_servers: "localhost:9092".to_string(),
            group_id: "atlas-consumer".to_string(),
            auto_offset_reset: "latest".to_string(),
            enable_auto_commit: true,
            session_timeout_ms: 30000,
            security_protocol: "PLAINTEXT".to_string(),
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            additional_config: HashMap::new(),
        }
    }
}

/// Kafka consumer.
pub struct KafkaConsumerBridge {
    consumer: Arc<StreamConsumer>,
    serializer: Box<dyn MessageSerializer>,
    shutdown_tx: broadcast::Sender<()>,
}

impl KafkaConsumerBridge {
    /// Create a new Kafka consumer.
    pub fn new(config: KafkaConsumerConfig) -> Result<Self> {
        let mut client_config = ClientConfig::new();
        
        client_config
            .set("bootstrap.servers", &config.bootstrap_servers)
            .set("group.id", &config.group_id)
            .set("auto.offset.reset", &config.auto_offset_reset)
            .set("enable.auto.commit", config.enable_auto_commit.to_string())
            .set("session.timeout.ms", config.session_timeout_ms.to_string())
            .set("security.protocol", &config.security_protocol);
        
        if let Some(mechanism) = &config.sasl_mechanism {
            client_config.set("sasl.mechanism", mechanism);
        }
        if let Some(username) = &config.sasl_username {
            client_config.set("sasl.username", username);
        }
        if let Some(password) = &config.sasl_password {
            client_config.set("sasl.password", password);
        }
        
        for (key, value) in &config.additional_config {
            client_config.set(key, value);
        }
        
        let consumer: StreamConsumer = client_config
            .create()
            .map_err(|e| Error::Connection(e.to_string()))?;
        
        let (shutdown_tx, _) = broadcast::channel(1);
        
        Ok(Self {
            consumer: Arc::new(consumer),
            serializer: create_serializer(SerializerType::Rkyv),
            shutdown_tx,
        })
    }
    
    /// Subscribe to topics and process messages.
    pub async fn subscribe(
        &self,
        topics: &[&str],
        callback: impl Fn(Message) + Send + Sync + 'static,
    ) -> Result<()> {
        self.consumer.subscribe(topics)
            .map_err(|e| Error::Connection(e.to_string()))?;
        
        let consumer = Arc::clone(&self.consumer);
        let serializer = create_serializer(SerializerType::Rkyv);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    message = consumer.recv() => {
                        match message {
                            Ok(m) => {
                                if let Some(payload) = m.payload() {
                                    if let Ok(msg) = serializer.deserialize(payload) {
                                        callback(msg);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Kafka consume error: {}", e);
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop consuming.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(());
    }
}
```

---

## 7. PyO3 Bindings

```rust
// In crates/sr-python/src/bridge.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use sr_messaging::Message;
use sr_messaging_bridge::redis::{RedisPubSubBridge, RedisConfig};
use sr_messaging_bridge::kafka::{KafkaProducerConfig, KafkaConsumerConfig};
use sr_messaging_bridge::bridge::{Bridge, BridgeConfig, BridgeStatus};

/// Python wrapper for Redis bridge.
#[pyclass(name = "RedisBridge")]
pub struct PyRedisBridge {
    inner: std::sync::Arc<RedisPubSubBridge>,
    runtime: tokio::runtime::Handle,
}

#[pymethods]
impl PyRedisBridge {
    /// Create a new Redis bridge.
    #[new]
    #[pyo3(signature = (url, name="redis", compress=true))]
    fn new(url: &str, name: &str, compress: bool) -> PyResult<Self> {
        let redis_config = RedisConfig {
            url: url.to_string(),
            ..Default::default()
        };
        
        let bridge_config = BridgeConfig {
            name: name.to_string(),
            compress,
            ..Default::default()
        };
        
        let bridge = RedisPubSubBridge::new(name, bridge_config, redis_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let runtime = tokio::runtime::Handle::current();
        
        Ok(Self {
            inner: std::sync::Arc::new(bridge),
            runtime,
        })
    }
    
    /// Connect to Redis.
    fn connect<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let bridge = std::sync::Arc::clone(&self.inner);
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            bridge.connect().await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
    
    /// Publish a message.
    fn publish<'py>(
        &self,
        py: Python<'py>,
        channel: String,
        payload: Bound<'_, PyDict>,
    ) -> PyResult<&'py PyAny> {
        let bridge = std::sync::Arc::clone(&self.inner);
        
        // Convert Python dict to JSON
        let json_str = Python::with_gil(|py| {
            let json = py.import_bound("json")?;
            let result = json.call_method1("dumps", (&payload,))?;
            result.extract::<String>()
        })?;
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let json_value: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            let message = Message::new(&channel, sr_messaging::Payload::Json(json_value));
            
            bridge.publish(&message).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            Ok(())
        })
    }
    
    /// Get bridge status.
    fn status(&self) -> &str {
        match self.inner.status() {
            BridgeStatus::Disconnected => "disconnected",
            BridgeStatus::Connecting => "connecting",
            BridgeStatus::Connected => "connected",
            BridgeStatus::Reconnecting => "reconnecting",
            BridgeStatus::Failed => "failed",
        }
    }
    
    /// Get metrics.
    fn metrics(&self) -> PyResult<PyObject> {
        let m = self.inner.metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("published", m.published)?;
            dict.set_item("received", m.received)?;
            dict.set_item("errors", m.errors)?;
            dict.set_item("reconnects", m.reconnects)?;
            dict.set_item("avg_latency_us", m.avg_latency_us)?;
            Ok(dict.into())
        })
    }
    
    /// Health check.
    fn health_check<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let bridge = std::sync::Arc::clone(&self.inner);
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let healthy = bridge.health_check().await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(healthy)
        })
    }
}

/// Register bridge module.
pub fn register_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRedisBridge>()?;
    Ok(())
}
```

---

## 8. Integration with sr-messaging

```rust
// src/routing.rs

use std::sync::Arc;

use sr_messaging::{Message, MessageBus};

use crate::bridge::{Bridge, BridgeRegistry};

/// Router that forwards messages to appropriate bridges.
pub struct BridgeRouter {
    bus: Arc<MessageBus>,
    registry: Arc<BridgeRegistry>,
    route_rules: Vec<RouteRule>,
}

/// Routing rule.
pub struct RouteRule {
    /// Channel pattern (glob).
    pub channel_pattern: String,
    
    /// Bridge name.
    pub bridge_name: String,
    
    /// Forward to external only (don't deliver locally).
    pub external_only: bool,
}

impl BridgeRouter {
    /// Create a new router.
    pub fn new(bus: Arc<MessageBus>, registry: Arc<BridgeRegistry>) -> Self {
        Self {
            bus,
            registry,
            route_rules: Vec::new(),
        }
    }
    
    /// Add a routing rule.
    pub fn add_route(&mut self, rule: RouteRule) {
        self.route_rules.push(rule);
    }
    
    /// Process a message and route to bridges.
    pub async fn route(&self, message: &Message) -> Vec<Result<(), crate::error::Error>> {
        let mut results = Vec::new();
        
        for rule in &self.route_rules {
            if self.matches_pattern(&message.channel, &rule.channel_pattern) {
                if let Some(bridge) = self.registry.get(&rule.bridge_name) {
                    results.push(bridge.publish(message).await);
                }
            }
        }
        
        results
    }
    
    /// Check if channel matches pattern.
    fn matches_pattern(&self, channel: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.ends_with("*") {
            let prefix = &pattern[..pattern.len() - 1];
            channel.starts_with(prefix)
        } else {
            channel == pattern
        }
    }
}
```

---

## 9. Acceptance Criteria

- [ ] Redis pub/sub achieves 100K+ messages/second
- [ ] Kafka producer achieves 200K+ messages/second with batching
- [ ] Compression reduces payload size by 50%+ for typical messages
- [ ] Automatic reconnection within 500ms
- [ ] Health checks complete within 100ms
- [ ] No message loss during reconnection (for persistent channels)
- [ ] PyO3 async bindings integrate cleanly with asyncio
- [ ] Bridge metrics are accurate and low-overhead

---

## 10. Testing

### Docker Compose for Tests

```yaml
# tests/docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
  
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
```

### Integration Tests

```rust
#[tokio::test]
async fn test_redis_roundtrip() {
    let redis_config = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        ..Default::default()
    };
    
    let bridge = RedisPubSubBridge::new("test", BridgeConfig::default(), redis_config).unwrap();
    bridge.connect().await.unwrap();
    
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    bridge.subscribe("test-channel", Box::new(move |msg| {
        let _ = tx.send(msg);
    })).await.unwrap();
    
    let message = Message::new("test-channel", Payload::bytes(b"hello".to_vec()));
    bridge.publish(&message).await.unwrap();
    
    let received = tokio::time::timeout(Duration::from_secs(1), rx)
        .await
        .unwrap()
        .unwrap();
    
    assert_eq!(received.channel, "test-channel");
}
```

---

## 11. Open Questions

1. **Schema Registry:** Should we integrate with Confluent Schema Registry for Kafka?
   - Recommendation: Optional, add as feature flag

2. **Message ordering:** How to preserve ordering across bridge reconnections?
   - Recommendation: Use partition keys in Kafka, implement sequence numbers

3. **Backpressure:** How to handle slow external systems?
   - Recommendation: Circuit breaker pattern with configurable thresholds
