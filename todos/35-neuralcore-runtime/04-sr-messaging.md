# 04 - sr-messaging: Lock-Free Message Bus Core

**Phase:** 4  
**Duration:** 2-3 weeks  
**Priority:** High (replaces NCB hot paths)  
**Dependencies:** 01-project-setup  

## Objective

Implement the core message bus functionality in Rust, providing lock-free priority queues, zero-copy message routing for in-process communication, efficient serialization, SQLite persistence, and atomic metrics. This replaces the performance-critical hot paths in `core/messaging/NCB.py` while maintaining the same semantics.

## Deliverables

- [ ] Lock-free multi-producer/multi-consumer priority queue
- [ ] Channel management (create, subscribe, unsubscribe)
- [ ] Message envelope with priority, metadata, trace IDs
- [ ] Zero-copy in-process routing (no serialization for local subscribers)
- [ ] Fast serialization for persistence/bridging (rkyv or bincode)
- [ ] SQLite persistence for critical channels
- [ ] Atomic metrics counters
- [ ] Rate limiting (token bucket)
- [ ] Idempotency deduplication
- [ ] Dead-letter queue support
- [ ] PyO3 bindings with async support

---

## 1. Performance Targets

| Operation | Current (Python NCB) | Target (Rust) | Improvement |
| --------- | ------------------- | ------------- | ----------- |
| Publish (in-process) | 100μs | 2μs | 50x |
| Dispatch to subscriber | 50μs | 1μs | 50x |
| Serialize message (msgpack) | 50μs | 2μs | 25x |
| Deserialize message | 40μs | 1μs | 40x |
| Priority queue push | 5μs | 0.1μs | 50x |
| Priority queue pop | 5μs | 0.1μs | 50x |
| Idempotency check | 10μs | 0.2μs | 50x |
| SQLite persist | 200μs | 50μs | 4x |

**Throughput targets:**

| Metric | Python NCB | Rust Target |
| ------ | ---------- | ----------- |
| Messages/sec (single channel) | 10K | 500K |
| Messages/sec (36 channels) | 50K | 2M |
| Concurrent subscribers | 100 | 10K |
| Memory per message | 2KB | 200B |

---

## 2. Crate Structure

```Text
crates/sr-messaging/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── bus.rs              # MessageBus main struct
│   ├── channel.rs          # Channel management
│   ├── message.rs          # Message envelope
│   ├── queue/
│   │   ├── mod.rs          # Queue abstraction
│   │   ├── priority.rs     # Lock-free priority queue
│   │   └── bounded.rs      # Bounded queue with backpressure
│   ├── routing/
│   │   ├── mod.rs          # Routing logic
│   │   ├── dispatch.rs     # Message dispatch
│   │   └── filter.rs       # Subscriber filters
│   ├── serde/
│   │   ├── mod.rs          # Serialization abstraction
│   │   ├── rkyv_impl.rs    # rkyv zero-copy
│   │   ├── bincode_impl.rs # bincode fallback
│   │   └── json_impl.rs    # JSON for debugging
│   ├── persist/
│   │   ├── mod.rs          # Persistence abstraction
│   │   ├── sqlite.rs       # SQLite backend
│   │   └── replay.rs       # Message replay logic
│   ├── metrics.rs          # Atomic counters
│   ├── limiter.rs          # Rate limiting
│   ├── idempotency.rs      # Deduplication
│   ├── dlq.rs              # Dead-letter queue
│   └── config.rs           # Configuration
└── benches/
    └── messaging_bench.rs
```

---

## 3. Core Types

### 3.1 Message Envelope

```rust
// src/message.rs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use rkyv::{Archive, Deserialize, Serialize};

/// Unique message identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct MessageId(pub String);

impl MessageId {
    /// Generate a new unique message ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
    
    /// Create from existing string.
    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Message priority (lower = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Archive, Serialize, Deserialize)]
pub struct Priority(pub u8);

impl Priority {
    pub const CRITICAL: Self = Self(0);
    pub const HIGH: Self = Self(25);
    pub const NORMAL: Self = Self(50);
    pub const LOW: Self = Self(75);
    pub const BACKGROUND: Self = Self(100);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Message metadata.
#[derive(Debug, Clone, Default, Archive, Serialize, Deserialize)]
pub struct MessageMeta {
    /// Trace ID for distributed tracing.
    pub trace_id: Option<String>,
    
    /// Correlation ID for request/response patterns.
    pub correlation_id: Option<String>,
    
    /// Source module/component.
    pub source: Option<String>,
    
    /// Custom key-value metadata.
    pub custom: HashMap<String, String>,
}

/// Message envelope containing payload and metadata.
#[derive(Debug, Clone)]
pub struct Message {
    /// Unique message ID.
    pub id: MessageId,
    
    /// Target channel.
    pub channel: String,
    
    /// Message priority.
    pub priority: Priority,
    
    /// Timestamp (nanos since epoch).
    pub timestamp_ns: u64,
    
    /// Message payload.
    pub payload: Payload,
    
    /// Metadata.
    pub meta: MessageMeta,
}

impl Message {
    /// Create a new message.
    pub fn new(channel: impl Into<String>, payload: Payload) -> Self {
        Self {
            id: MessageId::new(),
            channel: channel.into(),
            priority: Priority::default(),
            timestamp_ns: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            payload,
            meta: MessageMeta::default(),
        }
    }
    
    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set trace ID.
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.meta.trace_id = Some(trace_id.into());
        self
    }
}

/// Message payload variants.
#[derive(Debug, Clone)]
pub enum Payload {
    /// Raw bytes (zero-copy for in-process).
    Bytes(Arc<[u8]>),
    
    /// JSON value (for Python interop).
    Json(serde_json::Value),
    
    /// Already serialized (for forwarding).
    Serialized {
        format: SerializationFormat,
        data: Arc<[u8]>,
    },
}

impl Payload {
    /// Create from JSON-serializable value.
    pub fn json<T: serde::Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::Json(serde_json::to_value(value)?))
    }
    
    /// Create from raw bytes.
    pub fn bytes(data: impl Into<Arc<[u8]>>) -> Self {
        Self::Bytes(data.into())
    }
    
    /// Get payload as JSON value.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get payload as bytes.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            Self::Serialized { data, .. } => Some(data),
            _ => None,
        }
    }
}

/// Serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    Rkyv,
    Bincode,
    Json,
    MessagePack,
}
```

### 3.2 Channel Configuration

```rust
// src/channel.rs

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::config::ChannelConfig;
use crate::limiter::RateLimiter;
use crate::message::{Message, Priority};
use crate::queue::PriorityQueue;

/// A message channel with queue and subscribers.
pub struct Channel {
    /// Channel name.
    pub name: String,
    
    /// Configuration.
    pub config: ChannelConfig,
    
    /// Priority queue.
    queue: PriorityQueue<Message>,
    
    /// Subscribers.
    subscribers: RwLock<Vec<SubscriberHandle>>,
    
    /// Rate limiter (optional).
    rate_limiter: Option<RateLimiter>,
    
    /// Dead-letter channel name (optional).
    dead_letter: Option<String>,
}

impl Channel {
    /// Create a new channel.
    pub fn new(name: impl Into<String>, config: ChannelConfig) -> Self {
        let name = name.into();
        let rate_limiter = config.rate_limit_per_sec.map(|rate| {
            RateLimiter::new(rate, config.rate_limit_burst.unwrap_or(rate * 2.0))
        });
        
        Self {
            name,
            queue: PriorityQueue::new(config.max_queue_size),
            subscribers: RwLock::new(Vec::new()),
            rate_limiter,
            dead_letter: config.dead_letter_channel.clone(),
            config,
        }
    }
    
    /// Publish a message to this channel.
    pub fn publish(&self, message: Message) -> Result<(), PublishError> {
        // Check rate limit
        if let Some(limiter) = &self.rate_limiter {
            if !limiter.try_acquire(1) {
                return Err(PublishError::RateLimited);
            }
        }
        
        // Push to queue
        match self.queue.push(message) {
            Ok(()) => Ok(()),
            Err(queue_err) => {
                match self.config.drop_policy.as_str() {
                    "drop_new" => Err(PublishError::QueueFull),
                    "drop_old" => {
                        // Drop oldest and retry
                        self.queue.pop();
                        self.queue.push(queue_err.message).map_err(|_| PublishError::QueueFull)
                    }
                    _ => Err(PublishError::QueueFull),
                }
            }
        }
    }
    
    /// Subscribe to this channel.
    pub fn subscribe(&self, subscriber: SubscriberHandle) -> SubscriptionId {
        let id = SubscriptionId::new();
        let mut subs = self.subscribers.write();
        subs.push(subscriber);
        id
    }
    
    /// Unsubscribe.
    pub fn unsubscribe(&self, id: SubscriptionId) {
        let mut subs = self.subscribers.write();
        subs.retain(|s| s.id != id);
    }
    
    /// Get queue size.
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }
    
    /// Pop next message for dispatch.
    pub fn pop(&self) -> Option<Message> {
        self.queue.pop()
    }
}

/// Unique subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

impl SubscriptionId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Handle to a subscriber.
pub struct SubscriberHandle {
    pub id: SubscriptionId,
    pub module_name: String,
    pub filter: Option<Box<dyn Fn(&Message) -> bool + Send + Sync>>,
    pub callback: Box<dyn Fn(Message) + Send + Sync>,
}

/// Publish error.
#[derive(Debug, Clone)]
pub enum PublishError {
    QueueFull,
    RateLimited,
    ChannelNotFound,
    SerializationFailed(String),
}
```

---

## 4. Lock-Free Priority Queue

```rust
// src/queue/priority.rs

use std::sync::atomic::{AtomicUsize, Ordering};
use std::cmp::Reverse;

use crossbeam::queue::SegQueue;
use parking_lot::Mutex;

use crate::message::{Message, Priority};

/// Lock-free priority queue using multiple level queues.
/// 
/// Uses a combination of lock-free queues (one per priority level)
/// to achieve O(1) push and approximate priority ordering.
pub struct PriorityQueue<T> {
    /// Queues per priority level (0-255, grouped into 8 buckets).
    levels: [SegQueue<T>; 8],
    
    /// Current size.
    size: AtomicUsize,
    
    /// Maximum size (0 = unbounded).
    max_size: usize,
    
    /// Bitmask of non-empty levels (for fast scanning).
    non_empty: AtomicUsize,
}

impl<T> PriorityQueue<T> {
    /// Create a new priority queue.
    pub fn new(max_size: usize) -> Self {
        Self {
            levels: [
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
            ],
            size: AtomicUsize::new(0),
            max_size,
            non_empty: AtomicUsize::new(0),
        }
    }
    
    /// Push an item with given priority.
    pub fn push_with_priority(&self, item: T, priority: u8) -> Result<(), QueueFullError<T>> {
        // Check capacity
        if self.max_size > 0 && self.size.load(Ordering::Relaxed) >= self.max_size {
            return Err(QueueFullError { message: item });
        }
        
        // Map priority 0-255 to bucket 0-7
        let bucket = (priority / 32).min(7) as usize;
        
        self.levels[bucket].push(item);
        self.size.fetch_add(1, Ordering::Relaxed);
        
        // Mark bucket as non-empty
        self.non_empty.fetch_or(1 << bucket, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Pop the highest priority item.
    pub fn pop(&self) -> Option<T> {
        // Scan from highest priority (bucket 0) to lowest
        for bucket in 0..8 {
            if let Some(item) = self.levels[bucket].pop() {
                self.size.fetch_sub(1, Ordering::Relaxed);
                
                // Update non-empty mask if bucket is now empty
                if self.levels[bucket].is_empty() {
                    self.non_empty.fetch_and(!(1 << bucket), Ordering::Relaxed);
                }
                
                return Some(item);
            }
        }
        
        None
    }
    
    /// Get current size.
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Drain all items.
    pub fn drain(&self) -> Vec<T> {
        let mut items = Vec::with_capacity(self.len());
        while let Some(item) = self.pop() {
            items.push(item);
        }
        items
    }
}

impl PriorityQueue<Message> {
    /// Push a message using its priority.
    pub fn push(&self, message: Message) -> Result<(), QueueFullError<Message>> {
        let priority = message.priority.0;
        self.push_with_priority(message, priority)
    }
}

/// Error when queue is full.
#[derive(Debug)]
pub struct QueueFullError<T> {
    pub message: T,
}
```

---

## 5. Message Bus

```rust
// src/bus.rs

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use crossbeam::channel::{self, Sender, Receiver};
use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::channel::{Channel, ChannelConfig, PublishError, SubscriberHandle, SubscriptionId};
use crate::config::BusConfig;
use crate::idempotency::IdempotencyStore;
use crate::message::{Message, MessageId, Payload, Priority};
use crate::metrics::Metrics;
use crate::persist::Persistence;

/// High-performance message bus.
pub struct MessageBus {
    /// Configuration.
    config: BusConfig,
    
    /// Channels by name.
    channels: RwLock<HashMap<String, Arc<Channel>>>,
    
    /// Metrics.
    metrics: Arc<Metrics>,
    
    /// Idempotency store.
    idempotency: Option<IdempotencyStore>,
    
    /// Persistence backend.
    persistence: Option<Arc<dyn Persistence + Send + Sync>>,
    
    /// Background dispatch workers.
    dispatch_workers: Vec<thread::JoinHandle<()>>,
    
    /// Shutdown signal.
    shutdown_tx: Option<Sender<()>>,
    
    /// Running flag.
    running: std::sync::atomic::AtomicBool,
}

impl MessageBus {
    /// Create a new message bus.
    pub fn new(config: BusConfig) -> Self {
        Self {
            config,
            channels: RwLock::new(HashMap::new()),
            metrics: Arc::new(Metrics::new()),
            idempotency: None,
            persistence: None,
            dispatch_workers: Vec::new(),
            shutdown_tx: None,
            running: std::sync::atomic::AtomicBool::new(false),
        }
    }
    
    /// Create with persistence.
    pub fn with_persistence(mut self, persistence: impl Persistence + Send + Sync + 'static) -> Self {
        self.persistence = Some(Arc::new(persistence));
        self
    }
    
    /// Create with idempotency checking.
    pub fn with_idempotency(mut self, ttl_seconds: f64) -> Self {
        self.idempotency = Some(IdempotencyStore::new(ttl_seconds));
        self
    }
    
    /// Start the bus (spawns dispatch workers).
    pub fn start(&mut self, worker_count: usize) {
        use std::sync::atomic::Ordering;
        
        if self.running.swap(true, Ordering::SeqCst) {
            return; // Already running
        }
        
        let (shutdown_tx, shutdown_rx) = channel::bounded(1);
        self.shutdown_tx = Some(shutdown_tx);
        
        // Spawn dispatch workers
        for i in 0..worker_count {
            let channels = Arc::clone(&self.channels);
            let metrics = Arc::clone(&self.metrics);
            let shutdown_rx = shutdown_rx.clone();
            
            let handle = thread::spawn(move || {
                dispatch_worker_loop(i, channels, metrics, shutdown_rx);
            });
            
            self.dispatch_workers.push(handle);
        }
    }
    
    /// Stop the bus.
    pub fn stop(&mut self) {
        use std::sync::atomic::Ordering;
        
        if !self.running.swap(false, Ordering::SeqCst) {
            return; // Already stopped
        }
        
        // Signal shutdown
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        
        // Join workers
        for handle in self.dispatch_workers.drain(..) {
            let _ = handle.join();
        }
    }
    
    /// Create a channel.
    pub fn create_channel(&self, name: impl Into<String>, config: ChannelConfig) {
        let name = name.into();
        let channel = Arc::new(Channel::new(&name, config));
        
        let mut channels = self.channels.write();
        channels.insert(name, channel);
    }
    
    /// Publish a message.
    pub fn publish(&self, message: Message) -> Result<(), PublishError> {
        let start = std::time::Instant::now();
        
        // Check idempotency
        if let Some(store) = &self.idempotency {
            if !store.check_and_mark(&message.id) {
                // Duplicate, skip silently
                return Ok(());
            }
        }
        
        // Get channel
        let channels = self.channels.read();
        let channel = channels.get(&message.channel)
            .ok_or(PublishError::ChannelNotFound)?;
        
        // Persist if needed
        if channel.config.persistent {
            if let Some(persistence) = &self.persistence {
                persistence.persist(&message)?;
            }
        }
        
        // Publish to channel
        let result = channel.publish(message);
        
        // Record metrics
        let elapsed = start.elapsed();
        self.metrics.record_publish(&channel.name, elapsed, result.is_ok());
        
        result
    }
    
    /// Publish with JSON payload (convenience method).
    pub fn publish_json<T: serde::Serialize>(
        &self,
        channel: impl Into<String>,
        payload: &T,
    ) -> Result<(), PublishError> {
        let payload = Payload::json(payload)
            .map_err(|e| PublishError::SerializationFailed(e.to_string()))?;
        let message = Message::new(channel, payload);
        self.publish(message)
    }
    
    /// Subscribe to a channel.
    pub fn subscribe(
        &self,
        channel: &str,
        module_name: impl Into<String>,
        callback: impl Fn(Message) + Send + Sync + 'static,
    ) -> Result<SubscriptionId, SubscribeError> {
        let channels = self.channels.read();
        let channel = channels.get(channel)
            .ok_or(SubscribeError::ChannelNotFound)?;
        
        let subscriber = SubscriberHandle {
            id: SubscriptionId::new(),
            module_name: module_name.into(),
            filter: None,
            callback: Box::new(callback),
        };
        
        let id = channel.subscribe(subscriber);
        Ok(id)
    }
    
    /// Get metrics snapshot.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
    
    /// Get channel queue size.
    pub fn queue_size(&self, channel: &str) -> Option<usize> {
        let channels = self.channels.read();
        channels.get(channel).map(|c| c.queue_size())
    }
}

impl Drop for MessageBus {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Dispatch worker loop.
fn dispatch_worker_loop(
    worker_id: usize,
    channels: Arc<RwLock<HashMap<String, Arc<Channel>>>>,
    metrics: Arc<Metrics>,
    shutdown_rx: Receiver<()>,
) {
    loop {
        // Check for shutdown
        if shutdown_rx.try_recv().is_ok() {
            break;
        }
        
        // Round-robin dispatch across channels
        let channel_names: Vec<String> = {
            let channels = channels.read();
            channels.keys().cloned().collect()
        };
        
        let mut dispatched = false;
        
        for name in channel_names {
            let channels = channels.read();
            if let Some(channel) = channels.get(&name) {
                if let Some(message) = channel.pop() {
                    drop(channels); // Release lock before dispatch
                    
                    let start = std::time::Instant::now();
                    
                    // Dispatch to subscribers
                    let channels = channels.read();
                    if let Some(channel) = channels.get(&name) {
                        let subscribers = channel.subscribers.read();
                        for subscriber in subscribers.iter() {
                            // Check filter
                            if let Some(filter) = &subscriber.filter {
                                if !filter(&message) {
                                    continue;
                                }
                            }
                            
                            // Call subscriber
                            (subscriber.callback)(message.clone());
                        }
                    }
                    
                    let elapsed = start.elapsed();
                    metrics.record_dispatch(&name, elapsed);
                    dispatched = true;
                }
            }
        }
        
        // If no work, sleep briefly
        if !dispatched {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
}

/// Subscribe error.
#[derive(Debug)]
pub enum SubscribeError {
    ChannelNotFound,
}

/// Metrics snapshot.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub published: u64,
    pub dispatched: u64,
    pub dropped: u64,
    pub errors: u64,
    pub avg_publish_latency_us: f64,
    pub avg_dispatch_latency_us: f64,
}
```

---

## 6. Serialization

```rust
// src/serde/mod.rs

use crate::message::Message;
use crate::error::Result;

pub mod rkyv_impl;
pub mod bincode_impl;
pub mod json_impl;

/// Serialization backend trait.
pub trait MessageSerializer: Send + Sync {
    /// Serialize a message to bytes.
    fn serialize(&self, message: &Message) -> Result<Vec<u8>>;
    
    /// Deserialize a message from bytes.
    fn deserialize(&self, data: &[u8]) -> Result<Message>;
}

/// Serialization format selection.
#[derive(Debug, Clone, Copy, Default)]
pub enum SerializerType {
    /// rkyv (zero-copy, fastest).
    #[default]
    Rkyv,
    /// bincode (fast, compact).
    Bincode,
    /// JSON (human-readable, slowest).
    Json,
}

/// Create a serializer by type.
pub fn create_serializer(serializer_type: SerializerType) -> Box<dyn MessageSerializer> {
    match serializer_type {
        SerializerType::Rkyv => Box::new(rkyv_impl::RkyvSerializer),
        SerializerType::Bincode => Box::new(bincode_impl::BincodeSerializer),
        SerializerType::Json => Box::new(json_impl::JsonSerializer),
    }
}
```

```rust
// src/serde/rkyv_impl.rs

use rkyv::{Archive, Deserialize, Serialize, archived_root, to_bytes};

use crate::message::Message;
use crate::error::{Error, Result};
use super::MessageSerializer;

/// rkyv-based zero-copy serializer.
pub struct RkyvSerializer;

impl MessageSerializer for RkyvSerializer {
    fn serialize(&self, message: &Message) -> Result<Vec<u8>> {
        // For rkyv, we need an archivable message type
        let archivable = ArchivableMessage::from(message);
        let bytes = to_bytes::<_, 256>(&archivable)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        Ok(bytes.to_vec())
    }
    
    fn deserialize(&self, data: &[u8]) -> Result<Message> {
        let archived = unsafe { archived_root::<ArchivableMessage>(data) };
        let archivable: ArchivableMessage = archived.deserialize(&mut rkyv::Infallible)
            .map_err(|_| Error::Serialization("Deserialization failed".into()))?;
        Ok(archivable.into())
    }
}

/// Archivable version of Message for rkyv.
#[derive(Archive, Serialize, Deserialize)]
pub struct ArchivableMessage {
    pub id: String,
    pub channel: String,
    pub priority: u8,
    pub timestamp_ns: u64,
    pub payload: Vec<u8>,
    pub trace_id: Option<String>,
}

impl From<&Message> for ArchivableMessage {
    fn from(msg: &Message) -> Self {
        let payload = match &msg.payload {
            crate::message::Payload::Bytes(b) => b.to_vec(),
            crate::message::Payload::Json(v) => serde_json::to_vec(v).unwrap_or_default(),
            crate::message::Payload::Serialized { data, .. } => data.to_vec(),
        };
        
        Self {
            id: msg.id.0.clone(),
            channel: msg.channel.clone(),
            priority: msg.priority.0,
            timestamp_ns: msg.timestamp_ns,
            payload,
            trace_id: msg.meta.trace_id.clone(),
        }
    }
}

impl From<ArchivableMessage> for Message {
    fn from(am: ArchivableMessage) -> Self {
        Message {
            id: crate::message::MessageId(am.id),
            channel: am.channel,
            priority: crate::message::Priority(am.priority),
            timestamp_ns: am.timestamp_ns,
            payload: crate::message::Payload::Bytes(am.payload.into()),
            meta: crate::message::MessageMeta {
                trace_id: am.trace_id,
                ..Default::default()
            },
        }
    }
}
```

---

## 7. SQLite Persistence

```rust
// src/persist/sqlite.rs

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::{Connection, params};

use crate::message::Message;
use crate::serde::{MessageSerializer, SerializerType, create_serializer};
use crate::error::{Error, Result};
use super::Persistence;

/// SQLite-based message persistence.
pub struct SqlitePersistence {
    conn: Mutex<Connection>,
    serializer: Box<dyn MessageSerializer>,
}

impl SqlitePersistence {
    /// Create a new SQLite persistence backend.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| Error::Persistence(e.to_string()))?;
        
        // Create table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                priority INTEGER NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        
        // Create index
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)",
            [],
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        
        Ok(Self {
            conn: Mutex::new(conn),
            serializer: create_serializer(SerializerType::Rkyv),
        })
    }
    
    /// Create in-memory for testing.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| Error::Persistence(e.to_string()))?;
        
        conn.execute(
            "CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                priority INTEGER NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        
        Ok(Self {
            conn: Mutex::new(conn),
            serializer: create_serializer(SerializerType::Rkyv),
        })
    }
}

impl Persistence for SqlitePersistence {
    fn persist(&self, message: &Message) -> Result<()> {
        let data = self.serializer.serialize(message)?;
        
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO messages (id, channel, priority, timestamp_ns, data)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                message.id.0,
                message.channel,
                message.priority.0,
                message.timestamp_ns,
                data,
            ],
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        
        Ok(())
    }
    
    fn replay(&self, channel: &str, after_timestamp: u64) -> Result<Vec<Message>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT data FROM messages WHERE channel = ?1 AND timestamp_ns > ?2 ORDER BY timestamp_ns"
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        
        let messages = stmt.query_map(params![channel, after_timestamp], |row| {
            let data: Vec<u8> = row.get(0)?;
            Ok(data)
        }).map_err(|e| Error::Persistence(e.to_string()))?;
        
        let mut result = Vec::new();
        for msg_data in messages {
            let data = msg_data.map_err(|e| Error::Persistence(e.to_string()))?;
            let message = self.serializer.deserialize(&data)?;
            result.push(message);
        }
        
        Ok(result)
    }
    
    fn delete(&self, message_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM messages WHERE id = ?1", params![message_id])
            .map_err(|e| Error::Persistence(e.to_string()))?;
        Ok(())
    }
    
    fn prune_before(&self, timestamp_ns: u64) -> Result<usize> {
        let conn = self.conn.lock();
        let count = conn.execute(
            "DELETE FROM messages WHERE timestamp_ns < ?1",
            params![timestamp_ns],
        ).map_err(|e| Error::Persistence(e.to_string()))?;
        Ok(count)
    }
}
```

---

## 8. Metrics

```rust
// src/metrics.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Atomic metrics counters.
pub struct Metrics {
    pub published: AtomicU64,
    pub dispatched: AtomicU64,
    pub dropped: AtomicU64,
    pub errors: AtomicU64,
    pub publish_latency_sum_us: AtomicU64,
    pub dispatch_latency_sum_us: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            published: AtomicU64::new(0),
            dispatched: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            publish_latency_sum_us: AtomicU64::new(0),
            dispatch_latency_sum_us: AtomicU64::new(0),
        }
    }
    
    pub fn record_publish(&self, _channel: &str, latency: Duration, success: bool) {
        if success {
            self.published.fetch_add(1, Ordering::Relaxed);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        
        self.publish_latency_sum_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed,
        );
    }
    
    pub fn record_dispatch(&self, _channel: &str, latency: Duration) {
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        self.dispatch_latency_sum_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed,
        );
    }
    
    pub fn record_error(&self, _channel: &str) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn snapshot(&self) -> crate::bus::MetricsSnapshot {
        let published = self.published.load(Ordering::Relaxed);
        let dispatched = self.dispatched.load(Ordering::Relaxed);
        
        crate::bus::MetricsSnapshot {
            published,
            dispatched,
            dropped: self.dropped.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            avg_publish_latency_us: if published > 0 {
                self.publish_latency_sum_us.load(Ordering::Relaxed) as f64 / published as f64
            } else {
                0.0
            },
            avg_dispatch_latency_us: if dispatched > 0 {
                self.dispatch_latency_sum_us.load(Ordering::Relaxed) as f64 / dispatched as f64
            } else {
                0.0
            },
        }
    }
}
```

---

## 9. PyO3 Bindings

```rust
// In crates/sr-python/src/messaging.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyString};

use sr_messaging::{
    MessageBus, Message, Payload, Priority, ChannelConfig, BusConfig,
};
use sr_messaging::persist::SqlitePersistence;

/// Python wrapper for MessageBus.
#[pyclass(name = "MessageBus")]
pub struct PyMessageBus {
    inner: std::sync::Arc<std::sync::RwLock<MessageBus>>,
}

#[pymethods]
impl PyMessageBus {
    /// Create a new message bus.
    #[new]
    #[pyo3(signature = (persistence_path=None, worker_count=4))]
    fn new(persistence_path: Option<&str>, worker_count: usize) -> PyResult<Self> {
        let config = BusConfig::default();
        let mut bus = MessageBus::new(config);
        
        if let Some(path) = persistence_path {
            let persistence = SqlitePersistence::new(path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            bus = bus.with_persistence(persistence);
        }
        
        bus.start(worker_count);
        
        Ok(Self {
            inner: std::sync::Arc::new(std::sync::RwLock::new(bus)),
        })
    }
    
    /// Create a channel.
    fn create_channel(&self, name: &str, max_queue_size: Option<usize>) -> PyResult<()> {
        let config = ChannelConfig {
            max_queue_size: max_queue_size.unwrap_or(1000),
            ..Default::default()
        };
        
        let bus = self.inner.read().unwrap();
        bus.create_channel(name, config);
        Ok(())
    }
    
    /// Publish a message.
    fn publish(
        &self,
        channel: &str,
        payload: &Bound<'_, PyDict>,
        priority: Option<u8>,
        trace_id: Option<&str>,
    ) -> PyResult<()> {
        // Convert Python dict to JSON
        let json_str = Python::with_gil(|py| {
            let json = py.import_bound("json")?;
            let result = json.call_method1("dumps", (payload,))?;
            result.extract::<String>()
        })?;
        
        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let mut message = Message::new(channel, Payload::Json(json_value));
        
        if let Some(p) = priority {
            message = message.with_priority(Priority(p));
        }
        
        if let Some(tid) = trace_id {
            message = message.with_trace_id(tid);
        }
        
        let bus = self.inner.read().unwrap();
        bus.publish(message)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    /// Get metrics.
    fn metrics(&self) -> PyResult<PyObject> {
        let bus = self.inner.read().unwrap();
        let snapshot = bus.metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("published", snapshot.published)?;
            dict.set_item("dispatched", snapshot.dispatched)?;
            dict.set_item("dropped", snapshot.dropped)?;
            dict.set_item("errors", snapshot.errors)?;
            dict.set_item("avg_publish_latency_us", snapshot.avg_publish_latency_us)?;
            dict.set_item("avg_dispatch_latency_us", snapshot.avg_dispatch_latency_us)?;
            Ok(dict.into())
        })
    }
    
    /// Get queue size for a channel.
    fn queue_size(&self, channel: &str) -> PyResult<Option<usize>> {
        let bus = self.inner.read().unwrap();
        Ok(bus.queue_size(channel))
    }
    
    /// Stop the bus.
    fn stop(&self) -> PyResult<()> {
        let mut bus = self.inner.write().unwrap();
        bus.stop();
        Ok(())
    }
}

/// Register messaging functions in the module.
pub fn register_messaging(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMessageBus>()?;
    Ok(())
}
```

---

## 10. Acceptance Criteria

- [ ] Lock-free queue handles 500K+ push/pop operations per second
- [ ] Message publish latency < 5μs for in-process routing
- [ ] Serialization (rkyv) is ≥20x faster than msgpack
- [ ] SQLite persistence handles 10K+ messages/second
- [ ] No memory leaks under sustained load (test with valgrind)
- [ ] Thread-safe for concurrent publishers and subscribers
- [ ] PyO3 bindings integrate with existing ATLAS NCB patterns
- [ ] Metrics are accurate within 1% under load

---

## 11. Testing Strategy

### Load Tests

```rust
#[test]
fn test_high_throughput() {
    let mut bus = MessageBus::new(BusConfig::default());
    bus.create_channel("test", ChannelConfig::default());
    bus.start(4);
    
    let start = std::time::Instant::now();
    let count = 100_000;
    
    for i in 0..count {
        let msg = Message::new("test", Payload::bytes(format!("msg-{}", i).into_bytes()));
        bus.publish(msg).unwrap();
    }
    
    let elapsed = start.elapsed();
    let rate = count as f64 / elapsed.as_secs_f64();
    
    println!("Published {} messages in {:?} ({:.0} msg/sec)", count, elapsed, rate);
    assert!(rate > 100_000.0, "Expected >100K msg/sec, got {}", rate);
}
```

### Correctness Tests

```rust
#[test]
fn test_priority_ordering() {
    let queue = PriorityQueue::new(100);
    
    queue.push_with_priority("low", 100).unwrap();
    queue.push_with_priority("high", 0).unwrap();
    queue.push_with_priority("medium", 50).unwrap();
    
    assert_eq!(queue.pop(), Some("high"));
    assert_eq!(queue.pop(), Some("medium"));
    assert_eq!(queue.pop(), Some("low"));
}
```

---

## 12. Open Questions

1. **Subscriber callback execution:** Should callbacks run in the dispatch worker or be offloaded to a separate pool?
   - Recommendation: Separate pool to avoid blocking dispatch

2. **Backpressure strategy:** How to handle slow subscribers?
   - Recommendation: Per-subscriber queue with configurable size, drop oldest on overflow

3. **Message TTL:** Should messages expire automatically?
   - Recommendation: Yes, configurable per-channel

4. **Async Python integration:** How to handle async Python callbacks?
   - Recommendation: Use `pyo3-asyncio` with tokio runtime
