# Database Schema Documentation

This document describes the MongoDB collections used in the chat application, their purposes, field descriptions, and indexing information.

## Collection: chat_sessions

Purpose: Stores information about individual chat sessions.

### Fields:

| Field Name | Type | Description |
|------------|------|-------------|
| _id | ObjectId | Unique identifier for the chat session |
| conversation_history | Array | List of message objects in the conversation |
| created_at | Date | Timestamp of when the session was created |
| last_updated | Date | Timestamp of the last update to the session |

### Indexes:

- `_id`: Default index, provides quick lookup by session ID

### Sample Document:

```json
{
  "_id": ObjectId("5f8a7b2d9d3e7a1c9b8c7d6e"),
  "conversation_history": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "Hello! As an AI assistant, I don't have feelings, but I'm functioning well and ready to help you. How can I assist you today?"}
  ],
  "created_at": ISODate("2023-05-01T10:00:00Z"),
  "last_updated": ISODate("2023-05-01T10:05:00Z")
}
```

## Collection: messages

Purpose: Stores individual messages across all chat sessions.

### Fields:

| Field Name | Type | Description |
|------------|------|-------------|
| _id | ObjectId | Unique identifier for the message |
| message_id | String | Custom message identifier |
| message_number | Int32 | Sequential number of the message |
| content | String | The text content of the message |
| created_at | Date | Timestamp of when the message was created |

### Indexes:

- `_id`: Default index
- `message_id`: Unique index for quick lookup
- `message_number`: Index for sorting and pagination
- `content`: Text index for full-text search capabilities

### Sample Document:

```json
{
  "_id": ObjectId("5f8a7b2d9d3e7a1c9b8c7d6f"),
  "message_id": "msg_123456789",
  "message_number": 1,
  "content": "Hello, how are you?",
  "created_at": ISODate("2023-05-01T10:01:00Z")
}
```

## Collection: message_count

Purpose: Maintains a counter for generating sequential message numbers.

### Fields:

| Field Name | Type | Description |
|------------|------|-------------|
| _id | ObjectId | Unique identifier for the document |
| count_id | String | Identifier for the counter (always "message_count") |
| count | Int32 | The current message count |

### Indexes:

- `_id`: Default index
- `count_id`: Unique index

### Sample Document:

```json
{
  "_id": ObjectId("5f8a7b2d9d3e7a1c9b8c7d70"),
  "count_id": "message_count",
  "count": 1000
}
```

## Collection: archive

Purpose: Stores archived messages that have been removed from the active messages collection.

### Fields:

| Field Name | Type | Description |
|------------|------|-------------|
| _id | ObjectId | Unique identifier for the archived document |
| original_message_id | String | The message_id from the original messages collection |
| message | Object | The original message document |
| archived_at | Date | Timestamp of when the message was archived |

### Indexes:

- `_id`: Default index
- `original_message_id`: Index for quick lookup of archived messages
- `archived_at`: Index for date-based queries and cleanup operations

### Sample Document:

```json
{
  "_id": ObjectId("5f8a7b2d9d3e7a1c9b8c7d71"),
  "original_message_id": "msg_123456789",
  "message": {
    "message_id": "msg_123456789",
    "message_number": 1,
    "content": "Hello, how are you?",
    "created_at": ISODate("2023-05-01T10:01:00Z")
  },
  "archived_at": ISODate("2023-06-01T00:00:00Z")
}
```

## Collection: operation_logs

Purpose: Stores logs of database operations for auditing purposes.

### Fields:

| Field Name | Type | Description |
|------------|------|-------------|
| _id | ObjectId | Unique identifier for the log entry |
| operation | String | The type of operation performed |
| details | Object | Details about the operation |
| timestamp | Date | Timestamp of when the operation occurred |

### Indexes:

- `_id`: Default index
- `operation`: Index for querying specific types of operations
- `timestamp`: Index for date-based queries

### Sample Document:

```json
{
  "_id": ObjectId("5f8a7b2d9d3e7a1c9b8c7d72"),
  "operation": "create_chat_session",
  "details": {
    "session_id": "5f8a7b2d9d3e7a1c9b8c7d6e"
  },
  "timestamp": ISODate("2023-05-01T10:00:00Z")
}
```

## Query Optimization Tips

1. Use the appropriate indexes when querying. For example, when fetching a specific chat session, query by `_id` in the chat_sessions collection.

2. When searching for messages, utilize the text index on the content field:
   ```javascript
   db.messages.find({ $text: { $search: "keyword" } })
   ```

3. For range queries on dates (e.g., finding recent messages), use the index on `created_at`:
   ```javascript
   db.messages.find({ created_at: { $gte: ISODate("2023-05-01") } }).sort({ created_at: -1 })
   ```

4. When fetching messages in order, use the index on `message_number`:
   ```javascript
   db.messages.find().sort({ message_number: 1 })
   ```

5. For operations that require updating the message count, use findAndModify to ensure atomicity:
   ```javascript
   db.message_count.findAndModify({
     query: { count_id: "message_count" },
     update: { $inc: { count: 1 } },
     new: true
   })
   ```

By following these schema designs and query optimization tips, you can ensure efficient data storage and retrieval in your chat application.