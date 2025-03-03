# Database Manager Documentation

## File: `Database/DB_manager.py`

### Class: `DBManager`

The `DBManager` class is responsible for managing interactions with the MongoDB database. It handles operations related to chat sessions, messages, and other database-related tasks.

#### Initialization

```python
def __init__(self, config_manager: ConfigManager):
    self.config_manager = config_manager
    self.logger = self.config_manager.setup_logger('DB_manager')
    mongo_connection_string = self.config_manager.get_mongo_connection_string()
    self.mongo_client = AsyncIOMotorClient(mongo_connection_string)
    db_name = mongo_connection_string.rsplit('/', 1)[-1].split('?')[0]
    self.db = self.mongo_client[db_name]
    self.chat_sessions_collection = self.db.chat_sessions
    self.logger.info("DBManager initialized with database: " + db_name)
```

The constructor initializes the DBManager with a ConfigManager instance, sets up logging, and establishes a connection to the MongoDB database.

#### Key Methods

1. `connect(self)`:
   - Establishes a connection to the MongoDB database.
   - Performs a ping to verify the connection.

2. `ensure_indexes(self)`:
   - Creates necessary indexes on the database collections for efficient querying.

3. `create_chat_session(self, session_data: Dict[str, Any]) -> str`:
   - Creates a new chat session in the database.
   - Returns the session ID.

4. `get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]`:
   - Retrieves a chat session by its ID.

5. `update_chat_session(self, session_id: str, update_data: Dict[str, Any]) -> bool`:
   - Updates an existing chat session with new data.

6. `delete_chat_session(self, session_id: str) -> bool`:
   - Deletes a chat session from the database.

list_chat_sessions(self, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:

Retrieves a list of chat sessions, with pagination support.
Returns a list of session data dictionaries.


save_message(self, message_content: str, message_number: int) -> str:

Saves a new message to the database.
Returns the message ID.


get_message(self, message_id: str) -> Optional[Dict[str, Any]]:

Retrieves a message by its ID.


update_message(self, message_id: str, update_data: Dict[str, Any]) -> bool:

Updates an existing message with new data.


delete_message(self, message_id: str) -> bool:

Deletes a message from the database.


get_next_message_number(self) -> int:

Generates and returns the next available message number.


search_messages(self, keyword: str, limit: int = 10, skip: int = 0) -> List[Dict]:

Searches for messages containing a specific keyword.
Supports pagination.


check_similar_messages(self, message_content: str, threshold: float = 0.8) -> List[Dict]:

Finds messages similar to the given content using text similarity.


archive_message(self, message_id: str):

Moves a message to the archive collection.


retrieve_archived_message(self, message_id: str) -> Optional[Dict]:

Retrieves an archived message by its ID.


restore_archived_message(self, message_id: str):

Moves an archived message back to the main messages collection.


cleanup_old_archives(self, days: int = 30):

Removes archived messages older than the specified number of days.


log_operation(self, operation: str, details: Dict):

Logs database operations for auditing purposes.


close(self):

Closes the database connection.



Key Features

Asynchronous operations using Motor for non-blocking database interactions.
Comprehensive CRUD operations for chat sessions and messages.
Support for message archiving and restoration.
Text-based search functionality for messages.
Similar message detection using text similarity.
Automatic message numbering system.
Operation logging for auditing.

Usage Notes

The class relies on a ConfigManager for database connection details and logging setup.
Error handling is implemented for database operations.
The class uses logging to record information and errors during operation.
Indexes are created to optimize query performance.

Potential Improvements

Implement more advanced search capabilities, such as full-text search or semantic search.
Add support for bulk operations to improve performance when dealing with large datasets.
Implement a caching layer to reduce database load for frequently accessed data.
Add support for database migrations to handle schema changes.
Implement more sophisticated archiving strategies, such as data compression or cold storage for very old data.

Module-level Function

setup_db_manager(config_manager: ConfigManager):

An asynchronous function that creates a DBManager instance and establishes the database connection.
Returns the initialized DBManager object.



This DBManager class serves as a central point for all database interactions in the application, providing a robust and feature-rich interface for managing chat sessions, messages, and related data. Its asynchronous design ensures efficient handling of database operations, making it suitable for high-performance applications.