"""Migration script for knowledge store tables.

Creates the knowledge_bases, knowledge_documents, and knowledge_chunks tables
with pgvector support for semantic search.

Usage:
    from scripts.migrations.knowledge_store import upgrade, downgrade
    
    # Apply migration
    with engine.begin() as connection:
        upgrade(connection)
    
    # Rollback migration
    with engine.begin() as connection:
        downgrade(connection)
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection


SCHEMA = "public"


def upgrade(connection: Connection) -> None:
    """Apply the knowledge store schema."""
    dialect_name = getattr(connection.dialect, "name", "")
    
    # Ensure pgvector extension exists (PostgreSQL only)
    if dialect_name == "postgresql":
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    
    # Create knowledge_bases table
    connection.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.knowledge_bases (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                chunk_size INTEGER DEFAULT 512,
                chunk_overlap INTEGER DEFAULT 50,
                document_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                owner_id TEXT,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    )
    
    # Create knowledge_documents table
    connection.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.knowledge_documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                knowledge_base_id UUID NOT NULL 
                    REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                source_uri TEXT,
                document_type TEXT DEFAULT 'text',
                status TEXT DEFAULT 'pending',
                chunk_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{{}}',
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                indexed_at TIMESTAMPTZ
            )
        """)
    )
    
    # Create knowledge_chunks table with vector column
    # Using dimension 3072 to support most embedding models
    if dialect_name == "postgresql":
        connection.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.knowledge_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL 
                        REFERENCES {SCHEMA}.knowledge_documents(id) ON DELETE CASCADE,
                    knowledge_base_id UUID NOT NULL 
                        REFERENCES {SCHEMA}.knowledge_bases(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector(3072),
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    start_line INTEGER,
                    end_line INTEGER,
                    section TEXT,
                    language TEXT,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        )
    else:
        # Fallback for non-PostgreSQL (store embedding as JSON)
        connection.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    start_line INTEGER,
                    end_line INTEGER,
                    section TEXT,
                    language TEXT,
                    metadata TEXT DEFAULT '{{}}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
        )
    
    # Create indexes
    _create_indexes(connection, dialect_name)
    
    connection.commit()


def _create_indexes(connection: Connection, dialect_name: str) -> None:
    """Create indexes for the knowledge store tables."""
    
    # Knowledge bases indexes
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_kb_owner
            ON {SCHEMA}.knowledge_bases(owner_id)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_kb_name
            ON {SCHEMA}.knowledge_bases(name)
        """)
    )
    
    # Knowledge documents indexes
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_doc_kb
            ON {SCHEMA}.knowledge_documents(knowledge_base_id)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_doc_status
            ON {SCHEMA}.knowledge_documents(status)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_doc_hash
            ON {SCHEMA}.knowledge_documents(content_hash)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_doc_created
            ON {SCHEMA}.knowledge_documents(created_at)
        """)
    )
    
    # Knowledge chunks indexes
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_chunk_doc
            ON {SCHEMA}.knowledge_chunks(document_id)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_chunk_kb
            ON {SCHEMA}.knowledge_chunks(knowledge_base_id)
        """)
    )
    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_chunk_index
            ON {SCHEMA}.knowledge_chunks(document_id, chunk_index)
        """)
    )
    
    # Vector similarity index (PostgreSQL only)
    if dialect_name == "postgresql":
        connection.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS idx_chunk_embedding
                ON {SCHEMA}.knowledge_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        )
        
        # Full-text search on content (PostgreSQL)
        connection.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS idx_chunk_content_fts
                ON {SCHEMA}.knowledge_chunks
                USING gin (to_tsvector('english', content))
            """)
        )


def downgrade(connection: Connection) -> None:
    """Rollback the knowledge store schema."""
    
    # Drop tables in reverse order of creation (respecting foreign keys)
    connection.execute(
        text(f"DROP TABLE IF EXISTS {SCHEMA}.knowledge_chunks CASCADE")
    )
    connection.execute(
        text(f"DROP TABLE IF EXISTS {SCHEMA}.knowledge_documents CASCADE")
    )
    connection.execute(
        text(f"DROP TABLE IF EXISTS {SCHEMA}.knowledge_bases CASCADE")
    )
    
    connection.commit()


def check_migration_status(connection: Connection) -> dict:
    """Check if the knowledge store tables exist and their status.
    
    Returns:
        Dict with 'tables' and 'indexes' status information.
    """
    from sqlalchemy import inspect
    
    inspector = inspect(connection)
    tables = inspector.get_table_names(schema=SCHEMA)
    
    required_tables = ["knowledge_bases", "knowledge_documents", "knowledge_chunks"]
    
    status = {
        "tables": {t: t in tables for t in required_tables},
        "all_tables_exist": all(t in tables for t in required_tables),
    }
    
    if status["all_tables_exist"]:
        # Check for vector column in knowledge_chunks
        columns = {c["name"] for c in inspector.get_columns("knowledge_chunks", schema=SCHEMA)}
        status["has_embedding_column"] = "embedding" in columns
    
    return status


__all__ = [
    "upgrade",
    "downgrade",
    "check_migration_status",
    "SCHEMA",
]
