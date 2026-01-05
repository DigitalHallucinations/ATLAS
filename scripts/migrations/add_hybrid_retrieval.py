"""Migration: Add hybrid retrieval support to knowledge_chunks table.

This migration adds the following columns and indexes to support
hybrid (dense + lexical) retrieval:

Columns:
- tsv: tsvector for full-text search (BM25-style lexical search)
- content_length: Cached content length for BM25 normalization
- section_path: Hierarchical path for structured documents
- parent_chunk_id: Reference to parent chunk for hierarchical chunking
- is_parent: Boolean flag indicating parent chunks

Indexes:
- idx_chunk_tsv: GIN index on tsvector for fast full-text search
- idx_chunk_parent: Partial index for parent chunk lookups

Usage:
    python scripts/migrations/add_hybrid_retrieval.py --schema atlas
    python scripts/migrations/add_hybrid_retrieval.py --dry-run
    python scripts/migrations/add_hybrid_retrieval.py --rollback

Requirements:
    - PostgreSQL 12+ with pg_tsvector extension
    - Existing knowledge_chunks table
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Optional

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except ImportError:
    print("Error: sqlalchemy is required. Install with: pip install sqlalchemy")
    sys.exit(1)


# Migration version for tracking
MIGRATION_VERSION = "2024_001_hybrid_retrieval"
MIGRATION_DESCRIPTION = "Add hybrid retrieval support to knowledge_chunks"


def get_connection_string() -> str:
    """Get database connection string from environment or config."""
    import os

    # Try environment variable first
    conn_str = os.getenv("ATLAS_DATABASE_URL") or os.getenv("DATABASE_URL")
    if conn_str:
        return conn_str

    # Try to load from config
    try:
        from core.config import ConfigManager
        config = ConfigManager()
        # Try the conversation store URL method if available
        getter = getattr(config, "get_conversation_store_database_url", None)
        if getter:
            result = getter()
            if result:
                return str(result)
    except Exception:
        pass

    # Default for local development
    return "postgresql://atlas:atlas@localhost:5432/atlas"


def create_db_engine(connection_string: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine."""
    conn_str = connection_string or get_connection_string()
    return create_engine(conn_str)


def check_column_exists(engine: Engine, schema: str, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
              AND column_name = :column
        )
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {
            "schema": schema,
            "table": table,
            "column": column,
        })
        return bool(result.scalar())


def check_index_exists(engine: Engine, schema: str, index_name: str) -> bool:
    """Check if an index exists."""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = :schema
              AND indexname = :index_name
        )
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {
            "schema": schema,
            "index_name": index_name,
        })
        return bool(result.scalar())


def run_migration(
    engine: Engine,
    schema: str = "atlas",
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Run the migration to add hybrid retrieval support.

    Args:
        engine: SQLAlchemy engine.
        schema: Database schema name.
        dry_run: If True, print SQL without executing.
        verbose: Print progress messages.

    Returns:
        True if migration succeeded.
    """
    table = "knowledge_chunks"

    # Define columns to add
    columns = [
        ("tsv", "tsvector", None),
        ("content_length", "INTEGER", "0"),
        ("section_path", "TEXT", None),
        ("parent_chunk_id", "UUID", None),
        ("is_parent", "BOOLEAN", "FALSE"),
    ]

    # Define indexes to add
    indexes = [
        ("idx_chunk_tsv", f"CREATE INDEX idx_chunk_tsv ON {schema}.{table} USING GIN(tsv)"),
        ("idx_chunk_parent", f"CREATE INDEX idx_chunk_parent ON {schema}.{table}(parent_chunk_id) WHERE parent_chunk_id IS NOT NULL"),
        ("idx_chunk_section_path", f"CREATE INDEX idx_chunk_section_path ON {schema}.{table}(section_path) WHERE section_path IS NOT NULL"),
    ]

    # Foreign key for parent_chunk_id
    fk_sql = f"""
        ALTER TABLE {schema}.{table}
        ADD CONSTRAINT fk_parent_chunk
        FOREIGN KEY (parent_chunk_id) REFERENCES {schema}.{table}(id)
        ON DELETE SET NULL
    """

    # Trigger to auto-populate tsvector
    trigger_sql = f"""
        CREATE OR REPLACE FUNCTION {schema}.update_chunk_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
            NEW.content_length := LENGTH(COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS tr_chunk_tsv ON {schema}.{table};

        CREATE TRIGGER tr_chunk_tsv
        BEFORE INSERT OR UPDATE OF content ON {schema}.{table}
        FOR EACH ROW
        EXECUTE FUNCTION {schema}.update_chunk_tsv();
    """

    # Backfill existing data
    backfill_sql = f"""
        UPDATE {schema}.{table}
        SET
            tsv = to_tsvector('english', COALESCE(content, '')),
            content_length = LENGTH(COALESCE(content, ''))
        WHERE tsv IS NULL;
    """

    statements = []

    # Add columns
    for col_name, col_type, default in columns:
        if not check_column_exists(engine, schema, table, col_name):
            default_clause = f" DEFAULT {default}" if default else ""
            sql = f"ALTER TABLE {schema}.{table} ADD COLUMN {col_name} {col_type}{default_clause}"
            statements.append(("Add column", col_name, sql))
        elif verbose:
            print(f"  Column {col_name} already exists, skipping")

    # Add indexes
    for idx_name, idx_sql in indexes:
        if not check_index_exists(engine, schema, idx_name):
            statements.append(("Create index", idx_name, idx_sql))
        elif verbose:
            print(f"  Index {idx_name} already exists, skipping")

    # Add foreign key (check if exists)
    fk_check = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_schema = :schema
              AND constraint_name = 'fk_parent_chunk'
        )
    """
    with engine.connect() as conn:
        fk_exists = conn.execute(text(fk_check), {"schema": schema}).scalar()

    if not fk_exists:
        statements.append(("Add foreign key", "fk_parent_chunk", fk_sql))

    # Add trigger
    statements.append(("Create trigger", "tr_chunk_tsv", trigger_sql))

    # Backfill data
    statements.append(("Backfill tsvector", "existing rows", backfill_sql))

    if dry_run:
        print("\n=== DRY RUN - SQL statements that would be executed ===\n")
        for action, target, sql in statements:
            print(f"-- {action}: {target}")
            print(f"{sql};")
            print()
        return True

    # Execute migration
    print(f"\n=== Running migration: {MIGRATION_DESCRIPTION} ===\n")
    print(f"Schema: {schema}")
    print(f"Table: {table}")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print()

    try:
        with engine.begin() as conn:
            for action, target, sql in statements:
                if verbose:
                    print(f"  {action}: {target}...")
                conn.execute(text(sql))
                if verbose:
                    print(f"    Done")

        print(f"\n=== Migration completed successfully ===")
        print(f"Finished: {datetime.utcnow().isoformat()}")
        return True

    except Exception as exc:
        print(f"\n=== Migration failed ===")
        print(f"Error: {exc}")
        return False


def run_rollback(
    engine: Engine,
    schema: str = "atlas",
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Rollback the migration.

    Args:
        engine: SQLAlchemy engine.
        schema: Database schema name.
        dry_run: If True, print SQL without executing.
        verbose: Print progress messages.

    Returns:
        True if rollback succeeded.
    """
    table = "knowledge_chunks"

    statements = [
        ("Drop trigger", "tr_chunk_tsv", f"DROP TRIGGER IF EXISTS tr_chunk_tsv ON {schema}.{table}"),
        ("Drop function", "update_chunk_tsv", f"DROP FUNCTION IF EXISTS {schema}.update_chunk_tsv()"),
        ("Drop foreign key", "fk_parent_chunk", f"ALTER TABLE {schema}.{table} DROP CONSTRAINT IF EXISTS fk_parent_chunk"),
        ("Drop index", "idx_chunk_section_path", f"DROP INDEX IF EXISTS {schema}.idx_chunk_section_path"),
        ("Drop index", "idx_chunk_parent", f"DROP INDEX IF EXISTS {schema}.idx_chunk_parent"),
        ("Drop index", "idx_chunk_tsv", f"DROP INDEX IF EXISTS {schema}.idx_chunk_tsv"),
        ("Drop column", "is_parent", f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS is_parent"),
        ("Drop column", "parent_chunk_id", f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS parent_chunk_id"),
        ("Drop column", "section_path", f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS section_path"),
        ("Drop column", "content_length", f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS content_length"),
        ("Drop column", "tsv", f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS tsv"),
    ]

    if dry_run:
        print("\n=== DRY RUN - Rollback statements ===\n")
        for action, target, sql in statements:
            print(f"-- {action}: {target}")
            print(f"{sql};")
            print()
        return True

    print(f"\n=== Rolling back migration: {MIGRATION_DESCRIPTION} ===\n")

    try:
        with engine.begin() as conn:
            for action, target, sql in statements:
                if verbose:
                    print(f"  {action}: {target}...")
                conn.execute(text(sql))

        print(f"\n=== Rollback completed successfully ===")
        return True

    except Exception as exc:
        print(f"\n=== Rollback failed ===")
        print(f"Error: {exc}")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migration: Add hybrid retrieval support to knowledge_chunks"
    )
    parser.add_argument(
        "--schema",
        default="atlas",
        help="Database schema name (default: atlas)",
    )
    parser.add_argument(
        "--connection-string",
        help="Database connection string (default: from config/env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    engine = create_db_engine(args.connection_string)
    verbose = not args.quiet

    if args.rollback:
        success = run_rollback(engine, args.schema, args.dry_run, verbose)
    else:
        success = run_migration(engine, args.schema, args.dry_run, verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
