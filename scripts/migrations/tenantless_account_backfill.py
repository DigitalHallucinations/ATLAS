#!/usr/bin/env python3
"""One-time data migration to backfill tenantless user records.

This helper assigns a provided tenant identifier to any `users` or
`user_credentials` rows that are missing a tenant.  Run this before enabling
strict tenant enforcement in the conversation store.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from modules.conversation_store._shared import _normalize_tenant_id
from modules.conversation_store.models import User, UserCredential
from modules.conversation_store.schema import create_conversation_engine


def _detect_conflicts(
    session, tenant_id: str, credentials: Sequence[UserCredential], users: Sequence[User]
) -> dict[str, set[str]]:
    existing_usernames = set(
        session.execute(
            select(UserCredential.username).where(UserCredential.tenant_id == tenant_id)
        ).scalars()
    )
    existing_emails = set(
        session.execute(select(UserCredential.email).where(UserCredential.tenant_id == tenant_id)).scalars()
    )
    existing_external_ids = set(
        session.execute(select(User.external_id).where(User.tenant_id == tenant_id)).scalars()
    )

    conflicts: dict[str, set[str]] = {"usernames": set(), "emails": set(), "external_ids": set()}

    for credential in credentials:
        if credential.username in existing_usernames:
            conflicts["usernames"].add(credential.username)
        if credential.email in existing_emails:
            conflicts["emails"].add(credential.email)

    for user in users:
        if user.external_id in existing_external_ids:
            conflicts["external_ids"].add(user.external_id)

    return {key: value for key, value in conflicts.items() if value}


def backfill_tenantless_accounts(database_url: str, tenant_id: str, *, dry_run: bool = False) -> int:
    normalized_tenant = _normalize_tenant_id(tenant_id)
    if normalized_tenant is None:
        raise ValueError("Tenant identifier must not be empty")

    engine = create_conversation_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, future=True)

    with SessionLocal() as session:
        tenantless_credentials = session.execute(
            select(UserCredential).where(
                (UserCredential.tenant_id.is_(None)) | (UserCredential.tenant_id == "")
            )
        ).scalars().all()
        tenantless_users = session.execute(
            select(User).where((User.tenant_id.is_(None)) | (User.tenant_id == ""))
        ).scalars().all()

        conflicts = _detect_conflicts(session, normalized_tenant, tenantless_credentials, tenantless_users)
        if conflicts:
            print("Found conflicts; resolve before migrating:")
            for key, values in conflicts.items():
                label = key.replace("_", " ")
                print(f"- {label}: {sorted(values)}")
            session.rollback()
            return 1

        updated_credentials = 0
        updated_users = 0

        for credential in tenantless_credentials:
            credential.tenant_id = normalized_tenant
            updated_credentials += 1

        for user in tenantless_users:
            user.tenant_id = normalized_tenant
            updated_users += 1

        if dry_run:
            session.rollback()
        else:
            session.commit()

    print("Backfill complete")
    print(f" - tenantless user_credentials updated: {updated_credentials}")
    print(f" - tenantless users updated: {updated_users}")
    if dry_run:
        print("No changes were committed because --dry-run was supplied")
    return 0


def parse_args(argv: Iterable[str]) -> Mapping[str, object]:
    parser = argparse.ArgumentParser(
        description="Assign a tenant identifier to tenantless user rows in the conversation store.",
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="SQLAlchemy database URL for the conversation store",
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Tenant identifier to assign to tenantless rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates without committing changes",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        exit_code = backfill_tenantless_accounts(
            database_url=args.database_url,
            tenant_id=args.tenant_id,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Migration failed: {exc}")
        exit_code = 1
    sys.exit(exit_code)
