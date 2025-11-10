"""Account management helpers for the conversation store."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, ContextManager, Dict, List, Optional

from ._compat import (
    IntegrityError,
    Session,
    and_,
    delete,
    or_,
    select,
)
from ._shared import (
    _coerce_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_attempts,
    _normalize_tenant_id,
    _tenant_filter,
)
from .models import PasswordResetToken, UserCredential, UserLoginAttempt


class AccountStore:
    """Encapsulates user credential and authentication helpers."""

    def __init__(self, session_scope: Callable[[], ContextManager[Session]]) -> None:
        self._session_scope = session_scope

    # ------------------------------------------------------------------
    # Credential helpers

    def create_user_account(
        self,
        username: str,
        password_hash: str,
        email: str,
        *,
        tenant_id: Optional[Any] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
        user_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        cleaned_username = str(username).strip()
        if not cleaned_username:
            raise ValueError("Username must not be empty")
        cleaned_email = str(email).strip().lower()
        if not cleaned_email:
            raise ValueError("Email must not be empty")

        normalised_name = name.strip() if isinstance(name, str) and name.strip() else None
        normalised_dob = dob.strip() if isinstance(dob, str) and dob.strip() else None
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None
        tenant = _normalize_tenant_id(tenant_id)

        with self._session_scope() as session:
            if tenant is not None:
                legacy_credential = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned_username)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if legacy_credential is None:
                    legacy_credential = session.execute(
                        select(UserCredential)
                        .where(UserCredential.email == cleaned_email)
                        .where(UserCredential.tenant_id.is_(None))
                    ).scalar_one_or_none()
                if legacy_credential is not None:
                    legacy_credential.tenant_id = tenant
                    legacy_credential.username = cleaned_username
                    legacy_credential.password_hash = password_hash
                    legacy_credential.email = cleaned_email
                    legacy_credential.name = normalised_name
                    legacy_credential.dob = normalised_dob
                    legacy_credential.user_id = user_uuid
                    if legacy_credential.failed_attempts is None:
                        legacy_credential.failed_attempts = []
                    session.flush()
                    return _serialize_credential(legacy_credential)

            tenant_clause = _tenant_filter(UserCredential.tenant_id, tenant)
            username_conflict = session.execute(
                select(UserCredential.id)
                .where(UserCredential.username == cleaned_username)
                .where(tenant_clause)
            ).scalar_one_or_none()
            if username_conflict is not None:
                raise IntegrityError("duplicate username", params=None, orig=None)  # type: ignore[arg-type]

            email_conflict = session.execute(
                select(UserCredential.id)
                .where(UserCredential.email == cleaned_email)
                .where(tenant_clause)
            ).scalar_one_or_none()
            if email_conflict is not None:
                raise IntegrityError("duplicate email", params=None, orig=None)  # type: ignore[arg-type]

            record = UserCredential(
                user_id=user_uuid,
                tenant_id=tenant,
                username=cleaned_username,
                password_hash=password_hash,
                email=cleaned_email,
                name=normalised_name,
                dob=normalised_dob,
                failed_attempts=[],
            )
            session.add(record)
            try:
                session.flush()
            except IntegrityError:
                raise
            return _serialize_credential(record)

    def attach_credential(
        self, username: str, user_id: Any, *, tenant_id: Optional[Any] = None
    ) -> Optional[str]:
        cleaned_username = str(username).strip()
        if not cleaned_username:
            return None
        user_uuid = _coerce_uuid(user_id)
        tenant = _normalize_tenant_id(tenant_id)

        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential)
                .where(UserCredential.username == cleaned_username)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if credential is None and tenant is not None:
                legacy = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned_username)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if legacy is not None:
                    legacy.tenant_id = tenant
                    credential = legacy
            if credential is None:
                return None
            if credential.user_id == user_uuid:
                existing = credential.user_id
                return str(existing) if existing is not None else None
            credential.user_id = user_uuid
            session.flush()
            return str(credential.user_id) if credential.user_id is not None else None

    def get_user_account(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential)
                .where(UserCredential.username == cleaned)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if record is None and tenant is not None:
                legacy = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if legacy is not None:
                    legacy.tenant_id = tenant
                    record = legacy
            if record is None:
                return None
            return _serialize_credential(record)

    def get_user_account_by_email(
        self, email: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        cleaned = str(email).strip().lower()
        if not cleaned:
            return None
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential)
                .where(UserCredential.email == cleaned)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if record is None and tenant is not None:
                legacy = session.execute(
                    select(UserCredential)
                    .where(UserCredential.email == cleaned)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if legacy is not None:
                    legacy.tenant_id = tenant
                    record = legacy
            if record is None:
                return None
            return _serialize_credential(record)

    def get_username_for_email(
        self, email: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[str]:
        record = self.get_user_account_by_email(email, tenant_id=tenant_id)
        if not record:
            return None
        return record["username"]

    def list_user_accounts(
        self, *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = select(UserCredential)
            if tenant is None:
                stmt = stmt.where(UserCredential.tenant_id.is_(None))
            else:
                stmt = stmt.where(UserCredential.tenant_id == tenant)
            rows = session.execute(stmt).scalars().all()
            return [_serialize_credential(row) for row in rows]

    def search_user_accounts(
        self, query_text: Optional[str], *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        search_term = str(query_text or "").strip().lower()
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = select(UserCredential)
            if tenant is None:
                stmt = stmt.where(UserCredential.tenant_id.is_(None))
            else:
                stmt = stmt.where(UserCredential.tenant_id == tenant)
            if search_term:
                pattern = f"%{search_term}%"
                stmt = stmt.where(
                    or_(
                        UserCredential.username.ilike(pattern),
                        UserCredential.email.ilike(pattern),
                        UserCredential.name.ilike(pattern),
                    )
                )
            rows = session.execute(stmt).scalars().all()
            return [_serialize_credential(row) for row in rows]

    def update_user_account(
        self,
        username: str,
        *,
        tenant_id: Optional[Any] = None,
        **updates: Any,
    ) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential)
                .where(UserCredential.username == cleaned)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if record is None and tenant is not None:
                legacy = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if legacy is not None:
                    legacy.tenant_id = tenant
                    record = legacy
            if record is None:
                return None

            allowed = {
                "password_hash",
                "email",
                "name",
                "dob",
                "failed_attempts",
                "lockout_until",
                "last_login",
                "user_id",
            }
            for key, value in updates.items():
                if key not in allowed:
                    continue
                if key == "user_id" and value is not None:
                    value = _coerce_uuid(value)
                if key in {"lockout_until", "last_login"} and value is not None:
                    value = _coerce_dt(value)
                setattr(record, key, value)
            session.flush()
            return _serialize_credential(record)

    def delete_user_account(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> bool:
        cleaned = str(username).strip()
        if not cleaned:
            return False
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential)
                .where(UserCredential.username == cleaned)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if record is None and tenant is not None:
                record = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
            if record is None:
                return False
            session.delete(record)
            session.flush()
            return True

    def set_user_password(self, username: str, password_hash: str) -> bool:
        updated = self.update_user_account(username, password_hash=password_hash)
        return updated is not None

    def update_last_login(
        self, username: str, timestamp: Any, *, tenant_id: Optional[Any] = None
    ) -> bool:
        cleaned = str(username).strip()
        if not cleaned:
            return False
        moment = _coerce_dt(timestamp)
        tenant = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential)
                .where(UserCredential.username == cleaned)
                .where(_tenant_filter(UserCredential.tenant_id, tenant))
            ).scalar_one_or_none()
            if credential is None and tenant is not None:
                credential = session.execute(
                    select(UserCredential)
                    .where(UserCredential.username == cleaned)
                    .where(UserCredential.tenant_id.is_(None))
                ).scalar_one_or_none()
                if credential is not None:
                    credential.tenant_id = tenant
            if credential is None:
                return False
            credential.last_login = moment
            session.flush()
            return True

    def set_lockout_state(
        self,
        username: str,
        *,
        failed_attempts: Optional[List[Any]] = None,
        lockout_until: Optional[Any] = None,
    ) -> bool:
        cleaned = str(username).strip()
        if not cleaned:
            return False
        attempts = failed_attempts or []
        normalized_attempts = _normalize_attempts(attempts)
        lockout_dt = _coerce_dt(lockout_until) if lockout_until else None
        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if credential is None:
                return False
            credential.failed_attempts = normalized_attempts
            credential.lockout_until = lockout_dt
            session.flush()
            return True

    def clear_lockout_state(self, username: str) -> bool:
        return self.set_lockout_state(username, failed_attempts=[], lockout_until=None)

    def get_lockout_state(self, username: str) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if credential is None:
                return None
            return {
                "username": credential.username,
                "failed_attempts": list(credential.failed_attempts or []),
                "lockout_until": _dt_to_iso(credential.lockout_until),
            }

    def get_all_lockout_states(self) -> List[Dict[str, Any]]:
        accounts = self.list_user_accounts()
        states: List[Dict[str, Any]] = []
        for account in accounts:
            if account.get("failed_attempts") or account.get("lockout_until"):
                states.append(
                    {
                        "username": account["username"],
                        "failed_attempts": list(account.get("failed_attempts") or []),
                        "lockout_until": account.get("lockout_until"),
                    }
                )
        return states

    # ------------------------------------------------------------------
    # Login attempt helpers

    def record_login_attempt(
        self,
        username: Optional[str],
        timestamp: Any,
        successful: bool,
        reason: Optional[str],
    ) -> None:
        moment = _coerce_dt(timestamp)
        cleaned_username = None if username in (None, "") else str(username)
        trimmed_reason = None if reason in (None, "") else str(reason)
        with self._session_scope() as session:
            credential = None
            if cleaned_username:
                credential = session.execute(
                    select(UserCredential).where(UserCredential.username == cleaned_username)
                ).scalar_one_or_none()
            attempt = UserLoginAttempt(
                credential=credential,
                username=cleaned_username,
                attempted_at=moment,
                successful=bool(successful),
                reason=trimmed_reason,
            )
            session.add(attempt)

    def get_login_attempts(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned or limit <= 0:
            return []
        with self._session_scope() as session:
            stmt = (
                select(UserLoginAttempt)
                .where(UserLoginAttempt.username == cleaned)
                .order_by(UserLoginAttempt.attempted_at.desc(), UserLoginAttempt.id.desc())
                .limit(int(limit))
            )
            rows = session.execute(stmt).scalars().all()
        attempts: List[Dict[str, Any]] = []
        for row in rows:
            attempts.append(
                {
                    "timestamp": _dt_to_iso(row.attempted_at),
                    "successful": bool(row.successful),
                    "reason": row.reason if row.reason not in (None, "") else None,
                }
            )
        return attempts

    def prune_login_attempts(self, username: str, limit: int) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            return
        with self._session_scope() as session:
            if limit <= 0:
                session.execute(
                    delete(UserLoginAttempt).where(UserLoginAttempt.username == cleaned)
                )
                return

            subquery = (
                select(UserLoginAttempt.id)
                .where(UserLoginAttempt.username == cleaned)
                .order_by(UserLoginAttempt.attempted_at.desc(), UserLoginAttempt.id.desc())
                .limit(int(limit))
            )
            session.execute(
                delete(UserLoginAttempt).where(
                    and_(
                        UserLoginAttempt.username == cleaned,
                        ~UserLoginAttempt.id.in_(subquery),
                    )
                )
            )

    # ------------------------------------------------------------------
    # Password reset helpers

    def upsert_password_reset_token(
        self,
        username: str,
        token_hash: str,
        expires_at: Any,
        created_at: Optional[Any] = None,
    ) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            raise ValueError("Username must not be empty when storing reset token")
        expires_dt = _coerce_dt(expires_at) if expires_at is not None else None
        created_dt = (
            _coerce_dt(created_at)
            if created_at is not None
            else _coerce_dt(datetime.now(timezone.utc))
        )
        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if credential is None:
                raise ValueError(f"Unknown username '{cleaned}' for reset token")
            token = session.execute(
                select(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            ).scalar_one_or_none()
            if token is None:
                token = PasswordResetToken(
                    credential=credential,
                    username=cleaned,
                    token_hash=token_hash,
                    expires_at=expires_dt,
                    created_at=created_dt,
                )
                session.add(token)
            else:
                token.token_hash = token_hash
                token.expires_at = expires_dt
                token.created_at = created_dt

    def get_password_reset_token(self, username: str) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        with self._session_scope() as session:
            token = session.execute(
                select(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            ).scalar_one_or_none()
            if token is None:
                return None
            return {
                "username": token.username,
                "token_hash": token.token_hash,
                "expires_at": _dt_to_iso(token.expires_at),
            }

    def delete_password_reset_token(self, username: str) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            return
        with self._session_scope() as session:
            session.execute(
                delete(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            )


def _serialize_credential(record: UserCredential) -> Dict[str, Any]:
    attempts = list(record.failed_attempts or [])
    normalized_attempts = _normalize_attempts(attempts)
    data: Dict[str, Any] = {
        "id": int(record.id),
        "user_id": str(record.user_id) if record.user_id else None,
        "username": record.username,
        "password_hash": record.password_hash,
        "email": record.email,
        "name": record.name,
        "dob": record.dob,
        "last_login": _dt_to_iso(record.last_login),
        "failed_attempts": normalized_attempts,
        "lockout_until": _dt_to_iso(record.lockout_until),
        "created_at": _dt_to_iso(record.created_at),
        "updated_at": _dt_to_iso(record.updated_at),
        "tenant_id": record.tenant_id,
    }
    return data


__all__ = ["AccountStore"]
