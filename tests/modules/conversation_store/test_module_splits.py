import contextlib
import uuid
from datetime import datetime, timezone

import inspect

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker

if inspect.isfunction(sessionmaker):
    pytest.skip("SQLAlchemy is not available for module split tests", allow_module_level=True)

from modules.conversation_store.accounts import AccountStore
from modules.conversation_store.conversations import ConversationStore
from modules.conversation_store.graph import GraphStore
from modules.conversation_store.schema import create_schema, resolve_engine
from modules.conversation_store.vectors import VectorStore


TENANT = "tenant-test"


@pytest.fixture
def engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine):
    factory = sessionmaker(bind=engine, future=True)
    return factory


@pytest.fixture
def session_scope(session_factory):
    @contextlib.contextmanager
    def scope():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return scope


@pytest.fixture(autouse=True)
def apply_schema(session_factory):
    create_schema(session_factory)


def test_schema_helpers(session_factory, engine):
    resolved = resolve_engine(session_factory)
    assert resolved is engine

    inspector = sa_inspect(engine)
    tables = set(inspector.get_table_names())
    for name in {"users", "conversations", "messages", "message_vectors"}:
        assert name in tables


def test_account_store_round_trip(session_scope):
    accounts = AccountStore(session_scope)
    created = accounts.create_user_account(
        "tester",
        password_hash="hashed",
        email="tester@example.com",
        tenant_id=TENANT,
    )

    fetched = accounts.get_user_account("tester", tenant_id=TENANT)
    assert fetched == created

    accounts.record_login_attempt("tester", datetime.now(timezone.utc), True, None)
    attempts = accounts.get_login_attempts("tester")
    assert attempts and attempts[0]["successful"] is True


def test_conversation_store_integrates_with_vector_store(session_scope):
    vectors = VectorStore(session_scope)
    conversations = ConversationStore(session_scope, vectors)

    conversation_id = uuid.uuid4()
    conversations.ensure_conversation(conversation_id, tenant_id=TENANT)

    message = conversations.add_message(
        conversation_id,
        tenant_id=TENANT,
        role="assistant",
        content={"text": "hello"},
        vectors=[{"embedding": [0.1, 0.2, 0.3]}],
    )

    stored_vectors = list(
        vectors.query_message_vectors(
            conversation_ids=[conversation_id],
            tenant_id=TENANT,
        )
    )
    assert stored_vectors and stored_vectors[0][0]["id"] == message["id"]


def test_graph_store_upsert_and_query(session_scope):
    graph = GraphStore(session_scope)
    nodes = graph.upsert_graph_nodes(
        tenant_id=TENANT,
        nodes=[{"key": "alpha", "label": "Alpha"}, {"key": "beta", "label": "Beta"}],
    )
    assert {node["key"] for node in nodes} == {"alpha", "beta"}

    edges = graph.upsert_graph_edges(
        tenant_id=TENANT,
        edges=[{"source": "alpha", "target": "beta", "edge_type": "link"}],
    )
    assert edges and edges[0]["edge_type"] == "link"

    result = graph.query_graph(tenant_id=TENANT)
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
