from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import Base
from modules.conversation_store.models import Conversation, Session as ConversationSession, User
from modules.task_store import TaskStatus
from modules.task_store.repository import TaskStoreRepository
from modules.task_store.service import TaskService


def test_task_repository_contract(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repository = TaskStoreRepository(factory)
    repository.create_schema()

    with factory() as session:
        user = User(external_id="contract-user", display_name="Contract User")
        store_session = ConversationSession(user=user)
        conversation = Conversation(session=store_session, tenant_id="tenant-contract")
        session.add_all([user, store_session, conversation])
        session.commit()
        conversation_id = conversation.id

    record = repository.create_task(
        "Contract Task",
        tenant_id="tenant-contract",
        conversation_id=conversation_id,
    )

    other_repo = TaskStoreRepository(factory)
    tasks = other_repo.list_tasks(tenant_id="tenant-contract")
    assert tasks and tasks[0]["id"] == record["id"]

    engine.dispose()


def test_task_service_contract(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    repository = TaskStoreRepository(factory)
    repository.create_schema()
    service = TaskService(repository, event_emitter=lambda *_args, **_kwargs: None)

    with factory() as session:
        user = User(external_id="service-user", display_name="Service User")
        store_session = ConversationSession(user=user)
        conversation = Conversation(session=store_session, tenant_id="tenant-service")
        session.add_all([user, store_session, conversation])
        session.commit()
        conversation_id = conversation.id

    record = service.create_task(
        "Lifecycle",
        tenant_id="tenant-service",
        conversation_id=conversation_id,
    )

    ready = service.transition_task(
        record["id"], tenant_id="tenant-service", target_status=TaskStatus.READY
    )
    assert ready["status"] == TaskStatus.READY.value

    engine.dispose()
