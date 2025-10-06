import asyncio

from ATLAS.ATLAS import ATLAS


class _StubService:
    def __init__(self, users):
        self._users = users

    def list_users(self):
        return list(self._users)


def _make_atlas_with_service(users):
    atlas = ATLAS()
    atlas._user_account_service = _StubService(users)
    return atlas


def test_list_user_accounts_is_awaitable():
    atlas = _make_atlas_with_service([
        {"username": "alice", "name": "Alice"},
        {"username": "bob", "name": "Bob"},
    ])

    result = asyncio.run(atlas.list_user_accounts())

    assert result == [
        {"username": "alice", "name": "Alice"},
        {"username": "bob", "name": "Bob"},
    ]

