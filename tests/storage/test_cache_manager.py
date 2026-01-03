import pytest

from modules.Providers.HuggingFace.utils.cache_manager import CacheManager


def test_generate_cache_key_deterministic(tmp_path):
    cache_manager = CacheManager(str(tmp_path / "cache.json"))

    messages_a = [
        {
            "role": "user",
            "content": "Hello",
            "metadata": {"alpha": 1, "beta": 2},
        }
    ]
    messages_b = [
        {
            "content": "Hello",
            "metadata": {"beta": 2, "alpha": 1},
            "role": "user",
        }
    ]
    settings_a = {"temperature": 0.7, "max_new_tokens": 100}
    settings_b = {"max_new_tokens": 100, "temperature": 0.7}

    key_a = cache_manager.generate_cache_key(messages_a, "test-model", settings_a)
    key_b = cache_manager.generate_cache_key(messages_b, "test-model", settings_b)

    assert key_a == key_b


@pytest.mark.parametrize(
    "messages, other_messages",
    [
        ([{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Hello"}]),
        ([{"role": "user", "content": "Hi", "metadata": {"a": 1}}], [{"role": "user", "content": "Hi", "metadata": {"a": 2}}]),
    ],
)
def test_generate_cache_key_changes_with_payload(tmp_path, messages, other_messages):
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    settings = {"temperature": 0.7}

    key_a = cache_manager.generate_cache_key(messages, "test-model", settings)
    key_b = cache_manager.generate_cache_key(other_messages, "test-model", settings)

    assert key_a != key_b
