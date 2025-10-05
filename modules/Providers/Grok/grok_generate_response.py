# modules/Providers/Grok/grok_generate_response.py

import inspect
from typing import List, Dict, Union, AsyncIterator

from xai_sdk import Client
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger


class GrokGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_grok_api_key()
        if not self.api_key:
            self.logger.error("Grok API key not found in configuration")
            raise ValueError("Grok API key not found in configuration")

        # Initialize Grok Client
        self.client = Client(api_key=self.api_key)
        self.logger.info("Grok client initialized")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-2",
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generates a response from Grok's model.
        :param messages: List of message dicts containing "role" and "content" keys.
        :param model: The Grok model to use. Defaults to "grok-2".
        :param max_tokens: Maximum tokens for the response.
        :param stream: If True, enables streaming of responses.
        :return: Generated response or an async iterator if streaming.
        """
        self.logger.info(f"Generating response using Grok model {model}")
        try:
            if stream:
                return self._stream_response(messages, model, max_tokens)
            else:
                return await self._get_grok_response(messages, model, max_tokens)
        
        except Exception as e:
            self.logger.error(f"Error generating Grok response: {str(e)}")
            raise

    async def _get_grok_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int
    ) -> str:
        """
        Non-streaming response generation from Grok.
        """
        prompt = self._build_prompt_from_messages(messages)
        self.logger.debug(f"Sending prompt to Grok: {prompt}")
        result = await self.client.sampler.sample(
            prompt,
            max_len=max_tokens,
            model=model,
        )
        response = "".join([token.token_str for token in result])
        self.logger.info(f"Grok response received: {response}")
        return response

    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """
        Streaming response generation from Grok.
        """
        prompt = self._build_prompt_from_messages(messages)
        self.logger.debug(f"Streaming prompt to Grok: {prompt}")
        async for token in self.client.sampler.sample(
            prompt,
            max_len=max_tokens,
            model=model,
        ):
            yield token.token_str

    async def process_streaming_response(self, response: AsyncIterator[str]) -> str:
        """Consume a streaming Grok response and return the combined text.

        Grok currently provides token strings during streaming. Until the
        application supports incremental UI updates for Grok, we collapse the
        stream into a single string so downstream callers can reuse the same
        handling used for non-streaming responses.
        """
        collected_chunks = []
        async for chunk in response:
            if chunk:
                collected_chunks.append(chunk)

        final_response = "".join(collected_chunks)
        if collected_chunks:
            self.logger.debug(
                "Completed Grok streaming response with %d chunks", len(collected_chunks)
            )
        else:
            self.logger.debug("Received empty Grok streaming response")

        return final_response

    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts messages into a prompt string for Grok.
        """
        self.logger.debug(f"Building prompt from messages: {messages}")
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    async def unload_model(self) -> None:
        """Release the underlying Grok client resources safely."""

        client = getattr(self, "client", None)
        if client is None:
            self.logger.debug("Grok client already released; nothing to unload.")
            return

        async_close = getattr(client, "aclose", None)
        sync_close = getattr(client, "close", None)

        try:
            if callable(async_close):
                maybe_coro = async_close()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            elif callable(sync_close):
                maybe_result = sync_close()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to close Grok client cleanly: %s", exc, exc_info=True)
        finally:
            self.client = None
            self.logger.info("Grok client unloaded.")
