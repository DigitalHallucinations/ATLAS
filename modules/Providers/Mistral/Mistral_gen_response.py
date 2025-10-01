# modules/Providers/Mistral/Mistral_gen_response.py

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Union, AsyncIterator, Any, Tuple
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
import asyncio
import threading

class MistralGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_mistral_api_key()
        if not self.api_key:
            self.logger.error("Mistral API key not found in configuration")
            raise ValueError("Mistral API key not found in configuration")
        self.client = Mistral(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, messages: List[Dict[str, str]], model: str = "mistral-large-latest", max_tokens: int = 4096, temperature: float = 0.0, stream: bool = True, current_persona=None, functions=None) -> Union[str, AsyncIterator[str]]:
        try:
            mistral_messages = self.convert_messages_to_mistral_format(messages)
            self.logger.info(f"Generating response with Mistral AI using model: {model}")
            if stream:
                response_stream = await asyncio.to_thread(
                    self.client.chat.stream,
                    model=model,
                    messages=mistral_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return self.process_streaming_response(response_stream)

            response = await asyncio.to_thread(
                self.client.chat.complete,
                model=model,
                messages=mistral_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Mistral.APIError as e:
            self.logger.error(f"Mistral API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error with Mistral: {str(e)}")
            raise

    def convert_messages_to_mistral_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        mistral_messages = []
        for message in messages:
            mistral_messages.append({
                "role": message['role'],
                "content": message['content']
            })
        return mistral_messages

    async def process_streaming_response(self, response) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
        DONE = object()

        def iterate_stream():
            try:
                for chunk in response:
                    data = getattr(chunk, "data", None)
                    if not data or not getattr(data, "choices", None):
                        continue
                    choice = data.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("chunk", content)
                        )
                    if getattr(choice, "finish_reason", None):
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # pragma: no cover - defensive
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", DONE))

        threading.Thread(target=iterate_stream, daemon=True).start()

        while True:
            kind, payload = await queue.get()
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                break

    async def process_response(self, response: Union[str, AsyncIterator[str]]) -> str:
        if isinstance(response, str):
            return response
        else:
            full_response = ""
            async for chunk in response:
                full_response += chunk
            return full_response

def setup_mistral_generator(config_manager: ConfigManager):
    return MistralGenerator(config_manager)

async def generate_response(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "mistral-large-latest", max_tokens: int = 4096, temperature: float = 0.0, stream: bool = True, current_persona=None, functions=None) -> Union[str, AsyncIterator[str]]:
    generator = setup_mistral_generator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream, current_persona, functions)

async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response

    chunks: List[str] = []
    async for chunk in response:
        chunks.append(chunk)
    return "".join(chunks)

def generate_response_sync(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "mistral-large-latest", stream: bool = False) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        generate_response(
            config_manager,
            messages,
            model=model,
            stream=stream,
        )
    )
    if stream:
        return loop.run_until_complete(process_response(response))
    return response
