# modules/Providers/HuggingFace/components/response_generator.py

import asyncio
from typing import List, Dict, Union, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential

from .huggingface_model_manager import HuggingFaceModelManager
from ..utils.cache_manager import CacheManager
from ..utils.logger import setup_logger


class ResponseGenerator:
    def __init__(self, model_manager: HuggingFaceModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.logger = setup_logger()
        self.model_settings = self.model_manager.base_config.model_settings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = True
    ) -> Union[str, AsyncIterator[str]]:
        try:
            cache_key = self.cache_manager.generate_cache_key(messages, model, self.model_settings)
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.logger.info("Returning cached response")
                return cached_response

            if self.model_manager.current_model != model:
                await self.model_manager.load_model(model)

            response = await self._generate_local_response(messages, model, stream)

            if not stream:
                self.cache_manager.set(cache_key, response)

            return response
        except Exception as e:
            self.logger.error(f"Error in HuggingFace API call: {str(e)}")
            raise

    async def _generate_local_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool
    ) -> Union[str, AsyncIterator[str]]:
        prompt = self._convert_messages_to_prompt(messages)

        if stream:
            return self._stream_response(await self._generate_text(prompt))
        else:
            return await self._generate_text(prompt)

    async def _generate_text(self, prompt: str) -> str:
        generation_kwargs = self._get_generation_config()
        generation_kwargs.pop('prompt', None)  # Ensure 'prompt' isn't duplicated
        output = await asyncio.to_thread(self.model_manager.pipeline, prompt, **generation_kwargs)
        return output[0]['generated_text']

    async def _stream_response(self, text: str) -> AsyncIterator[str]:
        for token in text.split():
            yield token + " "
            await asyncio.sleep(0)  # Yield control to the event loop

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt.strip()

    def _get_generation_config(self) -> Dict:
        config = {
            "max_new_tokens": self.model_settings.get('max_tokens', 100),
            "temperature": self.model_settings.get('temperature', 0.7),
            "top_p": self.model_settings.get('top_p', 1.0),
            "top_k": self.model_settings.get('top_k', 50),
            "repetition_penalty": self.model_settings.get('repetition_penalty', 1.0),
            "length_penalty": self.model_settings.get('length_penalty', 1.0),
            "early_stopping": self.model_settings.get('early_stopping', False),
            "do_sample": self.model_settings.get('do_sample', False),
        }
        return config
