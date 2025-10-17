# modules/Providers/HuggingFace/HF_gen_response.py

import asyncio
import os
import shutil
from typing import List, Dict, Union, AsyncIterator, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from huggingface_hub import HfApi, hf_hub_download

from .config.base_config import BaseConfig
from .config.nvme_config import NVMeConfig
from .utils.cache_manager import CacheManager
from .components.huggingface_model_manager import HuggingFaceModelManager
from .components.response_generator import ResponseGenerator
from .utils.logger import setup_logger


class HuggingFaceGenerator:
    def __init__(self, config_manager):
        self.logger = setup_logger()
        self.base_config = BaseConfig(config_manager)
        self.nvme_config = NVMeConfig()
        cache_file = os.path.join(self.base_config.model_cache_dir, "response_cache.json")
        self.cache_manager = CacheManager(cache_file)
        self.model_manager = HuggingFaceModelManager(
            self.base_config,
            self.nvme_config,
            self.cache_manager
        )
        self.response_generator = ResponseGenerator(
            self.model_manager,
            self.cache_manager
        )
        self.installed_models_file = os.path.join(self.base_config.model_cache_dir, "installed_models.json")

    async def load_model(self, model_name: str, force_download: bool = False):
        await self.model_manager.load_model(model_name, force_download)

    def unload_model(self):
        self.model_manager.unload_model()

    def get_installed_models(self) -> List[str]:
        return self.model_manager.get_installed_models()
    
    def clear_model_cache(self):
        """
        Clears all files and directories within the model_cache directory.
        """
        model_cache_dir = self.base_config.model_cache_dir
        self.logger.info(f"Clearing all cache files in {model_cache_dir}")
        for filename in os.listdir(model_cache_dir):
            file_path = os.path.join(model_cache_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        self.logger.info("Model cache cleared successfully.")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = True,
        *,
        skill_signature: Optional[Any] = None,
    ) -> Union[str, AsyncIterator[str]]:
        return await self.response_generator.generate_response(
            messages,
            model,
            stream,
            skill_signature=skill_signature,
        )

    async def process_streaming_response(self, response: AsyncIterator[str]) -> str:
        """
        Processes a streaming response from the HuggingFace model.

        This method consumes an asynchronous iterator that yields pieces of the response
        (such as tokens or text chunks) and concatenates them into a single string.

        Args:
            response (AsyncIterator[str]): An asynchronous iterator that yields
                                           parts of the streamed response as strings.

        Returns:
            str: The complete, assembled response as a single string.
        """
        chunks = []
        async for chunk in response:
            # Each chunk is expected to be a string representing a portion of the response
            chunks.append(chunk)
        # Join all chunks to form the final response
        return "".join(chunks)

    def update_model_settings(self, new_settings: Dict):
        self.base_config.update_model_settings(new_settings)

    # NVMe Configuration Methods
    def set_nvme_offloading(self, enable: bool):
        self.nvme_config.enable_nvme_offloading(enable)

    def set_nvme_path(self, path: str):
        self.nvme_config.set_nvme_path(path)

    def set_nvme_buffer_count_param(self, count: int):
        self.nvme_config.set_nvme_buffer_count_param(count)

    def set_nvme_buffer_count_optimizer(self, count: int):
        self.nvme_config.set_nvme_buffer_count_optimizer(count)

    def set_nvme_block_size(self, size: int):
        self.nvme_config.set_nvme_block_size(size)

    def set_nvme_queue_depth(self, depth: int):
        self.nvme_config.set_nvme_queue_depth(depth)

    # Additional Feature Toggle Methods
    def set_quantization(self, quantization: str):
        self.base_config.set_quantization(quantization)

    def set_gradient_checkpointing(self, enable: bool):
        self.base_config.set_gradient_checkpointing(enable)

    def set_lora(self, enable: bool):
        self.base_config.set_lora(enable)

    def set_flash_attention(self, enable: bool):
        self.base_config.set_flash_attention(enable)

    def set_pruning(self, enable: bool):
        self.base_config.set_pruning(enable)

    def set_memory_mapping(self, enable: bool):
        self.base_config.set_memory_mapping(enable)

    def set_bfloat16(self, enable: bool):
        self.base_config.set_bfloat16(enable)

    def set_torch_compile(self, enable: bool):
        self.base_config.set_torch_compile(enable)

    

# Helper Functions

def setup_huggingface_generator(config_manager):
    return HuggingFaceGenerator(config_manager)


async def search_models(
    generator: HuggingFaceGenerator,
    search_query: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    """Search available Hugging Face models using the configured API token.

    Args:
        generator: Active ``HuggingFaceGenerator`` instance.
        search_query: Query string passed to the Hugging Face hub.
        filters: Optional dictionary of additional search filters.
        limit: Optional maximum number of results to return.

    Returns:
        A list of dictionaries describing the models.
    """

    filters = filters or {}
    token = None
    if hasattr(generator.base_config.config_manager, "get_huggingface_api_key"):
        token = generator.base_config.config_manager.get_huggingface_api_key()
    api = HfApi(token=token) if token else HfApi()

    def _fetch_models():
        return list(api.list_models(search=search_query, **filters))

    models = await asyncio.to_thread(_fetch_models)
    if limit is not None:
        models = models[:limit]

    serialised: List[Dict[str, Any]] = []
    for model in models:
        serialised.append(
            {
                "id": getattr(model, "modelId", ""),
                "tags": list(getattr(model, "tags", []) or []),
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
            }
        )
    return serialised


async def download_model(
    generator: HuggingFaceGenerator,
    model_id: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Download a model from Hugging Face into the local cache directory."""

    token = None
    if hasattr(generator.base_config.config_manager, "get_huggingface_api_key"):
        token = generator.base_config.config_manager.get_huggingface_api_key()
    api = HfApi(token=token) if token else HfApi()

    def _list_files() -> List[str]:
        return list(api.list_repo_files(repo_id=model_id))

    repo_files = await asyncio.to_thread(_list_files)

    async def _download_file(filename: str) -> str:
        return await asyncio.to_thread(
            hf_hub_download,
            repo_id=model_id,
            filename=filename,
            cache_dir=generator.base_config.model_cache_dir,
            force_download=force,
        )

    semaphore = asyncio.Semaphore(5)

    async def _bounded_download(filename: str) -> str:
        async with semaphore:
            return await _download_file(filename)

    tasks = [_bounded_download(file_name) for file_name in repo_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    downloaded_files: List[str] = []
    failures = []
    for file_name, result in zip(repo_files, results):
        if isinstance(result, Exception):
            failures.append((file_name, result))
        else:
            downloaded_files.append(result)

    if failures:
        failure_messages = ", ".join(f"{name}: {exc}" for name, exc in failures)
        raise RuntimeError(
            f"Failed to download one or more files for {model_id}: {failure_messages}"
        ) from failures[0][1]

    model_manager = getattr(generator, "model_manager", None)
    if model_manager is not None:
        if getattr(model_manager, "installed_models", None) is not None and model_id not in model_manager.installed_models:
            model_manager.installed_models.append(model_id)
        save_installed = getattr(model_manager, "_save_installed_models", None)
        if callable(save_installed):
            await asyncio.to_thread(save_installed, model_manager.installed_models)

    return {"model_id": model_id, "files": downloaded_files}


def update_model_settings(generator: HuggingFaceGenerator, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update the persisted Hugging Face model settings."""

    generator.update_model_settings(settings)
    return generator.base_config.model_settings.copy()


def clear_cache(generator: HuggingFaceGenerator) -> None:
    """Clear cached Hugging Face model artifacts."""

    generator.clear_model_cache()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(
    config_manager,
    messages: List[Dict[str, str]],
    model: str,
    stream: bool = True,
    *,
    skill_signature: Optional[Any] = None,
) -> Union[str, AsyncIterator[str]]:
    generator = setup_huggingface_generator(config_manager)
    await generator.load_model(model)
    return await generator.generate_response(
        messages,
        model,
        stream,
        skill_signature=skill_signature,
    )


async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response
    return "".join([chunk async for chunk in response])


def generate_response_sync(
    config_manager,
    messages: List[Dict[str, str]],
    model: str,
    stream: bool = False,
    *,
    skill_signature: Optional[Any] = None,
) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        generate_response(
            config_manager,
            messages,
            model,
            stream,
            skill_signature=skill_signature,
        )
    )
    if stream:
        return loop.run_until_complete(process_response(response))
    return response

