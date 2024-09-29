# ATLAS/provider_manager.py

from typing import List, Dict, Union, AsyncIterator
import asyncio
import time
import traceback
from ATLAS.model_manager import ModelManager
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from modules.Providers.HuggingFace.HF_gen_response import HuggingFaceGenerator
from modules.Providers.Grok.grok_generate_response import GrokGenerator

# Import other necessary provider generators
from modules.Providers.OpenAI.OA_gen_response import generate_response as openai_generate_response
from modules.Providers.Mistral.Mistral_gen_response import generate_response as mistral_generate_response
from modules.Providers.Google.GG_gen_response import generate_response as google_generate_response
from modules.Providers.Anthropic.Anthropic_gen_response import generate_response as anthropic_generate_response

class ProviderManager:
    """
    Manages interactions with different LLM providers, ensuring only one instance exists.
    """

    AVAILABLE_PROVIDERS = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]

    _instance = None  # Class variable to hold the singleton instance
    _lock = asyncio.Lock()  # Lock to ensure thread-safe instantiation

    def __init__(self, config_manager: ConfigManager):
        """
        Private constructor to prevent direct instantiation.
        """
        if ProviderManager._instance is not None:
            raise Exception("This class is a singleton! Use ProviderManager.create() to get the instance.")
        
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.model_manager = ModelManager(self.config_manager)
        self.current_provider_info = {
            "name": self.config_manager.get_default_provider(),
            "model": None,
            "generate_func": None,
            "stream_func": None
        }
        self.current_background_provider = self.config_manager.get_default_provider()
        self.huggingface_generator = None
        self.grok_generator = None
        self.current_functions = None
        self.providers = {}
        self.chat_session = None
        self.conversation_manager = None
        self.current_conversation_id = None

    @classmethod
    async def create(cls, config_manager: ConfigManager):
        """
        Asynchronous factory method to create or retrieve the singleton instance.
        
        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        
        Returns:
            ProviderManager: The singleton instance of ProviderManager.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config_manager)
                await cls._instance.initialize_all_providers()
            return cls._instance

    async def initialize_all_providers(self):
        """
        Initializes the provider managers and sets the default provider.
        """
        await self.set_current_provider(self.current_provider_info["name"])

    @classmethod
    def providers(cls) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return cls.AVAILABLE_PROVIDERS

    def get_default_model_for_provider(self, provider: str) -> str:
        """
        Get the default model for a specific provider.

        Args:
            provider (str): The name of the provider.

        Returns:
            str: The default model name for the specified provider.
        """
        provider_default_models = {
            "OpenAI": "gpt-4o",
            "Mistral": "mistral-large-latest",
            "Google": "gemini-1.5-pro-latest",
            "HuggingFace": "default_hf_model",
            "Anthropic": "claude-3-sonnet-20240229",
            "Grok": "grok-2"
        }

        default_model = provider_default_models.get(provider)
        if default_model is None:
            self.logger.warning(f"No default model found for provider {provider}. Using fallback model.")
            default_model = "gpt-4o"  # Fallback to a known model

        return default_model

    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return self.__class__.providers()

    async def set_current_provider(self, provider: str):
        """
        Set the current provider to the specified provider.

        Args:
            provider (str): The name of the provider to set.
        """
        if provider != self.current_provider_info["name"]:
            await self.switch_llm_provider(provider)
        else:
            self.logger.info(f"Provider {provider} is already set. No action taken.")

    async def switch_llm_provider(self, llm_provider: str):
        """
        Switches the current LLM provider to the specified provider.

        Args:
            llm_provider (str): The name of the LLM provider to switch to.
        """
        if llm_provider not in self.AVAILABLE_PROVIDERS:
            self.logger.warning(f"Provider {llm_provider} is not implemented. Reverting to default provider OpenAI.")
            llm_provider = "OpenAI"

        self.logger.info(f"Switching to provider: {llm_provider}")

        try:
            provider_settings = {
                "OpenAI": (openai_generate_response, None),
                "Mistral": (mistral_generate_response, None),
                "Google": (google_generate_response, None),
                "HuggingFace": (None, None),  # Will be set separately
                "Anthropic": (anthropic_generate_response, None),
                "Grok": (None, None)  # Will be set separately
            }

            generate_func, stream_func = provider_settings[llm_provider]

            if llm_provider == "HuggingFace":
                self.huggingface_generator = HuggingFaceGenerator(self.config_manager)
                generate_func = self.huggingface_generator.generate_response
                stream_func = self.huggingface_generator.process_response
            elif llm_provider == "Grok":
                self.grok_generator = GrokGenerator(self.config_manager)
                generate_func = self.grok_generator.generate_response
                stream_func = self.grok_generator.process_streaming_response

            default_model = self.get_default_model_for_provider(llm_provider)
            
            self.current_provider_info = {
                "name": llm_provider,
                "model": default_model,
                "generate_func": generate_func,
                "stream_func": stream_func
            }

            self.providers[llm_provider] = generate_func
            
            if llm_provider == "HuggingFace":
                await self.huggingface_generator.load_model(default_model)

            await self.set_model(default_model)

            self.logger.info(f"Switched to LLM provider: {llm_provider}")
            self.logger.info(f"Current model set to: {default_model}")

        except Exception as e:
            self.logger.error(f"Failed to switch to provider {llm_provider}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str or None: The name of the current model, or None if not set.
        """
        return self.current_provider_info["model"]

    def set_current_functions(self, functions):
        """
        Set the current functions for the provider.

        Args:
            functions: The functions to set.
        """
        self.current_functions = functions
        self.logger.debug(f"Updated current functions: {self.current_functions}")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        stream: bool = True,
        current_persona=None,
        functions=None,
        llm_call_type: str = None
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response using the current provider and model.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str, optional): The model to use. If None, uses the current model.
            max_tokens (int, optional): Maximum number of tokens. Defaults to 4000.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            stream (bool, optional): Whether to stream the response. Defaults to True.
            current_persona (optional): The current persona.
            functions (optional): Functions to use.
            llm_call_type (str, optional): The type of LLM call.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response or a stream of tokens.
        """
        if model and model != self.current_provider_info["model"]:
            await self.set_model(model)

        if functions is None:
            functions = self.current_functions

        start_time = time.time()
        self.logger.info(f"Starting API call to {self.current_provider_info['name']} with model {self.current_provider_info['model']} for {llm_call_type}")

        try:
            response = await self.current_provider_info["generate_func"](
                self.config_manager,
                messages,
                model=self.current_provider_info["model"],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                current_persona=current_persona,
                functions=functions
            )

            self.logger.info(f"API call completed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            self.logger.error(f"Error during API call to {self.current_provider_info['name']}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _use_fallback(self, messages: List[Dict[str, str]], llm_call_type: str, **kwargs) -> str:  
        """
        Use a fallback provider to generate a response.
        This method handles cases where the primary provider fails.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            llm_call_type (str): The type of LLM call.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated response from the fallback provider.
        """
        fallback_config = self.config_manager.get_llm_config('default')
        fallback_provider = fallback_config.get('provider')

        if not fallback_provider or fallback_provider not in self.providers:
            self.logger.error(f"Fallback provider {fallback_provider} is not configured or not available.")
            raise ValueError("Fallback provider configuration is missing or invalid.")

        fallback_function = self.providers.get(fallback_provider)
        if not fallback_function:
            self.logger.error(f"Fallback provider {fallback_provider} function not found.")
            raise ValueError("Fallback provider function is missing.")

        self.logger.info(f"Using fallback provider {fallback_provider} for {llm_call_type}")

        return await fallback_function(
            self.config_manager,
            messages=messages,
            model=fallback_config.get('model'),
            max_tokens=fallback_config.get('max_tokens', 4000),
            temperature=fallback_config.get('temperature', 0.0),
            stream=fallback_config.get('stream', True),
            current_persona=fallback_config.get('current_persona'),
            functions=fallback_config.get('functions'),
            llm_call_type=llm_call_type,
            **kwargs
        )

    async def process_streaming_response(self, response: AsyncIterator[Dict]) -> str:
        """
        Process a streaming response from the LLM provider.

        Args:
            response (AsyncIterator[Dict]): The streaming response.

        Returns:
            str: The full response.
        """
        try:
            if self.current_provider_info["stream_func"]:
                return await self.current_provider_info["stream_func"](response)
            else:
                raise ValueError(f"Streaming response processing not implemented for {self.current_provider_info['name']}")
        except Exception as e:
            self.logger.error(f"Error processing streaming response: {str(e)}")
            raise

    async def set_model(self, model: str):
        """
        Set the current model and load it if necessary.

        Args:
            model (str): The model to set.
        """
        if model != self.current_provider_info["model"]:
            self.current_provider_info["model"] = model
            self.model_manager.set_model(model, self.current_provider_info["name"])
            if self.current_provider_info["name"] == "HuggingFace" and self.huggingface_generator:
                await self.huggingface_generator.load_model(model)
            self.logger.info(f"Model set to {model}")

    def switch_background_provider(self, background_provider: str):
        """
        Switch the background provider.

        Args:
            background_provider (str): The name of the background provider.
        """
        if background_provider not in self.AVAILABLE_PROVIDERS:
            self.logger.warning(f"Background provider {background_provider} is not available.")
            return

        self.current_background_provider = background_provider
        self.logger.info(f"Switched background provider to: {self.current_background_provider}")

    def get_current_provider(self) -> str:
        """
        Get the current LLM provider.

        Returns:
            str: The name of the current LLM provider.
        """
        return self.current_provider_info["name"]

    def get_current_background_provider(self) -> str:
        """
        Get the current background provider.

        Returns:
            str: The name of the current background provider.
        """
        return self.current_background_provider

    async def get_available_models(self) -> List[str]:
        """
        Get available models for the current provider.

        Returns:
            List[str]: A list of available model names.
        """
        return self.model_manager.get_available_models(self.current_provider_info["name"]).get(self.current_provider_info["name"], [])

    def set_conversation_manager(self, conversation_manager):
        """
        Set the conversation manager.

        Args:
            conversation_manager: The conversation manager instance.
        """
        self.conversation_manager = conversation_manager

    def set_current_conversation_id(self, conversation_id: str):
        """
        Set the current conversation ID.

        Args:
            conversation_id (str): The conversation ID.
        """
        self.current_conversation_id = conversation_id

    async def close(self):
        """
        Perform cleanup operations, such as unloading models.
        """
        if self.huggingface_generator:
            self.huggingface_generator.unload_model()
        if self.grok_generator:
            await self.grok_generator.unload_model()
        # Add any additional cleanup here
        self.logger.info("ProviderManager closed and models unloaded.")