# ATLAS/provider_manager.py

from typing import List, Dict, Union, AsyncIterator
import time
import traceback
from ATLAS.model_manager import ModelManager
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from modules.Providers.HuggingFace.HF_gen_response import HuggingFaceGenerator
from modules.Providers.Grok.grok_generate_response import GrokGenerator

class ProviderManager:
    """
    Manages interactions with different LLM providers, including loading models,
    generating responses, and switching providers.
    """

    AVAILABLE_PROVIDERS = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the ProviderManager with a given ConfigManager.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.model_manager = ModelManager(self.config_manager)
        self.current_llm_provider = self.config_manager.get_default_provider()
        self.current_background_provider = self.config_manager.get_default_provider()
        self.current_model = None
        self.generate_response_func = None
        self.process_streaming_response_func = None
        self.huggingface_generator = HuggingFaceGenerator(self.config_manager)
        self.grok_generator = None
        self.current_functions = None
        self.providers = {}

    @classmethod
    async def create(cls, config_manager: ConfigManager):
        """
        Asynchronous factory method to create a ProviderManager instance.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager.

        Returns:
            ProviderManager: An initialized instance of ProviderManager.
        """
        self = cls(config_manager)
        await self.switch_llm_provider(self.current_llm_provider)
        return self

    @classmethod
    def providers(cls) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return cls.AVAILABLE_PROVIDERS

    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return self.__class__.providers()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def switch_llm_provider(self, llm_provider):
        """
        Switch the current LLM provider to the specified provider.

        Args:
            llm_provider (str): The name of the LLM provider to switch to.
        """
        provider_config = {
            "OpenAI": {
                "module": "modules.Providers.OpenAI.OA_gen_response",
                "default_model": "gpt-4o",
            },
            "Mistral": {
                "module": "modules.Providers.Mistral.Mistral_gen_response",
                "default_model": "mistral-large-latest",
            },
            "Google": {
                "module": "modules.Providers.Google.GG_gen_response",
                "default_model": "gemini-1.5-pro-latest",
            },
            "HuggingFace": {
                "module": "modules.Providers.HuggingFace.HF_gen_response",
                "default_model": None,
            },
            "Anthropic": {
                "module": "modules.Providers.Anthropic.Anthropic_gen_response",
                "default_model": "claude-3-sonnet-20240229",
            },
            "Grok": {
                "module": "modules.Providers.Grok.grok_generate_response",
                "default_model": "grok-2",
            }
        }

        if llm_provider not in provider_config:
            self.logger.warning(f"Provider {llm_provider} is not implemented. Reverting to default provider OpenAI.")
            llm_provider = "OpenAI"

        config = provider_config[llm_provider]
        module = __import__(config["module"], fromlist=["generate_response", "process_response"])

        self.generate_response_func = getattr(module, "generate_response")
        self.process_streaming_response_func = getattr(module, "process_response", None)

        self.current_llm_provider = llm_provider
        self.providers[llm_provider] = module

        if llm_provider == "Grok":
            self.grok_generator = GrokGenerator(self.config_manager)
            self.current_model = None
        elif llm_provider == "HuggingFace":
            self.huggingface_generator = HuggingFaceGenerator(self.config_manager)
            self.current_model = None
        else:
            self.grok_generator = None
            await self.set_model(config["default_model"])

        self.logger.info(f"Switched to LLM provider: {self.current_llm_provider}")
        if self.current_model:
            self.logger.info(f"Current model set to: {self.current_model}")

    async def set_current_provider(self, provider: str):
        """
        Set the current provider to the specified provider.

        Args:
            provider (str): The name of the provider to set.
        """
        await self.switch_llm_provider(provider)
        self.logger.info(f"Current provider set to {self.current_llm_provider}")
        if self.current_model:
            self.logger.info(f"Current model set to: {self.current_model}")

    def get_current_model(self):
        """
        Get the current model being used.

        Returns:
            str or None: The name of the current model, or None if not set.
        """
        if self.current_llm_provider == "HuggingFace":
            return self.huggingface_generator.current_model if self.huggingface_generator else None
        return self.current_model

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
        provider: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        stream: bool = True,
        current_persona=None,
        functions=None,
        llm_call_type: str = None
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response using the specified provider and model, or the current ones if not specified.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str, optional): The model to use. If None, uses the current model.
            provider (str, optional): The provider to use. If None, uses the current provider.
            max_tokens (int, optional): Maximum number of tokens. Defaults to 4000.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            stream (bool, optional): Whether to stream the response. Defaults to True.
            current_persona (optional): The current persona.
            functions (optional): Functions to use.
            llm_call_type (str, optional): The type of LLM call.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response or a stream of tokens.
        """
        if not provider:
            provider = self.current_llm_provider
        if not model:
            model = self.get_current_model()

        if provider != self.current_llm_provider:
            await self.switch_llm_provider(provider)

        if model != self.current_model:
            await self.set_model(model)

        if functions is None:
            functions = self.current_functions

        start_time = time.time()
        self.logger.info(f"Starting API call to {provider} with model {model} for {llm_call_type}")

        try:
            if provider == "HuggingFace":
                response = await self.huggingface_generator.generate_response(messages, model, stream)
            elif provider == "Grok":
                response = await self.grok_generator.generate_response(messages, model, stream)
            else:
                provider_module = self.providers.get(provider)
                if not provider_module:
                    raise ValueError(f"Provider {provider} not found")
                response = await self.generate_response_func(
                    self.config_manager,
                    messages,
                    model,
                    max_tokens,
                    temperature,
                    stream,
                    current_persona=current_persona,
                    functions=functions
                )

            self.logger.info(f"API call completed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            self.logger.error(f"Error during API call to {provider}: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
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
        fallback_provider = self.providers.get(fallback_config['provider'])
        
        if fallback_provider is None:
            self.logger.error(f"Fallback provider {fallback_config['provider']} not found.")
            raise ValueError("Fallback provider configuration is missing")
        
        self.logger.info(f"Using fallback provider {fallback_config['provider']} for {llm_call_type}")
        
        return await fallback_provider.generate_response(
            messages=messages,
            model=fallback_config['model'],
            max_tokens=fallback_config['max_tokens'],
            temperature=fallback_config['temperature'],
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
            if self.current_llm_provider == "HuggingFace":
                return await self.huggingface_generator.process_response(response)
            elif self.process_streaming_response_func is None:
                raise ValueError(f"Streaming response processing not implemented for {self.current_llm_provider}")
            else:
                return await self.process_streaming_response_func(response)
        except Exception as e:
            self.logger.error(f"Error processing streaming response: {str(e)}")
            raise

    async def set_model(self, model: str):
        """
        Set the current model and load it if necessary.

        Args:
            model (str): The model to set.
        """
        self.current_model = model
        self.model_manager.set_model(model, self.current_llm_provider)
        if self.current_llm_provider == "HuggingFace":
            await self.huggingface_generator.load_model(model)
        self.logger.info(f"Model set to {model}")

    def switch_background_provider(self, background_provider):
        """
        Switch the background provider.

        Args:
            background_provider (str): The name of the background provider.
        """
        self.current_background_provider = background_provider
        self.logger.info(f"Switched background provider to: {self.current_background_provider}")

    def get_current_provider(self):
        """
        Get the current LLM provider.

        Returns:
            str: The name of the current LLM provider.
        """
        return self.current_llm_provider

    def get_current_background_provider(self):
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
        return self.model_manager.get_available_models(self.current_llm_provider)[self.current_llm_provider]

    def set_conversation_manager(self, conversation_manager):
        """
        Set the conversation manager.

        Args:
            conversation_manager: The conversation manager instance.
        """
        self.conversation_manager = conversation_manager

    def set_current_conversation_id(self, conversation_id):
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