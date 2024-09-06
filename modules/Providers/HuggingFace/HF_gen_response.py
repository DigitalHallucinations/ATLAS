# modules/Providers/HuggingFace/HF_gen_response.py

import os
import json
import torch
import shutil
import hashlib
import asyncio
import traceback
from datasets import Dataset
from functools import lru_cache
from modules.config import ConfigManager
from typing import List, Dict, Union, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient, HfApi, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig


class HuggingFaceGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.logger
        self.api_key = self.config_manager.get_huggingface_api_key()
        if not self.api_key:
            self.logger.error("HuggingFace API key not found in configuration")
            raise ValueError("HuggingFace API key not found in configuration")
        self.client = InferenceClient(token=self.api_key)
        self.model = None
        self.tokenizer = None
        self.current_model = None
        self.pipeline = None
        self.model_cache_dir = self.config_manager.get_model_cache_dir()
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.response_cache = {}
        self.model_settings = {
            'temperature': 0.7,
            'top_p': 1.0,
            'max_tokens': 100,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0
        }
        self.quantization = None
        self.installed_models_file = os.path.join(self.model_cache_dir, "installed_models.json")
        self.installed_models = self._load_installed_models()

    def get_device_map(self, model, max_gpu_memory):
        """
        Create a custom device map for a model based on available GPU memory.

        Args:
        - model: The model to analyze.
        - max_gpu_memory: The maximum GPU memory available (in bytes).

        Returns:
        - device_map: A dictionary mapping layers to 'cpu' or 'cuda:0'.
        """
        device_map = {}
        current_gpu_memory = 0

        for name, param in model.named_parameters():
            param_memory = param.numel() * param.element_size()  # Calculate memory usage of the layer
            if current_gpu_memory + param_memory > max_gpu_memory:
                device_map[name] = 'cpu'
            else:
                device_map[name] = 'cuda:0'
                current_gpu_memory += param_memory

        return device_map

    def load_model_with_custom_device_map(self, model_name):
        # Get available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Generate device map
        device_map = self.get_device_map(model, max_gpu_memory=gpu_memory)

        # Load model with custom device map
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)

        return model, tokenizer

    def _load_installed_models(self) -> List[str]:
        if not os.path.exists(self.installed_models_file):
            self.logger.info(f"installed_models.json not found. Creating a new one at {self.installed_models_file}")
            self._save_installed_models([])
            return []
        
        try:
            with open(self.installed_models_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.error("Error decoding installed_models.json. Resetting to empty list.")
            self._save_installed_models([])
            return []

    def _save_installed_models(self, models: List[str]):
        try:
            with open(self.installed_models_file, 'w') as f:
                json.dump(models, f, indent=2)
            self.logger.info(f"Saved installed models to {self.installed_models_file}")
        except Exception as e:
            self.logger.error(f"Error saving installed models: {str(e)}")

    async def load_model(self, model_name: str, force_download: bool = False):
        model_path = os.path.join(self.model_cache_dir, "models--" + model_name.replace('/', '--'))
        model_loaded = False
        
        try:
            if not os.path.exists(model_path) or force_download:
                self.logger.info(f"Downloading model: {model_name}")
                try:
                    api = HfApi()
                    repo_files = await asyncio.to_thread(api.list_repo_files, repo_id=model_name)
                    for file in repo_files:
                        await asyncio.to_thread(hf_hub_download, repo_id=model_name, filename=file, cache_dir=self.model_cache_dir)
                    self.logger.info(f"Model downloaded and cached at: {model_path}")
                    
                    # Log actual cache directory contents
                    self.logger.info(f"Actual model cache directory contents:")
                    for root, dirs, files in os.walk(self.model_cache_dir):
                        for file in files:
                            self.logger.info(os.path.join(root, file))
                    
                except Exception as e:
                    self.logger.error(f"Error downloading model {model_name}: {str(e)}")
                    raise ValueError(f"Failed to download model {model_name}. Please check the model name and try again.")

            self.logger.info(f"Loading model: {model_name}")

            # Check for CUDA availability
            use_cuda = torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
            self.logger.info(f"Using device: {device}")

            # Load the config to determine the model type
            self.logger.info(f"Loading config for model: {model_name}")
            config = await asyncio.to_thread(AutoConfig.from_pretrained, model_name, cache_dir=self.model_cache_dir)
            self.logger.info(f"Loaded config: {config}")
            model_type = config.model_type
            self.logger.info(f"Model type: {model_type}")

            # Load the appropriate tokenizer based on the model type
            self.logger.info(f"Loading tokenizer for model type: {model_type}")
            try:
                if model_type == "phi":
                    self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_name, cache_dir=self.model_cache_dir)
                elif model_type == "llama":
                    from transformers import LlamaTokenizer
                    self.tokenizer = await asyncio.to_thread(LlamaTokenizer.from_pretrained, model_name, cache_dir=self.model_cache_dir)
                else:
                    # Default to AutoTokenizer for other model types
                    self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_name, cache_dir=self.model_cache_dir)
                self.logger.info(f"Tokenizer loaded successfully: {type(self.tokenizer)}")
            except ImportError as e:
                self.logger.error(f"Error importing tokenizer: {str(e)}")
                self.logger.info("Falling back to AutoTokenizer")
                self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_name, cache_dir=self.model_cache_dir)
            
            self.logger.info(f"Loaded tokenizer for model {model_name}")

            # Initialize model_kwargs
            model_kwargs = {}

            # Load the model
            try:
                # Prepare quantization config
                quantization_config = None
                if self.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                elif self.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

                # Calculate available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)

                # Load the model initially to calculate device map
                self.logger.info(f"Calculating device map for model: {model_name}")
                initial_model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, cache_dir=self.model_cache_dir)
                device_map = self.get_device_map(initial_model, max_gpu_memory=gpu_memory)

                # Adjust for CPU offloading
                if not use_cuda or any(device == 'cpu' for device in device_map.values()):
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
                    model_kwargs['device_map'] = device_map

                # Common kwargs for model loading
                model_kwargs.update({
                    "cache_dir": self.model_cache_dir,
                    "trust_remote_code": True,
                    "device_map": device_map,  # Use the custom device map
                    "torch_dtype": torch.float16,  # or torch.float32 based on your needs
                })

                # Ensure quantization_config is not None before adding it to kwargs
                if quantization_config is not None:
                    model_kwargs["quantization_config"] = quantization_config

                self.logger.info(f"Loading model with custom device map and CPU offloading")
                self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
                self.logger.info(f"Model loaded successfully with custom device map and CPU offloading: {type(self.model)}")

                # Set up the pipeline
                self.pipeline = await asyncio.to_thread(pipeline, "text-generation", model=self.model, tokenizer=self.tokenizer)
                self.logger.info(f"Pipeline set up successfully: {type(self.pipeline)}")

            except TypeError as e:
                if "load_in_8bit_fp32_cpu_offload" in str(e):
                    self.logger.warning(f"Removing unsupported 'load_in_8bit_fp32_cpu_offload' argument for model: {model_name}")
                    model_kwargs.pop("load_in_8bit_fp32_cpu_offload", None)
                    self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
                    self.logger.info(f"Model loaded successfully after removing unsupported argument: {type(self.model)}")
                else:
                    self.logger.error(f"Failed to load model: {str(e)}")
                    raise

            self.logger.info(f"Attempting to load model with kwargs: {model_kwargs}")

            # Attempt to load the model with different settings if initial load fails
            try:
                self.logger.info("Loading model with appropriate class")
                if model_type == "llama":
                    from transformers import LlamaForCausalLM
                    self.model = await asyncio.to_thread(LlamaForCausalLM.from_pretrained, model_name, **model_kwargs)
                else:
                    self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
                self.logger.info(f"Model loaded successfully: {type(self.model)}")
            except Exception as e:
                self.logger.error(f"Failed to load model with initial settings. Error: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error args: {e.args}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")

                # Try loading without quantization
                model_kwargs.pop("quantization_config", None)
                self.logger.info(f"Attempting to load model without quantization. New kwargs: {model_kwargs}")
                try:
                    if model_type == "llama":
                        from transformers import LlamaForCausalLM
                        self.model = await asyncio.to_thread(LlamaForCausalLM.from_pretrained, model_name, **model_kwargs)
                    else:
                        self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
                    self.logger.info(f"Model loaded successfully without quantization: {type(self.model)}")
                except Exception as e:
                    self.logger.error(f"Failed to load model without quantization. Error: {str(e)}")
                    self.logger.error(f"Error type: {type(e).__name__}")
                    self.logger.error(f"Error args: {e.args}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

                    # Try loading with CPU offload
                    model_kwargs["device_map"] = {"": torch.device("cpu")}
                    self.logger.info(f"Attempting to load model with CPU offload. New kwargs: {model_kwargs}")
                    if model_type == "llama":
                        from transformers import LlamaForCausalLM
                        self.model = await asyncio.to_thread(LlamaForCausalLM.from_pretrained, model_name, **model_kwargs)
                    else:
                        self.model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name, **model_kwargs)
                    self.logger.info(f"Model loaded successfully with CPU offload: {type(self.model)}")

            # Set up the pipeline
            try:
                self.logger.info("Setting up the pipeline")
                self.pipeline = await asyncio.to_thread(pipeline, "text-generation", model=self.model, tokenizer=self.tokenizer)
                self.logger.info(f"Pipeline set up successfully: {type(self.pipeline)}")
                self.current_model = model_name
                model_loaded = True
                self.logger.info(f"Loaded model: {model_name} on device: {device}")
            except Exception as e:
                self.logger.error(f"Error setting up the pipeline: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error args: {e.args}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to load model {model_name}. Error: {str(e)}")

        finally:
            # Add the model to installed_models.json even if loading failed
            if model_name not in self.installed_models:
                self.installed_models.append(model_name)
                self._save_installed_models(self.installed_models)
                self.logger.info(f"Added {model_name} to installed models list")

            if model_loaded:
                self.logger.info(f"Model {model_name} loaded and ready")
            else:
                self.logger.warning(f"Model {model_name} added to installed list, but failed to load")

            # Log model files for debugging
            self.logger.info(f"Model files present in {model_path}:")
            if os.path.exists(model_path):
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        self.logger.info(f" - {os.path.join(root, file)}")
            else:
                self.logger.warning(f"Model path {model_path} does not exist")

        if not model_loaded:
            raise ValueError(f"Failed to load model {model_name}. See logs for details.")

    def unload_model(self):
        if self.model:
            del self.model
            del self.tokenizer
            del self.pipeline
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.current_model = None
            self.logger.info("Model unloaded and CUDA cache cleared")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, messages: List[Dict[str, str]], model: str, stream: bool = True) -> Union[str, AsyncIterator[str]]:
        try:
            cache_key = self._get_cache_key(messages, model)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                self.logger.info("Returning cached response")
                return cached_response

            if self.current_model != model:
                await self.load_model(model)

            response = await self._generate_local_response(messages, model, stream)

            if not stream:
                self.response_cache[cache_key] = response

            return response
        except Exception as e:
            self.logger.error(f"Error in HuggingFace API call: {str(e)}")
            raise

    async def _generate_local_response(self, messages: List[Dict[str, str]], model: str, stream: bool) -> Union[str, AsyncIterator[str]]:
        prompt = self._convert_messages_to_prompt(messages)
        
        if stream:
            return self._stream_response(await self._generate_text(prompt))
        else:
            return await self._generate_text(prompt)

    async def _generate_text(self, prompt: str) -> str:
        generation_kwargs = self._get_generation_config()
        # Remove 'prompt' from generation_kwargs if it exists
        generation_kwargs.pop('prompt', None)
        # Pass the prompt as the first argument to the pipeline
        output = await asyncio.to_thread(self.pipeline, prompt, **generation_kwargs)
        # The output is typically a list of dictionaries, so we need to extract the generated text
        return output[0]['generated_text']

    async def _stream_response(self, text: str) -> AsyncIterator[str]:
        for token in text.split():
            yield token + " "
            await asyncio.sleep(0)  # Allow other coroutines to run

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

    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([message['content'] for message in messages if message['role'] == 'user'])

    def _get_schema_from_messages(self, messages: List[Dict[str, str]]) -> Dict:
        for message in messages:
            if message['role'] == 'system':
                try:
                    return json.loads(message['content'])
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse schema from system message")
                    return {}
        return {}

    def _get_cache_key(self, messages: List[Dict[str, str]], model: str) -> str:
        cache_data = json.dumps({"messages": messages, "model": model, "settings": self.model_settings})
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_generation_config(self, model_name: str = None) -> Dict:
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

    @lru_cache(maxsize=10)
    def get_model_info(self, model_name: str) -> Dict:
        try:
            model_info = self.client.model_info(model_name)
            return {
                "pipeline_tag": model_info.pipeline_tag,
                "tags": model_info.tags,
                "num_parameters": model_info.num_parameters
            }
        except Exception as e:
            self.logger.error(f"Error fetching model info for {model_name}: {str(e)}")
            return {}

    def get_model_settings(self) -> Dict:
        return self.model_settings

    def update_model_settings(self, new_settings: Dict):
        valid_settings = ['temperature', 'top_p', 'max_tokens', 'presence_penalty', 'repetition_penalty']
        self.model_settings.update({k: v for k, v in new_settings.items() if k in valid_settings})
        self.logger.info(f"Model settings updated: {self.model_settings}")

    def set_quantization(self, quantization: str):
        # Updated to include 'none' as a valid option
        if quantization not in [None, "4bit", "8bit", "none"]:
            raise ValueError("Quantization must be None, 'none', '4bit', or '8bit'")
        self.quantization = None if quantization == "none" else quantization
        self.logger.info(f"Quantization set to: {self.quantization}")

    async def fine_tune_model(self, base_model: str, train_data: List[Dict], output_dir: str, num_train_epochs: int = 3, per_device_train_batch_size: int = 8):
        self.logger.info(f"Fine-tuning model {base_model} with {len(train_data)} samples")

        await self.load_model(base_model)

        dataset = Dataset.from_dict({"text": [item["text"] for item in train_data]})
        tokenized_dataset = await asyncio.to_thread(dataset.map, self._tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=1000,
            save_total_limit=2,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        await asyncio.to_thread(trainer.train)

        await asyncio.to_thread(self.model.save_pretrained, output_dir)
        await asyncio.to_thread(self.tokenizer.save_pretrained, output_dir)

        self.logger.info(f"Fine-tuned model saved to {output_dir}")

        self.current_model = output_dir
        self.logger.info(f"Current model updated to fine-tuned version: {output_dir}")

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def get_installed_models(self) -> List[str]:
        return self.installed_models
    
    def remove_installed_model(self, model_name: str):
        model_path = os.path.join(self.model_cache_dir, model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            self.installed_models.remove(model_name)
            self._save_installed_models(self.installed_models)
            self.logger.info(f"Model {model_name} removed from installed models")
        else:
            self.logger.warning(f"Model {model_name} not found in installed models")


def setup_huggingface_generator(config_manager: ConfigManager):
    return HuggingFaceGenerator(config_manager)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "numind/NuExtract", stream: bool = True) -> Union[str, AsyncIterator[str]]:
    generator = setup_huggingface_generator(config_manager)
    await generator.load_model(model)
    return await generator.generate_response(messages, model, stream)

async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response
    return "".join([chunk async for chunk in response])

def generate_response_sync(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "numind/NuExtract", stream: bool = False) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(generate_response(config_manager, messages, model, stream))
    if stream:
        return loop.run_until_complete(process_response(response))
    return response
