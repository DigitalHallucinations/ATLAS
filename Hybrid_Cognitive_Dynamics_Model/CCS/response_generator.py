# modules/Hybrid_Cognitive_Dynamics_Model/CCS/response_generator.py

import asyncio
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from modules.Hybrid_Cognitive_Dynamics_Model.CCS.consciousness_stream_interface import ResponseGeneratorInterface
from modules.Providers.provider_manager import ProviderManager
from modules.Config.config import ConfigManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.memory_system import MemorySystem


class ResponseGenerator(ResponseGeneratorInterface):
    """
    The ResponseGenerator class is responsible for generating responses based on current thoughts,
    system state, and goals. It utilizes the provider manager to interact with language models
    and handles memory updates, chunking large responses, and refining responses for coherence.

    Attributes:
        provider_manager (ProviderManager): Manages interaction with language model providers.
        memory_system (MemorySystem): The memory system for the consciousness stream (episodic and semantic memory).
        config_manager (ConfigManager): Manages configuration settings.
        logger: Logger instance for response generation.
        max_new_tokens (int): Maximum number of tokens for the response generation.
        temperature (float): Temperature setting for controlling randomness in the response generation.
    """

    def __init__(self, 
                 provider_manager: ProviderManager,
                 memory_system: MemorySystem,
                 config_manager: ConfigManager,
                 max_new_tokens: int = 100,
                 temperature: float = 0.7):
        """
        Initializes a new instance of the ResponseGenerator.

        Args:
            provider_manager (ProviderManager): Manages language model interactions.
            memory_system (MemorySystem): System for managing memory.
            config_manager (ConfigManager): Configuration management.
            max_new_tokens (int, optional): Maximum tokens for response generation. Defaults to 100.
            temperature (float, optional): Controls randomness of responses. Defaults to 0.7.
        """
        self.provider_manager = provider_manager
        self.memory_system = memory_system
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ResponseGenerator')
        self.update_generation_params(max_new_tokens, temperature)

    def update_generation_params(self, 
                                 max_new_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None):
        """
        Updates the parameters used for generating responses, such as max tokens and temperature.

        Args:
            max_new_tokens (Optional[int], optional): Maximum tokens for generation. Defaults to None.
            temperature (Optional[float], optional): Controls randomness in generation. Defaults to None.
        """
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        self.logger.info(f"Generation parameters updated: max_new_tokens={self.max_new_tokens}, "
                         f"temperature={self.temperature}")

    async def generate(self, 
                       thought: Dict[str, Any], 
                       state: Dict[str, Any],
                       current_goals: List[Dict[str, Any]],
                       max_new_tokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> str:
        """
        Generates a response based on the provided thought, state, and goals.

        Args:
            thought (Dict[str, Any]): Current thought to process.
            state (Dict[str, Any]): Current system state.
            current_goals (List[Dict[str, Any]]): Current goals being pursued.
            max_new_tokens (Optional[int], optional): Maximum tokens for the response. Defaults to None.
            temperature (Optional[float], optional): Randomness control. Defaults to None.

        Returns:
            str: The generated response.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Generating response for thought: {thought}")
                prompt = await self._create_prompt(thought, state, current_goals)
                self.logger.debug(f"Created prompt: {prompt}")
                
                messages = [{"role": "user", "content": prompt}]
                response = await self.provider_manager.generate_response(
                    messages, 
                    model=self.provider_manager.get_current_model(),
                    llm_call_type="main_generation",
                    max_tokens=max_new_tokens or self.max_new_tokens,
                    temperature=temperature or self.temperature
                )
                
                self.logger.debug(f"Generated raw response: {response}")
                processed_response = await self._post_process_response(response)  
                self.logger.debug(f"Processed response: {processed_response}")

                # Integrate Time-Aware Processing: Adjust importance based on temporal factors
                self.memory_system.time_decay.adjust_importance(processed_response)
                await self.memory_system.consolidate_memory()

                if hasattr(self.memory_system, 'working') and self.response_requires_chunking(processed_response):
                    self.logger.debug("Response requires chunking")
                    try:
                        chunks = self.create_response_chunks(processed_response)
                        chunk_sequence = []
                        for chunk_name, chunk_items in chunks.items():
                            await self.memory_system.working.create_chunk(chunk_items, chunk_name)
                            chunk_sequence.append(chunk_name)
                        
                        chunked_response = await self.assemble_chunked_response(chunks, chunk_sequence)
                        self.logger.debug(f"Assembled chunked response: {chunked_response}")
                        return chunked_response
                    except Exception as chunk_error:
                        self.logger.error(f"Error in chunking process: {str(chunk_error)}", exc_info=True)
                        return processed_response
                else:
                    self.logger.debug("Response does not require chunking")
                    return processed_response
            except Exception as e:
                self.logger.error(f"Error generating response (attempt {attempt + 1}/{max_retries}): {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    return "I apologize, but I'm having trouble formulating a response at the moment."
                await asyncio.sleep(1)

    async def _create_prompt(self, thought: Dict[str, Any], state: Any, current_goals: List[Dict[str, Any]]) -> str:
        """
        Creates a prompt for generating a response by combining thought, state, and goals.

        Args:
            thought (Dict[str, Any]): The current thought to process.
            state (Any): The current system state.
            current_goals (List[Dict[str, Any]]): List of current goals.

        Returns:
            str: The generated prompt string.
        """
        try:
            if asyncio.iscoroutine(state):
                state = await state

            prompt = f"Current thought: {thought.get('content', 'No content')}\n"
            prompt += f"Thought type: {thought.get('type', 'Unknown')}\n"

            if isinstance(state, dict) and 'emotional_state' in state:
                prompt += f"Emotion: {state['emotional_state']}\n"
            else:
                prompt += "Emotion: Neutral\n"

            relevant_episodes = await self.memory_system.long_term_episodic.get_relevant_episodes(3)
            prompt += f"Relevant memories: {', '.join(episode['content'] for episode in relevant_episodes)}\n"
            
            semantic_context = await self.memory_system.long_term_semantic.query(thought.get('content', ''), 3)
            prompt += f"Semantic context: {', '.join(concept for concept, _ in semantic_context)}\n"

            if current_goals:
                goal_descriptions = [goal.get('description', 'Unknown goal') for goal in current_goals]
                prompt += f"Current goals: {', '.join(goal_descriptions)}\n"
            else:
                prompt += "No current goals.\n"

            prompt += "Based on this information, generate an appropriate response:\n"
            return prompt
        except Exception as e:
            self.logger.error(f"Error creating prompt: {str(e)}", exc_info=True)
            return "Generate a general response:"

    async def _post_process_response(self, response: Union[str, AsyncIterator[str]]) -> str:
        """
        Post-processes the generated response to ensure proper formatting and length.

        Args:
            response (Union[str, AsyncIterator[str]]): The raw response from the model.

        Returns:
            str: The processed response.
        """
        try:
            if isinstance(response, AsyncIterator):
                # Handle streaming response
                full_response = ""
                async for chunk in response:
                    full_response += chunk
                response = full_response
            
            response = response.strip()
            if len(response) > 500:
                response = response[:497] + "..."
            if response and response[-1] not in ".!?":
                response += "."
            return response
        except Exception as e:
            self.logger.error(f"Error post-processing response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while processing my response."

    def update_memory_with_interaction(self, user_input: str, response: str):
        """
        Updates memory with the user's input and the system's generated response.

        Args:
            user_input (str): The user's input.
            response (str): The system's response.
        """
        try:
            interaction = f"User: {user_input}\nSystem: {response}"
            self.memory_system.process_input(interaction)
            self.logger.debug(f"Memory updated with interaction: {interaction[:100]}...")
        except Exception as e:
            self.logger.error(f"Error updating memory with interaction: {str(e)}", exc_info=True)

    def response_requires_chunking(self, response: str) -> bool:
        """
        Determines if a response requires chunking based on the working memory capacity.

        Args:
            response (str): The generated response.

        Returns:
            bool: True if chunking is required, otherwise False.
        """
        return len(response.split()) > self.memory_system.working.capacity

    def create_response_chunks(self, response: str) -> Dict[str, List[str]]:
        """
        Splits the response into chunks based on working memory capacity.

        Args:
            response (str): The full generated response to be chunked.

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a chunk name (e.g., "chunk_0", "chunk_1"),
            and the value is a list of words that form the chunk.
        """
        words = response.split()
        chunk_size = self.memory_system.working.capacity
        chunks = {}
        for i in range(0, len(words), chunk_size):
            chunk_name = f"chunk_{i//chunk_size}"
            chunks[chunk_name] = words[i:i+chunk_size]
        return chunks

    async def assemble_chunked_response(self, chunks: Dict[str, List[str]], chunk_sequence: List[str]) -> str:
        """
        Assembles chunked responses back into a coherent full response by fetching the content of each chunk.

        Args:
            chunks (Dict[str, List[str]]): Dictionary of chunked responses.
            chunk_sequence (List[str]): The sequence in which the chunks should be assembled.

        Returns:
            str: The fully assembled response as a single string.
        """
        assembled_response = []
        for chunk_name in chunk_sequence:
            chunk = await self.memory_system.working.expand_chunk(chunk_name)
            assembled_response.extend(chunk)
        return ' '.join(assembled_response)

    async def refine_response(self, initial_response: str, thought: Dict[str, Any], state: Dict[str, Any]) -> str:
        """
        Refines the initial generated response to better align with the current thought and state.

        Args:
            initial_response (str): The initial generated response.
            thought (Dict[str, Any]): The current thought being processed.
            state (Dict[str, Any]): The current system state.

        Returns:
            str: The refined response.
        """
        refinement_prompt = f"Initial response: {initial_response}\n"
        refinement_prompt += f"Current thought: {thought.get('content', '')}\n"
        refinement_prompt += f"Emotional state: {state.get('emotional_state', 'Neutral')}\n"
        refinement_prompt += "Refine the initial response to better align with the current thought and emotional state:"
        
        refined_response = await self.provider_manager.generate_response(refinement_prompt)
        return refined_response

    def evaluate_response_coherence(self, response: str, thought: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Evaluates the coherence of the generated response with the provided thought and state.

        Args:
            response (str): The generated response.
            thought (Dict[str, Any]): The current thought being processed.
            state (Dict[str, Any]): The current system state.

        Returns:
            float: A coherence score based on keyword overlap between the thought and the response.
        """
        thought_keywords = set(thought.get('content', '').lower().split())
        response_keywords = set(response.lower().split())
        overlap = len(thought_keywords.intersection(response_keywords))
        coherence_score = overlap / (len(thought_keywords) + len(response_keywords)) if (len(thought_keywords) + len(response_keywords)) > 0 else 0
        return coherence_score

    async def generate_with_refinement(self, thought: Dict[str, Any], state: Dict[str, Any], current_goals: List[Dict[str, Any]], max_iterations: int = 3) -> str:
        """
        Generates a response with multiple iterations of refinement until a satisfactory coherence level is reached.

        Args:
            thought (Dict[str, Any]): The current thought being processed.
            state (Dict[str, Any]): The current system state.
            current_goals (List[Dict[str, Any]]): Current goals being pursued.
            max_iterations (int, optional): Maximum number of refinement iterations. Defaults to 3.

        Returns:
            str: The final response after refinement.
        """
        response = await self.generate(thought, state, current_goals)
        coherence_score = self.evaluate_response_coherence(response, thought, state)
        
        iterations = 0
        while coherence_score < 0.5 and iterations < max_iterations:
            refined_response = await self.refine_response(response, thought, state)
            new_coherence_score = self.evaluate_response_coherence(refined_response, thought, state)
            
            if new_coherence_score > coherence_score:
                response = refined_response
                coherence_score = new_coherence_score
            
            iterations += 1
        
        return response
