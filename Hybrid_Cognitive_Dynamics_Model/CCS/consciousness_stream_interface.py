# modules/Hybrid_Cognitive_Dynamics_Model/CCS/consciousness_stream_interface.py  ### NEW ###

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ConsciousnessStreamInterface(ABC):
    @abstractmethod
    def start(self):
        """Start the continuous consciousness stream."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the continuous consciousness stream."""
        pass

    @abstractmethod
    def add_thought(self, thought: Dict[str, Any], priority: int = 1):
        """
        Add a thought to the consciousness stream.
        
        Args:
            thought (Dict[str, Any]): The thought to be added.
            priority (int): The priority of the thought (default is 1).
        """
        pass

    @abstractmethod
    def get_current_thought(self) -> Dict[str, Any]:
        """
        Get the current thought being processed.
        
        Returns:
            Dict[str, Any]: The current thought.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the consciousness stream.
        
        Returns:
            Dict[str, Any]: The current state.
        """
        pass

    @abstractmethod
    def inject_external_input(self, input_data: Dict[str, Any], priority: int = 0):
        """
        Inject external input into the consciousness stream.
        
        Args:
            input_data (Dict[str, Any]): The external input data.
            priority (int): The priority of the input (default is 0).
        """
        pass

class StateModelInterface(ABC):
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the state model with new data.
        
        Args:
            data (Dict[str, Any]): The data to update the state model with.
        
        Returns:
            Dict[str, Any]: The updated state.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the model.
        
        Returns:
            Dict[str, Any]: The current state.
        """
        pass

    @abstractmethod
    def update_environment(self, observation: str):
        """
        Update the environment based on an observation.
        
        Args:
            observation (str): The observation to update the environment with.
        """
        pass

    @abstractmethod
    def update_emotional_state(self, emotion: str):
        """
        Update the emotional state.
        
        Args:
            emotion (str): The emotion to update the state with.
        """
        pass

class MemorySystemInterface(ABC):
    @abstractmethod
    def process_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a thought and store it in memory.
        
        Args:
            thought (Dict[str, Any]): The thought to process.
        
        Returns:
            Dict[str, Any]: The result of processing the thought.
        """
        pass

    @abstractmethod
    def find_related_memories(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find memories related to given keywords.
        
        Args:
            keywords (List[str]): The keywords to search for.
        
        Returns:
            List[Dict[str, Any]]: A list of related memories.
        """
        pass

    @abstractmethod
    def add_to_long_term_memory(self, data: Any):
        """
        Add data to long-term memory.
        
        Args:
            data (Any): The data to add to long-term memory.
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state of memory.
        
        Returns:
            Dict[str, Any]: A summary of the memory state.
        """
        pass

class ResponseGeneratorInterface(ABC):
    @abstractmethod
    def generate(self, thought: Dict[str, Any], state: Dict[str, Any]) -> str:
        """
        Generate a response based on a thought and the current state.
        
        Args:
            thought (Dict[str, Any]): The thought to generate a response for.
            state (Dict[str, Any]): The current state of the system.
        
        Returns:
            str: The generated response.
        """
        pass

class GoalManagerInterface(ABC):
    @abstractmethod
    def update_goals(self, thought: Dict[str, Any], state: Dict[str, Any]):
        """
        Update goals based on a thought and the current state.
        
        Args:
            thought (Dict[str, Any]): The thought to update goals with.
            state (Dict[str, Any]): The current state of the system.
        """
        pass

    @abstractmethod
    def evaluate_goal_progress(self, goal: str) -> float:
        """
        Evaluate the progress of a specific goal.
        
        Args:
            goal (str): The goal to evaluate.
        
        Returns:
            float: The progress of the goal (0.0 to 1.0).
        """
        pass

    @abstractmethod
    def get_current_goals(self) -> List[Dict[str, Any]]:
        """
        Get the current goals.
        
        Returns:
            List[Dict[str, Any]]: A list of current goals.
        """
        pass

    @abstractmethod
    def add_goal(self, goal: str, priority: int):
        """
        Add a new goal.
        
        Args:
            goal (str): The goal to add.
            priority (int): The priority of the goal.
        """
        pass

    @abstractmethod
    def remove_goal(self, goal: str):
        """
        Remove a goal.
        
        Args:
            goal (str): The goal to remove.
        """
        pass

class LanguageModelInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the language model.
        
        Args:
            prompt (str): The input prompt for the language model.
        
        Returns:
            str: The generated response.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text (str): The text to encode.
        
        Returns:
            List[int]: The list of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs into text.
        
        Args:
            token_ids (List[int]): The list of token IDs to decode.
        
        Returns:
            str: The decoded text.
        """
        pass

class AttentionManagerInterface(ABC):
    @abstractmethod
    def update_attention(self, input_data: str):
        """
        Update the attention focus based on input data.
        
        Args:
            input_data (str): The input data to update attention with.
        """
        pass

    @abstractmethod
    def get_attention_vector(self) -> List[float]:
        """
        Get the current attention vector.
        
        Returns:
            List[float]: The current attention vector.
        """
        pass

class EmotionModelInterface(ABC):
    @abstractmethod
    def update_emotion(self, input_data: str):
        """
        Update the emotional state based on input data.
        
        Args:
            input_data (str): The input data to update emotion with.
        """
        pass

    @abstractmethod
    def get_emotional_state(self) -> Dict[str, float]:
        """
        Get the current emotional state.
        
        Returns:
            Dict[str, float]: The current emotional state.
        """
        pass