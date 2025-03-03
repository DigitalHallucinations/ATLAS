import time
import asyncio
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

# Importing required modules for configuration, memory, attention, etc.
from modules.Config.config import ConfigManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.memory_system import MemorySystem
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.working.working_memory import EnhancedWorkingMemory
from modules.Hybrid_Cognitive_Dynamics_Model.Attention.attention_focus_mechanism import AttentionManager
from modules.thread_orchestrator import ThreadOrchestrator, cpu_bound_task, io_bound_task
from modules.Hybrid_Cognitive_Dynamics_Model.CCS.consciousness_stream_interface import (
    ConsciousnessStreamInterface,
    StateModelInterface,
    ResponseGeneratorInterface,
    GoalManagerInterface
)
from .particle_filter import ParticleFilter

class ThoughtType(Enum):
    """
    Enumeration of different types of thoughts that can occur within the consciousness stream.
    """
    OBSERVATION = "observation"
    EXTERNAL_INPUT = "external_input"
    GOAL = "goal"
    ACTION = "action"
    REFLECTION = "reflection"
    IDLE = "idle"

class ContinuousConsciousnessStream(ConsciousnessStreamInterface):
    """
    The ContinuousConsciousnessStream class manages the flow of thoughts and internal processing
    within the system's consciousness. It handles the scheduling and processing of different thought
    types, interacts with memory systems, state models, and manages attention focus.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ContinuousConsciousnessStream with necessary components.

        Args:
            config (Dict[str, Any]): Configuration containing all necessary components like threading, providers, etc.
        """
        # Assigning each component from the configuration dictionary.
        self.state_model: StateModelInterface = config['state_model']
        self.config_manager: ConfigManager = config['config_manager']
        self.logger = self.config_manager.setup_logger('ContinuousConsciousnessStream')
        self.thread_orchestrator: ThreadOrchestrator = config['thread_orchestrator']
        self.provider_manager: ProviderManager = config['provider_manager']
        self.memory_system: MemorySystem = config['memory_system']
        self.response_generator: ResponseGeneratorInterface = config['response_generator']
        self.goal_manager: GoalManagerInterface = config['goal_manager']
        self.attention_manager: AttentionManager = config['attention_manager']

        # Load configuration for the consciousness stream
        cc_stream_config = self.config_manager.get_subsystem_config('continuous_consciousness_stream')
        self.thought_queue_size = cc_stream_config.get('thought_queue_size', 100)
        self.priority_levels = cc_stream_config.get('priority_levels', 5)
        self.working_memory = EnhancedWorkingMemory(
            capacity=cc_stream_config.get('working_memory_capacity', 100),
            total_resources=cc_stream_config.get('working_memory_resources', 1.0)
        )

        # Initialize the thought queue with priority handling
        self.thought_queue = asyncio.PriorityQueue(self.thought_queue_size)
        self.running = False
        self.lock = asyncio.Lock()
        self.current_thought: Optional[Dict[str, Any]] = None

        # Set up memory and predictive error tracking
        self.memory_system.set_consciousness_stream(self)
        self.predictive_error = 0

        # Initialize the particle filter for state estimation
        self.particle_filter = ParticleFilter(
            n_particles=100,
            config_manager=self.config_manager,
            provider_manager=self.provider_manager
        )

        # Control the execution limits for testing and performance
        self.max_iterations = 2  # Limit the number of iterations
        self.iteration_count = 0
        self.max_runtime = 10  # Maximum runtime in seconds

    async def start(self):
        """
        Starts the continuous consciousness stream by initiating the processing loop.
        """
        self.running = True
        self.start_time = time.time()
        return asyncio.create_task(self._run_stream())

    async def _run_stream(self):
        """
        The main processing loop of the consciousness stream. Continuously processes thoughts
        from the thought queue or generates idle thoughts when the queue is empty.
        """
        while self.running:
            try:
                await self._process_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in CCS: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop in case of repeated errors
        self.logger.info("Continuous Consciousness Stream stopped")

    async def stop(self):
        """
        Stops the continuous consciousness stream gracefully.
        """
        self.running = False
        
        # Wait for all tasks to finish
        await asyncio.gather(*asyncio.all_tasks())
        self.logger.info("All tasks completed. Consciousness stream stopped.")


    async def process_for_duration(self, duration):
        """
        Processes thoughts for a specified duration.

        Args:
            duration (float): The duration in seconds for which to process thoughts.
        """
        end_time = time.time() + duration
        while time.time() < end_time and self.running and self.iteration_count < self.max_iterations:
            await asyncio.sleep(0.1)  # Small delay to prevent tight loop
        self.logger.info(f"Finished processing for duration. Iterations: {self.iteration_count}")

    async def add_thought(self, thought: Dict[str, Any], priority: int = 1):
        """
        Adds a thought to the thought queue with the specified priority.

        Args:
            thought (Dict[str, Any]): The thought to be added.
            priority (int): The priority level of the thought (lower value means higher priority).
        """
        await self.thought_queue.put((priority, time.time(), thought))
        self.logger.debug(f"Thought added to queue: {thought['type']}")

    async def get_current_thought(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current thought being processed.

        Returns:
            Optional[Dict[str, Any]]: The current thought or None if not processing any.
        """
        async with self.lock:
            return self.current_thought

    @cpu_bound_task
    async def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the current state of the consciousness stream, including state model,
        memory summary, and current goals.

        Returns:
            Dict[str, Any]: A dictionary containing the state information.
        """
        state_model_state = await self.state_model.get_state()
        memory_summary = self.memory_system.get_memory_stats()
        current_goals = await self.goal_manager.get_current_goals()

        return {
            "current_thought": await self.get_current_thought(),
            "state_model": state_model_state,
            "memory_summary": memory_summary,
            "current_goals": current_goals
        }

    @io_bound_task
    async def inject_external_input(self, input_data: Dict[str, Any], priority: int = 0, llm_call_type: str = None):
        """
        Injects external input into the consciousness stream and processes it immediately.

        Args:
            input_data (Dict[str, Any]): The external input data.
            priority (int): The priority level for processing the input.
            llm_call_type (str): The type of LLM call if applicable.
        """
        thought = {
            "type": ThoughtType.EXTERNAL_INPUT.value,
            "content": input_data,
            "timestamp": time.time(),
            "llm_call_type": llm_call_type
        }
        await self.add_thought(thought, priority)

        # Process the input immediately
        await self._process_thought(thought)

        # Ensure the thought is being processed by waiting for a short time
        await asyncio.sleep(0.1)

        # Process the input through the memory system
        if isinstance(input_data, dict) and 'content' in input_data:
            content = input_data['content']
        else:
            content = str(input_data)

        await self.memory_system.process_input(content)

        # Explicitly add to episodic memory
        context = await self.state_model.get_current_state_context()
        await self.memory_system.long_term_episodic.add(content, context)

        # Update the state model
        await self.state_model.update({'content': content})

        # Trigger memory consolidation
        await self.memory_system.consolidate_memory()

        # Process external input in GoalManager
        await self.goal_manager.update_goals(thought, await self.state_model.get_state())

        self.logger.info(f"External input processed and stored: {content[:50]}...")

    async def _process_thought(self, thought: Dict[str, Any]):
        """
        Processes a single thought by updating the state model, attention focus, and interacting
        with the memory system and goal manager.

        Args:
            thought (Dict[str, Any]): The thought to be processed.
        """
        try:
            async with self.lock:
                self.current_thought = thought

            self.logger.debug(f"Processing thought: {thought['type']}")

            # Get current state from state model
            current_state = await self.state_model.get_state()

            # Ensure 'content' is a string
            if isinstance(thought['content'], dict) and 'content' in thought['content']:
                content = thought['content']['content']
            else:
                content = str(thought['content'])

            # Update attention focus based on the current thought
            attention_vector = self.attention_manager.compute_attention_vector(content)
            self.logger.info(f"Computed attention vector: {attention_vector}")

            # Explicitly update the state model's attention focus
            await self.state_model.update_attention_focus(attention_vector, attention_vector)

            # Compute PFC activation
            pfc_input = np.array([hash(word) % self.state_model.dim for word in content.split()])
            pfc_activation = self.state_model.pfc_layer.update(pfc_input, dt=0.001)

            # Process thought through memory system with time-aware consolidation
            memory_result = await self.memory_system.process_input_with_consciousness(content)
            self.logger.debug(f"Memory result: {memory_result}")

            # Explicitly add to episodic memory
            context = await self.state_model.get_current_state_context()
            await self.memory_system.long_term_episodic.add(content, context)

            # Update state model
            state_update = await self.state_model.update({'content': content})

            # Update particle filter with the new state
            self.particle_filter.predict(state_update)
            self.particle_filter.update(current_state)

            # Update goals based on the thought and current state
            await self.goal_manager.update_goals(thought, current_state)

            # Generate response considering current goals
            current_goals = await self.goal_manager.get_current_goals()
            response = await self.response_generator.generate(thought, current_state, current_goals)
            self.logger.debug(f"Generated response: {response}")

            if response:
                response_thought = {
                    "type": ThoughtType.ACTION.value,
                    "content": response,
                    "timestamp": time.time()
                }
                await self.add_thought(response_thought)
                self.logger.debug(f"Added response thought to queue: {response_thought}")

                # Update memory with response
                await self.memory_system.process_input_with_consciousness(response)

            # Trigger memory consolidation (Time-Aware Processing)
            await self.memory_system.consolidate_memory()

            # Log the updated attention focus
            updated_focus = await self.state_model.get_attention_focus()
            self.logger.info(f"Updated attention focus: {updated_focus}")

        except Exception as e:
            self.logger.error(f"Error processing thought: {str(e)}", exc_info=True)

        finally:
            # Ensure current_thought is cleared after processing
            async with self.lock:
                self.current_thought = None

            # Explicitly remove the processed thought from the queue
            await self.thought_queue.get()
            self.thought_queue.task_done()

    def get_best_particle_state(self):
        """
        Retrieves the state description of the best particle from the particle filter.

        Returns:
            Dict[str, Any]: The state description of the best particle.
        """
        best_particle = self.particle_filter.get_best_particle()
        return best_particle.state.get_state_description()

    def prepare_next_input(self, thought: Dict[str, Any]) -> np.ndarray:
        """
        Prepares the next input for the state model based on the current thought.

        Args:
            thought (Dict[str, Any]): The current thought.

        Returns:
            np.ndarray: The prepared input vector.
        """
        # Convert thought to a format suitable for prediction
        if 'content' in thought:
            return np.array([hash(word) % 1000 for word in thought['content'].split()])
        else:
            return np.zeros(1000)  # Default size, adjust as needed

    def compute_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Computes the prediction error between the predicted and actual state vectors.

        Args:
            predicted (np.ndarray): The predicted state vector.
            actual (np.ndarray): The actual state vector.

        Returns:
            float: The mean absolute prediction error.
        """
        # Ensure arrays are the same size
        min_size = min(len(predicted), len(actual))
        return np.mean(np.abs(predicted[:min_size] - actual[:min_size]))

    def determine_attention_focus(self, thought: Dict[str, Any], prediction_error: float) -> Optional[int]:
        """
        Determines where to focus attention based on the thought content and prediction error.

        Args:
            thought (Dict[str, Any]): The current thought.
            prediction_error (float): The computed prediction error.

        Returns:
            Optional[int]: The index of the word to focus on, or None if no focus change is needed.
        """
        # Determine where to focus attention based on thought content and prediction error
        if prediction_error > self.config_manager.get('attention_threshold', 0.5):
            # Focus on the most relevant part of the thought
            if 'content' in thought:
                words = thought['content'].split()
                return words.index(max(words, key=len)) if words else None
        return None

    async def update_state_from_prediction(self, prediction_error: float, attention_focus: Optional[int]) -> None:
        """
        Updates the state model based on prediction error and attention focus adjustments.

        Args:
            prediction_error (float): The computed prediction error.
            attention_focus (Optional[int]): The index of the new attention focus.
        """
        # Update the state model based on prediction error and attention focus
        state_update = {
            'prediction_error': prediction_error,
            'attention_focus': attention_focus
        }
        await self.state_model.update(state_update)

    @cpu_bound_task
    async def _process_stream(self):
        """
        Processes the thought stream by continuously fetching thoughts from the queue and processing them.
        Generates idle thoughts when the queue is empty.
        """
        self.logger.debug("Entering _process_stream")
        while self.running and self.iteration_count < self.max_iterations and (time.time() - self.start_time) < self.max_runtime:
            try:
                self.logger.debug(f"Processing iteration {self.iteration_count + 1}/{self.max_iterations}")
                _, _, thought = await asyncio.wait_for(self.thought_queue.get(), timeout=0.1)
                await self._process_thought(thought)
                self.thought_queue.task_done()
                self.iteration_count += 1
            except asyncio.TimeoutError:
                self.logger.debug("No thought in queue, generating idle thought")
                await self._generate_idle_thought()
            except Exception as e:
                self.logger.error(f"Error in _process_stream: {str(e)}", exc_info=True)
        self.logger.info(f"Exited _process_stream loop after {self.iteration_count} iterations")
        self.running = False

    @cpu_bound_task
    async def _generate_associations(self, content: str) -> List[str]:
        """
        Generates associations based on the content using pattern completion and episodic memory retrieval.

        Args:
            content (str): The content to generate associations from.

        Returns:
            List[str]: A list of associated concepts and episodes.
        """
        try:
            completed_concepts = self.memory_system.long_term_semantic.pattern_completion(content)
            associations = [concept for concept, similarity in completed_concepts[:5]]

            relevant_episodes = await self.memory_system.long_term_episodic.get_relevant_episodes(3)
            episodic_associations = [episode['content'] for episode in relevant_episodes]

            return associations + episodic_associations
        except Exception as e:
            self.logger.error(f"Error generating associations: {str(e)}", exc_info=True)
            return []

    def _enrich_thought(self, thought: Dict[str, Any], state_update: Dict[str, Any],
                        memory_result: Dict[str, Any], associations: List[str]) -> Dict[str, Any]:
        """
        Enriches a thought with additional information such as state updates, memory results, and associations.

        Args:
            thought (Dict[str, Any]): The original thought.
            state_update (Dict[str, Any]): The state updates from processing.
            memory_result (Dict[str, Any]): The result from memory processing.
            associations (List[str]): Associated concepts and episodes.

        Returns:
            Dict[str, Any]: The enriched thought.
        """
        enriched_thought = thought.copy()
        enriched_thought['state_update'] = state_update
        enriched_thought['memory_result'] = memory_result
        enriched_thought['associations'] = associations
        return enriched_thought

    @cpu_bound_task
    async def _take_action(self, thought: Dict[str, Any]):
        """
        Decides on actions to take based on the thought type and processes accordingly.

        Args:
            thought (Dict[str, Any]): The thought to act upon.
        """
        try:
            thought_type = ThoughtType(thought['type'])

            if thought_type == ThoughtType.OBSERVATION:
                await self._process_observation(thought)
            elif thought_type == ThoughtType.EXTERNAL_INPUT:
                await self._process_external_input(thought)
            elif thought_type == ThoughtType.GOAL:
                await self._process_goal(thought)
            elif thought_type == ThoughtType.ACTION:
                await self._process_action(thought)
            elif thought_type == ThoughtType.REFLECTION:
                await self._process_reflection(thought)

            response = await self.response_generator.generate(thought, await self.state_model.get_state())
            self.logger.debug(f"Generated response: {response}")

        except Exception as e:
            self.logger.error(f"Error in _take_action: {str(e)}", exc_info=True)

    @cpu_bound_task
    async def _reflect(self, thought: Dict[str, Any], state_update: Dict[str, Any], memory_result: Dict[str, Any]):
        """
        Creates a reflection based on the thought, state changes, and memory impact.

        Args:
            thought (Dict[str, Any]): The original thought.
            state_update (Dict[str, Any]): The state updates from processing.
            memory_result (Dict[str, Any]): The result from memory processing.
        """
        reflection = f"Reflected on {thought['type']} thought. State changes: {state_update}. Memory impact: {memory_result}"
        await self.add_thought({"type": ThoughtType.REFLECTION.value, "content": reflection}, priority=3)

    @cpu_bound_task
    async def _generate_idle_thought(self):
        """
        Generates an idle thought to maintain the consciousness stream when the thought queue is empty.
        """
        idle_thought = {
            "type": ThoughtType.IDLE.value,
            "content": "Maintaining consciousness stream",
            "timestamp": time.time()
        }
        await self._process_thought(idle_thought)

    def _extract_keywords(self, content: Any) -> List[str]:
        """
        Extracts keywords from the content for further processing.

        Args:
            content (Any): The content from which to extract keywords.

        Returns:
            List[str]: A list of extracted keywords.
        """
        if isinstance(content, dict):
            text = ' '.join(str(value) for value in content.values() if isinstance(value, str))
        elif isinstance(content, str):
            text = content
        else:
            self.logger.warning(f"Unexpected content type in _extract_keywords: {type(content)}")
            return []

        return [word.lower() for word in text.split() if len(word) > 3]

    def _generate_implications(self, observation: str) -> List[str]:
        """
        Generates potential implications from an observation.

        Args:
            observation (str): The observation content.

        Returns:
            List[str]: A list of potential implications.
        """
        return [f"Potential implication of '{observation}': Further analysis needed."]

    def _analyze_input(self, input_content: str) -> Tuple[str, str]:
        """
        Analyzes external input to determine intent and emotional tone.

        Args:
            input_content (str): The input content.

        Returns:
            Tuple[str, str]: A tuple containing the intent and emotion.
        """
        # Placeholder for actual analysis logic
        return "general_query", "neutral"

    def _generate_sub_goals(self, goal: str, progress: float) -> List[str]:
        """
        Generates sub-goals based on the main goal and current progress.

        Args:
            goal (str): The main goal.
            progress (float): The progress towards the goal.

        Returns:
            List[str]: A list of sub-goals.
        """
        return [f"Sub-goal for '{goal}': Gather more information."]

    @io_bound_task
    async def _simulate_action(self, action: str) -> Dict[str, Any]:
        """
        Simulates the execution of an action and returns the effects.

        Args:
            action (str): The action to simulate.

        Returns:
            Dict[str, Any]: The effects of the simulated action.
        """
        # Placeholder for actual action simulation logic
        return {"action_completed": True, "side_effects": None}

    def _reflect_on_action(self, action: str, effects: Dict[str, Any]) -> str:
        """
        Generates a reflection based on the action taken and its effects.

        Args:
            action (str): The action performed.
            effects (Dict[str, Any]): The effects of the action.

        Returns:
            str: The reflection text.
        """
        return f"Action '{action}' completed with effects: {effects}"

    @cpu_bound_task
    async def _analyze_reflection(self, reflection: str) -> List[str]:
        """
        Analyzes a reflection to extract insights.

        Args:
            reflection (str): The reflection content.

        Returns:
            List[str]: A list of insights derived from the reflection.
        """
        # Placeholder for actual reflection analysis logic
        return [f"Insight from reflection: {reflection}"]

    def _generate_goals_from_insights(self, insights: List[str]) -> List[str]:
        """
        Generates new goals based on the insights obtained from reflections.

        Args:
            insights (List[str]): A list of insights.

        Returns:
            List[str]: A list of new goals.
        """
        return [f"New goal based on insight: Investigate {insight}" for insight in insights]

    @io_bound_task
    async def _process_observation(self, thought: Dict[str, Any]):
        """
        Processes an observation thought by updating the environment state and generating implications.

        Args:
            thought (Dict[str, Any]): The observation thought.
        """
        self.logger.debug(f"Processing observation: {thought['content']}")
        await self.state_model.update_environment(thought['content'])
        implications = self._generate_implications(thought['content'])
        for implication in implications:
            await self.add_thought({"type": ThoughtType.REFLECTION.value, "content": implication}, priority=2)

    @io_bound_task
    async def _process_external_input(self, thought: Dict[str, Any]):
        """
        Processes external input by analyzing it and generating appropriate responses.

        Args:
            thought (Dict[str, Any]): The external input thought.
        """
        self.logger.debug(f"Processing external input: {thought['content']}")
        intent, emotion = self._analyze_input(thought['content'])
        response = await self.response_generator.generate(thought, await self.state_model.get_state(), await self.goal_manager.get_current_goals())
        await self.add_thought({"type": ThoughtType.ACTION.value, "content": response}, priority=1)

    @cpu_bound_task
    async def _process_goal(self, thought: Dict[str, Any]):
        """
        Processes a goal thought by evaluating progress and generating sub-goals.

        Args:
            thought (Dict[str, Any]): The goal thought.
        """
        self.logger.debug(f"Processing goal: {thought['content']}")
        progress = await self.goal_manager.evaluate_goal_progress(thought['content'])
        sub_goals = self._generate_sub_goals(thought['content'], progress)
        for sub_goal in sub_goals:
            await self.add_thought({"type": ThoughtType.GOAL.value, "content": sub_goal}, priority=2)

    @cpu_bound_task
    async def _process_action(self, thought: Dict[str, Any]):
        """
        Processes an action thought by simulating the action and reflecting on its effects.

        Args:
            thought (Dict[str, Any]): The action thought.
        """
        self.logger.debug(f"Processing action: {thought['content']}")
        effects = await self._simulate_action(thought['content'])
        await self.state_model.update(effects)
        reflection = self._reflect_on_action(thought['content'], effects)
        await self.add_thought({"type": ThoughtType.REFLECTION.value, "content": reflection}, priority=2)

    @cpu_bound_task
    async def _process_reflection(self, thought: Dict[str, Any]):
        """
        Processes a reflection thought by analyzing it for insights and generating new goals.

        Args:
            thought (Dict[str, Any]): The reflection thought.
        """
        self.logger.debug(f"Processing reflection: {thought['content']}")
        insights = await self._analyze_reflection(thought['content'])
        # Integrate Time-Aware Processing: Update memory importance based on insights
        for insight in insights:
            self.memory_system.time_decay.adjust_importance(insight)
        await self.memory_system.consolidate_memory()
        new_goals = self._generate_goals_from_insights(insights)
        for goal in new_goals:
            await self.add_thought({"type": ThoughtType.GOAL.value, "content": goal}, priority=3)

    @io_bound_task
    async def process_interaction(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Processes an interaction by adding it as an external input thought.

        Args:
            prompt (str): The user's input.
            response (str): The system's response.

        Returns:
            Dict[str, Any]: The result of processing the interaction.
        """
        self.logger.info(f"Processing interaction - Prompt: {prompt}, Response: {response}")
        interaction_thought = {
            "content": f"User said: {prompt}\nSystem responded: {response}",
            "type": ThoughtType.EXTERNAL_INPUT.value,
        }
        return await self._process_thought(interaction_thought)

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the consciousness stream.

        Returns:
            Dict[str, Any]: A dictionary containing current thought, queue size, and running status.
        """
        return {
            "current_thought": self.current_thought,
            "queue_size": self.thought_queue.qsize(),
            "is_running": self.running,
        }

    @cpu_bound_task
    async def initialize(self):
        """
        Initializes the consciousness stream and starts the processing loop.
        """
        self.logger.info("Initializing ContinuousConsciousnessStream")
        # Add any initial thoughts if necessary
        # for thought in []:
        #     await self.add_thought(**thought)
        await self.start()
        self.logger.info("ContinuousConsciousnessStream initialized and started")

    @cpu_bound_task
    async def monitor_performance(self):
        """
        Periodically monitors the performance and state of the consciousness stream.
        """
        while self.running:
            stats = self.get_stats()
            self.logger.info(f"ContinuousConsciousnessStream performance: {stats}")
            await asyncio.sleep(60)  # Monitor every minute

    @io_bound_task
    async def backup_state(self):
        """
        Backs up the current state of the consciousness stream for recovery purposes.
        """
        state = await self.get_state()
        # Implement state backup logic here
        self.logger.info("ContinuousConsciousnessStream state backed up")
