# modules/Hybrid_Cognitive_Dynamics_Model/CCS/goal_manager.py  

import time
import asyncio
import networkx as nx
from datetime import datetime, timedelta
from modules.Config.config import ConfigManager
from typing import Dict, Any, List, Optional
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.CCS.continuous_consciousness_stream import ThoughtType
from modules.Hybrid_Cognitive_Dynamics_Model.CCS.consciousness_stream_interface import GoalManagerInterface
from modules.thread_orchestrator import ThreadOrchestrator, TaskResource, cpu_bound_task, io_bound_task


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class Goal:
    """
    Represents a single goal within the GoalManager system.

    Attributes:
        description (str): A brief description of the goal.
        priority (int): The priority level of the goal.
        parent (Optional[Goal]): The parent goal, if any.
        children (List[Goal]): A list of sub-goals under this goal.
        prerequisites (List[Goal]): A list of prerequisite goals.
        progress (float): The current progress of the goal (0.0 to 1.0).
        created_at (datetime): Timestamp when the goal was created.
        last_updated (datetime): Timestamp of the last update to the goal.
        completion_count (int): Number of times the goal has been completed.
        config_manager (ConfigManager): Manager for configuration settings.
        logger: Logger instance for logging goal-related events.
        consciousness_stream: Associated consciousness stream (if any).
    """
    def __init__(self, description: str, priority: int, parent: Optional['Goal'] = None, config_manager: ConfigManager = None):
        """
        Initializes a new Goal instance.

        Args:
            description (str): Description of the goal.
            priority (int): Priority level of the goal.
            parent (Optional[Goal], optional): Parent goal. Defaults to None.
            config_manager (ConfigManager, optional): Configuration manager. Defaults to None.
        """
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('Goal') if self.config_manager else None
        self.consciousness_stream = None
        self.description = description
        self.priority = priority
        self.parent = parent
        self.children = []
        self.prerequisites = []
        self.progress = 0.0
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.completion_count = 0

        if self.logger:
            self.logger.debug(f"Goal created: {self.description}")

    def update_progress(self, amount: float):
        """
        Updates the progress of the goal.

        Args:
            amount (float): Amount to increment the progress by (0.0 to 1.0).
        """
        self.progress = min(1.0, max(0.0, self.progress + amount))
        self.last_updated = datetime.now()
        if self.logger:
            self.logger.debug(f"Goal progress updated: {self.description} - {self.progress:.2f}")

    def is_completed(self) -> bool:
        """
        Checks if the goal is completed.

        Returns:
            bool: True if progress is 1.0 or more, else False.
        """
        return self.progress >= 1.0

    def add_child(self, child: 'Goal'):
        """
        Adds a sub-goal to this goal, preventing circular dependencies.
        """
        if child == self or self._has_circular_dependency(child):
            if self.logger:
                self.logger.warning(f"Circular dependency detected. Cannot add {child.description} as a child of {self.description}.")
            return
        self.children.append(child)
        if self.logger:
            self.logger.debug(f"Child goal added to {self.description}: {child.description}")

    def add_prerequisite(self, prerequisite: 'Goal'):
        """
        Adds a prerequisite goal to this goal, preventing circular dependencies.
        """
        if prerequisite == self or self._has_circular_dependency(prerequisite):
            if self.logger:
                self.logger.warning(f"Circular dependency detected. Cannot add {prerequisite.description} as a prerequisite of {self.description}.")
            return
        self.prerequisites.append(prerequisite)
        if self.logger:
            self.logger.debug(f"Prerequisite added to {self.description}: {prerequisite.description}")

    def _has_circular_dependency(self, goal: 'Goal') -> bool:
        """
        Checks if adding a goal would create a circular dependency.
        """
        # Check if this goal is already in the dependency chain of the other goal
        if goal == self:
            return True
        for child in self.children:
            if child._has_circular_dependency(goal):
                return True
        return False


@singleton
class GoalManager(GoalManagerInterface):
    """
    Manages the lifecycle, prioritization, and relationships of goals.

    Attributes:
        thread_orchestrator (ThreadOrchestrator): Manages threading tasks.
        memory_system: System managing memory (semantic and episodic).
        provider_manager (ProviderManager): Manages external providers for tasks.
        config_manager (ConfigManager): Manages configuration settings.
        logger: Logger instance for logging goal manager events.
        max_goals (int): Maximum number of active goals.
        goal_review_interval (int): Interval in seconds for reviewing goals.
        goals (List[Goal]): List of current active goals.
        goal_graph (nx.DiGraph): Directed graph representing goal relationships.
        goal_templates (Dict[str, str]): Templates for generating goals.
        _last_goal_review_time (datetime): Timestamp of the last goal review.
        _last_maintenance_time (float): Timestamp of the last maintenance task.
        _last_goal_count (int): Number of goals at the last maintenance.
    """
    def __init__(self, thread_orchestrator: ThreadOrchestrator, memory_system, provider_manager: ProviderManager, config_manager: ConfigManager):
        """
        Initializes the GoalManager.

        Args:
            thread_orchestrator (ThreadOrchestrator): Thread orchestrator instance.
            memory_system: Memory system instance.
            provider_manager (ProviderManager): Provider manager instance.
            config_manager (ConfigManager): Configuration manager instance.
        """
        self.thread_orchestrator = thread_orchestrator
        self.memory_system = memory_system
        self.provider_manager = provider_manager
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('GoalManager')

        goal_manager_config = config_manager.get_subsystem_config('goal_manager')
        self.max_goals = goal_manager_config.get('max_goals', 10)
        self.goal_review_interval = goal_manager_config.get('goal_review_interval', 300)

        self.goals: List[Goal] = []
        self.goal_graph = nx.DiGraph()
        self.goal_templates = self._initialize_goal_templates()
        self._last_goal_review_time = datetime.now()
        self._last_maintenance_time = time.time()
        self._last_goal_count = len(self.goals)
        self.logger.info(f"GoalManager initialized with max_goals: {self.max_goals}, goal_review_interval: {self.goal_review_interval}")

    def set_consciousness_stream(self, consciousness_stream):
        """
        Sets the consciousness stream for the GoalManager.

        This allows the GoalManager to interact with the consciousness stream,
        enabling it to add thoughts and receive updates.
        
        Args:
            consciousness_stream: The consciousness stream object to be used.
        """
        self.consciousness_stream = consciousness_stream
        if self.logger:
            self.logger.debug("Consciousness stream set for GoalManager")

    async def initialize(self):
        """
        Initializes the GoalManager by restoring goals from backup or setting default goals,
        and optimizing the goal structure.
        """
        if self.logger:
            self.logger.info("Initializing GoalManager")
        try:
            await self.restore_goals_from_backup()
            if not self.goals:
                await self.initialize_default_goals()
            await self.optimize_goal_structure()
            self._last_maintenance_time = time.time()
            self._last_goal_count = len(self.goals)
            if self.logger:
                self.logger.info("GoalManager initialization complete")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Initialization failed: {str(e)}")

    async def initialize_default_goals(self):
        """
        Initialize default goals for the system.
        """
        if self.logger:
            self.logger.info("Initializing default goals")
        default_goals = [
            ("Understand and adapt to the current environment", 5),
            ("Provide helpful and accurate responses", 4),
            ("Continuously learn and improve knowledge", 3)
        ]
        for description, priority in default_goals:
            await self.add_goal(description, priority)
        if self.logger:
            self.logger.info(f"Initialized {len(default_goals)} default goals")

    def _initialize_goal_templates(self) -> Dict[str, str]:
        """
        Initializes goal templates used for generating new goals.

        Returns:
            Dict[str, str]: A dictionary of goal templates.
        """
        return {
            "respond": "Respond to: {input}",
            "investigate": "Investigate: {topic}",
            "learn": "Learn about: {subject}",
            "solve": "Solve problem: {problem}",
            "create": "Create: {item}",
            "improve": "Improve: {aspect}",
            "analyze": "Analyze: {target}"
        }

    async def _find_most_relevant_goal(self, description: str, goals: List[Goal]) -> Goal:
        """
        Finds the most relevant goal based on a description.

        Args:
            description (str): The description to match against goals.
            goals (List[Goal]): The list of goals to consider.

        Returns:
            Goal: The most relevant goal.
        """
        relevance_scores = []
        for goal in goals:
            relevance = await self._compute_relevance(goal, {'content': description}, {})
            relevance_scores.append((goal, relevance))
        if relevance_scores:
            return max(relevance_scores, key=lambda x: x[1])[0]
        else:
            return None

    @cpu_bound_task
    async def process_external_input(self, input_data: Dict[str, Any]):
        """
        Process external input that might affect the current goals.

        This method allows the GoalManager to react to external events or information
        that could impact goal priorities, progress, or relevance.

        Args:
            input_data (Dict[str, Any]): A dictionary containing the external input.
        """
        if self.logger:
            self.logger.debug(f"Processing external input: {input_data}")

        try:
            # Generate a thought based on the input
            thought = {
                "type": "EXTERNAL_INPUT",
                "content": input_data.get('content', ''),
                "timestamp": time.time()
            }

            # Directly attempt to generate a new goal
            new_goal = await self._generate_new_goals(thought, {})
            if new_goal:
                if self.logger:
                    self.logger.debug(f"New goal generated from external input: {new_goal.description}")

            # Update goals based on the new input
            await self.update_goals(thought, {})

            # Check if any goals have become irrelevant or need modification
            await self._reassess_goals_relevance(input_data)

            # Trigger maintenance if necessary
            await self.trigger_maintenance()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing external input: {str(e)}")

    async def _reassess_goals_relevance(self, context: Dict[str, Any]): 
        """
        Reassess the relevance of current goals based on new context.

        This method checks if any goals have become irrelevant or need modification
        due to changes in the system's context or external factors.

        Args:
            context (Dict[str, Any]): The new context to consider.
        """
        if self.logger:
            self.logger.debug("Reassessing goals relevance based on new context")
        goals_to_remove = []
        try:
            for goal in self.goals:
                relevance = await self._compute_relevance(goal, {"content": str(context)}, {})
                if relevance < 0.2:  # If relevance is very low
                    goals_to_remove.append(goal)
                elif relevance < 0.5:  # If relevance is moderate
                    # Attempt to modify the goal to make it more relevant
                    modified_description = await self._modify_goal_for_relevance(goal, context)
                    if modified_description:
                        goal.description = modified_description
                        if self.logger:
                            self.logger.debug(f"Modified goal for relevance: {goal.description}")

            for goal in goals_to_remove:
                await self.remove_goal(goal)
                if self.logger:
                    self.logger.debug(f"Removed irrelevant goal: {goal.description}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reassessing goals relevance: {str(e)}")

    @io_bound_task
    async def _modify_goal_for_relevance(self, goal: Goal, context: Dict[str, Any]) -> Optional[str]: 
        """
        Attempt to modify a goal to make it more relevant to the current context.

        Args:
            goal (Goal): The goal to modify.
            context (Dict[str, Any]): The current context.

        Returns:
            Optional[str]: A modified goal description if successful, None otherwise.
        """
        prompt = (
            f"Given the following goal and new context, suggest a modification to the goal to make it more relevant:\n\n"
            f"Goal: {goal.description}\n"
            f"New context: {context}\n\n"
            f"Provide a modified goal description that maintains the original intent but is more relevant to the new context.\n"
            f"If no modification is necessary or possible, respond with 'No modification needed'."
        )

        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in goal modification and relevance assessment."},
                    {"role": "user", "content": prompt}
                ],
                llm_call_type="goal_modification"
            )

            if response and response.strip().lower() != "no modification needed":
                if self.logger:
                    self.logger.debug(f"Goal modified: {response.strip()}")
                return response.strip()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error modifying goal for relevance: {str(e)}")
        return None

    async def add_goal(self, description: str, priority: int, parent: Optional['Goal'] = None) -> Optional['Goal']:
        """
        Add a new goal to the goal manager.

        Args:
            description (str): Description of the goal.
            priority (int): Priority level of the goal.
            parent (Optional[Goal], optional): Parent goal. Defaults to None.

        Returns:
            Optional[Goal]: The newly added goal if successful, else None.
        """
        try:
            goal = Goal(description, priority, parent, self.config_manager) 
            self.goals.append(goal)
            self.goal_graph.add_node(goal)
            if parent:
                parent.add_child(goal)
                self.goal_graph.add_edge(parent, goal)
            self.goals.sort(key=lambda x: x.priority, reverse=True)
            if self.logger:
                self.logger.debug(f"New goal added: {description}")
            if self.memory_system:
                await self.memory_system.process_input(f"New goal: {description}")
            if self.memory_system and self.memory_system.long_term_semantic:
                await self.memory_system.long_term_semantic.add(description, ["goal", "active"])
            return goal
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding goal '{description}': {str(e)}")
            return None

    async def remove_goal(self, goal: Goal):
        """
        Remove a goal from the goal manager.

        Args:
            goal (Goal): The goal to remove.
        """
        try:
            self.goals.remove(goal)
            self.goal_graph.remove_node(goal)
            if self.logger:
                self.logger.debug(f"Goal removed: {goal.description}")
            if self.memory_system:
                await self.memory_system.process_input(f"Goal removed: {goal.description}")
                if self.memory_system.long_term_semantic:
                    await self.memory_system.long_term_semantic.add(goal.description, ["goal", "completed"])
        except ValueError:
            if self.logger:
                self.logger.warning(f"Attempted to remove non-existent goal: {goal.description}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error removing goal '{goal.description}': {str(e)}")

    async def update_goals(self, thought: Dict[str, Any], state: Dict[str, Any]):
        """
        Update the current goals based on new thoughts and state.

        Args:
            thought (Dict[str, Any]): The current thought.
            state (Dict[str, Any]): The current state.
        """
        if self.logger:
            self.logger.debug("Updating goals based on new thought and state")
        try:
            await self._evaluate_current_goals(thought, state)
            new_goal = await self._generate_new_goals(thought, state)
            if new_goal:
                if self.logger:
                    self.logger.debug(f"New goal added: {new_goal.description}")
            self._prune_completed_goals()
            await self._resolve_goal_conflicts()
            self._apply_goal_decay()
            self._update_goal_priorities()
            goals_state = await self.get_current_goals()
            if self.memory_system and hasattr(self.memory_system, 'intermediate'):
                await self.memory_system.intermediate.add({"current_goals": goals_state})
            await self._reflect_on_goals()

            # Log the current goals after update at debug level to reduce verbosity
            if self.logger:
                self.logger.debug(f"Current goals after update: {goals_state}")

            # Trigger consciousness stream update with new goals
            if self.consciousness_stream:
                await self.consciousness_stream.add_thought({
                    "type": "GOAL_UPDATE",
                    "content": f"Goals updated: {goals_state}",
                    "timestamp": time.time()
                })
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating goals: {str(e)}")

    @cpu_bound_task
    def evaluate_goal_progress(self, goal: Goal) -> float:
        """
        Evaluate the progress of a specific goal.

        Args:
            goal (Goal): The goal to evaluate.

        Returns:
            float: The progress of the goal.
        """
        return goal.progress

    async def get_current_goals(self) -> List[Dict[str, Any]]:
        """
        Retrieve the current list of goals.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the current goals.
        """
        try:
            return [
                {"description": goal.description, "priority": goal.priority, "progress": goal.progress} 
                for goal in self.goals
            ]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving current goals: {str(e)}")
            return []

    def _evaluate_current_goals(self, thought: Dict[str, Any], state: Dict[str, Any]):
        """
        Evaluate the current goals based on thoughts and state.
        Adjust the review interval based on the importance and urgency of active goals.
        """
        current_time = datetime.now()
        if (current_time - self._last_goal_review_time).total_seconds() < self.goal_review_interval:
            if self.logger:
                self.logger.debug("Goal review interval not reached yet")
            return
        self._last_goal_review_time = current_time

        try:
            urgent_goals = [goal for goal in self.goals if goal.priority >= 8]
            if urgent_goals:
                # If urgent goals are present, review more frequently
                self.goal_review_interval = max(300, self.goal_review_interval / 2)
            else:
                # If no urgent goals, increase the interval
                self.goal_review_interval = min(900, self.goal_review_interval * 1.2)

            # Evaluate each goal's relevance and progress
            for goal in self.goals:
                relevance = self._compute_relevance(goal, thought, state)
                goal.update_progress(relevance * 0.1)

            if self.memory_system:
                asyncio.create_task(self.memory_system.process_input(f"Goal progress update: {goal.description} - {goal.progress:.2f}"))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error evaluating current goals: {str(e)}")

    async def _generate_new_goals(self, thought: Dict[str, Any], state: Dict[str, Any]):
        """
        Generate new goals based on thoughts and state.

        Args:
            thought (Dict[str, Any]): The current thought.
            state (Dict[str, Any]): The current state.

        Returns:
            Optional[Goal]: The newly generated goal if any.
        """
        if len(self.goals) >= self.max_goals:
            if self.logger:
                self.logger.debug("Maximum number of goals reached; cannot add new goals")
            return None

        new_goal = None
        content = thought.get('content', '')

        try:
            # Handle different thought types
            if thought['type'] == ThoughtType.EXTERNAL_INPUT.value:
                if 'learn' in content.lower():
                    new_goal = f"Learn about {content.split('learn about')[-1].strip()}"
                else:
                    new_goal = self._generate_goal_from_template(thought)
            elif thought['type'] == ThoughtType.OBSERVATION.value:
                new_goal = f"Investigate observation: {content[:50]}"
            elif thought['type'] == ThoughtType.REFLECTION.value:
                new_goal = f"Reflect on and act upon: {content[:50]}"
            elif thought['type'] == ThoughtType.ACTION.value:
                new_goal = f"Follow up on action: {content[:50]}"

            # If no direct match, try other methods
            if not new_goal:
                new_goal = await self.generate_goal_from_context(content)
            if not new_goal:
                new_goal = await self._generate_goal_from_thought(thought)

            if new_goal:
                priority = await self._compute_initial_priority(new_goal)
                goal = await self.add_goal(new_goal, priority)
                if goal:
                    if self.logger:
                        self.logger.debug(f"New goal added: {new_goal} with priority {priority}")

                    # Add to episodic memory
                    if self.memory_system and self.memory_system.long_term_episodic:
                        context = await self.memory_system.long_term_episodic.context_retrieval.get_context_vector()
                        await self.memory_system.long_term_episodic.add(
                            {"content": f"Generated new goal: {new_goal}", "type": "goal_generation"},
                            context
                        )

                    # Add to semantic memory
                    if self.memory_system and self.memory_system.long_term_semantic:
                        await self.memory_system.long_term_semantic.add(new_goal, ["goal", "active"])

                    # Trigger consciousness stream update with new goal
                    if self.consciousness_stream:
                        await self.consciousness_stream.add_thought({
                            "type": ThoughtType.GOAL.value,
                            "content": f"New goal generated: {new_goal}",
                            "timestamp": time.time()
                        })

                    return goal
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating new goals: {str(e)}")

        return None

    @io_bound_task
    async def _generate_goal_from_thought(self, thought: Dict[str, Any]) -> Optional[str]:
        """
        Generate a goal based on a thought.

        Args:
            thought (Dict[str, Any]): The thought to generate a goal from.

        Returns:
            Optional[str]: The generated goal description if successful, else None.
        """
        content = thought.get('content', '')
        if isinstance(content, dict):
            content = content.get('content', '')

        prompt = (
            f"Based on the following thought, generate a SMART goal: {content}"
        )

        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with generating meaningful and relevant goals based on the given thought. The goal should be specific, measurable, achievable, relevant, and time-bound (SMART)."},
                    {"role": "user", "content": prompt}
                ],
                llm_call_type="goal_generation"
            )

            if response:
                if self.logger:
                    self.logger.debug(f"Generated goal from thought: {response.strip()}")
                return response.strip()
            else:
                if self.logger:
                    self.logger.warning("Failed to generate goal from thought")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating goal from thought: {str(e)}")
        return None

    def _generate_goal_from_template(self, thought: Dict[str, Any]) -> Optional[str]:
        """
        Generate a goal from predefined templates based on thought type.

        Args:
            thought (Dict[str, Any]): The thought to generate a goal from.

        Returns:
            Optional[str]: The generated goal description if successful, else None.
        """
        thought_type = thought.get('type', '').lower()
        content = thought.get('content', '')

        try:
            if thought_type == 'external_input':
                return self.goal_templates["respond"].format(input=content[:50])
            elif thought_type == 'reflection':
                return self.goal_templates["investigate"].format(topic=content[:50])
            elif thought_type == 'observation':
                return self.goal_templates["analyze"].format(target=content[:50])
            elif thought_type == 'action':
                return self.goal_templates["improve"].format(aspect=content[:50])
            elif thought_type == 'goal':
                return self.goal_templates["create"].format(item=f"plan for {content[:50]}")

            # If no template matches, extract a goal from the content
            words = content.lower().split()
            if "goal" in words:
                goal_index = words.index("goal")
                if goal_index < len(words) - 1:
                    return f"Goal: {' '.join(words[goal_index+1:])}"
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating goal from template: {str(e)}")
        return None

    @cpu_bound_task
    async def _compute_initial_priority(self, goal: str) -> int:
        """
        Compute the initial priority of a new goal.

        Args:
            goal (str): The goal description.

        Returns:
            int: The computed priority.
        """
        base_priority = 5
        if "urgent" in goal.lower():
            base_priority += 3
        if "important" in goal.lower():
            base_priority += 2
        return min(base_priority, 10)

    def _prune_completed_goals(self):
        completed_goals = [goal for goal in self.goals if goal.is_completed()]
        for goal in completed_goals:
            asyncio.create_task(self.remove_goal(goal))
            goal.completion_count += 1
            if self.memory_system and self.memory_system.long_term_episodic:
                asyncio.create_task(
                    self.memory_system.long_term_episodic.add(
                        {"content": f"Completed and removed goal: {goal.description}", "type": "goal_completion"},
                        self.memory_system.long_term_episodic.context_retrieval.get_context_vector()
                    )
                )
        if len(self.goals) > self.max_goals:
            self.goals = self.goals[:self.max_goals]
            if self.logger:
                self.logger.debug(f"Pruned goals to maintain a maximum of {self.max_goals} active goals")

    async def _compute_relevance(self, goal: 'Goal', thought: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Compute the relevance of a goal to the current thought and state.

        Args:
            goal (Goal): The goal to compute relevance for.
            thought (Dict[str, Any]): The current thought.
            state (Dict[str, Any]): The current state.

        Returns:
            float: Relevance score between 0.0 and 1.0.
        """
        relevance = 0.0
        try:
            if goal.description.lower() in thought['content'].lower():
                relevance += 0.5
            if any(goal.description.lower() in str(v).lower() for v in state.values()):
                relevance += 0.3

            # Use asynchronous pattern_completion method
            if self.memory_system and self.memory_system.long_term_semantic:
                related_concepts = await self.memory_system.long_term_semantic.pattern_completion(goal.description, threshold=0.6)
                if any(concept[0].lower() in thought['content'].lower() for concept in related_concepts):
                    relevance += 0.2
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing relevance for goal '{goal.description}': {str(e)}")
        return relevance

    @cpu_bound_task
    def get_highest_priority_goal(self) -> Optional[Goal]:
        """
        Get the highest priority goal.

        Returns:
            Optional[Goal]: The goal with the highest priority if any.
        """
        return self.goals[0] if self.goals else None

    @cpu_bound_task
    def get_goal_context(self) -> str:
        """
        Get the context of the current goals.

        Returns:
            str: A string representing the context of current goals.
        """
        context = "Current Goals Context:\n"
        try:
            for goal in self.goals[:5]:
                if self.memory_system and self.memory_system.long_term_semantic:
                    semantic_info = self.memory_system.long_term_semantic.query(goal.description, 3)
                    context += f"Goal: {goal.description}\n"
                    context += f"  Priority: {goal.priority}, Progress: {goal.progress:.2f}\n"
                    context += f"  Related concepts: {', '.join(concept for concept, _ in semantic_info)}\n"

                if self.memory_system and self.memory_system.long_term_episodic:
                    episodes = self.memory_system.long_term_episodic.get_relevant_episodes(2)
                    relevant_episodes = [ep for ep in episodes if goal.description.lower() in ep['content'].lower()]
                    if relevant_episodes:
                        context += f"  Related memories: {', '.join(ep['content'] for ep in relevant_episodes[:2])}\n"

                if goal.prerequisites:
                    context += f"  Prerequisites: {', '.join(prereq.description for prereq in goal.prerequisites)}\n"
                if goal.children:
                    context += f"  Subgoals: {', '.join(child.description for child in goal.children)}\n"
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating goal context: {str(e)}")
        return context

    async def _resolve_goal_conflicts(self):
        """
        Identify and resolve conflicts or redundancies among goals.
        """
        if len(self.goals) < 2:
            return  # No conflicts possible with less than 2 goals

        goals_descriptions = [f"{i+1}. {goal.description}" for i, goal in enumerate(self.goals)]
        goals_text = "\n".join(goals_descriptions)

        prompt = f"""Analyze the following list of goals and identify any conflicts or redundancies:

{goals_text}

For each conflict or redundancy, provide:
1. The numbers of the conflicting goals
2. A brief explanation of the conflict
3. A suggestion for resolution (e.g., merge, remove one, modify one)

Format your response as a list of conflicts, like this:
- Conflict between goals 1 and 3: [Explanation]
Suggestion: [Your suggestion]
- Redundancy between goals 2 and 5: [Explanation]
Suggestion: [Your suggestion]

If there are no conflicts or redundancies, simply respond with "No conflicts detected."
"""

        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in goal analysis and conflict resolution."},
                    {"role": "user", "content": prompt}
                ],
                llm_call_type="goal_conflict_resolution"
            )

            if response.strip().lower() == "no conflicts detected.":
                if self.logger:
                    self.logger.debug("No goal conflicts detected.")
                return

            conflict_resolutions = self._parse_conflict_response(response)

            for resolution in conflict_resolutions:
                await self._apply_resolution(resolution)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error resolving goal conflicts: {str(e)}")

    @cpu_bound_task
    def _parse_conflict_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the conflict resolution response.

        Args:
            response (str): The response containing conflict resolutions.

        Returns:
            List[Dict[str, Any]]: A list of conflict resolutions.
        """
        lines = response.split('\n')
        resolutions = []
        current_resolution = {}

        for line in lines:
            line = line.strip()
            if line.startswith('- Conflict') or line.startswith('- Redundancy'):
                if current_resolution:
                    resolutions.append(current_resolution)
                current_resolution = {'type': 'conflict' if 'Conflict' in line else 'redundancy'}
                try:
                    parts = line.split(':', 1)
                    goals_part = parts[0].split('between goals')[1].strip()
                    goal_numbers = [int(num.strip()) for num in goals_part.split('and')]
                    current_resolution['goals'] = goal_numbers
                    current_resolution['explanation'] = parts[1].strip()
                except (IndexError, ValueError) as parse_error:
                    if self.logger:
                        self.logger.error(f"Error parsing conflict line '{line}': {str(parse_error)}")
            elif line.startswith('Suggestion:'):
                current_resolution['suggestion'] = line.split('Suggestion:')[1].strip()

        if current_resolution:
            resolutions.append(current_resolution)

        return resolutions

    @cpu_bound_task
    async def _apply_resolution(self, resolution: Dict[str, Any]):
        """
        Apply a resolution to a conflict or redundancy.

        Args:
            resolution (Dict[str, Any]): The resolution details.
        """
        try:
            goal1, goal2 = self.goals[resolution['goals'][0] - 1], self.goals[resolution['goals'][1] - 1]

            if 'merge' in resolution['suggestion'].lower():
                new_description = f"Combined goal: {goal1.description} and {goal2.description}"
                new_priority = max(goal1.priority, goal2.priority)
                new_goal = await self.add_goal(new_description, new_priority)
                if new_goal:
                    new_goal.progress = (goal1.progress + goal2.progress) / 2
                    await self.remove_goal(goal1)
                    await self.remove_goal(goal2)
                    if self.logger:
                        self.logger.debug(f"Merged goals: '{goal1.description}' and '{goal2.description}' into '{new_description}'")

            elif 'remove' in resolution['suggestion'].lower():
                goal_to_remove = goal1 if goal1.priority < goal2.priority else goal2
                await self.remove_goal(goal_to_remove)
                if self.logger:
                    self.logger.debug(f"Removed goal: '{goal_to_remove.description}' due to conflict")

            elif 'modify' in resolution['suggestion'].lower():
                goal_to_modify = goal1 if goal1.priority < goal2.priority else goal2
                modified_description = f"Modified: {goal_to_modify.description} (to resolve conflict with {goal1.description if goal_to_modify == goal2 else goal2.description})"
                goal_to_modify.description = modified_description
                if self.logger:
                    self.logger.debug(f"Modified goal: '{goal_to_modify.description}' to resolve conflict")

            else:
                if self.logger:
                    self.logger.warning(f"Unrecognized conflict resolution suggestion: {resolution['suggestion']}")
        except IndexError:
            if self.logger:
                self.logger.error(f"Invalid goal numbers in resolution: {resolution}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying resolution '{resolution}': {str(e)}")

    def _apply_goal_decay(self):
        """
        Apply decay to the priorities of goals over time.
        """
        current_time = datetime.now()
        try:
            for goal in self.goals:
                time_since_update = (current_time - goal.last_updated).total_seconds()
                decay_factor = 1 / (1 + time_since_update / 86400)  # 86400 seconds in a day
                goal.priority *= decay_factor
                if self.logger:
                    self.logger.debug(f"Applied decay to goal '{goal.description}': new priority {goal.priority:.2f}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying goal decay: {str(e)}")

    def _update_goal_priorities(self):
        """
        Update the priorities of all goals based on dynamic factors.
        """
        try:
            for goal in self.goals:
                goal.priority = self._compute_dynamic_priority(goal)
            self.goals.sort(key=lambda x: x.priority, reverse=True)
            if self.logger:
                self.logger.debug("Updated goal priorities based on dynamic factors")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating goal priorities: {str(e)}")

    @cpu_bound_task
    def _compute_dynamic_priority(self, goal: Goal) -> float:
        """
        Compute the dynamic priority of a goal.

        Args:
            goal (Goal): The goal to compute priority for.

        Returns:
            float: The dynamic priority.
        """
        base_priority = goal.priority
        try:
            time_factor = 1 / (1 + (datetime.now() - goal.created_at).days)
            progress_factor = 1 - goal.progress
            prereq_factor = 1 if all(prereq.is_completed() for prereq in goal.prerequisites) else 0.5
            dynamic_priority = base_priority * time_factor * progress_factor * prereq_factor
            return dynamic_priority
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing dynamic priority for goal '{goal.description}': {str(e)}")
            return base_priority  # Fallback to base priority

    @cpu_bound_task
    async def _reflect_on_goals(self):
        """
        Reflect on completed goals and suggest improvements.
        """
        try:
            completed_goals = [goal for goal in self.goals if goal.is_completed()]
            if completed_goals:
                reflection = f"Completed {len(completed_goals)} goals. Consider generating new high-level goals."
                if self.memory_system:
                    await self.memory_system.process_input(reflection)

                response = await self.provider_manager.generate_response(
                    messages=[
                        {"role": "system", "content": "Reflect on the current state of goals and suggest improvements"},
                        {"role": "user", "content": f"Current goals: {self.get_current_goals()}\nCompleted goals: {completed_goals}"}
                    ],
                    llm_call_type="goal_reflection"
                )
                if self.logger:
                    self.logger.debug(f"Goal reflection: {response}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reflecting on goals: {str(e)}")

    async def get_goal_forecast(self, goal_description: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the forecast for a specific goal based on its description.

        This method allows other parts of the system to query the forecast
        for a particular goal without having to generate forecasts for all goals.
        It's useful for targeted decision-making or when providing information
        about a specific goal to the user or other system components.

        Args:
            goal_description (str): The description of the goal to forecast.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the forecast information
            for the specified goal, including:
                - 'description': The goal description (str)
                - 'estimated_completion': The estimated completion date (datetime)
                - 'current_progress': The current progress as a float between 0 and 1

            Returns None if no forecast is found for the given goal description.

        Example:
            forecast = goal_manager.get_goal_forecast("Improve code documentation")
            if forecast:
                print(f"Estimated completion: {forecast['estimated_completion']}")
            else:
                print("No forecast available for this goal")
        """
        try:
            forecasted_goals = await self.forecast_goals()
            for forecast in forecasted_goals:
                if forecast['description'] == goal_description:
                    return forecast
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting forecast for goal '{goal_description}': {str(e)}")
        return None

    async def forecast_goals(self) -> List[Dict[str, Any]]:
        """
        Forecast the estimated completion times for all current goals.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing forecast information for each goal.
        """
        forecasted_goals = []
        try:
            for goal in self.goals:
                if goal.progress < 1.0:
                    estimated_completion = self._estimate_completion_time(goal)
                    forecasted_goals.append({
                        "description": goal.description,
                        "estimated_completion": estimated_completion,
                        "current_progress": goal.progress
                    })
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error forecasting goals: {str(e)}")
        return forecasted_goals

    async def _adjust_goals_based_on_forecast(self, forecasted_goals: List[Dict[str, Any]]):
        """
        Adjust goals based on their forecasted completion times.

        Args:
            forecasted_goals (List[Dict[str, Any]]): The list of forecasted goals.
        """
        try:
            for forecast in forecasted_goals:
                goal = next((g for g in self.goals if g.description == forecast['description']), None)
                if not goal:
                    continue
                days_to_completion = (forecast['estimated_completion'] - datetime.now()).days

                if days_to_completion > 30 and goal.priority < 8:
                    goal.priority += 1
                    if self.logger:
                        self.logger.debug(f"Increased priority of goal '{goal.description}' due to long estimated completion time")

                if days_to_completion < 7 and goal.progress < 0.5:
                    new_subgoal_desc = f"Accelerate progress on: {goal.description}"
                    await self.add_goal(new_subgoal_desc, goal.priority + 1)
                    if self.logger:
                        self.logger.debug(f"Added subgoal '{new_subgoal_desc}' to address slow progress")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adjusting goals based on forecast: {str(e)}")

    def _estimate_completion_time(self, goal: Goal) -> datetime:
        """
        Estimate the completion time for a goal.

        Args:
            goal (Goal): The goal to estimate completion time for.

        Returns:
            datetime: The estimated completion datetime.
        """
        try:
            if goal.progress == 0:
                return datetime.now() + timedelta(days=7)  # Default estimate
            time_since_creation = (datetime.now() - goal.created_at).total_seconds()
            if time_since_creation == 0:
                time_since_creation = 1  # Prevent division by zero
            progress_rate = goal.progress / time_since_creation
            remaining_time_seconds = (1 - goal.progress) / progress_rate
            return datetime.now() + timedelta(seconds=remaining_time_seconds)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error estimating completion time for goal '{goal.description}': {str(e)}")
            return datetime.now() + timedelta(days=7)  # Fallback estimate

    def generate_goal_from_context(self, context: str) -> Optional[Goal]:
        """
        Generate a new goal based on the current context using the memory system.

        Args:
            context (str): The current context.

        Returns:
            Optional[Goal]: The newly generated goal if successful, else None.
        """
        try:
            if self.memory_system and self.memory_system.long_term_semantic:
                related_concepts = self.memory_system.long_term_semantic.pattern_completion(context, threshold=0.7)
                if related_concepts:
                    most_relevant_concept = related_concepts[0][0]
                    new_goal_desc = f"Explore the concept of {most_relevant_concept} in relation to {context}"
                    asyncio.create_task(self.add_goal(new_goal_desc, self._compute_initial_priority(new_goal_desc)))
                    if self.logger:
                        self.logger.debug(f"Generated goal from context: {new_goal_desc}")
                    return None  # The actual goal is added asynchronously
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating goal from context '{context}': {str(e)}")
        return None

    async def analyze_goal_network(self) -> Dict[str, Any]:
        """
        Analyze the current goal network and return various metrics.

        Returns:
            Dict[str, Any]: A dictionary containing analysis metrics such as isolated goals,
            central goals, the longest path, and the most connected goal.
        """
        analysis = {
            "isolated_goals": [],
            "central_goals": [],
            "longest_path": [],
            "most_connected_goal": None
        }

        if not self.goals:
            if self.logger:
                self.logger.warning("No goals in the network to analyze")
            return analysis

        try:
            # Identify isolated goals
            analysis["isolated_goals"] = [goal.description for goal in self.goals if self.goal_graph.degree(goal) == 0]

            # Identify central goals using betweenness centrality
            if len(self.goals) > 1:
                centrality = nx.betweenness_centrality(self.goal_graph)
                analysis["central_goals"] = [goal.description for goal in sorted(centrality, key=centrality.get, reverse=True)[:3]]

            # Find the longest path in the goal graph
            if self.goal_graph and not self.goal_graph.is_empty():
                try:
                    longest_path = nx.dag_longest_path(self.goal_graph)
                    analysis["longest_path"] = [goal.description for goal in longest_path]
                except nx.NetworkXUnfeasible:
                    if self.logger:
                        self.logger.warning("Goal graph contains cycles; cannot determine longest path")

            # Identify the most connected goal
            if self.goals and not self.goal_graph.is_empty():
                most_connected = max(self.goal_graph.degree, key=lambda x: x[1])[0]
                analysis["most_connected_goal"] = most_connected.description
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing goal network: {str(e)}")

        return analysis

    async def optimize_goal_structure(self):
        """
        Optimize the structure of the goal network by addressing isolated goals,
        breaking down long paths, and balancing connections.
        """
        if self.logger:
            self.logger.debug("Starting optimize_goal_structure")
        try:
            analysis = await self.analyze_goal_network()
            if self.logger:
                self.logger.debug(f"Analysis result: {analysis}")

            # Connect isolated goals
            for isolated_goal_desc in analysis['isolated_goals']:
                isolated_goal = next((goal for goal in self.goals if goal.description == isolated_goal_desc), None)
                if isolated_goal:
                    most_relevant_goal = await self._find_most_relevant_goal(isolated_goal.description, self.goals)
                    if most_relevant_goal and most_relevant_goal != isolated_goal:
                        isolated_goal.add_prerequisite(most_relevant_goal)
                        self.goal_graph.add_edge(most_relevant_goal, isolated_goal)
                        if self.logger:
                            self.logger.debug(f"Connected isolated goal '{isolated_goal.description}' to '{most_relevant_goal.description}'")

            # Break down long goal paths
            if len(analysis['longest_path']) > 5:  # If the longest path is more than 5 goals
                middle_index = len(analysis['longest_path']) // 2
                middle_goal_desc = analysis['longest_path'][middle_index]
                middle_goal = next((goal for goal in self.goals if goal.description == middle_goal_desc), None)
                if middle_goal:
                    new_subgoal = await self.add_goal(f"Achieve intermediate step for {middle_goal.description}", middle_goal.priority - 1)
                    if new_subgoal:
                        middle_goal.add_prerequisite(new_subgoal)
                        self.goal_graph.add_edge(new_subgoal, middle_goal)
                        if self.logger:
                            self.logger.debug(f"Added subgoal '{new_subgoal.description}' to break down the path for '{middle_goal.description}'")

            # Balance the goal network (simplified approach)
            if self.goals:  # Add this check
                avg_connections = sum(dict(self.goal_graph.degree()).values()) / len(self.goals)
                for goal in self.goals:
                    if self.goal_graph.degree(goal) > 2 * avg_connections:
                        # Remove some connections
                        connections = list(self.goal_graph.edges(goal))
                        connections_to_remove = connections[int(avg_connections):]
                        self.goal_graph.remove_edges_from(connections_to_remove)
                        if self.logger:
                            self.logger.debug(f"Removed {len(connections_to_remove)} excess connections from goal '{goal.description}'")
            else:
                if self.logger:
                    self.logger.warning("No goals to optimize")

            if self.logger:
                self.logger.info("Goal structure optimized")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error optimizing goal structure: {str(e)}")

    def adjust_goal_priorities_based_on_context(self, context: Dict[str, Any]):
        """
        Adjust goal priorities based on the current context of the system.

        Args:
            context (Dict[str, Any]): The current context.
        """
        try:
            for goal in self.goals:
                context_relevance = self._compute_context_relevance(goal, context)
                goal.priority = (goal.priority + context_relevance) / 2
            self._update_goal_priorities()
            if self.logger:
                self.logger.debug("Adjusted goal priorities based on context")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adjusting goal priorities based on context: {str(e)}")

    def _compute_context_relevance(self, goal: Goal, context: Dict[str, Any]) -> float:
        """
        Compute the relevance of a goal based on the current context.

        Args:
            goal (Goal): The goal to compute relevance for.
            context (Dict[str, Any]): The current context.

        Returns:
            float: The context relevance score.
        """
        relevance = 0.0
        try:
            if 'current_task' in context and goal.description.lower() in context['current_task'].lower():
                relevance += 0.3
            if 'user_input' in context and goal.description.lower() in context['user_input'].lower():
                relevance += 0.4
            if 'system_state' in context:
                state_relevance = sum(goal.description.lower() in str(v).lower() for v in context['system_state'].values()) / len(context['system_state'])
                relevance += state_relevance * 0.3
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing context relevance for goal '{goal.description}': {str(e)}")
        return min(relevance, 1.0)

    async def generate_progress_report(self, include_forecast: bool = False) -> str:
        """
        Generate a detailed progress report on all current goals.

        Args:
            include_forecast (bool, optional): Whether to include forecast information. Defaults to False.

        Returns:
            str: A formatted progress report.
        """
        report = "Goal Progress Report:\n\n"
        try:
            for goal in self.goals:
                report += f"Goal: {goal.description}\n"
                report += f"Priority: {goal.priority:.2f}\n"
                report += f"Progress: {goal.progress:.2%}\n"
                report += f"Time since creation: {(datetime.now() - goal.created_at).days} days\n"
                if goal.prerequisites:
                    report += "Prerequisites:\n"
                    for prereq in goal.prerequisites:
                        report += f"  - {prereq.description} (Completed: {prereq.is_completed()})\n"
                if goal.children:
                    report += "Sub-goals:\n"
                    for child in goal.children:
                        report += f"  - {child.description} (Progress: {child.progress:.2%})\n"
                report += f"Estimated completion: {self._estimate_completion_time(goal)}\n\n"

            if include_forecast:
                report += "\nGoal Forecast:\n"
                forecasted_goals = await self.forecast_goals()
                for forecast in forecasted_goals:
                    report += f"Goal: {forecast['description']}\n"
                    report += f"Estimated completion: {forecast['estimated_completion']}\n"
                    report += f"Current progress: {forecast['current_progress']:.2%}\n\n"
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating progress report: {str(e)}")
        return report

    @io_bound_task
    async def backup_goals(self):
        """
        Backup the current goal structure to persistent storage.
        """
        try:
            goals_data = [
                {
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "created_at": goal.created_at.isoformat(),
                    "last_updated": goal.last_updated.isoformat(),
                    "completion_count": goal.completion_count,
                    "parent": goal.parent.description if goal.parent else None,
                    "children": [child.description for child in goal.children],
                    "prerequisites": [prereq.description for prereq in goal.prerequisites]
                }
                for goal in self.goals
            ]

            await self.config_manager.save_goal_backup(goals_data)
            if self.logger:
                self.logger.debug(f"Goals backed up successfully. Total goals: {len(goals_data)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error backing up goals: {str(e)}")

    @io_bound_task
    async def restore_goals_from_backup(self):
        """
        Restore goals from the last backup.
        """
        try:
            goals_data = await self.config_manager.load_goal_backup()
            if not goals_data:
                if self.logger:
                    self.logger.warning("No goal backup found to restore from.")
                return

            self.goals.clear()
            self.goal_graph.clear()

            # First pass: Create all goals without relationships
            goal_map = {}
            for goal_data in goals_data:
                goal = Goal(
                    description=goal_data["description"],
                    priority=goal_data["priority"]
                )
                goal.progress = goal_data["progress"]
                goal.created_at = datetime.fromisoformat(goal_data["created_at"])
                goal.last_updated = datetime.fromisoformat(goal_data["last_updated"])
                goal.completion_count = goal_data["completion_count"]
                self.goals.append(goal)
                goal_map[goal.description] = goal

            # Second pass: Restore relationships
            for goal_data in goals_data:
                goal = goal_map.get(goal_data["description"])
                if not goal:
                    continue
                if goal_data["parent"]:
                    parent_goal = goal_map.get(goal_data["parent"])
                    if parent_goal:
                        goal.parent = parent_goal
                        self.goal_graph.add_edge(parent_goal, goal)
                for child_desc in goal_data["children"]:
                    child_goal = goal_map.get(child_desc)
                    if child_goal:
                        goal.children.append(child_goal)
                        self.goal_graph.add_edge(goal, child_goal)
                for prereq_desc in goal_data["prerequisites"]:
                    prereq_goal = goal_map.get(prereq_desc)
                    if prereq_goal:
                        goal.prerequisites.append(prereq_goal)

            if self.logger:
                self.logger.debug(f"Goals restored successfully. Total goals: {len(self.goals)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error restoring goals from backup: {str(e)}")

    def adjust_goal_review_interval(self):
        """
        Dynamically adjusts the goal review interval based on system load.
        """
        current_load = self.thread_orchestrator.get_system_load()  # Example method to get system load
        # Adjust interval based on the load
        if current_load > 0.75:  # High system load
            self.goal_review_interval = max(600, self.goal_review_interval * 1.5)  # Increase the interval to reduce frequency
        elif current_load < 0.25:  # Low system load
            self.goal_review_interval = min(300, self.goal_review_interval / 1.5)  # Decrease the interval to review more frequently
        if self.logger:
            self.logger.debug(f"Adjusted goal review interval based on system load: {self.goal_review_interval} seconds")

    async def periodic_maintenance(self):
        """
        Perform periodic maintenance tasks on the goal system.
        """
        self.adjust_goal_review_interval()
        try:
            if self.logger:
                self.logger.debug("Starting periodic maintenance")
            await self.update_goals({}, {})  # Update with empty thought and state
            await self.optimize_goal_structure()

            forecasted_goals = await self.forecast_goals()
            await self._adjust_goals_based_on_forecast(forecasted_goals)

            await self.backup_goals()
            report = await self.generate_progress_report(include_forecast=True)
            if self.logger:
                self.logger.debug(f"Goal Progress Report:\n{report}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during periodic maintenance: {str(e)}")

    async def start_maintenance_task(self, duration: int = 3600):  # default to 1 hour
        """
        Start the periodic maintenance task for a specified duration.

        Args:
            duration (int): The duration in seconds for which to run the maintenance task.
        """
        try:
            self.maintenance_task = self.thread_orchestrator.submit_task(
                self._run_maintenance_for_duration,
                args=(duration,),
                priority=3,
                resources=TaskResource(cpu_cores=1, memory_mb=100)
            )
            if self.logger:
                self.logger.debug(f"Started periodic goal maintenance task for {duration} seconds")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error starting maintenance task: {str(e)}")

    async def _run_maintenance_for_duration(self, duration: int):
        """
        Run maintenance tasks for a specified duration.

        Args:
            duration (int): Duration in seconds to run maintenance tasks.
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            await self.periodic_maintenance()
            await asyncio.sleep(self.goal_review_interval)

    async def trigger_maintenance(self):
        """
        Trigger the maintenance task based on certain conditions.
        """
        current_time = time.time()

        try:
            # Check if it's been more than 24 hours since the last maintenance
            if current_time - self._last_maintenance_time > 86400:  # 86400 seconds = 24 hours
                await self.start_maintenance_task(3600)  # Run for 1 hour
                return

            # Check if there are any high-priority goals that haven't been updated in a while
            for goal in self.goals:
                if goal.priority > 7 and (current_time - goal.last_updated.timestamp()) > 43200:  # 12 hours
                    await self.start_maintenance_task(1800)  # Run for 30 minutes
                    return

            # Check if the number of goals has changed significantly
            if abs(len(self.goals) - self._last_goal_count) > 5:
                await self.start_maintenance_task(1800)  # Run for 30 minutes
                return

            # If none of the conditions are met, just run a quick maintenance
            await self.periodic_maintenance()

            self._last_maintenance_time = current_time
            self._last_goal_count = len(self.goals)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error triggering maintenance: {str(e)}")

    def export_goals(self) -> List[Dict[str, Any]]:
        """
        Export all current goals in a format suitable for external analysis.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a goal
            with all its attributes and relationships.
        """
        exported_goals = []
        try:
            for goal in self.goals:
                exported_goal = {
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "created_at": goal.created_at.isoformat(),
                    "last_updated": goal.last_updated.isoformat(),
                    "completion_count": goal.completion_count,
                    "parent": goal.parent.description if goal.parent else None,
                    "children": [child.description for child in goal.children],
                    "prerequisites": [prereq.description for prereq in goal.prerequisites]
                }
                exported_goals.append(exported_goal)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error exporting goals: {str(e)}")
        return exported_goals
