# modules/Hybrid_Cognitive_Dynamics_Model/SSM/cognitive_temporal_state.py

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple


class CognitiveTemporalStateEnum(Enum):
    """
    Enum representing different cognitive temporal states based on cognitive and emotional factors.
    """
    IMMEDIATE = 1
    REFLECTIVE = 2
    EMOTIONAL = 3
    DEEP_LEARNING = 4
    SOCIAL = 5
    REACTIVE = 6
    ANALYTICAL = 7
    CREATIVE = 8
    FOCUSED = 9  # Newly added state


@dataclass
class CognitiveTemporalStateConfig:
    """
    Configuration for CognitiveTemporalState, encapsulating all configurable parameters.
    """
    alpha: float  # Smoothing factor for scaling updates
    scaling_bounds: Tuple[float, float]  # (min_scaling, max_scaling)
    state_transition_rules: Dict[CognitiveTemporalStateEnum, Tuple[float, float]]
    initial_state: CognitiveTemporalStateEnum = CognitiveTemporalStateEnum.IMMEDIATE
    initial_scaling: float = 1.0


class CognitiveTemporalState:
    """
    Represents the subjective perception of time within the model, managing different cognitive temporal states.
    """

    def __init__(self, config: CognitiveTemporalStateConfig):
        """
        Initializes the CognitiveTemporalState with a specific configuration.
        
        Args:
            config (CognitiveTemporalStateConfig): Configuration parameters.
        """
        self.state = config.initial_state
        self.scaling_factor = config.initial_scaling
        self.config = config
        self.logger = logging.getLogger('CognitiveTemporalState')
        self.logger.setLevel(logging.DEBUG)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """
        Validates the configuration parameters.
        """
        min_bound, max_bound = self.config.scaling_bounds
        if not (0 < self.config.alpha < 1):
            raise ValueError("Alpha must be between 0 and 1 (exclusive).")
        if min_bound >= max_bound:
            raise ValueError("Minimum scaling bound must be less than maximum scaling bound.")
        # Ensure all state_transition_rules have valid ranges
        for state, (lower, upper) in self.config.state_transition_rules.items():
            if not (min_bound <= lower < upper <= max_bound):
                raise ValueError(
                    f"Invalid scaling range for state {state.name}: ({lower}, {upper}) "
                    f"must be within ({min_bound}, {max_bound})."
                )

    def update_state(self, influence_factor: float):
        """
        Updates the CognitiveTemporalState based on an influence factor.
        
        Args:
            influence_factor (float): Factor derived from aLIF network activity.
                                       Positive values may speed up time perception,
                                       negative values may slow it down.
        """
        # Exponential Moving Average update for scaling factor
        new_scaling = (1 - self.config.alpha) * self.scaling_factor + self.config.alpha * (1 + influence_factor)
        # Clamp the scaling factor within configured bounds
        min_scaling, max_scaling = self.config.scaling_bounds
        self.scaling_factor = max(min_scaling, min(new_scaling, max_scaling))
        self.logger.debug(f"CognitiveTemporalState scaling_factor updated to: {self.scaling_factor}")

        # Determine new state based on updated scaling_factor using transition rules
        previous_state = self.state
        for state, (lower, upper) in self.config.state_transition_rules.items():
            if lower <= self.scaling_factor < upper:
                self.state = state
                break
        else:
            # If no rule matches, retain the previous state
            self.logger.warning(
                f"No state transition rule matched for scaling_factor={self.scaling_factor}. "
                f"Retaining previous state: {self.state.name}"
            )

        if self.state != previous_state:
            self.logger.info(f"CognitiveTemporalState changed from {previous_state.name} to {self.state.name}")

    def get_time_scaling(self) -> float:
        """
        Retrieves the current time scaling factor.
        
        Returns:
            float: The current time scaling factor.
        """
        return self.scaling_factor

    def get_current_state(self) -> CognitiveTemporalStateEnum:
        """
        Retrieves the current cognitive temporal state.
        
        Returns:
            CognitiveTemporalStateEnum: The current cognitive temporal state.
        """
        return self.state
