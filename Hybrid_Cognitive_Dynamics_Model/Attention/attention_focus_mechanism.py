# modules/Hybrid_Cognitive_Dynamics_Model/Attention/attention_focus_mechanism.py

import threading
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.Config.config import ConfigManager

if TYPE_CHECKING:
    from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_space_model import StateSpaceModel


# Determine the device for computation (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Singleton decorator
def singleton(cls):
    """
    Singleton decorator to apply thread-safe Singleton pattern.
    """
    cls._instance_lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = cls(*args, **kwargs)
        return cls._instance

    return wrapper


@singleton
class AttentionManager(nn.Module):
    """
    Manages the attention mechanism, integrating with the StateSpaceModel.
    """
    def __init__(self, state_model: 'StateSpaceModel', config_manager: ConfigManager):
        """
        Initializes the AttentionManager with the given state model and configuration manager.

        Args:
            state_model (StateSpaceModel): The state space model instance.
            config_manager (ConfigManager): The configuration manager for retrieving settings.
        """
        super(AttentionManager, self).__init__()
        self.state_model = state_model
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('AttentionManager')

        # Use settings from ConfigManager
        attention_config = config_manager.get_subsystem_config('attention_mechanism')
        num_attention_heads = attention_config.get('num_attention_heads', 4)
        dropout_prob = attention_config.get('dropout_prob', 0.1)
        blending_weights = attention_config.get('blending_weights', [0.7, 0.3])
        activation_multiplier = attention_config.get('activation_multiplier', 2.0)
        activation_function = attention_config.get('activation_function', 'tanh')
        learning_rate = attention_config.get('learning_rate', 0.001)
        attention_mlp_hidden_size = attention_config.get('attention_mlp_hidden_size', 64)
        loss_function_name = attention_config.get('loss_function', 'MSELoss')
        weight_decay = attention_config.get('weight_decay', 1e-5)  # L2 regularization

        self.blending_weights = blending_weights
        self.activation_multiplier = activation_multiplier
        self.activation_function = activation_function

        # Set up loss function
        if loss_function_name == 'MSELoss':
            self.selectivity_gate_loss_fn = nn.MSELoss()
        elif loss_function_name == 'CrossEntropyLoss':
            self.selectivity_gate_loss_fn = nn.CrossEntropyLoss()
        else:
            self.logger.warning(f"Unsupported loss function '{loss_function_name}'. Defaulting to MSELoss.")
            self.selectivity_gate_loss_fn = nn.MSELoss()

        # Set up attention mechanism
        self.ATTENTION_VECTOR_SIZE = self.state_model.dim
        self.attention_mechanism = AttentionFocusMechanism(
            hidden_size=self.ATTENTION_VECTOR_SIZE,
            num_attention_heads=num_attention_heads,
            attention_mlp_hidden_size=attention_mlp_hidden_size,
            dropout_prob=dropout_prob,
            activation_function=activation_function,
            config_manager=self.config_manager
        ).to(DEVICE)

        # Initialize the selectivity gate as a neural network layer
        self.selectivity_gate = SelectivityGate(self.ATTENTION_VECTOR_SIZE).to(DEVICE)
        self.selectivity_gate_optimizer = torch.optim.Adam(
            self.selectivity_gate.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        self.selectivity_gate_scheduler = ReduceLROnPlateau(
            self.selectivity_gate_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        self.logger.info(f"AttentionManager initialized with vector size: {self.ATTENTION_VECTOR_SIZE}")

    def forward(self, saliency: torch.Tensor, current_focus: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention mechanism.

        Args:
            saliency (torch.Tensor): The saliency tensor of shape [batch_size, hidden_size].
            current_focus (torch.Tensor): The current focus tensor of shape [batch_size, hidden_size].

        Returns:
            torch.Tensor: The updated attention tensor of shape [batch_size, hidden_size].
        """
        try:
            self.logger.debug(f"Saliency tensor shape: {saliency.shape}")
            self.logger.debug(f"Current focus tensor shape: {current_focus.shape}")

            # Apply attention mechanism
            attention_output = self.attention_mechanism(saliency, current_focus)

            self.logger.debug(f"Attention output shape: {attention_output.shape}")

            return attention_output
        except Exception as e:
            self.logger.error(f"Error in AttentionManager forward pass: {str(e)}", exc_info=True)
            return current_focus

    def update_attention(self, input_data: Any):
        """
        Updates the attention vector based on the input data.

        Args:
            input_data (Any): The input data to process. Expected to be a dictionary with 'content' key.
        """
        try:
            self.logger.info(f"Updating attention with input data: {input_data}")

            # Compute saliency
            saliency = self.compute_attention_vector(input_data)  # Returns torch.Tensor
            current_focus = self.state_model.attention_focus.unsqueeze(0).to(DEVICE)  # Shape: [1, hidden_size]

            self.logger.debug(f"Saliency shape: {saliency.shape}")
            self.logger.debug(f"Current focus shape: {current_focus.shape}")

            if saliency.shape[0] != 1 or saliency.shape[1] != self.ATTENTION_VECTOR_SIZE:
                self.logger.error(f"Shape mismatch: saliency {saliency.shape}, expected [1, {self.ATTENTION_VECTOR_SIZE}]")
                return

            # Forward pass through attention mechanism
            attention_output = self.forward(saliency, current_focus)

            self.logger.debug(f"Attention output shape: {attention_output.shape}")

            # Blend the current focus with the new attention output
            blend_weight_current, blend_weight_new = self.blending_weights

            with torch.no_grad():
                updated_attention = blend_weight_current * self.state_model.attention_focus + blend_weight_new * attention_output.squeeze(0)
                # Apply non-linear activation to enhance attention differences
                updated_attention = torch.tanh(updated_attention * self.activation_multiplier)

                # Normalize the attention vector
                norm = torch.norm(updated_attention, p=2)
                if norm > 0:
                    updated_attention = updated_attention / norm
                else:
                    updated_attention = torch.ones(self.ATTENTION_VECTOR_SIZE).to(DEVICE) / self.ATTENTION_VECTOR_SIZE

                # Update the state model's attention_focus
                self.state_model.attention_focus = updated_attention.cpu()

            self.logger.info(f"Updated attention vector: {self.state_model.attention_focus}")
            self.logger.info(f"Max attention value: {self.state_model.attention_focus.max().item()}")
            self.logger.debug(f"Updated state model attention focus: {self.state_model.attention_focus}")

            # Train the selectivity gate
            self.train_selectivity_gate(self.state_model.attention_focus, saliency)
        except Exception as e:
            self.logger.error(f"Error in update_attention: {str(e)}", exc_info=True)

    def train_selectivity_gate(self, attention_vector: torch.Tensor, saliency: torch.Tensor):
        """
        Trains the selectivity gate using backpropagation.

        Args:
            attention_vector (torch.Tensor): The updated attention vector.
            saliency (torch.Tensor): The computed saliency vector.
        """
        try:
            self.selectivity_gate.train()
            self.selectivity_gate_optimizer.zero_grad()

            # Ensure tensors are on the correct device
            input_tensor = saliency.to(DEVICE)  # Shape: [1, hidden_size]
            target_tensor = attention_vector.unsqueeze(0).to(DEVICE)  # Shape: [1, hidden_size]

            # Forward pass through the selectivity gate
            output = self.selectivity_gate(input_tensor)

            # Compute loss
            if isinstance(self.selectivity_gate_loss_fn, nn.CrossEntropyLoss):
                # For classification tasks, target needs to be class indices
                # Assuming a single class prediction here as an example
                # Modify as per actual use-case
                target_classes = torch.argmax(target_tensor, dim=1)
                loss = self.selectivity_gate_loss_fn(output, target_classes)
            else:
                # For regression tasks like MSELoss
                loss = self.selectivity_gate_loss_fn(output, target_tensor)

            # Backward pass
            loss.backward()

            # Optimize
            self.selectivity_gate_optimizer.step()

            # Step the scheduler
            self.selectivity_gate_scheduler.step(loss)

            self.logger.debug(f"Selectivity Gate Training Loss: {loss.item()}")
        except Exception as e:
            self.logger.error(f"Error in training selectivity gate: {str(e)}", exc_info=True)

    def compute_attention_vector(self, data: Any) -> torch.Tensor:
        """
        Compute an attention vector based on the input data.

        Args:
            data (Any): The input data to analyze.

        Returns:
            torch.Tensor: The computed attention vector of shape [1, hidden_size].
        """
        self.logger.debug(f"Computing attention vector for data: {data}")
        try:
            if isinstance(data, dict) and 'content' in data:
                content = data['content']
            else:
                content = str(data)

            words = content.split()
            attention_vector = torch.zeros(self.ATTENTION_VECTOR_SIZE, dtype=torch.float32).to(DEVICE)

            # Use TF-IDF-like weighting for words
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            for i, word in enumerate(words):
                if i >= self.ATTENTION_VECTOR_SIZE:
                    break
                # Term frequency * Inverse document frequency (approximated)
                tf = word_counts[word] / len(words)
                idf = math.log(len(words) / word_counts[word]) if word_counts[word] > 0 else 0
                attention_vector[i] = tf * idf

            # Normalize the attention vector
            attention_sum = torch.sum(attention_vector)
            if attention_sum > 0:
                attention_vector = attention_vector / attention_sum
            else:
                attention_vector = torch.ones(self.ATTENTION_VECTOR_SIZE, dtype=torch.float32).to(DEVICE) / self.ATTENTION_VECTOR_SIZE

            self.logger.debug(f"Computed attention vector: {attention_vector}")
            self.logger.debug(f"Max attention value: {attention_vector.max().item()}")
            return attention_vector.unsqueeze(0)  # Shape: [1, hidden_size]
        except Exception as e:
            self.logger.error(f"Error in compute_attention_vector: {str(e)}", exc_info=True)
            # Return a uniform attention vector in case of error
            return torch.ones(1, self.ATTENTION_VECTOR_SIZE, dtype=torch.float32).to(DEVICE) / self.ATTENTION_VECTOR_SIZE


class AttentionFocusMechanism(nn.Module):
    """
    Implements the attention focus mechanism using multi-head attention and an MLP for enhanced processing.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AttentionFocusMechanism, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_mlp_hidden_size: int,
        dropout_prob: float = 0.1,
        activation_function: str = 'tanh',
        config_manager: Optional[ConfigManager] = None
    ):
        if self.__initialized:
            return
        super(AttentionFocusMechanism, self).__init__()
        self.logger = logging.getLogger('AttentionFocusMechanism')
        self.config_manager = config_manager

        self.activation_function = activation_function

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, attention_mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_mlp_hidden_size, hidden_size)
        )

        self.logger.info(
            f"AttentionFocusMechanism initialized with hidden_size: {hidden_size}, "
            f"num_attention_heads: {num_attention_heads}, "
            f"attention_mlp_hidden_size: {attention_mlp_hidden_size}"
        )
        self.__initialized = True

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension into (num_attention_heads, attention_head_size).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size].

        Returns:
            torch.Tensor: Tensor reshaped to [batch_size, num_attention_heads, seq_length, attention_head_size].
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_size]

    def forward(self, saliency: torch.Tensor, current_focus: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention mechanism with MLP processing.

        Args:
            saliency (torch.Tensor): The saliency tensor of shape [batch_size, hidden_size].
            current_focus (torch.Tensor): The current focus tensor of shape [batch_size, hidden_size].

        Returns:
            torch.Tensor: The updated attention tensor of shape [batch_size, hidden_size].
        """
        try:
            self.logger.debug(f"Saliency shape: {saliency.shape}")
            self.logger.debug(f"Current focus shape: {current_focus.shape}")

            mixed_query_layer = self.query(current_focus)
            mixed_key_layer = self.key(saliency)
            mixed_value_layer = self.value(saliency)

            query_layer = self.split_heads(mixed_query_layer)
            key_layer = self.split_heads(mixed_key_layer)
            value_layer = self.split_heads(mixed_value_layer)

            self.logger.debug(f"Query layer shape: {query_layer.shape}")
            self.logger.debug(f"Key layer shape: {key_layer.shape}")
            self.logger.debug(f"Value layer shape: {value_layer.shape}")

            # Compute attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply softmax to get attention probabilities
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Compute context layer
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # Apply output transformation
            output = self.dense(context_layer)
            output = self.dropout(output)
            output = self.layer_norm(output + current_focus)

            # Apply MLP
            output = self.mlp(output)

            # Apply activation function if specified
            if self.activation_function == 'tanh':
                output = torch.tanh(output)
            elif self.activation_function == 'relu':
                output = torch.relu(output)
            elif self.activation_function == 'sigmoid':
                output = torch.sigmoid(output)
            # Add other activation functions if needed

            self.logger.debug(f"Output shape: {output.shape}")

            return output
        except Exception as e:
            self.logger.error(f"Error in AttentionFocusMechanism forward pass: {str(e)}", exc_info=True)
            return current_focus


class SelectivityGate(nn.Module):
    """
    Implements the Selectivity Gate as a simple feedforward neural network.
    """
    def __init__(self, hidden_size: int):
        """
        Initializes the SelectivityGate with a linear layer and activation function.

        Args:
            hidden_size (int): The size of the hidden layer.
        """
        super(SelectivityGate, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.logger = logging.getLogger('SelectivityGate')
        self.logger.info(f"SelectivityGate initialized with hidden_size: {hidden_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the selectivity gate.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, hidden_size].

        Returns:
            torch.Tensor: Output tensor after applying linear transformation and activation.
        """
        try:
            return self.activation(self.linear(x))
        except Exception as e:
            self.logger.error(f"Error in SelectivityGate forward pass: {str(e)}", exc_info=True)
            return x
