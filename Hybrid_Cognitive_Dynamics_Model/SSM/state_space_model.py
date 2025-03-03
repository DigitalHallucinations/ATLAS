# modules/Hybrid_Cognitive_Dynamics_Model/StateSpaceModel/state_space_model.py

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, AsyncIterator
import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import math

# Assuming the HuggingFaceGenerator has a transformer_encode method
from modules.Providers.provider_manager import ProviderManager
from modules.Config.config import ConfigManager

from modules.Hybrid_Cognitive_Dynamics_Model.Time_Processing.time_aware_processing import (
    TimeDecay,
    SpacedRepetition,
    MemoryConsolidationThread,
    MemoryType
)

from modules.Hybrid_Cognitive_Dynamics_Model.Time_Processing.cognitive_temporal_state import (
    CognitiveTemporalState,
    CognitiveTemporalStateConfig,
    CognitiveTemporalStateEnum
)

from modules.Hybrid_Cognitive_Dynamics_Model.Attention.attention_focus_mechanism import AttentionManager

from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_measurement import StateMeasurement


class PyTorchUnscentedKalmanFilter(nn.Module):
    """
    A PyTorch-based Unscented Kalman Filter (UKF) implementation.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        fx: Callable[[torch.Tensor, float], torch.Tensor],
        hx: Callable[[torch.Tensor], torch.Tensor],
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = -1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the PyTorch Unscented Kalman Filter.

        Args:
            dim_x (int): Dimensionality of the state.
            dim_z (int): Dimensionality of the measurement.
            dt (float): Time step.
            fx (Callable): State transition function.
            hx (Callable): Measurement function.
            alpha (float, optional): UKF scaling parameter. Defaults to 0.1.
            beta (float, optional): UKF scaling parameter. Defaults to 2.0.
            kappa (float, optional): UKF scaling parameter. Defaults to -1.0.
            process_noise (float, optional): Process noise covariance scalar. Defaults to 0.01.
            measurement_noise (float, optional): Measurement noise covariance scalar. Defaults to 0.1.
            device (Optional[torch.device], optional): Device to run computations on. Defaults to None.
        """
        super(PyTorchUnscentedKalmanFilter, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.device = device if device else torch.device('cpu')

        # State estimate
        self.x = torch.zeros(dim_x, dtype=torch.float32, device=self.device)
        # Covariance estimate
        self.P = torch.eye(dim_x, dtype=torch.float32, device=self.device) * 0.2

        # Process noise covariance
        self.Q = torch.eye(dim_x, dtype=torch.float32, device=self.device) * process_noise

        # Measurement noise covariance
        self.R = torch.eye(dim_z, dtype=torch.float32, device=self.device) * measurement_noise

        # Sigma points and weights
        self._compute_weights()

    def _compute_weights(self):
        lambda_ = self.alpha ** 2 * (self.dim_x + self.kappa) - self.dim_x
        self.lambda_ = lambda_
        self.gamma = math.sqrt(self.dim_x + self.lambda_)
        self.num_sigma = 2 * self.dim_x + 1

        self.weights_mean = torch.empty(self.num_sigma, dtype=torch.float32, device=self.device)
        self.weights_cov = torch.empty(self.num_sigma, dtype=torch.float32, device=self.device)
        self.weights_mean[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.weights_cov[0] = self.weights_mean[0] + (1 - self.alpha ** 2 + self.beta)
        self.weights_mean[1:] = 1.0 / (2 * (self.dim_x + self.lambda_))
        self.weights_cov[1:] = self.weights_mean[1:]

    def sigma_points(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Generates sigma points.

        Args:
            x (torch.Tensor): State vector.
            P (torch.Tensor): Covariance matrix.

        Returns:
            torch.Tensor: Sigma points matrix of shape (dim_x, 2*dim_x+1).
        """
        try:
            L = torch.linalg.cholesky(P)
            sigma_pts = torch.zeros((self.dim_x, self.num_sigma), dtype=torch.float32, device=self.device)
            sigma_pts[:, 0] = x
            for i in range(self.dim_x):
                sigma_pts[:, i + 1] = x + self.gamma * L[:, i]
                sigma_pts[:, self.dim_x + i + 1] = x - self.gamma * L[:, i]
            return sigma_pts
        except Exception as e:
            logging.getLogger(self.__class__.__name__).error(f"Error generating sigma points: {e}")
            raise

    def forward(self):
        """
        Placeholder to comply with nn.Module. The UKF operates via predict and update methods.
        """
        pass

    def predict_step(self):
        """
        Performs the prediction step of the UKF.
        """
        try:
            sigma_pts = self.sigma_points(self.x, self.P)
            sigma_pts_pred = torch.stack([self.fx(pt, self.dt) for pt in sigma_pts.t()], dim=1)  # Shape: (dim_x, num_sigma)

            # Predicted state mean
            x_pred = torch.sum(self.weights_mean.unsqueeze(0) * sigma_pts_pred, dim=1)

            # Predicted covariance
            y = sigma_pts_pred - x_pred.unsqueeze(1)
            P_pred = torch.zeros_like(self.P)
            for i in range(self.num_sigma):
                P_pred += self.weights_cov[i] * torch.ger(y[:, i], y[:, i])
            P_pred += self.Q

            # Update state and covariance
            self.x = x_pred
            self.P = P_pred

            # Store sigma points for update step
            self.sigma_pts_pred = sigma_pts_pred
        except Exception as e:
            logging.getLogger(self.__class__.__name__).error(f"Error in predict_step: {e}")
            raise

    def update_step(self, z: torch.Tensor):
        """
        Performs the update step of the UKF with measurement z.

        Args:
            z (torch.Tensor): Measurement vector.
        """
        try:
            sigma_pts_meas = torch.stack([self.hx(pt) for pt in self.sigma_pts_pred.t()], dim=1)  # Shape: (dim_z, num_sigma)

            # Predicted measurement mean
            z_pred = torch.sum(self.weights_mean.unsqueeze(0) * sigma_pts_meas, dim=1)

            # Measurement covariance
            y = sigma_pts_meas - z_pred.unsqueeze(1)
            S = torch.zeros((self.dim_z, self.dim_z), dtype=torch.float32, device=self.device)
            for i in range(self.num_sigma):
                S += self.weights_cov[i] * torch.ger(y[:, i], y[:, i])
            S += self.R

            # Cross covariance
            x_diff = self.sigma_pts_pred - self.x.unsqueeze(1)
            z_diff = y
            Pxz = torch.zeros((self.dim_x, self.dim_z), dtype=torch.float32, device=self.device)
            for i in range(self.num_sigma):
                Pxz += self.weights_cov[i] * torch.ger(x_diff[:, i], z_diff[:, i])

            # Kalman gain
            K = Pxz @ torch.linalg.inv(S)

            # Update state and covariance
            self.x = self.x + K @ (z - z_pred)
            self.P = self.P - K @ S @ K.t()
        except Exception as e:
            logging.getLogger(self.__class__.__name__).error(f"Error in update_step: {e}")
            raise


class PyTorchUKFModule(nn.Module):
    """
    PyTorch module encapsulating the Unscented Kalman Filter operations.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        fx: Callable[[torch.Tensor, float], torch.Tensor],
        hx: Callable[[torch.Tensor], torch.Tensor],
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = -1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the PyTorchUKFModule.

        Args:
            dim_x (int): Dimensionality of the state.
            dim_z (int): Dimensionality of the measurement.
            dt (float): Time step.
            fx (Callable): State transition function.
            hx (Callable): Measurement function.
            alpha (float, optional): UKF scaling parameter. Defaults to 0.1.
            beta (float, optional): UKF scaling parameter. Defaults to 2.0.
            kappa (float, optional): UKF scaling parameter. Defaults to -1.0.
            process_noise (float, optional): Process noise covariance scalar. Defaults to 0.01.
            measurement_noise (float, optional): Measurement noise covariance scalar. Defaults to 0.1.
            device (Optional[torch.device], optional): Device to run computations on. Defaults to None.
        """
        super(PyTorchUKFModule, self).__init__()
        self.ukf = PyTorchUnscentedKalmanFilter(
            dim_x=dim_x,
            dim_z=dim_z,
            dt=dt,
            fx=fx,
            hx=hx,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            device=device
        )

    def predict(self):
        """
        Performs the prediction step.
        """
        self.ukf.predict_step()

    def update(self, z: torch.Tensor):
        """
        Performs the update step with measurement z.

        Args:
            z (torch.Tensor): Measurement vector.
        """
        self.ukf.update_step(z)

    def get_state(self) -> torch.Tensor:
        """
        Retrieves the current state estimate.

        Returns:
            torch.Tensor: State vector.
        """
        return self.ukf.x

    def get_covariance(self) -> torch.Tensor:
        """
        Retrieves the current covariance estimate.

        Returns:
            torch.Tensor: Covariance matrix.
        """
        return self.ukf.P


class NeuralLayerBase(nn.Module):
    """
    Base class for neural network layers, encapsulating common functionalities.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initializes the base neural layer with shared attributes.

        Args:
            input_size (int): Size of the input vector.
            output_size (int): Size of the output vector.
            learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
            weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 1e-5.
        """
        super(NeuralLayerBase, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logger = logging.getLogger(self.__class__.__name__)

        # Define a simple linear layer followed by activation
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer. To be implemented by derived classes.

        Args:
            input_signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal.
        """
        raise NotImplementedError("Forward method must be implemented by the subclass.")

    def compute_loss(self, output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss. Default is Mean Squared Error.

        Args:
            output (torch.Tensor): The model's output.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target_output)
        return loss

    def backward_and_optimize(self, loss: torch.Tensor):
        """
        Backward pass to compute gradients and update weights.

        Args:
            loss (torch.Tensor): The computed loss.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_step(self, input_signal: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            input_signal (torch.Tensor): The input signal.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        output = self.forward(input_signal)
        loss = self.compute_loss(output, target_output)
        self.backward_and_optimize(loss)
        self.logger.debug(f"Training loss: {loss.item()}")
        return loss


class SelectiveSSM(NeuralLayerBase):
    """
    Implements a selective state space model that compresses state representations
    by selectively filtering relevant features from the input signal.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.001):
        """
        Initializes the SelectiveSSM with specified sizes and learning rate.

        Args:
            input_size (int): Size of the input vector.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output vector.
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.001.
        """
        super(SelectiveSSM, self).__init__(input_size, hidden_size, learning_rate)
        self.output_size = output_size  # Ensure output size is set correctly

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the selective SSM.

        Args:
            input_signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The state update.
        """
        self.state_update = torch.tanh(self.linear(input_signal))
        return self.state_update

    def generate_target(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Generates target output for self-supervised learning.

        Args:
            input_signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The target output.
        """
        # For self-supervised learning, reconstruct the input_signal
        return input_signal

    def train_step_custom(self, input_signal: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Perform a custom training step specific to SelectiveSSM.

        Args:
            input_signal (torch.Tensor): The input signal.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.compute_loss(self.forward(input_signal), target_output)
        self.backward_and_optimize(loss)
        return loss


class OscillatoryNeuralLayerHH(nn.Module):
    """
    Implements Hodgkin-Huxley-style spiking behavior for temporal dynamics using PyTorch.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        frequency: float,
        dt: float,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initializes the OscillatoryNeuralLayerHH with specified parameters.

        Args:
            input_size (int): Size of the input vector.
            output_size (int): Size of the output vector.
            frequency (float): Frequency of oscillation.
            dt (float): Time step for simulation.
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
        """
        super(OscillatoryNeuralLayerHH, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.frequency = frequency
        self.dt = dt
        self.logger = logging.getLogger(self.__class__.__name__)

        # Define parameters for Hodgkin-Huxley model
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 6.0

        # Initialize membrane potentials and recovery variables as buffers
        self.register_buffer('voltage', torch.zeros(output_size))
        self.register_buffer('recovery', torch.zeros(output_size))
        self.spike_threshold = 30.0

        # Define linear layer for synaptic weights
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Oscillation phase
        self.register_buffer('phase', torch.zeros(1))

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the HH layer.

        Args:
            input_signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The spiking output.
        """
        with torch.no_grad():
            # Update oscillation phase
            self.phase += 2 * math.pi * self.frequency * self.dt
            oscillatory_input = math.sin(self.phase)  # Scalar oscillatory input

            # Update membrane potentials
            dv = (0.04 * self.voltage ** 2) + (5 * self.voltage) + 140 - self.recovery + self.linear(input_signal) + oscillatory_input
            self.voltage += dv * self.dt

            # Update recovery variable
            dr = self.a * (self.b * self.voltage - self.recovery)
            self.recovery += dr * self.dt

            # Detect spikes
            spikes = self.voltage > self.spike_threshold
            self.voltage[spikes] = self.c
            self.recovery[spikes] += self.d

            # Convert spikes to float output
            self.output = spikes.float()

        return self.output

    def compute_loss(self, output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss. Default is Mean Squared Error.

        Args:
            output (torch.Tensor): The model's output.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target_output)
        return loss

    def backward_and_optimize(self, loss: torch.Tensor):
        """
        Backward pass to compute gradients and update weights.

        Args:
            loss (torch.Tensor): The computed loss.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_step_custom(self, input_signal: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Perform a custom training step specific to OscillatoryNeuralLayerHH.

        Args:
            input_signal (torch.Tensor): The input signal.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.compute_loss(self.forward(input_signal), target_output)
        self.backward_and_optimize(loss)
        self.logger.debug(f"Training loss: {loss.item()}")
        return loss


class AdaptiveLIFNeuralLayer(nn.Module):
    """
    Simulates an adaptive Leaky Integrate-and-Fire (aLIF) neural network for time-aware processing using PyTorch.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tau_m: float = 20.0,
        tau_ref: float = 2.0,
        dt: float = 0.001,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initializes the AdaptiveLIFNeuralLayer with specified parameters.

        Args:
            input_size (int): Size of the input vector.
            output_size (int): Size of the output vector.
            tau_m (float, optional): Membrane time constant in ms. Defaults to 20.0.
            tau_ref (float, optional): Refractory period in ms. Defaults to 2.0.
            dt (float, optional): Time step for simulation in seconds. Defaults to 0.001.
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
        """
        super(AdaptiveLIFNeuralLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        self.dt = dt
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize membrane potentials and refractory timers as buffers
        self.register_buffer('v', torch.zeros(output_size))
        self.register_buffer('refractory_timer', torch.zeros(output_size))
        self.v_threshold = 1.0  # Threshold for spiking
        self.v_reset = 0.0  # Reset potential after a spike

        # Define linear layer for synaptic weights
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute spiking output.

        Args:
            input_signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The spiking output.
        """
        with torch.no_grad():
            # Check for neurons in refractory period
            refractory = self.refractory_timer > 0
            self.v[refractory] = self.v_reset
            self.refractory_timer[refractory] -= self.dt

            # Update membrane potentials for non-refractory neurons
            dv = (-self.v + self.linear(input_signal)) / self.tau_m
            self.v += dv * self.dt

            # Detect spikes
            spikes = self.v >= self.v_threshold
            self.v[spikes] = self.v_reset
            self.refractory_timer[spikes] = self.tau_ref

            # Convert spikes to float output
            self.output = spikes.float()

        return self.output

    def compute_loss(self, output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss. Default is Mean Squared Error.

        Args:
            output (torch.Tensor): The model's output.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target_output)
        return loss

    def backward_and_optimize(self, loss: torch.Tensor):
        """
        Backward pass to compute gradients and update weights.

        Args:
            loss (torch.Tensor): The computed loss.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_step_custom(self, input_signal: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        Perform a custom training step specific to AdaptiveLIFNeuralLayer.

        Args:
            input_signal (torch.Tensor): The input signal.
            target_output (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.compute_loss(self.forward(input_signal), target_output)
        self.backward_and_optimize(loss)
        self.logger.debug(f"Training loss: {loss.item()}")
        return loss


class StateSpaceModel(nn.Module):
    """
    StateSpaceModel integrates multiple components to maintain and update the internal state,
    including selective state space modeling, Hodgkin-Huxley neurons, aLIF neurons for time-aware processing,
    attention mechanisms, and emotional state management.
    """

    def __init__(self, provider_manager: ProviderManager, config_manager: ConfigManager, device: Optional[torch.device] = None):
        """
        Initializes the StateSpaceModel with the specified provider manager and configuration manager.
        Sets up the Unscented Kalman Filter (UKF), Time Decay, Spaced Repetition, aLIF Layer, Cognitive Temporal State, and AttentionManager.

        Args:
            provider_manager (ProviderManager): The provider manager for LLM interactions.
            config_manager (ConfigManager): The configuration manager to retrieve settings.
            device (Optional[torch.device], optional): Device to run computations on. Defaults to None.
        """
        super(StateSpaceModel, self).__init__()
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('StateSpaceModel')
        self.provider_manager = provider_manager

        # Device management
        self.device = device if device else torch.device('cpu')

        # Retrieve subsystem configuration
        subsystem_config = self.config_manager.get_subsystem_config('state_space_model')
        time_aware_config = self.config_manager.get_subsystem_config('time_aware_processing')

        # Extract state space model parameters
        self.dim = subsystem_config.get('dimension', 50)  # Default to 50 if not specified
        self.update_interval = subsystem_config.get('update_interval', 1.0)
        self.pfc_frequency = subsystem_config.get('pfc_frequency', 5)
        self.striatum_frequency = subsystem_config.get('striatum_frequency', 40)
        self.learning_rate = subsystem_config.get('learning_rate', 0.001)
        self.ukf_alpha = subsystem_config.get('ukf_alpha', 0.1)
        self.ukf_beta = subsystem_config.get('ukf_beta', 2.0)
        self.ukf_kappa = subsystem_config.get('ukf_kappa', -1.0)
        self.process_noise = subsystem_config.get('process_noise', 0.01)
        self.measurement_noise = subsystem_config.get('measurement_noise', 0.1)
        self.dt = subsystem_config.get('dt', 0.001)  # Time step for HH neurons
        self.scaling_factor = subsystem_config.get('scaling_factor', 2.0)
        self.attention_mlp_hidden_size = subsystem_config.get('attention_mlp_hidden_size', 64)
        self.initial_confidence_threshold = subsystem_config.get('initial_confidence_threshold', 0.5)
        self.threshold_increment = subsystem_config.get('threshold_increment', 0.01)
        aLIF_params = subsystem_config.get('aLIF_parameters', {})
        self.aLIF_tau_m = aLIF_params.get('tau_m', 20.0)
        self.aLIF_tau_ref = aLIF_params.get('tau_ref', 2.0)
        self.aLIF_learning_rate = aLIF_params.get('learning_rate', 0.001)
        self.default_cognitive_temporal_state = subsystem_config.get('default_cognitive_temporal_state', 'IMMEDIATE')

        # Initialize state dimensions to include aLIF layer
        self.dim_x = 5 * self.dim + 8 + self.dim  # Existing dimensions + aLIF layer
        self.dim_z = self.dim + 8 + self.dim  # Existing measurement dimensions + aLIF layer

        self.logger.debug(f"StateSpaceModel initialized with dim={self.dim}, dim_x={self.dim_x}, dim_z={self.dim_z}")

        # Initialize the Unscented Kalman Filter with updated dimensions
        self.ukf_module = PyTorchUKFModule(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.update_interval,
            fx=self.fx_with_selection,
            hx=self.hx,
            alpha=self.ukf_alpha,
            beta=self.ukf_beta,
            kappa=self.ukf_kappa,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            device=self.device
        )

        # Initialize UKF state vector
        self.ukf_module.ukf.x[:self.dim] = torch.randn(self.dim, dtype=torch.float32, device=self.device)
        self.ukf_module.ukf.x[5 * self.dim:6 * self.dim] = torch.randn(self.dim, dtype=torch.float32, device=self.device)
        self.ukf_module.ukf.x[-8:] = torch.ones(8, dtype=torch.float32, device=self.device) * 0.5

        # Initialize covariance matrices (already handled in UKF class)

        # Initialize TimeDecay and SpacedRepetition
        self.time_decay = TimeDecay(system_state=self, config_manager=config_manager)
        self.spaced_repetition = SpacedRepetition(memory_store=self, config_manager=config_manager)

        # Initialize MemoryConsolidationThread
        self.memory_consolidation_thread = MemoryConsolidationThread(
            memory_store=self,
            spaced_repetition=self.spaced_repetition,
            provider_manager=provider_manager,
            config_manager=config_manager,
            system_state=self
        )
        self.memory_consolidation_thread.start()

        # Initialize Selective State Space Modeling
        self.selective_ssm = SelectiveSSM(
            input_size=self.dim,
            hidden_size=self.dim,
            output_size=self.dim,
            learning_rate=self.learning_rate
        ).to(self.device)

        # Initialize HH neural layers
        self.pfc_layer = OscillatoryNeuralLayerHH(
            input_size=self.dim,
            output_size=self.dim,
            frequency=self.pfc_frequency,
            dt=self.dt,
            learning_rate=self.learning_rate
        ).to(self.device)
        self.striatum_layer = OscillatoryNeuralLayerHH(
            input_size=self.dim,
            output_size=2,  # Assuming output_size=2 for go/no-go signals
            frequency=self.striatum_frequency,
            dt=self.dt,
            learning_rate=self.learning_rate
        ).to(self.device)

        # Initialize aLIF neural layer for Time-Aware Processing (TAP)
        self.tap_layer = AdaptiveLIFNeuralLayer(
            input_size=self.dim,
            output_size=self.dim,
            tau_m=self.aLIF_tau_m,
            tau_ref=self.aLIF_tau_ref,
            dt=self.dt,
            learning_rate=self.aLIF_learning_rate
        ).to(self.device)

        # Initialize AttentionManager
        self.attention_manager = AttentionManager(self, config_manager=self.config_manager).to(self.device)

        # Initialize emotional state components
        self._emotional_state = {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        self.consciousness_level = 0.5

        # Initialize confidence estimation parameters
        self.previous_variance = torch.tensor(float('inf'), device=self.device)

        # Initialize CognitiveTemporalState with configuration
        cognitive_temporal_states_config = time_aware_config.get('cognitive_temporal_states', {})
        state_transition_rules = {
            CognitiveTemporalStateEnum[state_name]: tuple(state_config['decay_rates_multiplier'].values())
            for state_name, state_config in cognitive_temporal_states_config.items()
            if state_name in CognitiveTemporalStateEnum.__members__
        }

        cognitive_temporal_state_config = CognitiveTemporalStateConfig(
            alpha=time_aware_config.get('alpha', 0.1),
            scaling_bounds=tuple(time_aware_config.get('scaling_bounds', [0.1, 5.0])),
            state_transition_rules=state_transition_rules,
            initial_state=CognitiveTemporalStateEnum.get(
                time_aware_config.get('default_cognitive_temporal_state', 'IMMEDIATE').upper(),
                CognitiveTemporalStateEnum.IMMEDIATE
            ),
            initial_scaling=time_aware_config.get('initial_scaling', 1.0)
        )

        self.current_cognitive_temporal_state = CognitiveTemporalState(config=cognitive_temporal_state_config)
        self.logger.info(f"Initial CognitiveTemporalState set to: {self.current_cognitive_temporal_state.get_current_state().name}")

        # Move UKF module to device
        self.ukf_module.to(self.device)

    def fx_with_selection(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State transition function with selective state space modeling and aLIF integration.

        Args:
            x (torch.Tensor): The current state vector.
            dt (float): The time step.

        Returns:
            torch.Tensor: The updated state vector.
        """
        try:
            new_x = torch.zeros_like(x, device=self.device)

            # Extract input for SelectiveSSM
            selective_input = x[:self.dim]

            # Update SelectiveSSM
            selected_output = self.selective_ssm.forward(selective_input)
            new_x[:self.dim] = selected_output + torch.sin(x[3 * self.dim:4 * self.dim]) * dt

            # Update other state components
            new_x[self.dim:2 * self.dim] = x[self.dim:2 * self.dim] + torch.cos(x[3 * self.dim:4 * self.dim]) * dt
            new_x[2 * self.dim:3 * self.dim] = x[2 * self.dim:3 * self.dim] + dt
            new_x[3 * self.dim:4 * self.dim] = x[3 * self.dim:4 * self.dim] + dt
            new_x[4 * self.dim:5 * self.dim] = x[4 * self.dim:5 * self.dim] + dt
            new_x[5 * self.dim:6 * self.dim] = selected_output  # Update aLIF layer state
            new_x[-8:] = x[-8:]  # Keep scalar values unchanged

            return new_x
        except Exception as e:
            self.logger.error(f"Error in state transition function fx_with_selection: {str(e)}", exc_info=True)
            raise

    def hx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measurement function.

        Args:
            x (torch.Tensor): The current state vector.

        Returns:
            torch.Tensor: The measurement vector.
        """
        try:
            # Concatenate outputs from PFC, Striatum, aLIF (TAP), Attention, and scalar state variables
            pfc_output = x[:self.dim]
            striatum_output = x[self.dim:2 * self.dim]
            tap_output = x[5 * self.dim:6 * self.dim]  # aLIF layer outputs
            attention_vector = x[2 * self.dim:3 * self.dim]
            scalar_outputs = x[-8:]
            measurement = torch.cat([pfc_output, striatum_output, tap_output, attention_vector, scalar_outputs], dim=0)
            return measurement
        except Exception as e:
            self.logger.error(f"Error in measurement function hx: {str(e)}", exc_info=True)
            raise

    def ensure_positive_definite(self):
        """Ensures that the covariance matrices P and Q are positive definite."""
        self.logger.debug("Ensuring positive definiteness of P and Q matrices")
        try:
            self.ukf_module.ukf.P = self.nearest_positive_definite_efficient(self.ukf_module.ukf.P)
            self.ukf_module.ukf.Q = self.nearest_positive_definite_efficient(self.ukf_module.ukf.Q)
        except ValueError as ve:
            self.logger.error(f"Failed to ensure positive definiteness: {ve}", exc_info=True)
            raise

    @staticmethod
    def nearest_positive_definite_efficient(A: torch.Tensor) -> torch.Tensor:
        """Finds the nearest positive-definite matrix to A."""
        try:
            B = (A + A.t()) / 2
            eigvals = torch.linalg.eigvalsh(B)
            if torch.all(eigvals > 0):
                return B
            min_eig = torch.min(eigvals).item()
            return B + (-min_eig * torch.eye(B.size(0), device=B.device) + torch.eye(B.size(0), device=B.device) * 1e-8)
        except Exception as e:
            raise ValueError(f"Error in computing nearest positive-definite matrix: {str(e)}")

    async def initialize(self):
        """
        Performs asynchronous initialization tasks for the StateSpaceModel.
        """
        try:
            self.logger.info("Starting StateSpaceModel initialization")

            # Initialize StateMeasurement
            self.state_measurement = StateMeasurement(self.provider_manager)

            # Initialize AttentionManager
            self.attention_manager = AttentionManager(self, config_manager=self.config_manager).to(self.device)

            self.logger.info("StateSpaceModel initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Error during StateSpaceModel initialization: {str(e)}")
            raise ValueError(f"Failed to initialize StateSpaceModel: {str(e)}")

    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the state vector based on the given data.

        Args:
            data (Dict[str, Any]): The input data for the update.

        Returns:
            Dict[str, Any]: The updated state.
        """
        self.logger.debug(f"Updating state with data: {data}")
        self.logger.debug(f"State before update: {self.ukf_module.ukf.x}")
        self.logger.debug(f"Covariance before update: {self.ukf_module.ukf.P}")

        try:
            # Prepare inputs for HH and aLIF layers
            pfc_input = self.prepare_pfc_input(data)
            striatum_input = self.prepare_striatum_input(data)
            tap_input = self.prepare_tap_input(data)  # New method for TAP input

            # Update HH layers
            pfc_activation = self.pfc_layer.forward(pfc_input)
            striatum_activation = self.striatum_layer.forward(striatum_input)

            # Update aLIF layer
            tap_activation = self.tap_layer.forward(tap_input)

            # Estimate confidence
            confidence = self.estimate_confidence()
            self.logger.debug(f"Estimated confidence: {confidence}, Threshold: {self.initial_confidence_threshold}")

            if confidence < self.initial_confidence_threshold:
                # Query external sources
                content = data.get('content', '')
                external_info = await self.provider_manager.generate_response(
                    messages=[{"role": "user", "content": content}],
                    model=self.provider_manager.get_current_model(),
                    max_tokens=50,
                    temperature=0.0,
                    stream=False,
                    current_persona=None,
                    functions=None,
                    llm_call_type='transformer_encode'  # Adjust as necessary
                )
                # Incorporate external_info into the state update
                await self.update_with_external_info(external_info)

            # Training steps
            target_pfc_output = self.pfc_layer.generate_target(pfc_input)
            loss_pfc = self.pfc_layer.train_step(pfc_input, target_pfc_output)
            self.logger.debug(f"PFC Layer Training Loss: {loss_pfc.item()}")

            target_striatum_output = self.striatum_layer.generate_target(striatum_input)
            loss_striatum = self.striatum_layer.train_step(striatum_input, target_striatum_output)
            self.logger.debug(f"Striatum Layer Training Loss: {loss_striatum.item()}")

            target_tap_output = self.tap_layer.generate_target(tap_input)
            loss_tap = self.tap_layer.train_step(tap_input, target_tap_output)  # Train aLIF layer
            self.logger.debug(f"aLIF Layer Training Loss: {loss_tap.item()}")

            # Adjust learning rates (handled by Adam optimizer)

            # Update attention focus using the attention mechanism
            await self.attention_manager.update_attention(data)

            # Use the updated attention focus from the AttentionManager
            attention_vector = self.attention_focus

            # Update UKF state based on neural activations and attention
            measurement = torch.cat([pfc_activation, striatum_activation, tap_activation, attention_vector, self.ukf_module.ukf.x[-8:]], dim=0)

            # Ensure measurement size matches UKF dimensions
            if measurement.shape[0] != self.ukf_module.dim_z:
                self.logger.warning(
                    f"Measurement shape mismatch. Expected {self.ukf_module.dim_z}, got {measurement.shape[0]}. Resizing measurement."
                )
                measurement = measurement.view(-1)[:self.ukf_module.dim_z]
                if measurement.shape[0] < self.ukf_module.dim_z:
                    padding = torch.ones(self.ukf_module.dim_z - measurement.shape[0], dtype=measurement.dtype, device=measurement.device) * 0.5
                    measurement = torch.cat([measurement, padding], dim=0)

            # Perform UKF prediction and update
            self.ukf_module.predict()
            self.ukf_module.update(measurement)

            # Update other state components
            await self.update_emotional_state(data)
            self.update_consciousness_level(striatum_activation)

            # Update confidence threshold
            self.update_confidence_threshold()

            # Dynamically adjust consolidation interval based on CognitiveTemporalState
            consolidation_interval = self.time_decay.get_consolidation_interval()
            self.memory_consolidation_thread.consolidation_interval = consolidation_interval
            self.logger.debug(f"Consolidation interval updated to: {consolidation_interval} seconds based on CognitiveTemporalState")

            self.ensure_positive_definite()

            # Evaluate and switch CognitiveTemporalState based on current conditions
            await self.evaluate_and_switch_cognitive_temporal_state(data)

            self.logger.debug(f"Updated state shape: {self.ukf_module.ukf.x.shape}")
            self.logger.debug(f"State after update: {self.ukf_module.ukf.x}")
            self.logger.debug(f"Covariance after update: {self.ukf_module.ukf.P}")

            return await self.get_state()
        except Exception as e:
            self.logger.error(f"Error in state update: {str(e)}", exc_info=True)
            raise

    def prepare_pfc_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares input for the PFC (Prefrontal Cortex) layer.

        Args:
            data (Dict[str, Any]): The input data.

        Returns:
            torch.Tensor: The prepared PFC input.
        """
        self.logger.debug(f"Preparing PFC input with data: {data}")
        try:
            if 'content' in data:
                content = data['content']
                # Use transformer_encode method from ProviderManager
                transformer_output = self.provider_manager.huggingface_generator.transformer_encode(content)
                if not isinstance(transformer_output, torch.Tensor):
                    transformer_output = torch.tensor(transformer_output, dtype=torch.float32, device=self.device)
                # Ensure transformer_output is the correct size
                if transformer_output.numel() > self.dim:
                    transformer_output = transformer_output[:self.dim]
                elif transformer_output.numel() < self.dim:
                    padding = torch.ones(self.dim - transformer_output.numel(), dtype=torch.float32, device=self.device) * 0.5
                    transformer_output = torch.cat([transformer_output, padding], dim=0)
                pfc_input = transformer_output
            else:
                pfc_input = torch.ones(self.dim, dtype=torch.float32, device=self.device) * 0.5
            return pfc_input
        except Exception as e:
            self.logger.error(f"Error in preparing PFC input: {str(e)}", exc_info=True)
            raise

    def prepare_striatum_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares input for the Striatum layer (e.g., go/no-go signals).

        Args:
            data (Dict[str, Any]): The input data.

        Returns:
            torch.Tensor: The prepared Striatum input.
        """
        self.logger.debug(f"Preparing Striatum input with data: {data}")
        try:
            go_signal = 1.0 if data.get('action_required', False) else 0.0
            striatum_input = torch.tensor([go_signal, 1.0 - go_signal], dtype=torch.float32, device=self.device)
            return striatum_input
        except Exception as e:
            self.logger.error(f"Error in preparing Striatum input: {str(e)}", exc_info=True)
            raise

    def prepare_tap_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Extracts relevant features from data for TAP (aLIF) input.

        Args:
            data (Dict[str, Any]): The input data.

        Returns:
            torch.Tensor: The prepared TAP input.
        """
        self.logger.debug(f"Preparing TAP input with data: {data}")
        try:
            tap_input = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
            # Example: Use temporal features from data
            if 'timestamp' in data:
                current_time = time.time()
                time_diff = current_time - data['timestamp']
                tap_input[0] = time_diff  # Example feature
            return tap_input
        except Exception as e:
            self.logger.error(f"Error in preparing TAP input: {str(e)}", exc_info=True)
            raise

    async def update_with_external_info(self, external_info: Any):
        """
        Incorporates external information into the state update.

        Args:
            external_info (Any): The external information to incorporate.
        """
        try:
            # Example: Use external sentiment analysis
            sentiment = external_info.get('sentiment', 0.0)
            # Update emotional state based on external sentiment
            await self.update_emotional_state({'content': '', 'sentiment': sentiment})
            # Optionally, use external info to adjust other state components
        except Exception as e:
            self.logger.error(f"Error in updating with external info: {str(e)}", exc_info=True)
            raise

    def estimate_confidence(self) -> float:
        """
        Estimates the confidence level of the SSM's internal processing.

        Returns:
            float: The estimated confidence level.
        """
        try:
            # Use the variance from the UKF as an uncertainty measure
            variance = torch.mean(torch.diag(self.ukf_module.ukf.P))
            confidence = 1.0 / (1.0 + variance.item())
            return confidence
        except Exception as e:
            self.logger.error(f"Error in estimating confidence: {str(e)}", exc_info=True)
            raise

    def update_confidence_threshold(self):
        """
        Updates the confidence threshold dynamically.
        """
        try:
            self.initial_confidence_threshold = min(1.0, self.initial_confidence_threshold + self.threshold_increment)
            self.logger.debug(f"Confidence threshold updated to: {self.initial_confidence_threshold}")
        except Exception as e:
            self.logger.error(f"Error in updating confidence threshold: {str(e)}", exc_info=True)
            raise

    async def update_emotional_state(self, data: Dict[str, Any]):
        """
        Asynchronously updates the emotional state based on the input data.

        Args:
            data (Dict[str, Any]): The input data.
        """
        self.logger.debug(f"Updating emotional state with data: {data}")
        try:
            content = data.get('content', '')
            sentiment = await self.state_measurement.analyze_text(content)
            valence = sentiment[1] if sentiment[0] == 'POSITIVE' else -sentiment[1]
            arousal = sentiment[2]
            dominance = sentiment[3]

            # Apply time-based decay to long-term episodic memory influenced by CognitiveTemporalState
            decayed_memory = self.time_decay.decay(
                memory_type=MemoryType.LONG_TERM_EPISODIC,
                time_elapsed=1.0,  # Example time elapsed; adjust as needed
                importance=valence
            )

            # Update emotional state using TimeDecay influence
            new_valence = 0.9 * self.emotional_state['valence'] + 0.1 * decayed_memory
            new_arousal = 0.9 * self.emotional_state['arousal'] + 0.1 * arousal
            new_dominance = 0.9 * self.emotional_state['dominance'] + 0.1 * dominance

            self._emotional_state = {
                'valence': new_valence,
                'arousal': new_arousal,
                'dominance': new_dominance
            }

            # Update the UKF state vector with the new emotional values
            self.ukf_module.ukf.x[self.dim:2 * self.dim] = new_valence * torch.ones(self.dim, dtype=torch.float32, device=self.device)
            self.ukf_module.ukf.x[-8] = torch.tensor(new_arousal, dtype=torch.float32, device=self.device)
            self.ukf_module.ukf.x[-7] = torch.tensor(new_dominance, dtype=torch.float32, device=self.device)

            self.logger.debug(f"Emotional state updated: {self._emotional_state}")
        except Exception as e:
            self.logger.error(f"Error in updating emotional state: {str(e)}", exc_info=True)
            raise

    def update_consciousness_level(self, striatum_activation: torch.Tensor):
        """
        Updates the consciousness level based on striatum activation.

        Args:
            striatum_activation (torch.Tensor): The striatum activation vector.
        """
        self.logger.debug(f"Updating consciousness level with striatum activation")
        try:
            go_signal = torch.mean(striatum_activation).item()
            self.consciousness_level = 0.9 * self.consciousness_level + 0.1 * go_signal
            self.ukf_module.ukf.x[-6] = torch.tensor(self.consciousness_level, dtype=torch.float32, device=self.device)
            self.logger.debug(f"Consciousness level updated: {self.consciousness_level}")

            # Influence CognitiveTemporalState based on consciousness_level or other factors
            influence_factor = self.consciousness_level  # Example influence
            self.current_cognitive_temporal_state.update_state(influence_factor)
        except Exception as e:
            self.logger.error(f"Error in updating consciousness level: {str(e)}", exc_info=True)
            raise

    async def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the current state of the model.

        Returns:
            Dict[str, Any]: The current state of the model.
        """
        self.logger.debug("Retrieving current state of the model")
        try:
            state = {
                'ukf_state': self.ukf_module.get_state().cpu().numpy().tolist(),
                'emotional_state': self.emotional_state,
                'attention_focus': self.attention_focus.cpu().numpy().tolist(),
                'consciousness_level': self.consciousness_level,
                'cognitive_temporal_state': self.current_cognitive_temporal_state.get_current_state().name
            }
            self.logger.debug(f"State retrieved: {state}")
            return state
        except Exception as e:
            self.logger.error(f"Error in getting state: {str(e)}", exc_info=True)
            raise

    @property
    def emotional_state(self) -> Dict[str, float]:
        """
        Retrieves the emotional state from the state vector.

        Returns:
            Dict[str, float]: The emotional state with valence, arousal, and dominance.
        """
        try:
            valence = torch.mean(self.ukf_module.ukf.x[self.dim:2 * self.dim]).item() if self.ukf_module.ukf.x[self.dim:2 * self.dim].numel() > 0 else 0.0
            return {
                'valence': float(valence),
                'arousal': float(self.ukf_module.ukf.x[-8].item()),
                'dominance': float(self.ukf_module.ukf.x[-7].item()),
            }
        except Exception as e:
            self.logger.error(f"Error in retrieving emotional state: {str(e)}", exc_info=True)
            raise

    @property
    def consciousness_level(self) -> float:
        """
        Retrieves the consciousness level from the state vector.

        Returns:
            float: The consciousness level.
        """
        try:
            return float(self.ukf_module.ukf.x[-6].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving consciousness level: {str(e)}", exc_info=True)
            raise

    @consciousness_level.setter
    def consciousness_level(self, value: float):
        """
        Sets the consciousness level in the state vector.

        Args:
            value (float): The new consciousness level.
        """
        try:
            self.ukf_module.ukf.x[-6] = torch.tensor(value, dtype=torch.float32, device=self.device)
        except Exception as e:
            self.logger.error(f"Error in setting consciousness level: {str(e)}", exc_info=True)
            raise

    @property
    def attention_focus(self) -> torch.Tensor:
        """
        Retrieves the attention focus from the state vector.

        Returns:
            torch.Tensor: The attention focus vector.
        """
        try:
            return self.ukf_module.ukf.x[:self.dim]
        except Exception as e:
            self.logger.error(f"Error in retrieving attention focus: {str(e)}", exc_info=True)
            raise

    @attention_focus.setter
    def attention_focus(self, value: torch.Tensor):
        """
        Sets the attention focus in the state vector.

        Args:
            value (torch.Tensor): The new attention focus vector.
        """
        try:
            if value.shape[0] != self.dim:
                raise ValueError(f"Attention focus size mismatch. Expected {self.dim}, got {value.shape[0]}")
            self.ukf_module.ukf.x[:self.dim] = value.to(self.device)
        except Exception as e:
            self.logger.error(f"Error in setting attention focus: {str(e)}", exc_info=True)
            raise

    @property
    def attention_allocation(self) -> torch.Tensor:
        """
        Retrieves the attention allocation from the state vector.

        Returns:
            torch.Tensor: The attention allocation vector.
        """
        try:
            return self.ukf_module.ukf.x[2 * self.dim:3 * self.dim]
        except Exception as e:
            self.logger.error(f"Error in retrieving attention allocation: {str(e)}", exc_info=True)
            raise

    @property
    def memory_activation(self) -> torch.Tensor:
        """
        Retrieves the memory activation from the state vector.

        Returns:
            torch.Tensor: The memory activation vector.
        """
        try:
            return self.ukf_module.ukf.x[3 * self.dim:4 * self.dim]
        except Exception as e:
            self.logger.error(f"Error in retrieving memory activation: {str(e)}", exc_info=True)
            raise

    @property
    def cognitive_control(self) -> torch.Tensor:
        """
        Retrieves the cognitive control from the state vector.

        Returns:
            torch.Tensor: The cognitive control vector.
        """
        try:
            return self.ukf_module.ukf.x[4 * self.dim:5 * self.dim]
        except Exception as e:
            self.logger.error(f"Error in retrieving cognitive control: {str(e)}", exc_info=True)
            raise

    @property
    def cognitive_load(self) -> float:
        """
        Retrieves the cognitive load from the state vector.

        Returns:
            float: The cognitive load.
        """
        try:
            return float(self.ukf_module.ukf.x[-5].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving cognitive load: {str(e)}", exc_info=True)
            raise

    @property
    def mental_energy(self) -> float:
        """
        Retrieves the mental energy from the state vector.

        Returns:
            float: The mental energy.
        """
        try:
            return float(self.ukf_module.ukf.x[-4].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving mental energy: {str(e)}", exc_info=True)
            raise

    @property
    def curiosity_level(self) -> float:
        """
        Retrieves the curiosity level from the state vector.

        Returns:
            float: The curiosity level.
        """
        try:
            return float(self.ukf_module.ukf.x[-3].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving curiosity level: {str(e)}", exc_info=True)
            raise

    @property
    def uncertainty(self) -> float:
        """
        Retrieves the uncertainty from the state vector.

        Returns:
            float: The uncertainty.
        """
        try:
            return float(self.ukf_module.ukf.x[-2].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving uncertainty: {str(e)}", exc_info=True)
            raise

    @property
    def creativity_index(self) -> float:
        """
        Retrieves the creativity index from the state vector.

        Returns:
            float: The creativity index.
        """
        try:
            return float(self.ukf_module.ukf.x[-1].item())
        except Exception as e:
            self.logger.error(f"Error in retrieving creativity index: {str(e)}", exc_info=True)
            raise

    def get_state_description(self) -> Dict[str, Any]:
        """
        Provides a description of the current state.

        Returns:
            Dict[str, Any]: The description of the current state.
        """
        self.logger.debug("Providing a description of the current state")
        try:
            desc = {
                "topic_focus": torch.norm(self.attention_focus).item() if self.attention_focus.numel() > 0 else 0.0,
                "emotional_valence": self.emotional_state['valence'],
                "emotional_arousal": self.emotional_state['arousal'],
                "emotional_dominance": self.emotional_state['dominance'],
                "attention_allocation": torch.norm(self.attention_allocation).item() if self.attention_allocation.numel() > 0 else 0.0,
                "memory_activation": torch.norm(self.memory_activation).item() if self.memory_activation.numel() > 0 else 0.0,
                "cognitive_control": torch.norm(self.cognitive_control).item() if self.cognitive_control.numel() > 0 else 0.0,
                "consciousness_level": self.consciousness_level,
                "cognitive_load": self.cognitive_load,
                "mental_energy": self.mental_energy,
                "curiosity_level": self.curiosity_level,
                "uncertainty": self.uncertainty,
                "creativity_index": self.creativity_index,
                "cognitive_temporal_state": self.current_cognitive_temporal_state.get_current_state().name
            }
            self.logger.debug(f"State description: {desc}")
            return desc
        except Exception as e:
            self.logger.error(f"Error in getting state description: {str(e)}", exc_info=True)
            raise

    def update_parameter(self, param_name: str, param_value: float):
        """
        Updates a specific parameter of the state space model.

        Args:
            param_name (str): The name of the parameter to update.
            param_value (float): The new value for the parameter.
        """
        self.logger.info(f"Updating state model parameter {param_name} to {param_value}")

        try:
            if param_name == 'process_noise':
                self.ukf_module.ukf.Q.fill_(0)
                torch.diag(self.ukf_module.ukf.Q)[:] = param_value
            elif param_name == 'measurement_noise':
                self.ukf_module.ukf.R.fill_(0)
                torch.diag(self.ukf_module.ukf.R)[:] = param_value
            elif param_name == 'alpha':
                self.ukf_module.ukf.alpha = param_value
                self.ukf_module.ukf._compute_weights()
            elif param_name == 'beta':
                self.ukf_module.ukf.beta = param_value
                self.ukf_module.ukf._compute_weights()
            elif param_name == 'kappa':
                self.ukf_module.ukf.kappa = param_value
                self.ukf_module.ukf._compute_weights()
            elif param_name.startswith('state_'):
                index = int(param_name.split('_')[1])
                if 0 <= index < self.dim_x:
                    self.ukf_module.ukf.x[index] = torch.tensor(param_value, dtype=torch.float32, device=self.device)
                else:
                    raise ValueError(f"Invalid state index: {index}")
            else:
                raise ValueError(f"Unknown parameter name: {param_name}")

            self.ensure_positive_definite()

            self.logger.info(f"Successfully updated {param_name} to {param_value}")
        except Exception as e:
            self.logger.error(f"Error updating parameter {param_name}: {str(e)}")
            raise

    async def derive_state_update(self, processed_info: Dict[str, Any], context_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Derives state updates from processed information.

        Args:
            processed_info (Dict[str, Any]): The processed information.
            context_vector (torch.Tensor): The context vector.

        Returns:
            Dict[str, Any]: A dictionary of state updates.
        """
        self.logger.info(f"Deriving state updates from processed information: {processed_info}")
        if not isinstance(processed_info, dict):
            self.logger.error(f"Expected dict for processed_info but got {type(processed_info)}")
            raise ValueError(f"Expected dict for processed_info but got {type(processed_info)}")

        try:
            state_update = {}
            state_update['attention_focus'] = await self._compute_attention_focus(processed_info)
            state_update['emotional_state'] = await self._compute_emotional_state(processed_info)
            state_update['context_vector'] = context_vector
            self.logger.info(f"Derived state updates: {state_update}")
            return state_update
        except Exception as e:
            self.logger.error(f"Error in deriving state update: {str(e)}", exc_info=True)
            raise

    async def _compute_attention_focus(self, processed_info: Dict[str, Any]) -> torch.Tensor:
        """
        Computes new attention focus based on the processed information.

        Args:
            processed_info (Dict[str, Any]): The processed information.

        Returns:
            torch.Tensor: The new attention focus vector.
        """
        self.logger.debug(f"Computing attention focus from processed information: {processed_info}")
        try:
            text_data = processed_info.get('content', '')
            await self.attention_manager.update_attention({'content': text_data})
            attention_focus = self.attention_focus
            self.logger.debug(f"Computed attention focus: {attention_focus}")
            return attention_focus
        except Exception as e:
            self.logger.error(f"Error in computing attention focus: {str(e)}", exc_info=True)
            raise

    async def _compute_emotional_state(self, processed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes new emotional state based on the processed information.

        Args:
            processed_info (Dict[str, Any]): The processed information.

        Returns:
            Dict[str, Any]: The new emotional state with valence, arousal, and dominance.
        """
        self.logger.debug(f"Computing emotional state from processed information: {processed_info}")
        try:
            content = processed_info.get('content', '')
            sentiment = await self.state_measurement.analyze_text(content)
            valence = sentiment[1] if sentiment[0] == 'POSITIVE' else -sentiment[1]
            arousal = sentiment[2]
            dominance = sentiment[3]

            # Apply time-based decay to long-term episodic memory influenced by CognitiveTemporalState
            decayed_memory = self.time_decay.decay(
                memory_type=MemoryType.LONG_TERM_EPISODIC,
                time_elapsed=1.0,  # Example time elapsed; adjust as needed
                importance=valence
            )

            # Update emotional state using TimeDecay influence
            new_valence = 0.9 * self.emotional_state['valence'] + 0.1 * decayed_memory
            new_arousal = 0.9 * self.emotional_state['arousal'] + 0.1 * arousal
            new_dominance = 0.9 * self.emotional_state['dominance'] + 0.1 * dominance

            self._emotional_state = {
                'valence': new_valence,
                'arousal': new_arousal,
                'dominance': new_dominance
            }

            # Update the UKF state vector with the new emotional values
            self.ukf_module.ukf.x[self.dim:2 * self.dim] = new_valence * torch.ones(self.dim, dtype=torch.float32, device=self.device)
            self.ukf_module.ukf.x[-8] = torch.tensor(new_arousal, dtype=torch.float32, device=self.device)
            self.ukf_module.ukf.x[-7] = torch.tensor(new_dominance, dtype=torch.float32, device=self.device)

            self.logger.debug(f"Emotional state updated: {self._emotional_state}")
            return self._emotional_state
        except Exception as e:
            self.logger.error(f"Error in computing emotional state: {str(e)}", exc_info=True)
            raise

    def update_cognitive_temporal_state(self, influence_factor: float):
        """
        Updates the CognitiveTemporalState based on an influence factor.

        Args:
            influence_factor (float): Factor derived from aLIF network activity.
                                       Positive values may speed up time perception,
                                       negative values may slow it down.
        """
        self.logger.debug(f"Updating CognitiveTemporalState with influence factor: {influence_factor}")
        try:
            self.current_cognitive_temporal_state.update_state(influence_factor)
            # Retrieve the current state after update
            current_state = self.current_cognitive_temporal_state.get_current_state()
            scaling = self.current_cognitive_temporal_state.get_time_scaling()
            self.logger.debug(f"CognitiveTemporalState after update: {current_state.name}, Scaling Factor: {scaling}")

            # Optionally, additional actions based on state can be implemented here
        except Exception as e:
            self.logger.error(f"Error in updating CognitiveTemporalState: {str(e)}", exc_info=True)
            raise

    async def update_with_cognitive_temporal_state(self):
        """
        Updates model parameters based on the current CognitiveTemporalState.
        """
        try:
            current_state = self.current_cognitive_temporal_state.get_current_state()
            self.logger.info(f"Updating model based on CognitiveTemporalState: {current_state.name}")

            # Apply state-specific configurations
            cognitive_temporal_states_config = self.config_manager.get_subsystem_config('time_aware_processing').get('cognitive_temporal_states', {})
            state_config = cognitive_temporal_states_config.get(current_state.name, {})

            # Update decay rates multipliers
            decay_multipliers = state_config.get('decay_rates_multiplier', {})
            for memory_type, multiplier in decay_multipliers.items():
                enum_memory_type = MemoryType[memory_type.upper()]
                self.time_decay.base_decay_rates[enum_memory_type] *= multiplier

            # Update consolidation interval if overridden
            consolidation_interval = state_config.get('consolidation_interval')
            if consolidation_interval:
                self.memory_consolidation_thread.consolidation_interval = consolidation_interval
                self.logger.debug(f"Consolidation interval updated to: {consolidation_interval} seconds")
        except Exception as e:
            self.logger.error(f"Error in updating with CognitiveTemporalState: {str(e)}", exc_info=True)
            raise

    async def evaluate_and_switch_cognitive_temporal_state(self, data: Dict[str, Any]):
        """
        Evaluates the current conditions and switches the CognitiveTemporalState accordingly.

        Args:
            data (Dict[str, Any]): The input data containing relevant information.
        """
        try:
            # Example condition: High emotional arousal switches to EMOTIONAL state
            arousal = self.emotional_state['arousal']
            cognitive_load = self.cognitive_load

            if arousal > 0.7:
                new_state = CognitiveTemporalStateEnum.EMOTIONAL
            elif cognitive_load > 0.8:
                new_state = CognitiveTemporalStateEnum.ANALYTICAL
            else:
                new_state = CognitiveTemporalStateEnum.IMMEDIATE  # Default state

            if new_state != self.current_cognitive_temporal_state.get_current_state():
                await self.switch_cognitive_temporal_state(new_state)
        except Exception as e:
            self.logger.error(f"Error in evaluate_and_switch_cognitive_temporal_state: {str(e)}", exc_info=True)
            raise

    async def switch_cognitive_temporal_state(self, new_state: CognitiveTemporalStateEnum):
        """
        Switches the CognitiveTemporalState to a new state.

        Args:
            new_state (CognitiveTemporalStateEnum): The new cognitive temporal state to switch to.
        """
        self.logger.info(f"Switching CognitiveTemporalState to: {new_state.name}")
        try:
            # Update CognitiveTemporalState instance with the new state
            cognitive_temporal_states_config = self.config_manager.get_subsystem_config('time_aware_processing').get('cognitive_temporal_states', {})
            state_config = cognitive_temporal_states_config.get(new_state.name, {})

            # Update CognitiveTemporalStateConfig with new state parameters
            updated_state_transition_rules = {
                CognitiveTemporalStateEnum[new_state.name]: tuple(state_config.get('decay_rates_multiplier', {}).values())
            }

            updated_cognitive_temporal_state_config = CognitiveTemporalStateConfig(
                alpha=self.current_cognitive_temporal_state.config.alpha,
                scaling_bounds=self.current_cognitive_temporal_state.config.scaling_bounds,
                state_transition_rules=updated_state_transition_rules,
                initial_state=new_state,
                initial_scaling=self.current_cognitive_temporal_state.scaling_factor
            )

            # Re-instantiate CognitiveTemporalState with the new configuration
            self.current_cognitive_temporal_state = CognitiveTemporalState(config=updated_cognitive_temporal_state_config)
            self.logger.info(f"CognitiveTemporalState changed to: {self.current_cognitive_temporal_state.get_current_state().name}")

            # Update model parameters based on the new CognitiveTemporalState
            await asyncio.create_task(self.update_with_cognitive_temporal_state())
        except Exception as e:
            self.logger.error(f"Error switching CognitiveTemporalState: {str(e)}", exc_info=True)
            raise

    async def stop(self):
        """
        Stops the memory consolidation thread gracefully.
        """
        self.memory_consolidation_thread.stop()
        self.memory_consolidation_thread.join()
        self.logger.info("Memory consolidation thread stopped.")

    # Additional methods and properties can be added here as needed to integrate with the Time-Aware Processing module.

