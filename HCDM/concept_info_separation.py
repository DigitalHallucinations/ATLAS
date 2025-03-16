context/information separation

###############################################################################
# neural_cognitive_bus.py
###############################################################################

"""
Neural Cognitive Bus (NCB)
====================================

This module implements a production–grade, multi–channel, asynchronous communication
system for the Hybrid Cognitive Dynamics Model (HCDM). It supports robust inter–module
data exchange with advanced features:
  • Scalability & Throughput: Channels are managed using asyncio queues with optional
    integration of a Neural Entanglement State Transfer (NEST) module for quantum–inspired
    nonlocal transformations. Concurrency is hardened via proper locking and asynchronous loops.
  • Filtering & Routing: Subscribers may register with topic–based filters or custom filter
    functions to receive only relevant messages.
  • Lifecycle Management: The NCB provides start/stop routines that clean up all background tasks,
    ensuring graceful shutdown.
  • Advanced Error Handling: Detailed logging and exception handling ensure high availability
    in production environments.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Callable, List, Optional
from torchdiffeq import odeint_adjoint as odeint

###############################################################################
# Helper functions for quantum operators
###############################################################################

def random_hermitian(dim: int) -> torch.Tensor:
    """Create a random Hermitian matrix of size (dim, dim)."""
    real_part = torch.randn(dim, dim)
    imag_part = torch.randn(dim, dim)
    A = real_part + 1j * imag_part
    H = (A + A.conj().t()) / 2.0
    return H

def lowering_operator(dim: int) -> torch.Tensor:
    """
    Create the generalized lowering operator for a d-dimensional Hilbert space.
    It acts as: L |i> = |i-1> for i >= 1, and L |0> = 0.
    """
    L = torch.zeros(dim, dim, dtype=torch.cfloat)
    for j in range(1, dim):
        L[j-1, j] = 1.0
    return L

###############################################################################
# NEST Module
###############################################################################

class NESTModule(nn.Module):
    """
    Neural Entanglement State Transfer (NEST) module performs a quantum–inspired nonlocal
    transformation. Given a classical state vector x (of shape [batch_size, dim]), it:
      1. Constructs a density matrix from x.
      2. Evolves the density matrix via a modified Lindblad master equation over a fixed time T.
      3. Flattens the final density matrix.
      4. Applies a learnable linear projection to produce an output of shape [batch_size, dim].

    The module is fully differentiable and production–ready.
    """
    def __init__(self, dim: int, T: float = 1.0):
        """
        Args:
            dim: The dimension of the state vector (and the Hilbert space).
            T: The evolution time for the Lindblad dynamics.
        """
        super(NESTModule, self).__init__()
        self.dim = dim
        self.T = T
        H_init = random_hermitian(dim)
        self.H = nn.Parameter(H_init)
        self.log_kappa = nn.Parameter(torch.randn(1))
        self.register_buffer("L_base", lowering_operator(dim))
        self.out_layer = nn.Linear(dim * dim, dim)

    def _lindblad_rhs(self, t, rho_flat, H, L, kappa):
        dim = self.dim
        rho = rho_flat.view(dim, dim)
        commutator = torch.matmul(H, rho) - torch.matmul(rho, H)
        coherent = -1j * commutator
        L_rho = torch.matmul(L, rho)
        dissipative_term = torch.matmul(L_rho, L.conj().t())
        LL = torch.matmul(L.conj().t(), L)
        anticommutator = torch.matmul(LL, rho) + torch.matmul(rho, LL)
        dissipative = kappa * (dissipative_term - 0.5 * anticommutator)
        drho_dt = coherent + dissipative
        return drho_dt.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        kappa = F.softplus(self.log_kappa)
        for i in range(batch_size):
            psi = x[i]
            psi = psi / (torch.norm(psi) + 1e-8)
            rho = torch.outer(psi, psi.conj())
            rho_flat = rho.view(-1)
            def ode_func(t, rho_flat):
                return self._lindblad_rhs(t, rho_flat, self.H, self.L_base, kappa)
            t_span = torch.tensor([0.0, self.T], dtype=torch.float32)
            rho_t = odeint(ode_func, rho_flat, t_span, method='rk4')
            rho_final = rho_t[-1].view(self.dim, self.dim)
            rho_final = 0.5 * (rho_final + rho_final.conj().t())
            trace_rho = torch.trace(rho_final)
            if torch.abs(trace_rho) > 1e-8:
                rho_final = rho_final / trace_rho
            rho_flat_final = rho_final.view(-1)
            y = self.out_layer(rho_flat_final.real)
            outputs.append(y)
        out = torch.stack(outputs, dim=0)
        return out

###############################################################################
# Neural Cognitive Bus (NCB)
###############################################################################

class NeuralCognitiveBus(nn.Module):
    """
    Neural Cognitive Bus (NCB)
    ============================

    This module implements a production–grade, multi–channel, asynchronous communication
    system for HCDM modules. It supports:
      • Scalable multi–channel messaging with each channel implemented as an asyncio.Queue.
      • Optional integration of a Neural Entanglement State Transfer (NEST) module per channel
        for quantum–inspired data transformation.
      • Robust topic–based and filter–based routing: subscribers can register custom filter functions.
      • Lifecycle management: start/stop methods ensure all asynchronous tasks are cancelled gracefully.
      • Advanced error handling and logging for high–availability in enterprise environments.
    """
    def __init__(self, config_manager: Optional[Any] = None):
        super(NeuralCognitiveBus, self).__init__()
        self.config_manager = config_manager
        self.logger = (config_manager.setup_logger("NCB")
                       if config_manager else logging.getLogger("NCB"))
        self.channels: Dict[str, Dict[str, Any]] = {}   # channel_name -> { 'dim', 'tensor', 'queue' }
        self.subscribers: Dict[str, List[Dict[str, Any]]] = {}  # channel_name -> list of subscriber dicts
        self.nest_modules: Dict[str, nn.Module] = {}      # Optional NEST modules per channel
        self.running = False
        self.process_task: Optional[asyncio.Task] = None
        self.logger.info("Neural Cognitive Bus initialized.")

    def create_channel(self, channel_name: str, dim: int, use_nest: bool = True):
        if channel_name in self.channels:
            self.logger.warning(f"Channel '{channel_name}' already exists.")
            return
        self.channels[channel_name] = {
            "dim": dim,
            "tensor": torch.zeros(dim, dtype=torch.float32),
            "queue": asyncio.Queue(),
        }
        self.subscribers[channel_name] = []
        self.logger.debug(f"Channel '{channel_name}' created with dim={dim}.")
        if use_nest:
            nest_mod = NESTModule(dim)
            self.nest_modules[channel_name] = nest_mod
            self.logger.debug(f"NEST module attached to channel '{channel_name}'.")
        else:
            self.logger.debug(f"No NEST module attached to channel '{channel_name}'.")

    async def start(self):
        self.running = True
        self.process_task = asyncio.create_task(self._process_incoming_updates())
        self.logger.info("NCB started background processing.")

    async def stop(self):
        self.running = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                self.logger.debug("NCB process task cancelled cleanly.")
        self.logger.info("NCB stopped.")

    async def register_subscriber(
        self,
        channel_name: str,
        module_name: str,
        callback_fn: Callable[[Any], None],
        filter_fn: Optional[Callable[[Any], bool]] = None
    ):
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' does not exist.")
        self.subscribers[channel_name].append({
            "module_name": module_name,
            "callback": callback_fn,
            "filter_fn": filter_fn,
        })
        self.logger.debug(f"Subscriber '{module_name}' registered on channel '{channel_name}'.")

    async def publish(self, channel_name: str, data: Any):
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' does not exist.")
        # If data is a tensor and a NEST module is attached, process it.
        if isinstance(data, torch.Tensor):
            dim = self.channels[channel_name]["dim"]
            if data.dim() == 2 and data.shape[1] != dim:
                data = self._reshape_data(data, dim)
            elif data.dim() == 1 and data.shape[0] != dim:
                data = self._reshape_data(data.unsqueeze(0), dim).squeeze(0)
            if channel_name in self.nest_modules:
                data = self.nest_modules[channel_name](data)
            await self.channels[channel_name]["queue"].put(data.clone())
        else:
            await self.channels[channel_name]["queue"].put(data)
        self.logger.debug(
            f"Published data to channel '{channel_name}' (queue size: {self.channels[channel_name]['queue'].qsize()})."
        )

    async def _process_incoming_updates(self):
        try:
            while self.running:
                for chan_name, chan_info in self.channels.items():
                    queue = chan_info["queue"]
                    while not queue.empty():
                        new_data = await queue.get()
                        if isinstance(new_data, torch.Tensor):
                            chan_info["tensor"] = new_data
                        for sub in self.subscribers[chan_name]:
                            filt = sub["filter_fn"]
                            if (filt is None) or (filt(new_data)):
                                try:
                                    sub["callback"](new_data)
                                except Exception as e:
                                    self.logger.error(f"Error in subscriber callback on channel '{chan_name}': {e}", exc_info=True)
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            self.logger.debug("NCB update loop cancelled.")
        except Exception as e:
            self.logger.error(f"Exception in _process_incoming_updates: {e}", exc_info=True)

    def _reshape_data(self, data: torch.Tensor, dim: int) -> torch.Tensor:
        if data.shape[1] > dim:
            return data[:, :dim]
        else:
            pad = dim - data.shape[1]
            return torch.cat([data, torch.zeros(data.shape[0], pad, dtype=data.dtype)], dim=1)


####################################################################################
# dynamic_attention_routing.py (DAR)
####################################################################################
"""
Dynamic Attention Routing (DAR)
=================================

This module implements a production–grade, dynamic multi–route decision mechanism that
integrates environmental context, high–level gating signals from the Executive Function Module (EFM),
and advanced exploration–exploitation modulation. Instead of using a simplistic discrete
routing approach, it employs a continuous neural network (EnvContextNet) that outputs routing
logits for a scalable number of channels. The DAR module uses a PPO–style update mechanism to
adjust its policy based on a robust reward function drawn from external performance feedback.

Enhancements:
  • Integration with EFM: The module accepts an external gating signal (in [0,1]) from the EFM,
    which is used to modulate the routing logits.
  • Multi–Module Use: Supports a scalable number of channels and outputs a probability distribution
    over many routes.
  • Robust Reward Function: Rewards are based on how well the selected route improves the overall
    performance (e.g. by comparing predicted versus actual outcomes).
  • Scalability: The network is designed to handle many channels via a learned embedding and MLP,
    rather than a fixed small set of discrete routes.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from torch.distributions import Categorical

class EnvContextNet(nn.Module):
    """
    EnvContextNet: High–capacity network for producing routing logits and a critic value.
    It integrates embeddings for channel and source IDs, continuous features, and environmental
    context into a unified representation.
    """
    def __init__(
        self,
        max_channels: int,
        max_sources: int,
        embed_dim: int,
        cont_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_routes: int
    ):
        super().__init__()
        self.logger = logging.getLogger("EnvContextNet")
        self.channel_embedding = nn.Embedding(max_channels, embed_dim)
        self.source_embedding = nn.Embedding(max_sources, embed_dim)
        input_dim = (embed_dim * 2) + cont_dim + context_dim
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.route_head = nn.Linear(hidden_dim, num_routes)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self,
                channel_ids: torch.Tensor,
                source_ids: torch.Tensor,
                cont_feats: torch.Tensor,
                env_ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            ch_emb = self.channel_embedding(channel_ids)
            src_emb = self.source_embedding(source_ids)
            x = torch.cat([ch_emb, src_emb, cont_feats, env_ctx], dim=-1)
            h = self.relu(self.fc_in(x))
            h = self.relu(self.fc_hidden(h))
            route_logits = self.route_head(h)  # shape: (batch, num_routes)
            value = self.value_head(h)         # shape: (batch, 1)
            return route_logits, value
        except Exception as e:
            self.logger.error(f"Error in EnvContextNet.forward: {e}", exc_info=True)
            raise

@dataclass
class Transition:
    """
    Transition: Stores one transition in the rollout buffer for PPO updates.
    """
    obs: Dict[str, Any]
    route: int
    logp: float
    value: float
    reward: float
    next_obs: Dict[str, Any]
    done: bool

class RolloutBuffer:
    """
    RolloutBuffer: Buffer to accumulate transitions for PPO updates.
    """
    def __init__(self, gamma: float, lam: float, capacity: int = 64):
        self.gamma = gamma
        self.lam = lam
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0

    def size(self) -> int:
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def compute_gae(self, final_value: float = 0.0) -> Tuple[List[float], List[float]]:
        advantages = []
        returns = []
        values = np.array([t.value for t in self.transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.bool_)
        next_values = np.concatenate([values[1:], np.array([final_value], dtype=np.float32)], axis=0)
        gae = 0.0
        for i in reversed(range(len(self.transitions))):
            mask = 1.0 - dones[i].astype(np.float32)
            delta = rewards[i] + self.gamma * next_values[i] * mask - values[i]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        for i in range(len(self.transitions)):
            returns.append(values[i] + advantages[i])
        return advantages, returns

class DAR(nn.Module):
    """
    Dynamic Attention Routing (DAR)
    ================================

    This module implements a production–grade, dynamic multi–route decision mechanism that
    integrates environmental context, high–level gating signals from the Executive Function Module (EFM),
    and advanced exploration–exploitation modulation. Instead of using a simplistic discrete
    routing approach, it employs a continuous neural network (EnvContextNet) that outputs routing
    logits for a scalable number of channels. The DAR module uses a PPO–style update mechanism to
    adjust its policy based on a robust reward function drawn from external performance feedback.

    Enhancements:
      • Integration with EFM: The module accepts an external gating signal (in [0,1]) from the EFM,
        which is used to modulate the routing logits.
      • Multi–Module Use: Supports a scalable number of channels and outputs a probability distribution
        over many routes.
      • Robust Reward Function: Rewards are based on how well the selected route improves the overall
        performance (e.g. by comparing predicted versus actual outcomes).
      • Scalability: The network is designed to handle many channels via a learned embedding and MLP,
        rather than a fixed small set of discrete routes.

    Author: Jeremy Shows – Digital Hallucinations
    Date: Feb 14 2025
    """
    def __init__(
        self,
        max_channels: int = 20,
        max_sources: int = 20,
        embed_dim: int = 16,
        cont_dim: int = 1,
        context_dim: int = 2,
        hidden_dim: int = 64,
        num_routes: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
        capacity: int = 256,
        ppo_update_interval: int = 32,
        config_manager: Optional[ConfigManager] = None,
        efm: Optional[Any] = None
    ):
        super(DAR, self).__init__()
        self.logger = (config_manager.setup_logger("DAR")
                       if config_manager else logging.getLogger("DAR"))
        self.device = torch.device("cpu")
        self.num_routes = num_routes
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.ppo_update_interval = ppo_update_interval
        self.capacity = capacity
        self.efm = efm  # External gating signal provider

        # Instantiate the context network.
        self.context_net = EnvContextNet(max_channels, max_sources, embed_dim, cont_dim, context_dim, hidden_dim, num_routes).to(self.device)
        self.optimizer = optim.Adam(self.context_net.parameters(), lr=lr)

        # Initialize a rollout buffer for PPO.
        self.rollout_buffer = RolloutBuffer(gamma=self.gamma, lam=self.lam, capacity=capacity)
        self.logger.info("DynamicAttentionRouting initialized with {} routes.".format(num_routes))

    def forward(self, channel_ids: torch.Tensor, source_ids: torch.Tensor,
                salience: torch.Tensor, env_ctx: torch.Tensor,
                efm_gating: Optional[torch.Tensor] = None) -> Tuple[Categorical, torch.Tensor]:
        try:
            route_logits, value = self.context_net(channel_ids, source_ids, salience, env_ctx)
            if efm_gating is not None:
                # Modulate logits with external gating signal.
                gating = efm_gating.unsqueeze(1)  # (batch, 1)
                route_logits = route_logits * gating
            dist = Categorical(logits=route_logits)
            return dist, value
        except Exception as e:
            self.logger.error(f"Error in DAR.forward: {e}", exc_info=True)
            raise

    def route_data(self, obs: Dict[str, Any], next_obs: Optional[Dict[str, Any]] = None, done: bool = False) -> int:
        if next_obs is None:
            next_obs = obs

        channel_id = torch.tensor([obs.get("channel_id", 0)], dtype=torch.long, device=self.device)
        source_id = torch.tensor([obs.get("source_id", 0)], dtype=torch.long, device=self.device)
        sal_val = float(obs.get("salience", 0.0))
        sal_tensor = torch.tensor([[sal_val]], dtype=torch.float32, device=self.device)
        env_ctx_list = obs.get("env_context", [0.0, 0.0])
        env_ctx = torch.tensor([env_ctx_list], dtype=torch.float32, device=self.device)

        efm_gate = None
        if self.efm and hasattr(self.efm, "get_gating_signal"):
            try:
                gate_value = float(self.efm.get_gating_signal())
                efm_gate = torch.tensor([gate_value], dtype=torch.float32, device=self.device)
            except Exception as e:
                self.logger.error(f"Error obtaining gating signal from EFM: {e}", exc_info=True)

        with torch.no_grad():
            dist, value = self.forward(channel_id, source_id, sal_tensor, env_ctx, efm_gate)
            route_choice = dist.sample()
        logp = float(dist.log_prob(route_choice).item())
        route_int = int(route_choice.item())
        transition = Transition(
            obs=obs,
            route=route_int,
            logp=logp,
            value=value.item(),
            reward=0.0,
            next_obs=next_obs,
            done=done
        )
        self.rollout_buffer.add_transition(transition)
        if self.rollout_buffer.size() >= self.ppo_update_interval:
            self._ppo_update()
        self.logger.debug(f"Route selected: {route_int} for channel_id {channel_id.item()}")
        return route_int

    def give_reward(self, reward: float):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].reward += reward
            self.logger.debug(f"Reward {reward} assigned to latest transition.")

    def finalize_step(self, next_obs: Dict[str, Any], done: bool):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].next_obs = next_obs
            self.rollout_buffer.transitions[-1].done = done
            self.logger.debug("Finalized the latest transition.")

    def end_of_episode(self, final_value: float = 0.0):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].done = True
        self._ppo_update(final_value=final_value)

    def _ppo_update(self, final_value: float = 0.0):
        if self.rollout_buffer.is_empty():
            return
        self.logger.info("Starting PPO update for DAR.")
        advantages, returns = self.rollout_buffer.compute_gae(final_value=final_value)

        obs_channels = []
        obs_sources = []
        obs_saliences = []
        obs_env_ctxs = []
        old_logps = []
        old_values = []
        routes = []
        rewards = []
        dones = []

        for t in self.rollout_buffer.transitions:
            obs_channels.append(t.obs.get("channel_id", 0))
            obs_sources.append(t.obs.get("source_id", 0))
            obs_saliences.append(t.obs.get("salience", 0.0))
            obs_env_ctxs.append(t.obs.get("env_context", [0.0, 0.0]))
            old_logps.append(t.logp)
            old_values.append(t.value)
            routes.append(t.route)
            rewards.append(t.reward)
            dones.append(t.done)

        channel_ids_t = torch.tensor(obs_channels, dtype=torch.long, device=self.device)
        source_ids_t = torch.tensor(obs_sources, dtype=torch.long, device=self.device)
        salience_t = torch.tensor(obs_saliences, dtype=torch.float32, device=self.device).unsqueeze(-1)
        env_ctx_t = torch.tensor(obs_env_ctxs, dtype=torch.float32, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        routes_t = torch.tensor(routes, dtype=torch.long, device=self.device)

        data_size = self.rollout_buffer.size()
        indices = np.arange(data_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, data_size, self.mini_batch_size):
                batch_idx = indices[start:start+self.mini_batch_size]
                ch_b = channel_ids_t[batch_idx]
                src_b = source_ids_t[batch_idx]
                sal_b = salience_t[batch_idx]
                ctx_b = env_ctx_t[batch_idx]
                old_logp_b = old_logps_t[batch_idx]
                adv_b = advantages_t[batch_idx]
                ret_b = returns_t[batch_idx]
                route_b = routes_t[batch_idx]

                # For DAR, we assume no external gating during update (or use ones)
                gating_dummy = torch.ones((ch_b.shape[0],), dtype=torch.float32, device=self.device)
                route_logits, value_b = self.context_net(ch_b, src_b, sal_b, ctx_b)
                dist = Categorical(logits=route_logits)
                new_logps = dist.log_prob(route_b)
                ratio = torch.exp(new_logps - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = F.mse_loss(value_b.squeeze(-1), ret_b)
                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.logger.debug(f"Epoch {epoch+1}/{self.n_epochs} completed: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}")
        self.logger.info("PPO update for DAR completed.")
        self.rollout_buffer.clear()

    def get_gating_signal(self) -> float:
        if not self.rollout_buffer.transitions:
            return 1.0
        avg_reward = np.mean([t.reward for t in self.rollout_buffer.transitions])
        gating = 1.0 / (1.0 + np.exp(-avg_reward))
        self.logger.debug(f"Computed gating signal: {gating:.4f}")
        return gating

    async def async_route_data(self, obs: Dict[str, Any], efm_gating: Optional[float] = None) -> int:
        return await asyncio.to_thread(self.route_data, obs, None, False)

    async def async_update(self, batch: Dict[str, torch.Tensor],
                           gamma: float = 0.99, lam: float = 0.95, ppo_epochs: int = 4) -> Dict[str, float]:
        return await asyncio.to_thread(self._ppo_update, 0.0)


# AAN - advanced_attention_networks.py

"""
Advanced Attention Networks (AAN)
----------------------------------

This module implements an advanced attention mechanism that:
  • Combines multi–modal (cross–modal) saliency signals from the Sensory Processing Module.
  • Integrates both bottom–up (saliency) and top–down (gating from DAR/EFM) signals.
  • Computes multi–head attention via AdvancedAttentionNetworks.
  • Blends the new attention focus with the current state.
  • Broadcasts the final attention mask to all relevant modules via the Neural Cognitive Bus (NCB).
  • Trains a SelectivityGate using both supervised and reinforcement signals.
  
All operations are performed in a robust, asynchronous and fashion.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
"""

import math
import threading
import logging
import asyncio
import time
from functools import wraps
from typing import Optional, Callable, Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local imports (adjust the paths as necessary)
try:
    from neural_cognitive_bus import NeuralCognitiveBus
    from DAR import DAR
except ImportError:
    NeuralCognitiveBus = None
    DAR = None

# Assume we have a configuration manager in our project.
from modules.Config.config import ConfigManager

# =============================================================================
# Singleton Decorator
# =============================================================================

def singleton(cls):
    """
    Thread–safe singleton decorator.
    """
    cls._instance_lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = cls(*args, **kwargs)
        return cls._instance
    return wrapper

# =============================================================================
# Advanced Attention Networks: Multi–Head Self–Attention
# =============================================================================

"""
Advanced Attention Networks (AAN) Module
==========================================

This module implements an advanced attention mechanism that integrates:
  • Cross–modal saliency: It accepts multi–modal feature inputs (e.g., from text, vision, and audio)
    and projects each modality into a common embedding space.
  • Multi–head self–attention: The projected modalities are combined via multi–head self–attention,
    with robust scaling and dropout.
  • Top–down gating integration: External gating signals (from the EFM/DAR) are used to modulate
    the attention output.
  • A SelectivityGate: A learnable feedforward network that refines the computed attention mask,
    trained continuously using a compound loss that includes a reinforcement signal.
  • Global broadcasting: The final attention mask is published in real time over the Neural
    Cognitive Bus (NCB) to all interested modules.

The module is designed for a real–time, asynchronous environment and is fully instrumented
with thorough error handling and logging.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
"""

import math
import asyncio
import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume the configuration manager and Neural Cognitive Bus (NCB) are provided by your project.
from modules.Config.config import ConfigManager
from neural_cognitive_bus import NeuralCognitiveBus


# -----------------------------------------------------------------------------
# Advanced Attention Networks: Multi–Head Self–Attention for Cross–Modal Integration
# -----------------------------------------------------------------------------
class AdvancedAttentionNetworks(nn.Module):
    """
    Multi–head self–attention mechanism that integrates multi–modal saliency inputs.
    
    Each modality is first projected to a common embedding space using a learned linear layer.
    The resulting features are stacked to form a sequence, which is then processed with standard
    multi–head self–attention. Top–down gating is applied to the final output before being passed
    through a feed–forward network.
    """
    def __init__(self,
                 modalities: List[str],
                 projection_dim: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_mlp_hidden_size: int,
                 dropout_prob: float,
                 activation_function: str,
                 config_manager: ConfigManager):
        """
        Args:
            modalities: List of modality names (e.g., ['visual', 'auditory', 'text']).
            projection_dim: Target dimension for each modality’s projection.
            hidden_size: Dimension of the hidden representation used in attention.
            num_attention_heads: Number of attention heads.
            attention_mlp_hidden_size: Hidden layer size in the post-attention MLP.
            dropout_prob: Dropout probability.
            activation_function: Activation function name ('tanh', 'relu', or 'sigmoid').
            config_manager: Instance of ConfigManager for logging and parameters.
        """
        super(AdvancedAttentionNetworks, self).__init__()
        self.logger = config_manager.setup_logger("AdvancedAttentionNetworks")
        self.modalities = modalities
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.dropout_prob = dropout_prob
        self.activation_function = activation_function

        # Create learned projection layers for each modality.
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(in_features=projection_dim, out_features=hidden_size)
            for modality in modalities
        })

        # Multi-head attention layers.
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout_prob)
        
        # Final MLP and layer normalization.
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, attention_mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_mlp_hidden_size, hidden_size)
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.logger.info(
            f"AdvancedAttentionNetworks initialized with modalities: {modalities}, "
            f"hidden_size={hidden_size}, heads={num_attention_heads}."
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_size) and transpose.
        
        Input shape: (batch, seq_len, hidden_size)
        Output shape: (batch, num_heads, seq_len, head_size)
        """
        batch, seq_len, hidden = x.size()
        new_shape = (batch, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)  # (batch, seq_len, num_heads, head_size)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_size)

    def forward(self, modality_features: Dict[str, torch.Tensor],
                top_down_gating: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi–modal saliency integration.
        
        Args:
            modality_features: A dictionary mapping modality names to feature tensors.
              Each tensor is expected to have shape (batch, projection_dim).
            top_down_gating: Optional tensor of shape (batch, hidden_size) representing external gating.
        
        Returns:
            final_attention: Tensor of shape (batch, hidden_size) representing the computed attention mask.
        """
        try:
            # Project each modality to the hidden space.
            projected_list = []
            for modality in self.modalities:
                if modality not in modality_features:
                    self.logger.warning(f"Missing modality '{modality}' in input; using zeros.")
                    batch_size = next(iter(modality_features.values())).size(0)
                    proj = torch.zeros((batch_size, self.projection_dim), device=next(self.parameters()).device)
                else:
                    proj = modality_features[modality]
                # Project to hidden_size.
                proj = self.modality_projections[modality](proj)  # (batch, hidden_size)
                projected_list.append(proj.unsqueeze(1))  # (batch, 1, hidden_size)
            
            # Stack projected modalities to form a sequence.
            # Shape: (batch, n_modalities, hidden_size)
            modality_seq = torch.cat(projected_list, dim=1)
            
            # Compute query, key, value.
            Q = self.query_layer(modality_seq)  # (batch, n_modalities, hidden_size)
            K = self.key_layer(modality_seq)      # (batch, n_modalities, hidden_size)
            V = self.value_layer(modality_seq)    # (batch, n_modalities, hidden_size)
            
            # Split heads.
            Q = self.split_heads(Q)  # (batch, num_heads, n_modalities, head_size)
            K = self.split_heads(K)  # (batch, num_heads, n_modalities, head_size)
            V = self.split_heads(V)  # (batch, num_heads, n_modalities, head_size)
            
            # Compute scaled dot-product attention.
            attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, n_modalities, n_modalities)
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attention_dropout(attn_probs)
            context = torch.matmul(attn_probs, V)  # (batch, num_heads, n_modalities, head_size)
            # Concatenate heads.
            context = context.permute(0, 2, 1, 3).contiguous()  # (batch, n_modalities, num_heads, head_size)
            context = context.view(context.size(0), context.size(1), self.hidden_size)  # (batch, n_modalities, hidden_size)
            # Aggregate over modalities (e.g., weighted sum using learned parameters).
            aggregated = torch.mean(context, dim=1)  # (batch, hidden_size)
            
            # Apply top-down gating if provided.
            if top_down_gating is not None:
                if top_down_gating.dim() == 1:
                    top_down_gating = top_down_gating.unsqueeze(0)
                aggregated = aggregated * top_down_gating  # Elementwise multiplication.
            
            # Feed through final projection, residual connection, and MLP.
            out = self.output_proj(aggregated)
            out = self.attn_layer_norm(out + aggregated)
            mlp_out = self.mlp(out)
            final_out = self.final_layer_norm(mlp_out + out)
            
            # Apply activation function.
            if self.activation_function.lower() == "tanh":
                final_attention = torch.tanh(final_out)
            elif self.activation_function.lower() == "relu":
                final_attention = F.relu(final_out)
            elif self.activation_function.lower() == "sigmoid":
                final_attention = torch.sigmoid(final_out)
            else:
                self.logger.warning("Unknown activation function; using tanh as default.")
                final_attention = torch.tanh(final_out)
            
            self.logger.debug("AdvancedAttentionNetworks forward pass completed.")
            return final_attention  # (batch, hidden_size)
        except Exception as e:
            self.logger.error("Error in AdvancedAttentionNetworks.forward", exc_info=True)
            raise


# -----------------------------------------------------------------------------
# Selectivity Gate: Learned Gating for Refining Attention
# -----------------------------------------------------------------------------
class SelectivityGate(nn.Module):
    """
    The SelectivityGate refines the raw attention mask output from the advanced attention module.
    It is trained with a compound loss that includes both an MSE term (comparing gate outputs to a target)
    and a reinforcement term (reflecting performance feedback). This module ensures that the final attention
    distribution is both selective and optimized for downstream tasks.
    """
    def __init__(self, hidden_size: int, config_manager: ConfigManager):
        super(SelectivityGate, self).__init__()
        self.logger = config_manager.setup_logger("SelectivityGate")
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.logger.info(f"SelectivityGate initialized with hidden_size={hidden_size}.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gate.
        
        Args:
            x: Input tensor of shape (batch, hidden_size)
        
        Returns:
            Output tensor of shape (batch, hidden_size)
        """
        try:
            out = self.activation(self.linear(x))
            return out
        except Exception as e:
            self.logger.error("Error in SelectivityGate.forward", exc_info=True)
            raise

    def train_update(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, 
                     reinforcement_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update the gate using a compound loss function:
          Loss = MSE(input_transformed, target) + lambda * ReinforcementLoss,
        where ReinforcementLoss = MSE(input_transformed, reinforcement_signal) if provided.
        
        Args:
            input_tensor: Input tensor (batch, hidden_size)
            target_tensor: Target tensor (batch, hidden_size)
            reinforcement_signal: Optional reinforcement target tensor (batch, hidden_size)
        
        Returns:
            The computed loss.
        """
        try:
            output = self.forward(input_tensor)
            mse_loss = F.mse_loss(output, target_tensor)
            if reinforcement_signal is not None:
                reinforcement_loss = F.mse_loss(output, reinforcement_signal)
            else:
                reinforcement_loss = torch.tensor(0.0, device=input_tensor.device)
            lambda_factor = 0.5  # Weighting factor for reinforcement term.
            total_loss = mse_loss + lambda_factor * reinforcement_loss
            return total_loss
        except Exception as e:
            self.logger.error("Error in SelectivityGate.train_update", exc_info=True)
            raise


# -----------------------------------------------------------------------------
# Attention Manager (Singleton)
# -----------------------------------------------------------------------------
class AttentionManager(nn.Module):
    """
    The AttentionManager orchestrates the overall attention process. It subscribes to a
    saliency channel on the NCB to receive multi–modal features, queries external modules
    for top–down gating signals (via EFM or DAR), computes an attention mask using the
    AdvancedAttentionNetworks, refines it via the SelectivityGate (trained with reinforcement),
    and then publishes the final attention mask on a dedicated NCB channel.
    
    This is implemented as a singleton to ensure one global attention manager exists.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AttentionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 state_model: Any,
                 config_manager: ConfigManager,
                 ncb: NeuralCognitiveBus,
                 dar: Optional[Any] = None,
                 top_down_callback: Optional[callable] = None):
        """
        Args:
            state_model: The system’s state model (e.g., DSSM) to which attention is applied.
            config_manager: Provides configuration parameters and logging.
            ncb: Neural Cognitive Bus for inter–module communication.
            dar: Optional Dynamic Attention Routing module.
            top_down_callback: Optional callable that returns a top–down gating signal (tensor, shape (batch, hidden_size)).
        """
        super(AttentionManager, self).__init__()
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger("AttentionManager")
        self.ncb = ncb
        self.state_model = state_model
        self.dar = dar
        self.top_down_callback = top_down_callback  # This should return a tensor for gating.
        
        # Get configuration for the attention module.
        attn_cfg = self.config_manager.get_subsystem_config("attention_mechanism") or {}
        modalities = attn_cfg.get("modalities", ["visual", "auditory", "text"])
        projection_dim = attn_cfg.get("projection_dim", 512)
        hidden_size = attn_cfg.get("hidden_size", 256)
        num_heads = attn_cfg.get("num_attention_heads", 4)
        mlp_hidden_size = attn_cfg.get("attention_mlp_hidden_size", 128)
        dropout_prob = attn_cfg.get("dropout_prob", 0.1)
        activation_function = attn_cfg.get("activation_function", "tanh")
        
        # Instantiate the Advanced Attention module.
        self.advanced_attention = AdvancedAttentionNetworks(
            modalities=modalities,
            projection_dim=projection_dim,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_mlp_hidden_size=mlp_hidden_size,
            dropout_prob=dropout_prob,
            activation_function=activation_function,
            config_manager=config_manager
        ).to(next(self.parameters()).device)
        
        # Instantiate the SelectivityGate.
        self.selectivity_gate = SelectivityGate(hidden_size, config_manager).to(next(self.parameters()).device)
        
        # Internal state: current attention mask.
        self.current_attention: Optional[torch.Tensor] = None  # Shape: (batch, hidden_size)
        
        # Setup NCB channels.
        self.saliency_channel = attn_cfg.get("saliency_channel", "saliency_channel")
        self.attention_update_channel = attn_cfg.get("attention_update_channel", "attention_update_channel")
        # Create channels if not existing.
        self.ncb.create_channel(self.saliency_channel, projection_dim)
        self.ncb.create_channel(self.attention_update_channel, hidden_size)
        
        # Subscribe to saliency channel.
        asyncio.create_task(self._subscribe_to_saliency())
        
        self.logger.info("AttentionManager initialized and subscribed to saliency channel.")

    async def _subscribe_to_saliency(self) -> None:
        """
        Subscribe to the NCB saliency channel to receive multi–modal saliency inputs.
        """
        try:
            await self.ncb.register_subscriber(
                channel_name=self.saliency_channel,
                module_name="AttentionManager",
                callback_fn=self._saliency_callback
            )
            self.logger.info(f"Subscribed to NCB channel '{self.saliency_channel}'.")
        except Exception as e:
            self.logger.error("Error subscribing to saliency channel", exc_info=True)

    async def _saliency_callback(self, data: Any) -> None:
        """
        Callback for processing incoming saliency data from the NCB.
        Expects data to be a dictionary mapping modality names to lists (or tensors).
        """
        try:
            if not isinstance(data, dict):
                self.logger.error("Received saliency data is not a dictionary.")
                return
            # Convert all modality inputs to tensors.
            modality_features = {}
            for modality, value in data.items():
                if not isinstance(value, torch.Tensor):
                    modality_features[modality] = torch.tensor(value, dtype=torch.float32, device=next(self.parameters()).device)
                else:
                    modality_features[modality] = value.to(next(self.parameters()).device)
            # Query top–down gating signal if available.
            top_down_signal = None
            if self.top_down_callback is not None:
                try:
                    top_down_signal = self.top_down_callback()
                    if not isinstance(top_down_signal, torch.Tensor):
                        top_down_signal = torch.tensor(top_down_signal, dtype=torch.float32, device=next(self.parameters()).device)
                except Exception as e:
                    self.logger.error("Error obtaining top–down gating signal from callback", exc_info=True)
            # Alternatively, if DAR is available, query it.
            elif self.dar is not None and hasattr(self.dar, "get_gating_signal"):
                try:
                    gating = self.dar.get_gating_signal()
                    if not isinstance(gating, torch.Tensor):
                        gating = torch.tensor(gating, dtype=torch.float32, device=next(self.parameters()).device)
                    top_down_signal = gating
                except Exception as e:
                    self.logger.error("Error obtaining gating signal from DAR", exc_info=True)

            # Compute raw attention mask from advanced attention.
            raw_attention = self.advanced_attention(modality_features, top_down_gating=top_down_signal)
            self.logger.debug("Raw attention computed from multi–modal inputs.")

            # For training the SelectivityGate, determine a target.
            # In production, the target can come from the state model’s performance or a reinforcement signal.
            # Here we query the state model’s current attention focus as the target.
            if hasattr(self.state_model, "attention_focus"):
                target_focus = self.state_model.attention_focus
                if target_focus.dim() == 1:
                    target_focus = target_focus.unsqueeze(0)
            else:
                target_focus = raw_attention  # Fallback

            # Optionally, incorporate an external reinforcement signal (if available via NCB).
            # Here we assume that if a reinforcement signal was received, it is attached to the data.
            reinforcement_signal = None
            if "reinforcement_signal" in data:
                reinforcement_signal = data["reinforcement_signal"]
                if not isinstance(reinforcement_signal, torch.Tensor):
                    reinforcement_signal = torch.tensor(reinforcement_signal, dtype=torch.float32, device=raw_attention.device)
                if reinforcement_signal.dim() == 1:
                    reinforcement_signal = reinforcement_signal.unsqueeze(0)

            # Train the SelectivityGate with the compound loss.
            loss = self.selectivity_gate.train_update(raw_attention, target_focus, reinforcement_signal)
            # (In a full production system, you would backpropagate this loss using an optimizer.)
            # For demonstration, we perform an optimizer step:
            optimizer = torch.optim.Adam(self.selectivity_gate.parameters(), lr=1e-4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.logger.debug(f"SelectivityGate updated with loss: {loss.item():.6f}")

            # Compute the final refined attention mask.
            refined_attention = self.selectivity_gate(raw_attention)
            self.current_attention = refined_attention.detach()

            # Optionally, update the state model's attention focus.
            if hasattr(self.state_model, "attention_focus"):
                self.state_model.attention_focus = refined_attention.detach().cpu()

            # Broadcast the final attention mask via NCB.
            payload = {
                "attention_mask": refined_attention.detach().cpu().numpy().tolist(),
                "timestamp": time.time(),
                "source": "AttentionManager"
            }
            await self.ncb.publish(self.attention_update_channel, payload)
            self.logger.info("Final attention mask broadcast on channel '{}'.".format(self.attention_update_channel))
        except Exception as e:
            self.logger.error("Error in saliency callback processing", exc_info=True)

    def get_current_focus(self) -> Optional[torch.Tensor]:
        """
        Returns the current refined attention mask.
        """
        return self.current_attention

    async def update_top_down(self) -> torch.Tensor:
        """
        Query external sources (via top_down_callback or DAR) to obtain an updated top–down gating signal.
        """
        try:
            if self.top_down_callback is not None:
                gating = self.top_down_callback()
                if not isinstance(gating, torch.Tensor):
                    gating = torch.tensor(gating, dtype=torch.float32, device=next(self.parameters()).device)
                self.logger.info(f"Top–down gating signal updated via callback: {gating}")
                return gating
            elif self.dar is not None and hasattr(self.dar, "get_gating_signal"):
                gating = self.dar.get_gating_signal()
                if not isinstance(gating, torch.Tensor):
                    gating = torch.tensor(gating, dtype=torch.float32, device=next(self.parameters()).device)
                self.logger.info(f"Top–down gating signal updated via DAR: {gating}")
                return gating
            else:
                default = torch.ones((1, self.advanced_attention.hidden_size), device=next(self.parameters()).device)
                self.logger.info("No top–down gating source available; defaulting to ones.")
                return default
        except Exception as e:
            self.logger.error("Error obtaining top–down gating signal", exc_info=True)
            return torch.ones((1, self.advanced_attention.hidden_size), device=next(self.parameters()).device)

    async def initialize_subscriptions(self) -> None:
        """
        Ensure that the AttentionManager is subscribed to all necessary channels on the NCB.
        """
        try:
            await self._subscribe_to_saliency()
        except Exception as e:
            self.logger.error("Error during subscriptions initialization", exc_info=True)


# =============================================================================
# End of Advanced Attention Networks Module
# =============================================================================

if __name__ == "__main__":
    # For testing purposes, a basic async test harness is provided.
    import random
    async def test_main():
        # Setup dummy config manager.
        dummy_config = {
            "attention_mechanism": {
                "modalities": ["visual", "auditory", "text"],
                "projection_dim": 512,
                "hidden_size": 256,
                "num_attention_heads": 4,
                "attention_mlp_hidden_size": 128,
                "dropout_prob": 0.1,
                "activation_function": "tanh",
                "saliency_channel": "saliency_channel",
                "attention_update_channel": "attention_update_channel"
            }
        }
        class DummyConfigManager:
            def __init__(self, config):
                self.config = config
            def get_subsystem_config(self, name: str) -> Dict[str, Any]:
                return self.config.get(name, {})
            def setup_logger(self, name: str) -> logging.Logger:
                logger = logging.getLogger(name)
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)
                    logger.setLevel(logging.DEBUG)
                return logger
        config_manager = DummyConfigManager(dummy_config)
        
        # Dummy NCB that prints published payloads.
        class DummyNCB:
            def __init__(self):
                self.channels = {}
            def create_channel(self, channel_name: str, dim: int):
                self.channels[channel_name] = []
            async def publish(self, channel_name: str, data: Any):
                print(f"[NCB] Published on channel '{channel_name}': {data}")
            async def register_subscriber(self, channel_name: str, module_name: str, callback_fn: callable):
                print(f"[NCB] Registered subscriber '{module_name}' on channel '{channel_name}'")
        ncb = DummyNCB()
        
        # Dummy state model with an attention_focus property.
        class DummyStateModel:
            def __init__(self):
                self.attention_focus = torch.zeros((1, 256))
            def get_current_state(self):
                return {"attention_focus": self.attention_focus.tolist()}
        state_model = DummyStateModel()
        
        # Dummy top_down_callback returning a random gating tensor.
        def dummy_top_down():
            return torch.rand((1, 256))
        
        # Instantiate AttentionManager.
        attention_manager = AttentionManager(state_model, config_manager, ncb, top_down_callback=dummy_top_down)
        await attention_manager.initialize_subscriptions()
        
        # Simulate incoming saliency data.
        sample_saliency = {
            "visual": [random.random() for _ in range(512)],
            "auditory": [random.random() for _ in range(512)],
            "text": [random.random() for _ in range(512)],
            "reinforcement_signal": [random.random() for _ in range(256)]
        }
        await attention_manager._saliency_callback(sample_saliency)
        current_focus = attention_manager.get_current_focus()
        print("Current Attention Focus:", current_focus)
    
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_main())


###############################################################################
# enhanced_memory_model.py
###############################################################################

"""
Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
"""


import os
import json
import time
import torch
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

import aiofiles

# Import production–grade memory modules
from modules.HCDM.Memory.Sensory.sensory_memory import SensoryMemory
from modules.HCDM.Memory.Short_Term.short_term_memory import ShortTermMemory
from modules.HCDM.Memory.Working_Memory.working_memory import WorkingMemory
from modules.HCDM.Memory.Intermediate_Memory.intermediate_memory import IntermediateMemory
from modules.HCDM.Memory.Long_Term.Episodic.long_term_episodic_memory import EnhancedLongTermEpisodicMemory
from modules.HCDM.Memory.Long_Term.Semantic.long_term_semantic_memory import LongTermSemanticMemory
from modules.HCDM.Memory.Retrieval.context_aware_retrieval import ContextAwareRetrieval

# Time–based processes for consolidation and spaced repetition
from modules.HCDM.Time_Processing.circadian_sleep_processes_simulator import (
    TimeDecay,
    SpacedRepetition,
    MemoryConsolidationThread,
    MemoryType
)

# Emotional / Motivational module (enterprise–grade)
from modules.HCDM.Emo.emotional_motivational_module import EMoM

# Neural Cognitive Bus for inter–module publishing
from neural_cognitive_bus import NeuralCognitiveBus

# Dynamic State Space Model (DSSM)
from modules.HCDM.SSM.state_space_model import DSSM

# Configuration manager
from modules.Config.config import ConfigManager

# Replay Buffer (assumed production–grade)
from modules.Replay.replay_buffer import ReplayBuffer


class EMM:
    """
    Enhanced Memory Model (EMM)

    Integrates multiple memory subsystems (sensory, short-term, working memory,
    intermediate, and long-term episodic/semantic memory). It features a replay
    buffer, time-based consolidation via a dedicated thread, optional emotional
    modulation via an EMoM instance, and publishing via a Neural Cognitive Bus (NCB).
    """
    def __init__(
        self,
        state_model: Optional[DSSM] = None,
        file_path: Optional[str] = None,
        provider_manager: Optional[Any] = None,
        config_manager: Optional[ConfigManager] = None,
        ncb: Optional[NeuralCognitiveBus] = None,
        dar: Optional[Any] = None,
        emom: Optional[EMoM] = None
    ):
        self.config_manager = config_manager
        self.logger = (config_manager.setup_logger("EMM")
                       if config_manager else logging.getLogger("EMM"))
        self.state_model = state_model

        # File for persistence
        self.file_path = file_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'memory_store.json'
        )
        self.provider_manager = provider_manager
        self.ncb = ncb
        self.dar = dar
        self.emom = emom  # Either provided externally or initialized below
        if not self.emom:
            self.emom = self._maybe_init_emom()

        # Time–based processes (only if state_model is provided)
        if self.state_model:
            self.time_decay = TimeDecay(system_state=self.state_model, config_manager=config_manager)
            self.spaced_repetition = SpacedRepetition(memory_store=self, config_manager=config_manager)
        else:
            self.time_decay = None
            self.spaced_repetition = None
            self.logger.warning("No state model provided; time-aware processing disabled.")

        # Instantiate memory layers
        self.sensory = SensoryMemory(config_manager)
        self.short_term = ShortTermMemory(config_manager)
        self.working_memory = WorkingMemory(config_manager)
        self.intermediate = IntermediateMemory(config_manager)
        self.long_term_episodic = EnhancedLongTermEpisodicMemory(self.state_model, config_manager, self)
        self.long_term_semantic = LongTermSemanticMemory(config_manager)
        self.context_retrieval = (ContextAwareRetrieval(self.state_model, config_manager)
                                  if self.state_model else None)

        self.consciousness_stream = None

        # Replay Buffer
        mem_conf = self.config_manager.get_subsystem_config('memory') if self.config_manager else {}
        replay_capacity = mem_conf.get('replay_buffer_size', 200)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.max_memory_entries = mem_conf.get('max_memory_entries', 10000)

        # Launch memory consolidation thread if components are available
        if self.time_decay and self.spaced_repetition and self.provider_manager:
            self.memory_consolidation_thread = MemoryConsolidationThread(
                memory_store=self,
                spaced_repetition=self.spaced_repetition,
                provider_manager=self.provider_manager,
                config_manager=self.config_manager,
                system_state=self.state_model
            )
            self.memory_consolidation_thread.start()
            self.logger.debug("MemoryConsolidationThread started.")
        else:
            self.memory_consolidation_thread = None
            self.logger.warning("Memory consolidation thread not started (missing components).")

        self.logger.info("EMM initialized with all memory modules and replay logic.")

    def _maybe_init_emom(self) -> Optional[EMoM]:
        """
        Initialize an EMoM instance from configuration.
        """
        if not self.config_manager:
            return None
        emom_config = self.config_manager.get_subsystem_config("emom")
        if not emom_config:
            self.logger.warning("No EMoM configuration found; EMoM integration disabled.")
            return None
        try:
            external_input_dim = emom_config.get("external_input_dim", 50)
            internal_input_dim = emom_config.get("internal_input_dim", 10)
            affective_state_dim = emom_config.get("affective_state_dim", 3)
            hidden_dims = emom_config.get("hidden_dims", [128, 64])
            dropout = emom_config.get("dropout", 0.1)
            device = self.state_model.device if self.state_model else torch.device("cpu")
            new_emom = EMoM(
                config_manager=self.config_manager,
                external_input_dim=external_input_dim,
                internal_input_dim=internal_input_dim,
                affective_state_dim=affective_state_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                device=device
            )
            return new_emom
        except Exception as e:
            self.logger.error(f"Failed to initialize EMoM automatically: {e}", exc_info=True)
            return None

    async def initialize(self) -> "EMM":
        """
        Load memory from file (if exists) and prepare the model.
        """
        self.logger.info("Initializing EMM.")
        try:
            if os.path.exists(self.file_path):
                await self._load_memory()
            else:
                self.logger.info("No memory file found; starting with an empty store.")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory: {e}", exc_info=True)
        return self

    async def close(self) -> None:
        """
        Graceful shutdown: stop consolidation thread, save and backup memory.
        """
        self.logger.info("Closing EMM.")
        if self.memory_consolidation_thread:
            await self.memory_consolidation_thread.stop()
            self.memory_consolidation_thread.join()
            self.logger.debug("MemoryConsolidationThread stopped.")
        await self._save_memory()
        await self._backup_memory()
        self.logger.info("EMM fully closed.")

    async def _load_memory(self) -> None:
        """
        Load memory from JSON file.
        """
        try:
            self.logger.info(f"Loading memory from {self.file_path}")
            async with aiofiles.open(self.file_path, 'r') as infile:
                data = json.loads(await infile.read())
            for ep in data.get('episodic', []):
                ctx = torch.tensor(ep.get('context', []), dtype=torch.float32)
                await self.long_term_episodic.add(ep['content'], ctx)
            if self.long_term_semantic:
                for concept, related_list in data.get('semantic', {}).items():
                    await self.long_term_semantic.add(concept, related_list)
            self.logger.info("Memory loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading memory from {self.file_path}: {e}", exc_info=True)

    async def _save_memory(self) -> None:
        """
        Save memory to JSON file.
        """
        try:
            data = {
                'episodic': self.long_term_episodic.episodes if self.long_term_episodic else [],
                'semantic': {}
            }
            if self.long_term_semantic:
                data['semantic'] = {
                    node: list(self.long_term_semantic.knowledge_graph.neighbors(node))
                    for node in self.long_term_semantic.knowledge_graph.nodes()
                }
            async with aiofiles.open(self.file_path, 'w') as outfile:
                await outfile.write(json.dumps(data, indent=2))
            self.logger.info(f"Memory saved to {self.file_path}")
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}", exc_info=True)

    async def _backup_memory(self) -> None:
        """
        Create a backup copy of the memory file.
        """
        try:
            backup_path = self.file_path + ".bak"
            async with aiofiles.open(self.file_path, 'r') as infile:
                text_data = await infile.read()
            async with aiofiles.open(backup_path, 'w') as outfile:
                await outfile.write(text_data)
            self.logger.info(f"Backup of memory created at {backup_path}")
        except Exception as e:
            self.logger.error(f"Error during memory backup: {e}", exc_info=True)

    def set_consciousness_stream(self, stream: Any) -> None:
        """
        Connect a continuous consciousness stream.
        """
        self.consciousness_stream = stream
        self.logger.info("Consciousness stream set for EMM.")

    def set_ncb(self, ncb: NeuralCognitiveBus):
        """
        Attach a Neural Cognitive Bus instance.
        """
        self.ncb = ncb
        self.logger.debug("NCB set for EMM.")

    def set_dar(self, dar: Any):
        """
        Attach the Dynamic Attention Routing (DAR) module.
        """
        self.dar = dar
        self.logger.debug("DAR set for EMM.")

    def _wrap_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Standardize input data.
        """
        if isinstance(input_data, str):
            return {
                'content': input_data,
                'timestamp': time.time(),
                'emotional_state': 0.5,
                'salience': 1.0,
                'tags': []
            }
        elif isinstance(input_data, dict):
            wrapped = dict(input_data)
            wrapped.setdefault('timestamp', time.time())
            wrapped.setdefault('salience', 1.0)
            wrapped.setdefault('tags', [])
            return wrapped
        else:
            return {
                'content': str(input_data),
                'timestamp': time.time(),
                'emotional_state': 0.5,
                'salience': 1.0,
                'tags': []
            }

    async def process_input(self, input_data: Any) -> Any:
        """
        Ingest new input into the memory pipeline, apply emotional tagging,
        update memory layers, update state model, update replay buffer, and publish.
        """
        try:
            self.logger.debug(f"EMM processing input: {input_data}")
            wrapped = self._wrap_input(input_data)

            # Apply EMoM modulation if available
            if self.emom:
                content = wrapped.get("content", "")
                # Use provider_manager for a high-quality embedding (if available)
                external_signal = self.provider_manager.huggingface_generator.transformer_encode(content)
                if not isinstance(external_signal, torch.Tensor):
                    external_signal = torch.tensor(external_signal, dtype=torch.float32, device=self.emom.device)
                internal_signal = torch.ones((1, 10), dtype=torch.float32, device=self.emom.device) * 0.5
                affective_state_tensor = self.emom(external_signal, internal_signal)
                wrapped["emotional_state"] = affective_state_tensor.squeeze(0).tolist()
                # Adjust salience based on affect intensity
                affect_weight = max(abs(wrapped["emotional_state"][0]), abs(wrapped["emotional_state"][1]))
                wrapped["salience"] = 1.0 + affect_weight

            # Feed through memory layers
            self.sensory.add(wrapped)
            self.short_term.add(wrapped)
            self.working_memory.add(wrapped)
            self.intermediate.add(wrapped)

            if self.state_model:
                await self.state_model.update({'new_input': wrapped})

            priority = wrapped.get("salience", 1.0)
            self.replay_buffer.add(wrapped, priority=priority)

            await self.publish_to_ncb(wrapped)
            return wrapped

        except Exception as e:
            self.logger.error(f"Error in EMM.process_input: {e}", exc_info=True)
            return None

    def _convert_to_tensor(self, data: Any, dim: int = 256) -> torch.Tensor:
        """
        Convert input data to a fixed-length 1D float tensor.
        In production, use a robust embedding service.
        """
        if isinstance(data, dict) and 'content' in data:
            s = data['content']
        else:
            s = str(data)
        arr = [float(ord(c)) for c in s]
        t = torch.tensor(arr, dtype=torch.float32)
        if t.shape[0] > dim:
            t = t[:dim]
        else:
            pad_len = dim - t.shape[0]
            t = torch.cat([t, torch.zeros(pad_len, dtype=torch.float32)])
        return t

    async def publish_to_ncb(self, memory_signal: Any) -> None:
        """
        Publish the memory update to the NCB on the "memory_channel".
        """
        if not self.ncb:
            self.logger.debug("No NCB set; skipping publish.")
            return
        try:
            data_tensor = self._convert_to_tensor(memory_signal, dim=256)
            await self.ncb.publish("memory_channel", data_tensor)
            self.logger.debug("EMM published update to 'memory_channel'.")
        except Exception as e:
            self.logger.error(f"Failed to publish to NCB: {e}", exc_info=True)

    async def consolidate_memory(self) -> None:
        """
        Consolidate short-term and intermediate memories into long-term episodic memory.
        """
        try:
            context_vector = await self.get_current_state_context()
            items_to_consolidate = self.short_term.retrieve() + self.intermediate.retrieve()
            threshold = self.adjust_consolidation_threshold()
            for m in items_to_consolidate:
                content_str = m.get('content', '')
                consolidated_content = f"Consolidated: {content_str}"
                await self.long_term_episodic.add(consolidated_content, context_vector)
            self.short_term.clear()
            self.intermediate.clear()
            await self._save_memory()
            self.logger.info("Memory consolidation complete.")
        except Exception as e:
            self.logger.error(f"Error in consolidate_memory: {e}", exc_info=True)

    async def get_current_state_context(self) -> torch.Tensor:
        """
        Retrieve the current context vector from the state model.
        """
        try:
            if self.context_retrieval:
                return await self.context_retrieval.get_context_vector()
            return torch.zeros(256, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"Error in get_current_state_context: {e}", exc_info=True)
            return torch.zeros(256, dtype=torch.float32)

    def adjust_consolidation_threshold(self) -> float:
        """
        Adjust consolidation threshold based on neuromodulatory modulation.
        """
        base_threshold = 0.7
        if not self.emom:
            return base_threshold
        try:
            # Assume EMoM provides a modulation factor via its affective state.
            affective_state = self.emom.get_current_affective_state()
            # For enterprise use, a more robust mapping would be used.
            modulation = 1.0 - 0.2 * (affective_state[0] - 0.5)
            return base_threshold * modulation
        except Exception as e:
            self.logger.error(f"Error adjusting threshold: {e}", exc_info=True)
            return base_threshold

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Return memory usage statistics.
        """
        try:
            stats = {
                "sensory_size": len(self.sensory.retrieve()),
                "short_term_size": len(self.short_term.retrieve()),
                "working_memory_size": len(self.working_memory.retrieve()),
                "intermediate_size": len(self.intermediate.retrieve()),
                "long_term_episodic_size": len(self.long_term_episodic.episodes),
                "long_term_semantic_size": len(self.long_term_semantic.knowledge_graph.nodes()),
                "replay_buffer_size": self.replay_buffer.size()
            }
            total = sum(stats.values())
            if total > self.max_memory_entries:
                self.logger.warning(f"Memory usage {total} exceeds limit {self.max_memory_entries}!")
            self.logger.debug(f"EMM memory stats: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error in get_memory_stats: {e}", exc_info=True)
            return {}

    async def cleanup_memory(self, threshold: float = 0.1) -> None:
        """
        Remove old or low-salience items from long-term episodic memory.
        """
        try:
            if not self.long_term_episodic or not self.time_decay:
                return
            current_time = time.time()
            orig_count = len(self.long_term_episodic.episodes)
            new_episodes = []
            for ep in self.long_term_episodic.episodes:
                ts = ep.get('timestamp')
                if ts is None:
                    new_episodes.append(ep)
                    continue
                time_elapsed = current_time - float(ts)
                importance = ep.get('importance', 1.0)
                val = self.time_decay.decay(MemoryType.LONG_TERM_EPISODIC, time_elapsed, importance)
                if val >= threshold:
                    new_episodes.append(ep)
            removed = orig_count - len(new_episodes)
            self.long_term_episodic.episodes = new_episodes
            if removed > 0:
                self.logger.info(f"Cleaned {removed} episodes from LTM (value below {threshold}).")
        except Exception as e:
            self.logger.error(f"Error in cleanup_memory: {e}", exc_info=True)
        return

    def adapt_from_rpe(self, rpe: float) -> None:
        """
        Force consolidation of short-term memory into intermediate memory if RPE is high.
        """
        try:
            if abs(rpe) > 0.5:
                new_items = self.short_term.retrieve()
                for item in new_items:
                    self.intermediate.add(item, importance=1.0)
                self.short_term.clear()
                self.logger.debug(f"Adaptation from RPE {rpe:.3f}: forced consolidation.")
        except Exception as e:
            self.logger.error(f"Error in adapt_from_rpe: {e}", exc_info=True)

    async def get_recent_context(self) -> str:
        """
        Retrieve recent sensory memory items as a concatenated string.
        """
        try:
            recent = self.sensory.retrieve()[-5:]
            lines = [str(item.get("content", "")) for item in recent]
            return " | ".join(lines)
        except Exception as e:
            self.logger.error(f"Error in get_recent_context: {e}", exc_info=True)
            return ""

    def get_state_vector(self) -> List[float]:
        """
        Provide a brief state vector for integration with other modules.
        """
        try:
            stats = self.get_memory_stats()
            return [
                float(stats.get("sensory_size", 0)),
                float(stats.get("short_term_size", 0)),
                float(stats.get("replay_buffer_size", 0))
            ]
        except Exception as e:
            self.logger.error(f"Error in get_state_vector: {e}", exc_info=True)
            return [0.0, 0.0, 0.0]


# replay_buffer.py

import random
import logging
from typing import Any, List, Tuple

class ReplayBuffer:
    """
    A robust replay buffer that supports priority sampling and batch retrieval.
    Each memory entry is stored as a tuple: (memory_item, priority).
    """
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, float]] = []
        self.logger = logging.getLogger("ReplayBuffer")
    
    def add(self, memory_item: Any, priority: float = 1.0) -> None:
        """
        Add a memory item with an associated priority.
        If capacity is exceeded, the item with the lowest priority is removed.
        """
        self.buffer.append((memory_item, priority))
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda x: x[1])
            removed_item, removed_priority = self.buffer.pop(0)
            self.logger.debug(f"ReplayBuffer capacity exceeded, removed item with priority {removed_priority}.")
        self.logger.debug(f"Added memory to ReplayBuffer with priority {priority}.")
    
    def sample(self, batch_size: int = 32) -> List[Any]:
        """
        Sample a batch of memory items using probability proportional to their priority.
        """
        if not self.buffer:
            return []
        priorities = [priority for (_, priority) in self.buffer]
        total_priority = sum(priorities)
        if total_priority == 0:
            probabilities = [1/len(self.buffer)] * len(self.buffer)
        else:
            probabilities = [p / total_priority for p in priorities]
        sampled_items = random.choices(self.buffer, weights=probabilities, k=min(batch_size, len(self.buffer)))
        return [item for (item, _) in sampled_items]

    def clear(self) -> None:
        """
        Clear the replay buffer.
        """
        self.buffer.clear()
        self.logger.debug("ReplayBuffer cleared.")
    
    def size(self) -> int:
        return len(self.buffer)


# working_memory.py

import time
import logging
from typing import Any, List
from modules.Config.config import ConfigManager

class WorkingMemory:
    """
    Working Memory module:
      - Acts as a temporary, actively manipulated storage.
      - Implements a standardized API: add(item), retrieve(), and clear().
    """
    def __init__(self, config_manager: ConfigManager, capacity: int = 50):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger("WorkingMemory")
        self.capacity = capacity
        self.items: List[Any] = []
        self.logger.info(f"Initialized WorkingMemory with capacity {self.capacity}.")

    def add(self, item: Any) -> None:
        """
        Add an item to working memory. If capacity is exceeded, remove the oldest item.
        """
        self.items.append(item)
        if len(self.items) > self.capacity:
            removed = self.items.pop(0)
            self.logger.debug(f"WorkingMemory capacity exceeded, removed oldest item: {removed}")
        self.logger.debug(f"Added item to WorkingMemory: {item}")

    def retrieve(self) -> List[Any]:
        """
        Retrieve a copy of all items currently stored.
        """
        self.logger.debug("Retrieving items from WorkingMemory.")
        return self.items.copy()

    def clear(self) -> None:
        """
        Clear all items from working memory.
        """
        count = len(self.items)
        self.items.clear()
        self.logger.debug(f"Cleared WorkingMemory, removed {count} items.")


# intermediate_memory.py

import time
from typing import List, Any, Dict
import asyncio
import logging
import torch
from modules.Config.config import ConfigManager
from modules.HCDM.Time_Processing.circadian_sleep_processes_simulator import TimeDecay, SpacedRepetition, MemoryType

class IntermediateMemory:
    """
    Intermediate Memory buffers items for eventual consolidation into long-term memory.
    Standard API: add(item), retrieve(), and clear().
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('IntermediateMemory')
        memory_config = config_manager.get_subsystem_config('memory')
        self.capacity = memory_config.get('intermediate_memory', {}).get('capacity', 1000)
        self.consolidation_threshold = memory_config.get('consolidation_threshold', 0.7)
        self.memories: List[Dict[str, Any]] = []
        self.logger.info(f"Initialized IntermediateMemory with capacity: {self.capacity}")
        self.time_decay = TimeDecay(system_state=None, config_manager=self.config_manager)
        self.spaced_repetition = SpacedRepetition(memory_store=self, config_manager=self.config_manager)
    
    def add(self, memory: Any, importance: float = 1.0) -> None:
        if len(self.memories) >= self.capacity:
            self.consolidate_oldest()
        memory_entry = {
            'content': memory,
            'timestamp': time.time(),
            'importance': importance
        }
        self.memories.append(memory_entry)
        preview = memory[:50] + "..." if isinstance(memory, str) and len(memory) > 50 else str(memory)
        self.logger.debug(f"Added memory: {preview} with importance {importance}")
    
    def retrieve(self) -> List[Any]:
        self.logger.debug("Retrieving memories from IntermediateMemory.")
        return self.memories.copy()
    
    def clear(self) -> None:
        original_count = len(self.memories)
        self.memories = self.memories[-(self.capacity // 2):]
        self.logger.debug(f"Cleared {original_count - len(self.memories)} consolidated memories from IntermediateMemory.")
    
    def consolidate_oldest(self) -> Dict[str, Any]:
        if not self.memories:
            self.logger.warning("No memories available for consolidation.")
            return {}
        oldest_memory = min(self.memories, key=lambda m: m['timestamp'])
        self.memories.remove(oldest_memory)
        preview = oldest_memory['content'][:50] + "..." if isinstance(oldest_memory['content'], str) and len(oldest_memory['content']) > 50 else str(oldest_memory['content'])
        self.logger.debug(f"Consolidating memory: {preview}")
        time_elapsed = time.time() - oldest_memory['timestamp']
        decayed_strength = self.time_decay.decay(memory_type=MemoryType.LONG_TERM_EPISODIC,
                                                  time_elapsed=time_elapsed,
                                                  importance=oldest_memory.get('importance', 1.0))
        if decayed_strength > self.consolidation_threshold:
            review_time = time.time() + self.spaced_repetition.sm2_params.get("interval", 1) * 86400
            self.spaced_repetition.schedule_review(memory=oldest_memory, review_time=review_time, emotion_factor=1.0)
            self.logger.debug(f"Memory scheduled for spaced repetition: {preview}")
        else:
            self.logger.debug(f"Memory discarded due to low strength: {preview}")
        return oldest_memory
    
    async def process_memories(self) -> None:
        memories_to_cons = []
        current_time = time.time()
        for memory in self.memories[:]:
            time_elapsed = current_time - memory['timestamp']
            strength = self.time_decay.decay(memory_type=MemoryType.LONG_TERM_EPISODIC,
                                               time_elapsed=time_elapsed,
                                               importance=memory.get('importance', 1.0))
            if strength > self.consolidation_threshold:
                memories_to_cons.append(memory)
                self.memories.remove(memory)
                preview = memory['content'][:50] + "..." if isinstance(memory['content'], str) and len(memory['content']) > 50 else str(memory['content'])
                self.logger.debug(f"Memory marked for consolidation: {preview}")
        for memory in memories_to_cons:
            quality = await self.simulate_review_quality(memory)
            if quality >= 3:
                self.spaced_repetition.review(memory, quality)
                preview = memory['content'][:50] + "..." if isinstance(memory['content'], str) and len(memory['content']) > 50 else str(memory['content'])
                self.logger.debug(f"Memory reviewed successfully: {preview}")
            else:
                preview = memory['content'][:50] + "..." if isinstance(memory['content'], str) and len(memory['content']) > 50 else str(memory['content'])
                self.logger.debug(f"Memory review quality low ({quality}) for memory: {preview}")
        self.clear()
    
    async def simulate_review_quality(self, memory: Dict[str, Any]) -> int:
        import random
        quality = random.randint(0, 5)
        self.logger.debug(f"Simulated review quality: {quality}")
        return quality


# LTSM - long_term_semantic_memory.py

import torch
import networkx as nx
import logging
import asyncio
from typing import Dict, Any, List, Optional

from modules.Config.config import ConfigManager

class LongTermSemanticMemory:
    """
    Long-term semantic memory stores concepts and their related semantic embeddings.
    This module uses a knowledge graph to represent inter-concept relationships.
    All numerical data is represented as PyTorch tensors.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = config_manager.setup_logger('LongTermSemanticMemory')
        self.knowledge_graph = nx.Graph()
        # Memory vectors: keys are concept names, values are PyTorch tensor embeddings
        self.memory_vectors: Dict[str, torch.Tensor] = {}
        # Configuration for embedding size
        semantic_config = self.config_manager.get_subsystem_config('semantic_memory') or {}
        self.embedding_dim = semantic_config.get('embedding_dim', 128)
        self.logger.info(f"Initialized LongTermSemanticMemory with embedding_dim: {self.embedding_dim}")

    async def add(self, concept: str, related_concepts: List[str]) -> None:
        """
        Adds a new semantic concept and its related concepts to the knowledge graph.
        Generates an embedding for the concept if one does not already exist.
        
        Args:
            concept (str): The semantic concept.
            related_concepts (List[str]): List of related concept names.
        """
        if concept not in self.memory_vectors:
            # Initialize embedding with random tensor (or load a pretrained embedding)
            self.memory_vectors[concept] = torch.randn(self.embedding_dim, dtype=torch.float32)
            self.logger.debug(f"Generated new embedding for concept: {concept}")

        if concept not in self.knowledge_graph:
            self.knowledge_graph.add_node(concept)

        for rel in related_concepts:
            if rel not in self.memory_vectors:
                self.memory_vectors[rel] = torch.randn(self.embedding_dim, dtype=torch.float32)
                self.logger.debug(f"Generated new embedding for related concept: {rel}")
            self.knowledge_graph.add_edge(concept, rel)
            self.logger.debug(f"Added edge between {concept} and {rel}")

    async def query(self, concept: str, n: int = 5) -> List[tuple]:
        """
        Queries the knowledge graph for the most related concepts to the given concept.
        The similarity is computed as cosine similarity between embeddings.
        
        Args:
            concept (str): The concept to query.
            n (int, optional): Number of related concepts to return. Defaults to 5.
        
        Returns:
            List[tuple]: A list of (concept, similarity) tuples.
        """
        import torch.nn.functional as F

        if concept not in self.memory_vectors:
            self.logger.warning(f"Concept {concept} not found in semantic memory.")
            return []
        query_embedding = self.memory_vectors[concept]
        similarities = []
        for other_concept, emb in self.memory_vectors.items():
            if other_concept == concept:
                continue
            sim = F.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0), dim=1).item()
            similarities.append((other_concept, sim))
        similarities.sort(key=lambda tup: tup[1], reverse=True)
        result = similarities[:n]
        self.logger.debug(f"Query result for concept {concept}: {result}")
        return result

    async def preload_LTMS(self, preload_data: Dict[str, List[str]]) -> None:
        """
        Preloads semantic memory from provided data.
        
        Args:
            preload_data (Dict[str, List[str]]): Mapping from concept to related concepts.
        """
        self.logger.info("Preloading Long-Term Semantic Memory...")
        for concept, related in preload_data.items():
            await self.add(concept, related)
        self.logger.info("Preloading completeed.")

    def pattern_separation(self, concept1: str, concept2: str) -> Optional[float]:
        """
        Computes a separation metric between two concepts. Lower values indicate higher similarity.
        
        Args:
            concept1 (str): The first concept.
            concept2 (str): The second concept.
        
        Returns:
            Optional[float]: The separation value (e.g., 1 - cosine similarity) or None if one concept is missing.
        """
        import torch.nn.functional as F
        if concept1 not in self.memory_vectors or concept2 not in self.memory_vectors:
            self.logger.warning(f"One or both concepts not found: {concept1}, {concept2}")
            return None
        emb1 = self.memory_vectors[concept1]
        emb2 = self.memory_vectors[concept2]
        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
        separation = 1.0 - cos_sim
        return separation


# CAR - context_aware_retrieval.py

import torch
from modules.Config.config import ConfigManager
from modules.HCDM.SSM.state_space_model import DSSM

class ContextAwareRetrieval:
    """
    Retrieves the current context vector from the state model and computes cosine similarity
    between stored context and the current context using PyTorch operations.
    """

    def __init__(self, state_model: DSSM, config_manager: ConfigManager):
        self.state_model = state_model
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ContextAwareRetrieval')

    async def get_context_vector(self) -> torch.Tensor:
        return await self.state_model.get_current_state_context()

    async def context_similarity(self, memory_context: torch.Tensor, current_context: torch.Tensor) -> float:
        # Use torch.nn.functional.cosine_similarity
        import torch.nn.functional as F
        similarity = F.cosine_similarity(memory_context.unsqueeze(0), current_context.unsqueeze(0), dim=1)
        return similarity.item()


# SM - sensory_memory.py

import time
from typing import Any, List, Dict
import torch
from modules.Config.config import ConfigManager

class SensoryMemory:
    """
    Sensory Memory stores preprocessed sensory inputs.
    Standard API: add(item), retrieve(), and clear().
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('SensoryMemory')
        memory_config = self.config_manager.get_subsystem_config('memory')
        sensory_config = memory_config.get('sensory_memory', {})
        self.max_size = sensory_config.get('max_size', 100)
        self.decay_rate = sensory_config.get('decay_rate', 0.1)
        self.buffer: List[Dict[str, Any]] = []
        self.logger.info(f"Initialized SensoryMemory with max_size: {self.max_size} and decay_rate: {self.decay_rate}")
    
    def add(self, input_data: Any) -> None:
        processed = self._preprocess_input(input_data)
        timestamp = time.time()
        entry = {"data": processed, "timestamp": timestamp, "salience": 1.0}
        self.buffer.append(entry)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        self.logger.debug(f"Added processed input to SensoryMemory: {processed}")
    
    def retrieve(self) -> List[Any]:
        self.update()
        self.logger.debug("Retrieving items from SensoryMemory.")
        return [item["data"] for item in self.buffer]
    
    def clear(self) -> None:
        count = len(self.buffer)
        self.buffer.clear()
        self.logger.debug(f"Cleared SensoryMemory, removed {count} items.")
    
    def _preprocess_input(self, input_data: Any) -> Any:
        if isinstance(input_data, str):
            return self._process_text(input_data)
        elif isinstance(input_data, torch.Tensor):
            return self._process_visual(input_data)
        return input_data
    
    def _process_text(self, text: str) -> str:
        return ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    
    def _process_visual(self, image: torch.Tensor) -> torch.Tensor:
        min_val = torch.min(image)
        max_val = torch.max(image)
        return (image - min_val) / (max_val - min_val + 1e-8)
    
    def update(self) -> None:
        current_time = time.time()
        for item in self.buffer:
            dt = current_time - item["timestamp"]
            item["salience"] *= torch.exp(torch.tensor(-self.decay_rate * dt, dtype=torch.float32)).item()
            item["salience"] = max(item["salience"], 0.1)
        self.logger.debug("SensoryMemory updated with decay.")


# STM - short_term_memory.py

from typing import List, Any, Optional
from modules.Config.config import ConfigManager

class ShortTermMemory:
    """
    Short-Term Memory for transient storage.
    Standard API: add(item), retrieve(), and clear().
    """
    def __init__(self, config_manager: ConfigManager, capacity: Optional[int] = None):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ShortTermMemory')
        memory_config = config_manager.get_subsystem_config('memory')
        self.capacity = capacity or memory_config.get('short_term_memory', {}).get('capacity', 100)
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.logger.info(f"Initialized ShortTermMemory with capacity: {self.capacity}")
        self.items: List[Any] = []
    
    def add(self, item: Any) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]
        self.logger.debug(f"Added item to ShortTermMemory: {item}")
    
    def retrieve(self) -> List[Any]:
        self.logger.debug("Retrieving items from ShortTermMemory.")
        return self.items.copy()
    
    def clear(self) -> None:
        count = len(self.items)
        self.items.clear()
        self.logger.debug(f"Cleared ShortTermMemory, removed {count} items.")


###############################################################################
# circadian_sleep_processes_simulator.py
###############################################################################

"""
Circadian and Sleep Processes Simulator (CSPS)

This module implements a robust circadian/sleep processing system for a neuromodulatory framework.
It integrates:
  • A detailed TimeDecay module that computes a circadian multiplier from a 24–hour sinusoidal schedule,
    with configurable sleep windows.
  • A complete SM2–based SpacedRepetition system that schedules and triggers offline replay during sleep.
  • A MemoryConsolidationThread that:
       – Periodically consolidates memory (via memory_system.consolidate_memory())
       – Triggers offline replay of high–priority memories during sleep periods
       – Dynamically adjusts its sleep interval based on the current circadian multiplier.
  • A top–level CircadianSleepProcessesSimulator (CSPS) that starts/stops all processes and reports the current
    circadian state, notifying connected modules (such as an Executive Function Module) when sleep mode is active.
  """

import math
import time
import datetime
import logging
import asyncio
import threading
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch

# configuration manager
from modules.Config.config import ConfigManager


###############################################################################
# TimeDecay
###############################################################################
class TimeDecay:
    """
    TimeDecay implements time–based decay of memory traces modulated by realistic
    circadian (day/night) cycles. It computes a circadian multiplier using a sinusoidal
    function over a 24–hour period, with a fixed multiplier during a configurable sleep window.
    This multiplier is used downstream to adjust decay rates, learning rates, and other parameters.
    """
    def __init__(self, config_manager: ConfigManager):
        self.logger = config_manager.setup_logger("TimeDecay")
        config = config_manager.get_subsystem_config("time_aware_processing") or {}
        # Base decay rates for different memory types.
        self.base_decay_rates: Dict[str, float] = config.get("decay_rates", {
            "sensory": 0.1,
            "short_term": 0.01,
            "long_term_epidodic": 0.001,
            "long_term_semantic": 0.0001
        })
        # Circadian parameters.
        self.circadian_period: float = 86400  # 24 hours in seconds.
        self.circadian_min: float = config.get("circadian_min", 0.5)   # Minimum multiplier (e.g., during sleep).
        self.circadian_max: float = config.get("circadian_max", 1.5)   # Maximum multiplier (daytime peak).
        # Sleep window definitions.
        self.sleep_start: str = config.get("sleep_start", "22:00")
        self.sleep_end: str = config.get("sleep_end", "06:00")
        self.logger.info(f"TimeDecay initialized with circadian_min={self.circadian_min}, "
                         f"circadian_max={self.circadian_max}, sleep window={self.sleep_start} to {self.sleep_end}")

    def _parse_time(self, time_str: str) -> datetime.time:
        """Parse a time string 'HH:MM' into a datetime.time object."""
        try:
            hour, minute = map(int, time_str.strip().split(":"))
            return datetime.time(hour=hour, minute=minute)
        except Exception as e:
            self.logger.error(f"Error parsing time string '{time_str}': {e}", exc_info=True)
            return datetime.time(hour=0, minute=0)

    def is_nighttime(self, current_time: Optional[float] = None) -> bool:
        """
        Determine whether the current time falls within the sleep window.
        """
        try:
            if current_time is None:
                current_time = time.time()
            now = datetime.datetime.fromtimestamp(current_time)
            sleep_start = self._parse_time(self.sleep_start)
            sleep_end = self._parse_time(self.sleep_end)
            if sleep_start < sleep_end:
                return sleep_start <= now.time() <= sleep_end
            else:
                return now.time() >= sleep_start or now.time() <= sleep_end
        except Exception as e:
            self.logger.error(f"Error in is_nighttime: {e}", exc_info=True)
            return False

    def get_circadian_multiplier(self, current_time: Optional[float] = None) -> float:
        """
        Compute a circadian multiplier based on the current time.
        If within the sleep window, returns a fixed multiplier (circadian_min).
        Otherwise, computes a sinusoidal value between circadian_min and circadian_max.
        """
        try:
            if current_time is None:
                current_time = time.time()
            now = datetime.datetime.fromtimestamp(current_time)
            if self.is_nighttime(current_time):
                multiplier = self.circadian_min
                self.logger.debug(f"Current time {now.time()} is within sleep window; multiplier set to {multiplier}")
                return multiplier
            # Compute seconds since midnight.
            midnight = datetime.datetime.combine(now.date(), datetime.time(0, 0))
            seconds_since_midnight = (now - midnight).total_seconds()
            phase = (2 * math.pi * seconds_since_midnight) / self.circadian_period
            sin_value = math.sin(phase)
            normalized = (sin_value + 1) / 2  # Normalize to [0, 1].
            multiplier = self.circadian_min + normalized * (self.circadian_max - self.circadian_min)
            self.logger.debug(f"Circadian multiplier at {now.time()}: {multiplier:.3f}")
            return multiplier
        except Exception as e:
            self.logger.error(f"Error in get_circadian_multiplier: {e}", exc_info=True)
            return 1.0

    def decay(self, memory_type: str, time_elapsed: float, importance: float) -> float:
        """
        Compute the decayed strength of a memory trace.
        The decay is exponential and modulated by the current circadian multiplier.
        """
        try:
            base_rate = self.base_decay_rates.get(memory_type, 0.1)
            multiplier = self.get_circadian_multiplier()
            effective_rate = base_rate * multiplier
            decayed_value = math.exp(-effective_rate * time_elapsed) * importance
            self.logger.debug(
                f"Decaying '{memory_type}' memory: time_elapsed={time_elapsed:.2f}s, base_rate={base_rate}, "
                f"multiplier={multiplier:.3f}, effective_rate={effective_rate:.3f}, "
                f"importance={importance}, decayed_value={decayed_value:.3f}"
            )
            return decayed_value
        except Exception as e:
            self.logger.error(f"Error in decay: {e}", exc_info=True)
            return 0.0

    def get_consolidation_interval(self) -> float:
        """
        Dynamically compute the memory consolidation interval (in seconds) based on the circadian multiplier.
        A lower multiplier (e.g., during sleep) yields a shorter interval (more frequent consolidation).
        """
        base_interval = 3600.0  # 1 hour base.
        multiplier = self.get_circadian_multiplier()
        # Invert multiplier (ensuring no division by zero) so that a lower multiplier means a shorter interval.
        inv_multiplier = 1.0 / max(multiplier, 0.1)
        interval = base_interval * inv_multiplier
        self.logger.debug(f"Consolidation interval computed: {interval:.2f} seconds (base_interval={base_interval}, multiplier={multiplier:.3f})")
        return interval


###############################################################################
# SpacedRepetition
###############################################################################
class SpacedRepetition:
    """
    Implements a spaced repetition system using an enhanced SM2 algorithm.
    Memories are scheduled for review based on performance feedback. During sleep,
    the system triggers offline replay for consolidation.
    """
    def __init__(self, config_manager: ConfigManager):
        self.logger = config_manager.setup_logger("SpacedRepetition")
        config = config_manager.get_subsystem_config("time_aware_processing") or {}
        self.initial_interval: float = config.get("sm2_initial_interval", 86400)  # 1 day by default.
        self.factor: float = config.get("sm2_factor", 2.5)
        self.review_queue: List[Tuple[float, Dict[str, Any]]] = []
        self.logger.info(
            f"SpacedRepetition initialized with initial_interval={self.initial_interval}, factor={self.factor}"
        )

    def schedule_review(self, memory: Dict[str, Any], performance: float) -> None:
        """
        Schedule a memory for review using an SM2–style update.
        """
        try:
            performance = max(0.0, min(performance, 1.0))
            if performance < 0.3:
                interval = self.initial_interval
            else:
                interval = self.initial_interval * (self.factor ** (performance * 5))
            review_time = time.time() + interval
            self.review_queue.append((review_time, memory))
            self.review_queue.sort(key=lambda x: x[0])
            self.logger.info(
                f"Scheduled review for memory '{memory.get('content', '')[:30]}...' in {interval:.2f} seconds (performance={performance:.2f})"
            )
        except Exception as e:
            self.logger.error(f"Error in schedule_review: {e}", exc_info=True)

    async def process_reviews(self, memory_system: Any) -> None:
        """
        Process all memory reviews whose scheduled time has arrived by invoking memory_system.replay_memory.
        """
        try:
            now = time.time()
            ready = [item for item in self.review_queue if item[0] <= now]
            if not ready:
                return
            for review_time, memory in ready:
                self.logger.info(
                    f"Processing review for memory '{memory.get('content', '')[:30]}...' scheduled at {review_time}"
                )
                await memory_system.replay_memory(memory)
            self.review_queue = [item for item in self.review_queue if item[0] > now]
        except Exception as e:
            self.logger.error(f"Error in process_reviews: {e}", exc_info=True)


###############################################################################
# MemoryConsolidationThread
###############################################################################
class MemoryConsolidationThread(threading.Thread):
    """
    Runs a background asynchronous loop that:
      - Consolidates short-term and intermediate memories into long-term episodic memory.
      - Triggers offline replay of selected memories during sleep.
      - Dynamically adjusts its sleep interval based on the circadian multiplier.
      - Notifies the Executive Function Module (EFM) to switch modes.
    """
    def __init__(
        self,
        memory_system: Any,
        spaced_repetition: SpacedRepetition,
        config_manager: ConfigManager,
        time_decay: TimeDecay,
        efm: Optional[Any] = None
    ):
        super(MemoryConsolidationThread, self).__init__()
        self.memory_system = memory_system
        self.spaced_repetition = spaced_repetition
        self.config_manager = config_manager
        self.logger = config_manager.setup_logger("MemoryConsolidationThread")
        self.time_decay = time_decay
        self.efm = efm
        self.running = True
        self.daemon = True
        self.base_interval: float = config_manager.get_subsystem_config("time_aware_processing").get("base_consolidation_interval", 3600)
        self.logger.info(f"MemoryConsolidationThread initialized with base_interval={self.base_interval} seconds.")

    def run(self) -> None:
        self.logger.info("MemoryConsolidationThread started.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while self.running:
                loop.run_until_complete(self._consolidate_and_replay())
                # Dynamically adjust sleep interval.
                multiplier = self.time_decay.get_circadian_multiplier()
                circadian_max = self.config_manager.get_subsystem_config("time_aware_processing").get("circadian_max", 1.5)
                interval = self.base_interval * (multiplier / circadian_max)
                self.logger.info(f"MemoryConsolidationThread sleeping for {interval:.2f} seconds.")
                time.sleep(interval)
        except Exception as e:
            self.logger.error(f"Error in MemoryConsolidationThread run loop: {e}", exc_info=True)
        finally:
            loop.close()
            self.logger.info("MemoryConsolidationThread terminated.")

    async def _consolidate_and_replay(self) -> None:
        try:
            self.logger.info("Starting memory consolidation cycle.")
            await self.memory_system.consolidate_memory()
            self.logger.info("Memory consolidation completed.")
            if self.time_decay.is_nighttime():
                self.logger.info("Nighttime detected; triggering offline replay.")
                top_memories = await self.memory_system.retrieve_top_memories(limit=10)
                for memory in top_memories:
                    performance = memory.get("performance", 0.5)
                    self.spaced_repetition.schedule_review(memory, performance)
                await self.spaced_repetition.process_reviews(self.memory_system)
                if self.efm and hasattr(self.efm, "enter_sleep_mode"):
                    self.efm.enter_sleep_mode()
                    self.logger.info("Notified EFM to enter sleep mode.")
            else:
                if self.efm and hasattr(self.efm, "exit_sleep_mode"):
                    self.efm.exit_sleep_mode()
                    self.logger.info("Notified EFM to exit sleep mode.")
        except Exception as e:
            self.logger.error(f"Error during consolidation and replay: {e}", exc_info=True)

    async def stop(self) -> None:
        self.running = False
        self.logger.info("MemoryConsolidationThread stop requested.")


###############################################################################
# CircadianSleepProcessesSimulator (CSPS)
###############################################################################
class CircadianSleepProcessesSimulator:
    """
    Orchestrates all circadian and sleep–related processes by tying together TimeDecay,
    SpacedRepetition, and MemoryConsolidationThread. It provides start/stop interfaces and
    reports the current circadian state while notifying connected modules (e.g. EFM) when
    entering/exiting sleep mode.
    """
    def __init__(self, config_manager: ConfigManager, memory_system: Any, efm: Optional[Any] = None):
        self.config_manager = config_manager
        self.logger = config_manager.setup_logger("CSPS")
        self.memory_system = memory_system
        self.efm = efm
        self.time_decay = TimeDecay(config_manager)
        self.spaced_repetition = SpacedRepetition(config_manager)
        self.consolidation_thread = MemoryConsolidationThread(
            memory_system, self.spaced_repetition, config_manager, self.time_decay, efm
        )
        self.running = False

    def start(self) -> None:
        self.logger.info("Starting CircadianSleepProcessesSimulator.")
        self.consolidation_thread.start()
        self.running = True

    def stop(self) -> None:
        self.logger.info("Stopping CircadianSleepProcessesSimulator.")
        self.consolidation_thread.running = False
        self.running = False

    def get_current_circadian_state(self) -> Dict[str, Any]:
        try:
            current_time = time.time()
            multiplier = self.time_decay.get_circadian_multiplier(current_time)
            is_night = self.time_decay.is_nighttime(current_time)
            now = datetime.datetime.fromtimestamp(current_time)
            sleep_start = self.time_decay._parse_time(self.time_decay.sleep_start)
            sleep_end = self.time_decay._parse_time(self.time_decay.sleep_end)
            if is_night:
                sleep_end_dt = datetime.datetime.combine(now.date(), sleep_end)
                if sleep_end_dt < now:
                    sleep_end_dt += datetime.timedelta(days=1)
                time_until_phase = (sleep_end_dt - now).total_seconds()
                next_phase = "wake"
            else:
                sleep_start_dt = datetime.datetime.combine(now.date(), sleep_start)
                if sleep_start_dt < now:
                    sleep_start_dt += datetime.timedelta(days=1)
                time_until_phase = (sleep_start_dt - now).total_seconds()
                next_phase = "sleep"
            return {
                "multiplier": multiplier,
                "is_nighttime": is_night,
                "time_until_next_phase": time_until_phase,
                "next_phase": next_phase,
                "current_time": now.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting circadian state: {e}", exc_info=True)
            return {"multiplier": 1.0, "is_nighttime": False, "time_until_next_phase": 0, "next_phase": "unknown"}


# =============================================================================
# Test Script (for standalone testing)
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Dummy configuration manager (replace with actual enterprise ConfigManager)
    class DummyConfigManager:
        def __init__(self):
            self.config = {
                "time_aware_processing": {
                    "decay_rates": {
                        "sensory": 0.1,
                        "short_term": 0.01,
                        "long_term_epidodic": 0.001,
                        "long_term_semantic": 0.0001
                    },
                    "circadian_min": 0.5,
                    "circadian_max": 1.5,
                    "sleep_start": "22:00",
                    "sleep_end": "06:00",
                    "base_consolidation_interval": 3600,
                    "sm2_initial_interval": 86400,
                    "sm2_factor": 2.5
                }
            }
        def get_subsystem_config(self, subsystem_name: str) -> Dict[str, Any]:
            return self.config.get(subsystem_name, {})
        def get(self, key: str, default: Any = None) -> Any:
            return self.config.get(key, default)
        def setup_logger(self, name: str) -> logging.Logger:
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.DEBUG)
            return logger

    # Dummy memory system with minimal implementations.
    class DummyMemorySystem:
        async def consolidate_memory(self):
            print("Memory consolidated.")
        async def retrieve_top_memories(self, limit: int = 10):
            return [{"content": "Sample memory", "performance": 0.7} for _ in range(limit)]
        async def replay_memory(self, memory: Dict[str, Any]):
            print(f"Replaying memory: {memory.get('content', '')}")

    dummy_cm = DummyConfigManager()
    dummy_memory = DummyMemorySystem()
    csps = CircadianSleepProcessesSimulator(config_manager=dummy_cm, memory_system=dummy_memory)
    csps.start()
    state = csps.get_current_circadian_state()
    print("Current circadian state:", state)
    # Let the thread run briefly, then stop.
    time.sleep(5)
    csps.stop()

# dynamic_state_space_model.py (DSSM)

"""
Dynamic State Space Model (DSSM)

This module implements a robust state–estimation system based on an Unscented Kalman Filter (UKF),
a selective state transformation network, and additional neural sub–modules. It also incorporates
time–aware processing and cognitive temporal state management (with configurable transition rules)
and integrates a memory consolidation thread with spaced repetition for offline replay.
"""

import time
import math
import threading
from enum import Enum
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.Config.config import ConfigManager  # configuration manager


# =============================================================================
# Cognitive Temporal State
# =============================================================================

class CognitiveTemporalStateEnum(Enum):
    IMMEDIATE = 1
    EMOTIONAL = 2
    ANALYTICAL = 3


class CognitiveTemporalStateConfig:
    def __init__(self, 
                 alpha: float,
                 scaling_bounds: Tuple[float, float],
                 state_transition_rules: Dict[CognitiveTemporalStateEnum, Dict[str, Any]],
                 initial_state: CognitiveTemporalStateEnum,
                 initial_scaling: float):
        """
        Parameters:
            alpha (float): Smoothing factor for state updates.
            scaling_bounds (tuple): Minimum and maximum scaling factors (e.g. (0.5, 2.0)).
            state_transition_rules (dict): Mapping from each state to a set of rules. For each state, specify:
                - 'arousal_upper': threshold above which transition to EMOTIONAL occurs.
                - 'arousal_lower': threshold below which transition to ANALYTICAL occurs.
                - 'cognitive_load_threshold': threshold on cognitive load.
                - 'scaling_multiplier': multiplier applied to the base scaling factor.
                - 'transition_delay': minimum time in seconds before a new transition.
            initial_state (CognitiveTemporalStateEnum): The starting state.
            initial_scaling (float): The starting scaling factor.
        """
        self.alpha = alpha
        self.scaling_bounds = scaling_bounds
        self.state_transition_rules = state_transition_rules
        self.initial_state = initial_state
        self.initial_scaling = initial_scaling


class CognitiveTemporalState:
    def __init__(self, config: CognitiveTemporalStateConfig):
        """
        Initializes the cognitive temporal state system.
        """
        self.config = config
        self.current_state = config.initial_state
        self.scaling_factor = config.initial_scaling
        self.last_transition_time = time.time()

    def update(self, arousal: float, cognitive_load: float) -> None:
        """
        Update the temporal state based on measured arousal and cognitive load.
        
        Parameters:
            arousal (float): A value in [0, 1] indicating arousal level.
            cognitive_load (float): A value in [0, 1] indicating cognitive load.
        
        This method consults the configured transition rules and only changes state if a minimum
        delay has elapsed.
        """
        current_time = time.time()
        time_since_transition = current_time - self.last_transition_time
        rules = self.config.state_transition_rules.get(self.current_state, {})
        min_delay = rules.get('transition_delay', 10)
        if time_since_transition < min_delay:
            return  # Do not change state if transition delay has not elapsed.

        if arousal > rules.get('arousal_upper', 0.7) and self.current_state != CognitiveTemporalStateEnum.EMOTIONAL:
            self.current_state = CognitiveTemporalStateEnum.EMOTIONAL
        elif arousal < rules.get('arousal_lower', 0.3) and self.current_state != CognitiveTemporalStateEnum.ANALYTICAL:
            self.current_state = CognitiveTemporalStateEnum.ANALYTICAL
        else:
            self.current_state = CognitiveTemporalStateEnum.IMMEDIATE

        self.last_transition_time = current_time
        multiplier = self.config.state_transition_rules[self.current_state].get('scaling_multiplier', 1.0)
        base, top = self.config.scaling_bounds
        self.scaling_factor = ((base + top) / 2) * multiplier

    def get_current_state(self) -> CognitiveTemporalStateEnum:
        """Return the current cognitive temporal state."""
        return self.current_state

    def get_scaling_factor(self) -> float:
        """Return the current scaling factor for state modulation."""
        return self.scaling_factor


# =============================================================================
# Spaced Repetition
# =============================================================================

class SpacedRepetition:
    def __init__(self, config: Dict[str, Any]):
        """
        Implements spaced repetition using an SM2-based algorithm.
        
        Parameters:
            config (dict): Configuration parameters including:
                - 'sm2_initial_interval': Initial review interval in seconds.
                - 'sm2_factor': Exponential factor for increasing the interval.
        """
        self.sm2_initial_interval = config.get("sm2_initial_interval", 86400)
        self.sm2_factor = config.get("sm2_factor", 2.5)
    
    def schedule_review(self, memory: Dict[str, Any], quality: int) -> float:
        """
        Compute the next review interval for a memory based on its quality.
        
        Parameters:
            memory (dict): Memory item.
            quality (int): Review quality score (0 to 5).
            
        Returns:
            float: Next review interval in seconds.
        """
        if quality < 3:
            interval = self.sm2_initial_interval
        else:
            interval = self.sm2_initial_interval * (self.sm2_factor ** quality)
        return interval


# =============================================================================
# Memory Consolidation Thread
# =============================================================================

class MemoryConsolidationThread(threading.Thread):
    def __init__(self,
                 memory_store: Any,
                 spaced_repetition: SpacedRepetition,
                 provider_manager: Any,
                 config_manager: ConfigManager,
                 system_state: Any):
        """
        Initializes the memory consolidation thread.
        
        Parameters:
            memory_store: An object with a consolidate_memory() method.
            spaced_repetition: Instance of the spaced repetition system.
            provider_manager: External provider manager (for any required API calls).
            config_manager: Configuration manager for logging and settings.
            system_state: The current system state (e.g., DSSM) for time–aware signals.
        """
        super(MemoryConsolidationThread, self).__init__()
        self.memory_store = memory_store
        self.spaced_repetition = spaced_repetition
        self.provider_manager = provider_manager
        self.config_manager = config_manager
        self.system_state = system_state
        self.logger = config_manager.setup_logger("MemoryConsolidationThread")
        tdec_config = config_manager.get_subsystem_config("time_aware_processing")
        self.base_interval = tdec_config.get("base_consolidation_interval", 3600)
        self.running = True
        self.daemon = True

    def run(self) -> None:
        self.logger.info("MemoryConsolidationThread started.")
        while self.running:
            try:
                # Consolidate memory (this should be a robust, call)
                self.memory_store.consolidate_memory()
                # Process short-term memories: evaluate and schedule reviews.
                memories = self.memory_store.short_term.retrieve()
                if memories:
                    for memory in memories:
                        quality = self._evaluate_memory(memory)
                        next_interval = self.spaced_repetition.schedule_review(memory, quality)
                        memory['next_review'] = time.time() + next_interval
                        self.logger.info(f"Scheduled review for memory in {next_interval:.2f} seconds.")
                    self.memory_store.short_term.clear()
                # Use a time–aware consolidation interval if available.
                interval = self.base_interval
                if hasattr(self.system_state, 'time_decay'):
                    interval = self.system_state.time_decay.get_consolidation_interval()
                self.logger.info(f"Sleeping for consolidation interval: {interval:.2f} seconds.")
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error during memory consolidation: {e}", exc_info=True)
                time.sleep(self.base_interval)
        self.logger.info("MemoryConsolidationThread terminated.")

    def _evaluate_memory(self, memory: Dict[str, Any]) -> int:
        """
        Evaluate a memory's quality based on its salience.
        
        In a production system, this would use retrieval performance metrics.
        """
        salience = memory.get("salience", 1.0)
        quality = int(min(max(salience * 5, 0), 5))
        return quality

    def stop(self) -> None:
        self.running = False


# =============================================================================
# UKF Module
# =============================================================================

class PyTorchUKFModule(nn.Module):
    """
    A robust, implementation of an Unscented Kalman Filter (UKF) in PyTorch.
    This module includes complete sigma–point generation, prediction, and update steps with error handling.
    """
    def __init__(self,
                 dim_x: int,
                 dim_z: int,
                 dt: float,
                 fx: Callable[[torch.Tensor, float], torch.Tensor],
                 hx: Callable[[torch.Tensor], torch.Tensor],
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1,
                 device: Optional[torch.device] = None):
        """
        Args:
            dim_x: Dimension of the state.
            dim_z: Dimension of the measurement.
            dt: Time step.
            fx: Nonlinear state transition function.
            hx: Nonlinear measurement function.
            alpha, beta, kappa: UKF scaling parameters.
            process_noise: Scalar for process noise covariance.
            measurement_noise: Scalar for measurement noise covariance.
            device: Computation device.
        """
        super(PyTorchUKFModule, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = math.sqrt(self.dim_x + self.lambda_)
        self.device = device if device is not None else torch.device("cpu")

        self.x = torch.zeros(self.dim_x, device=self.device)
        self.P = torch.eye(self.dim_x, device=self.device) * process_noise
        self.Q = torch.eye(self.dim_x, device=self.device) * process_noise
        self.R = torch.eye(self.dim_z, device=self.device) * measurement_noise

        # Precompute weights
        self.Wm = torch.full((2 * self.dim_x + 1,), 1.0 / (2 * (self.dim_x + self.lambda_)), device=self.device)
        self.Wc = self.Wm.clone()
        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        self.logger = torch.log(self.device)
        self.logger = torch.log(self.device)  # (dummy assignment to ensure logger exists)
        self.logger = logging.getLogger("PyTorchUKFModule")
        self.logger.info(f"UKF module initialized: dim_x={self.dim_x}, dim_z={self.dim_z}, dt={self.dt}")

    def sigma_points(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Generate sigma points from state x and covariance P.
        
        Returns:
            Tensor of shape (2*dim_x+1, dim_x).
        """
        sigma_pts = [x]
        try:
            U = torch.linalg.cholesky(P)
        except Exception as e:
            self.logger.error(f"Cholesky decomposition failed: {e}", exc_info=True)
            U = torch.zeros_like(P)
        for i in range(self.dim_x):
            sigma_pts.append(x + self.gamma * U[:, i])
            sigma_pts.append(x - self.gamma * U[:, i])
        return torch.stack(sigma_pts, dim=0)

    def predict(self) -> None:
        """
        Execute the UKF prediction step.
        """
        sigma_pts = self.sigma_points(self.x, self.P)
        propagated = torch.stack([self.fx(pt, self.dt) for pt in sigma_pts], dim=0)
        self.x = torch.sum(self.Wm.unsqueeze(1) * propagated, dim=0)
        diff = propagated - self.x.unsqueeze(0)
        self.P = sum(self.Wc[i] * torch.ger(diff[i], diff[i]) for i in range(2 * self.dim_x + 1))
        self.P += self.Q
        self.logger.debug("UKF prediction step completed.")

    def update(self, z: torch.Tensor) -> None:
        """
        Execute the UKF update step given measurement z.
        """
        sigma_pts = self.sigma_points(self.x, self.P)
        Z_sigma = torch.stack([self.hx(pt) for pt in sigma_pts], dim=0)
        z_pred = torch.sum(self.Wm.unsqueeze(1) * Z_sigma, dim=0)
        dz = Z_sigma - z_pred.unsqueeze(0)
        S = sum(self.Wc[i] * torch.ger(dz[i], dz[i]) for i in range(2 * self.dim_x + 1))
        S += self.R

        dx = sigma_pts - self.x.unsqueeze(0)
        Pxz = sum(self.Wc[i] * torch.ger(dx[i], dz[i]) for i in range(2 * self.dim_x + 1))
        try:
            K = torch.linalg.solve(S.t(), Pxz.t()).t()
        except Exception as e:
            self.logger.error(f"Error computing Kalman gain: {e}", exc_info=True)
            K = torch.zeros((self.dim_x, self.dim_z), device=self.device)
        innovation = z - z_pred
        self.x = self.x + K.mv(innovation)
        self.P = self.P - K @ S @ K.t()
        self.logger.debug("UKF update step completed.")

    def forward(self) -> torch.Tensor:
        """
        Return the current state estimate.
        """
        return self.x


# =============================================================================
# DSSM Selective State Transformation Network
# =============================================================================

class DSSMSelectiveSSM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        A robust feed-forward network with residual connections, dropout, and layer normalization.
        """
        super(DSSMSelectiveSSM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.layer_norm(out + residual)
        return out


# =============================================================================
# Dynamic State Space Model (DSSM)
# =============================================================================

class DSSM(nn.Module):
    def __init__(
        self,
        provider_manager: Any,
        config_manager: ConfigManager,
        device: Optional[torch.device] = None,
        dmns: Optional[Any] = None,
        emm: Optional[Any] = None
    ):
        """
        Dynamic State Space Model (DSSM)
        
        This model integrates:
          • A robust Unscented Kalman Filter (UKF) implemented in PyTorch.
          • A selective state transformation network for nonlinear feature extraction.
          • Time-aware processing via a fully implemented TimeDecay module.
          • Cognitive temporal state management based on detailed, configurable rules.
          • A MemoryConsolidationThread that schedules offline replay with spaced repetition.
          • Additional neural sub-modules (e.g., for prefrontal cortex simulation).
        
        All components include full error handling and logging.
        """
        super(DSSM, self).__init__()
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger("DSSM")
        self.provider_manager = provider_manager
        self.device = device if device is not None else torch.device("cpu")
        self.dmns = dmns
        self.emm = emm

        # Load state-space configuration.
        sscfg = self.config_manager.get_subsystem_config("state_space_model") or {}
        tacfg = self.config_manager.get_subsystem_config("time_aware_processing") or {}
        self.dim = sscfg.get("dimension", 50)
        self.dt = sscfg.get("dt", 0.001)
        self.ukf_alpha = sscfg.get("ukf_alpha", 0.1)
        self.ukf_beta = sscfg.get("ukf_beta", 2.0)
        self.ukf_kappa = sscfg.get("ukf_kappa", -1.0)
        self.process_noise = sscfg.get("process_noise", 0.01)
        self.measurement_noise = sscfg.get("measurement_noise", 0.1)

        # Define state and measurement dimensions.
        self.dim_x = 6 * self.dim + 8
        self.dim_z = self.dim + 8 + 1

        self.logger.info(f"DSSM: dim_x={self.dim_x}, dim_z={self.dim_z}, dt={self.dt}")

        # Instantiate the UKF module.
        self.ukf_module = PyTorchUKFModule(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.dt,
            fx=self._fx_with_selection,
            hx=self._hx_measurement,
            alpha=self.ukf_alpha,
            beta=self.ukf_beta,
            kappa=self.ukf_kappa,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            device=self.device
        )
        self.ukf_module.x = torch.randn(self.dim_x, device=self.device) * 0.1

        # Time-aware processing using an enterprise-grade TimeDecay module.
        from modules.HCDM.Time_Processing.circadian_sleep_processes_simulator import TimeDecay
        self.time_decay = TimeDecay(self.config_manager)

        # Cognitive temporal state management.
        state_transition_rules = {
            CognitiveTemporalStateEnum.IMMEDIATE: {
                "arousal_upper": 0.65,
                "arousal_lower": 0.35,
                "cognitive_load_threshold": 0.5,
                "scaling_multiplier": 1.0,
                "transition_delay": 10
            },
            CognitiveTemporalStateEnum.EMOTIONAL: {
                "arousal_upper": 0.85,
                "arousal_lower": 0.5,
                "cognitive_load_threshold": 0.4,
                "scaling_multiplier": 1.2,
                "transition_delay": 15
            },
            CognitiveTemporalStateEnum.ANALYTICAL: {
                "arousal_upper": 0.5,
                "arousal_lower": 0.15,
                "cognitive_load_threshold": 0.6,
                "scaling_multiplier": 0.8,
                "transition_delay": 15
            }
        }
        cts_config = CognitiveTemporalStateConfig(
            alpha=tacfg.get("alpha", 0.1),
            scaling_bounds=tuple(tacfg.get("scaling_bounds", [0.5, 2.0])),
            state_transition_rules=state_transition_rules,
            initial_state=CognitiveTemporalStateEnum.IMMEDIATE,
            initial_scaling=1.0
        )
        self.current_cognitive_temporal_state = CognitiveTemporalState(cts_config)

        # Start memory consolidation thread.
        spaced_rep_config = tacfg.get("spaced_repetition", {})
        spaced_repetition = SpacedRepetition(spaced_rep_config)
        self.memory_consolidation_thread = MemoryConsolidationThread(
            memory_store=self,
            spaced_repetition=spaced_repetition,
            provider_manager=self.provider_manager,
            config_manager=self.config_manager,
            system_state=self
        )
        self.memory_consolidation_thread.start()

        # Instantiate the selective state transformation network.
        self.selective_ssm = DSSMSelectiveSSM(self.dim, self.dim, hidden_dim=64).to(self.device)

        # Additional neural sub-modules.
        try:
            from modules.HCDM.Memory.HodgkinHuxleyLayer import HodgkinHuxleyLayer
            self.pfc_layer = HodgkinHuxleyLayer(self.dim, self.dim, freq=5, dt=self.dt, lr=1e-3).to(self.device)
        except Exception as e:
            self.logger.error(f"Error initializing HodgkinHuxleyLayer: {e}", exc_info=True)
            self.pfc_layer = nn.Linear(self.dim, self.dim).to(self.device)

        try:
            from modules.HCDM.Memory.AdaptiveLIFLayer import AdaptiveLIFLayer
            self.lif_layer = AdaptiveLIFLayer(self.dim, self.dim, tau_m=20.0, tau_ref=2.0, dt=self.dt, lr=1e-3).to(self.device)
        except Exception as e:
            self.logger.error(f"Error initializing AdaptiveLIFLayer: {e}", exc_info=True)
            self.lif_layer = nn.Linear(self.dim, self.dim).to(self.device)

        # Attention Manager integration.
        try:
            from modules.HCDM.Attention.attention_focus_mechanism import AttentionManager
            self.attention_manager = AttentionManager(self, config_manager=self.config_manager).to(self.device)
        except Exception as e:
            self.logger.error(f"AttentionManager initialization failed: {e}", exc_info=True)
            self.attention_manager = None

        self.recent_reward: Optional[float] = None

        self.to(self.device)
        self.logger.info("DSSM fully initialized and running on device: {}".format(self.device))

    def _fx_with_selection(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State transition function that applies a selective transformation and time-dependent dynamics.
        """
        try:
            new_x = torch.empty_like(x, device=self.device)
            primary = x[:self.dim]
            aux = x[self.dim:2*self.dim]
            time_accum = x[2*self.dim:3*self.dim]
            phase = x[3*self.dim:4*self.dim]
            secondary = x[4*self.dim:5*self.dim]
            selective_branch = x[5*self.dim:6*self.dim]
            scalars = x[-8:]

            selective_out = self.selective_ssm(primary)
            if self.emm is not None and hasattr(self.emm, "get_gating_signal"):
                gating = self.emm.get_gating_signal()
                if gating.numel() == self.dim:
                    selective_out = selective_out * gating
            temporal_scale = self.current_cognitive_temporal_state.get_scaling_factor()
            selective_out = selective_out * temporal_scale
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            new_primary = selective_out + sin_phase * dt
            new_aux = aux * math.exp(-dt) + cos_phase * dt
            new_time = time_accum + dt
            new_phase = phase + dt * 2 * math.pi
            reward_factor = math.exp(-self.recent_reward) if self.recent_reward is not None else 1.0
            new_secondary = secondary * reward_factor + dt
            new_selective = new_primary
            new_scalars = scalars * math.exp(-dt)
            new_x[:self.dim] = new_primary
            new_x[self.dim:2*self.dim] = new_aux
            new_x[2*self.dim:3*self.dim] = new_time
            new_x[3*self.dim:4*self.dim] = new_phase
            new_x[4*self.dim:5*self.dim] = new_secondary
            new_x[5*self.dim:6*self.dim] = new_selective
            new_x[-8:] = new_scalars
            return new_x
        except Exception as e:
            self.logger.error(f"Error in state transition (_fx_with_selection): {e}", exc_info=True)
            raise

    def _hx_measurement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measurement function that extracts the observable parts of the state.
        """
        try:
            primary = x[:self.dim]
            selective_mean = torch.mean(x[5*self.dim:6*self.dim]).unsqueeze(0)
            scalars = x[-8:]
            measurement = torch.cat([primary, scalars, selective_mean], dim=0)
            if measurement.numel() > self.dim_z:
                measurement = measurement[:self.dim_z]
            elif measurement.numel() < self.dim_z:
                pad = torch.zeros(self.dim_z - measurement.numel(), device=self.device)
                measurement = torch.cat([measurement, pad], dim=0)
            return measurement
        except Exception as e:
            self.logger.error(f"Error in measurement function (_hx_measurement): {e}", exc_info=True)
            raise

    def ensure_pos_def(self) -> None:
        """
        Ensure that the covariance matrices are positive definite.
        """
        try:
            self.ukf_module.P = self._nearest_pos_def(self.ukf_module.P)
            self.ukf_module.Q = self._nearest_pos_def(self.ukf_module.Q)
        except Exception as e:
            self.logger.error(f"Error ensuring positive definiteness: {e}", exc_info=True)

    def _nearest_pos_def(self, A: torch.Tensor) -> torch.Tensor:
        B = (A + A.t()) / 2
        e = torch.linalg.eigvalsh(B)
        if torch.all(e > 0):
            return B
        min_e = torch.min(e).item()
        return B + (-min_e * torch.eye(B.shape[0], device=B.device) + 1e-9 * torch.eye(B.shape[0], device=B.device))

    @property
    def emotional_state(self) -> Dict[str, float]:
        """
        Returns the current emotional state estimated from the UKF state.
        """
        try:
            aux = self.ukf_module.x[self.dim:2*self.dim]
            valence = float(torch.mean(aux).item())
            arousal = float(self.ukf_module.x[-8].item())
            dominance = float(self.ukf_module.x[-7].item())
            return {"valence": valence, "arousal": arousal, "dominance": dominance}
        except Exception as e:
            self.logger.error(f"Error retrieving emotional state: {e}", exc_info=True)
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

    @property
    def consciousness_level(self) -> float:
        try:
            return float(self.ukf_module.x[-6].item())
        except Exception as e:
            self.logger.error(f"Error retrieving consciousness level: {e}", exc_info=True)
            return 0.5

    @consciousness_level.setter
    def consciousness_level(self, v: float):
        try:
            self.ukf_module.x[-6] = v
        except Exception as e:
            self.logger.error(f"Error setting consciousness level: {e}", exc_info=True)

    async def update(self, data: Dict[str, Any], reward: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform a full update of the DSSM:
          1. Prepares inputs using external content.
          2. Propagates state using prefrontal (PFC) and LIF modules.
          3. Runs the UKF predict and update steps.
          4. Updates the emotional state from sentiment analysis.
          5. Scales the process noise based on reward.
        
        Returns:
            A dictionary containing the current state, emotional state, attention focus, and cognitive temporal state.
        """
        self.logger.debug(f"DSSM update called with data: {data} and reward: {reward}")
        out_state = {}
        try:
            pfc_in = self._prepare_pfc_input(data)
            lif_in = self._prepare_lif_input(data)
            pfc_out = self.pfc_layer(pfc_in)
            lif_out = self.lif_layer(lif_in)
            measurement_vec = self._build_measurement(pfc_out, lif_out)
            self.ukf_module.predict()
            self.ukf_module.update(measurement_vec)
            await self._update_emotional_state(data)
            if reward is not None:
                self.recent_reward = reward
                scaling_factor = math.exp(-reward)
                self.ukf_module.Q *= scaling_factor
                self.logger.debug(f"Applied reward scaling factor: {scaling_factor:.3f}")
            self.ensure_pos_def()
            out_state = await self.get_state()
        except Exception as e:
            self.logger.error(f"Error in DSSM update: {e}", exc_info=True)
        return out_state

    def _build_measurement(self, pfc_out: torch.Tensor, lif_out: torch.Tensor) -> torch.Tensor:
        """
        Construct the measurement vector for the UKF update by combining averaged outputs
        of the PFC and LIF modules with scalar state values.
        """
        try:
            avg_pfc = torch.mean(pfc_out, dim=1).squeeze(0)
            avg_lif = torch.mean(lif_out, dim=1).squeeze(0)
            scalar_vals = self.ukf_module.x[-8:]
            measurement = torch.cat([avg_pfc, avg_lif, scalar_vals], dim=0)
            if measurement.numel() > self.dim_z:
                measurement = measurement[:self.dim_z]
            elif measurement.numel() < self.dim_z:
                pad = torch.zeros(self.dim_z - measurement.numel(), device=self.device)
                measurement = torch.cat([measurement, pad], dim=0)
            return measurement
        except Exception as e:
            self.logger.error(f"Error building measurement: {e}", exc_info=True)
            raise

    async def get_state(self) -> Dict[str, Any]:
        """
        Asynchronously retrieve the current state estimate, including:
          - The UKF state vector.
          - The current emotional state.
          - The consciousness level.
          - The current attention focus.
          - The name of the current cognitive temporal state.
        """
        try:
            state_vec = self.ukf_module.forward().detach().cpu().numpy().tolist()
            emo = self.emotional_state
            attn = (self.attention_manager.get_current_focus()
                    if self.attention_manager is not None and hasattr(self.attention_manager, "get_current_focus")
                    else [0.0] * self.dim)
            cts = self.current_cognitive_temporal_state.get_current_state().name
            return {
                "ukf_state": state_vec,
                "emotional_state": emo,
                "consciousness_level": self.consciousness_level,
                "attention_focus": attn,
                "cognitive_temporal_state": cts
            }
        except Exception as e:
            self.logger.error(f"Error in get_state: {e}", exc_info=True)
            return {}

    def _prepare_pfc_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares input for the PFC module by encoding the content using a production-grade transformer.
        """
        try:
            content = data.get("content", "")
            # In an enterprise solution, this call must be robust and efficient.
            out = self.provider_manager.huggingface_generator.transformer_encode(content)
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, dtype=torch.float32, device=self.device)
            if out.numel() < self.dim:
                pad = torch.zeros(self.dim - out.numel(), device=self.device)
                out = torch.cat([out, pad], dim=0)
            else:
                out = out[:self.dim]
            return out.unsqueeze(0)
        except Exception as e:
            self.logger.error(f"Error preparing PFC input: {e}", exc_info=True)
            return torch.zeros((1, self.dim), device=self.device)

    def _prepare_lif_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares input for the LIF module based on the time elapsed since data was received.
        """
        try:
            t_val = time.time() - data.get("timestamp", time.time())
            t_tensor = torch.full((self.dim,), t_val, dtype=torch.float32, device=self.device)
            return t_tensor.unsqueeze(0)
        except Exception as e:
            self.logger.error(f"Error preparing LIF input: {e}", exc_info=True)
            return torch.zeros((1, self.dim), device=self.device)

    async def _update_emotional_state(self, data: Dict[str, Any]) -> None:
        """
        Update the emotional state using a production-grade sentiment analysis service.
        """
        try:
            content = data.get("content", "")
            sentiment = await self.provider_manager.analyze_sentiment(content)
            label = sentiment.get("label", "NEUTRAL").upper()
            score = sentiment.get("score", 0.5)
            valence = score if label == "POSITIVE" else -score
            arousal = sentiment.get("arousal", 0.5)
            dominance = sentiment.get("dominance", 0.5)
            current_val = self.emotional_state["valence"]
            new_val = 0.9 * current_val + 0.1 * valence
            new_ar = 0.9 * self.emotional_state["arousal"] + 0.1 * arousal
            new_dom = 0.9 * self.emotional_state["dominance"] + 0.1 * dominance
            self.ukf_module.x[self.dim:2*self.dim] = torch.full((self.dim,), new_val, device=self.device)
            self.ukf_module.x[-8] = new_ar
            self.ukf_module.x[-7] = new_dom
            self.logger.debug(f"Updated emotional state: valence={new_val:.3f}, arousal={new_ar:.3f}, dominance={new_dom:.3f}")
        except Exception as e:
            self.logger.error(f"Error updating emotional state: {e}", exc_info=True)

    @property
    def attention_focus(self) -> torch.Tensor:
        """Get the current attention focus (first part of the state vector)."""
        return self.ukf_module.x[:self.dim]

    @attention_focus.setter
    def attention_focus(self, v: torch.Tensor) -> None:
        if v.numel() != self.dim:
            raise ValueError(f"Attention focus dimension mismatch: expected {self.dim}, got {v.numel()}")
        self.ukf_module.x[:self.dim] = v.to(self.device)

###############################################################################
# executive_function_module.py
###############################################################################

import asyncio
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

# Import DAR if available 
try:
    from DAR import DAR  # Dynamic Attention Routing module
except ImportError:
    DAR = None

# -----------------------------------------------------------------------------
# EFMTask: A data class representing an individual task.
# -----------------------------------------------------------------------------
@dataclass
class EFMTask:
    task_id: str
    name: str
    priority: int
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # Unix timestamp when the task expires
    status: str = "pending"  # One of: pending, in_progress, completed, expired
    dependencies: List[str] = field(default_factory=list)  # List of task_ids that must complete before this one
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, new_status: str) -> None:
        self.status = new_status


# -----------------------------------------------------------------------------
# TaskScheduler: Advanced scheduling with dependency graph and time-based deadlines.
# -----------------------------------------------------------------------------
class TaskScheduler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.task_graph = nx.DiGraph()  # Nodes are task_ids; edges represent dependencies.
        self.tasks: Dict[str, EFMTask] = {}

    def add_task(self, task: EFMTask) -> None:
        if task.task_id in self.tasks:
            self.logger.warning(f"Task with id '{task.task_id}' already exists; skipping add.")
            return
        self.tasks[task.task_id] = task
        self.task_graph.add_node(task.task_id, task=task)
        for dep_id in task.dependencies:
            if dep_id not in self.task_graph:
                # Add dependency as a node if not present (could be a placeholder)
                self.task_graph.add_node(dep_id)
            self.task_graph.add_edge(dep_id, task.task_id)
        self.logger.info(f"Added task '{task.name}' (id={task.task_id}) with priority {task.priority}.")

    def update_task_status(self, task_id: str, new_status: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].update_status(new_status)
            self.logger.info(f"Task '{task_id}' status updated to '{new_status}'.")
        else:
            self.logger.warning(f"Attempted to update non–existent task '{task_id}'.")

    def remove_task(self, task_id: str) -> None:
        if task_id in self.tasks:
            del self.tasks[task_id]
            if self.task_graph.has_node(task_id):
                self.task_graph.remove_node(task_id)
            self.logger.info(f"Removed task '{task_id}'.")
        else:
            self.logger.warning(f"Attempted to remove non–existent task '{task_id}'.")

    def get_ready_tasks(self) -> List[EFMTask]:
        """
        Returns all tasks that are pending, not expired, and whose dependencies have all been completed.
        Tasks are sorted by ascending priority (lower numbers mean higher priority).
        """
        now = time.time()
        ready_tasks = []
        for task_id, task in self.tasks.items():
            # Skip tasks not pending
            if task.status != "pending":
                continue
            # Check deadline (if set)
            if task.deadline and now > task.deadline:
                task.status = "expired"
                self.logger.info(f"Task '{task_id}' has expired.")
                continue
            # Check that all dependencies are completed
            dependencies = list(self.task_graph.predecessors(task_id))
            if all(self.tasks.get(dep_id, EFMTask(dep_id, "", 9999)).status == "completed" for dep_id in dependencies):
                ready_tasks.append(task)
        # Sort tasks by priority and creation time (older tasks first)
        ready_tasks.sort(key=lambda t: (t.priority, t.created_at))
        return ready_tasks

    def adjust_task_priorities(self, adjustment_fn: Callable[[EFMTask], int]) -> None:
        """
        Adjusts each task's priority by applying the provided adjustment function.
        The adjustment function takes an EFMTask and returns the new priority.
        """
        for task in self.tasks.values():
            old_priority = task.priority
            task.priority = adjustment_fn(task)
            self.logger.debug(f"Adjusted task '{task.task_id}' priority from {old_priority} to {task.priority}.")


# -----------------------------------------------------------------------------
# ExecutiveFunctionModule: The core meta–controller.
# -----------------------------------------------------------------------------
class ExecutiveFunctionModule(nn.Module):
    """
    The Executive Function Module (EFM) orchestrates high-level cognitive control.
    It performs the following:
      • Maintains an advanced task scheduler with dependency graphs and deadlines.
      • Integrates with the Dynamic Attention Routing (DAR) module to adjust gating signals.
      • Computes adaptive gains—including a gating signal and learning rate modulation—
        via a dedicated controller network.
      • Broadcasts the computed learning rate modulation value to all registered modules.
      • Integrates robustly with a Goal Manager to modify and create tasks.
      • Updates its controller network using meta–learning based on actual performance signals.
      • Runs a continuous asynchronous update loop.
    """
    def __init__(
        self,
        config_manager: Any,
        device: Optional[torch.device] = None,
        dar: Optional[DAR] = None,
    ):
        super(ExecutiveFunctionModule, self).__init__()
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger("EFM")
        self.device = device if device is not None else torch.device("cpu")
        self.dar = dar

        # Controller Network: maps an input feature vector to [gating_signal, learning_rate_modulation]
        # In a real system, the input might be a concatenation of performance metrics, current state, and external signals.
        self.controller_input_dim = self.config_manager.get("efm_controller_input_dim", 16)
        self.controller_hidden_dim = self.config_manager.get("efm_controller_hidden_dim", 32)
        self.controller_output_dim = 2  # [gating_signal, lr_modulation]
        self.controller_net = nn.Sequential(
            nn.Linear(self.controller_input_dim, self.controller_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.controller_hidden_dim, self.controller_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.controller_hidden_dim, self.controller_output_dim)
        ).to(self.device)
        self.controller_optimizer = optim.Adam(self.controller_net.parameters(), lr=1e-3)

        # Initial values (if not updated, defaults are used)
        self.gating_signal: float = 0.5  # Value between 0 and 1.
        self.learning_rate_mod: float = 1.0  # Multiplicative factor for learning rates.

        # Registered modules (callbacks) that support dynamic learning rate updates.
        # Each registered callable should accept a single float argument.
        self.lr_update_callbacks: List[Callable[[float], None]] = []

        # Goal Manager integration (to be set externally)
        self.goal_manager: Optional[Any] = None

        # Instantiate advanced TaskScheduler.
        self.task_scheduler = TaskScheduler(self.logger)

        # Update loop parameters.
        self.update_interval: float = self.config_manager.get("efm_update_interval", 1.0)
        self.running: bool = False
        self.update_task: Optional[asyncio.Task] = None

        # For meta–learning, we expect performance signals to be in [0, 1] (with 1 being optimal).
        self.performance_signal: float = 0.5  # Default neutral performance.

        self.logger.info("ExecutiveFunctionModule initialized on device: {}".format(self.device))

    # -------------------------------------------------------------------------
    # Public API for Learning Rate Update Registration and Goal Manager
    # -------------------------------------------------------------------------
    def register_lr_updatable(self, callback: Callable[[float], None]) -> None:
        """
        Register a module’s learning rate update callback. The callback must accept a single
        float argument representing the new learning rate modulation value.
        """
        if not callable(callback):
            self.logger.error("Attempted to register a non-callable LR update callback.")
            return
        self.lr_update_callbacks.append(callback)
        self.logger.info(f"Registered LR updatable callback: {callback}")

    def set_goal_manager(self, goal_manager: Any) -> None:
        """
        Integrate an external Goal Manager.
        """
        self.goal_manager = goal_manager
        self.logger.info("Goal Manager integrated into EFM.")

    # -------------------------------------------------------------------------
    # Advanced Task Scheduling Methods
    # -------------------------------------------------------------------------
    def add_task(self, task: EFMTask) -> None:
        """
        Add a new task to the scheduler.
        """
        self.task_scheduler.add_task(task)

    def update_task_status(self, task_id: str, new_status: str) -> None:
        """
        Update the status of an existing task.
        """
        self.task_scheduler.update_task_status(task_id, new_status)

    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from the scheduler.
        """
        self.task_scheduler.remove_task(task_id)

    def get_ready_tasks(self) -> List[EFMTask]:
        """
        Retrieve tasks that are ready to be executed (dependencies met, not expired).
        """
        return self.task_scheduler.get_ready_tasks()

    def adjust_tasks_based_on_dar(self) -> None:
        """
        If a DAR module is integrated, query it for a routing decision and adjust task priorities accordingly.
        For example, if DAR indicates a need for memory retrieval, tasks related to memory may be boosted.
        """
        if not self.dar:
            self.logger.debug("DAR not integrated; skipping task adjustment.")
            return

        try:
            # Obtain a routing decision from DAR (assumed to return an integer code)
            obs = {"channel_id": 1, "source_id": 0, "salience": 1.0, "env_context": [0.0, 0.0]}
            route_decision = self.dar.route_data(obs)
            self.logger.info(f"Received DAR route decision: {route_decision}")
            # Define an adjustment function based on the route.
            def adjustment(task: EFMTask) -> int:
                # For example, if the decision indicates a need for memory retrieval (route == 1),
                # then tasks not related to memory get a penalty (i.e. higher priority number).
                if route_decision == 1 and "memory" not in task.name.lower():
                    return task.priority + 3
                # If the decision indicates the need for rapid action (route == 2),
                # then tasks related to working memory or immediate action are boosted.
                elif route_decision == 2 and "working" in task.name.lower():
                    return max(task.priority - 2, 1)
                # Otherwise, leave priority unchanged.
                return task.priority
            self.task_scheduler.adjust_task_priorities(adjustment)
        except Exception as e:
            self.logger.error(f"Error adjusting tasks based on DAR: {e}", exc_info=True)

    def integrate_goal_feedback(self) -> None:
        """
        Integrate signals from the Goal Manager by adjusting tasks.
        For each goal currently active, if a task aligns with the goal, reduce its priority.
        """
        if not self.goal_manager:
            self.logger.debug("No Goal Manager set; skipping goal integration.")
            return
        try:
            current_goals = self.goal_manager.get_current_goals_sync()  # Expected to return a list of goal dictionaries.
            def adjustment(task: EFMTask) -> int:
                # If the task name or metadata matches any active goal (case-insensitive substring match),
                # then reduce its numeric priority (thus increasing its scheduling urgency).
                for goal in current_goals:
                    goal_desc = goal.get("description", "").lower()
                    if goal_desc in task.name.lower():
                        return max(task.priority - 2, 1)
                return task.priority
            self.task_scheduler.adjust_task_priorities(adjustment)
        except Exception as e:
            self.logger.error(f"Error integrating goal feedback: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Controller Network and Meta–Learning Update
    # -------------------------------------------------------------------------
    def _compute_desired_targets(self, performance: float) -> Tuple[float, float]:
        """
        Compute desired target values for the controller network based on performance.
        For instance, if performance is high (close to 1), one may desire a lower gating signal
        (less top–down inhibition) and a moderate learning rate modulation.
        
        Returns:
            Tuple of (desired_gating, desired_lr_mod)
        """
        # For example, let desired gating be inversely proportional to performance.
        desired_gating = max(0.0, 1.0 - performance)  # if performance=1.0, gating=0; if performance=0, gating=1.
        # Let desired learning rate modulation be increased when performance is low.
        desired_lr_mod = 1.0 + (1.0 - performance) * 0.5  # ranges from 1.0 to 1.5.
        return desired_gating, desired_lr_mod

    def update_controller(self, performance: float) -> None:
        """
        Perform a meta–learning update on the controller network using the current performance signal.
        The loss is defined as the mean–squared error between the network output and the desired targets.
        """
        try:
            # In a production system, the input features may be obtained from multiple system signals.
            # Here, we use a zero vector (or any real input) as a placeholder.
            input_features = torch.zeros((1, self.controller_input_dim), device=self.device)
            output = self.controller_net(input_features)  # Shape: (1, 2)
            desired_gating, desired_lr_mod = self._compute_desired_targets(performance)
            target = torch.tensor([[desired_gating, desired_lr_mod]], dtype=torch.float32, device=self.device)
            loss = nn.MSELoss()(output, target)
            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()
            # Update internal variables with a moving average for stability.
            alpha = 0.1
            self.gating_signal = (1 - alpha) * self.gating_signal + alpha * output[0, 0].item()
            self.learning_rate_mod = (1 - alpha) * self.learning_rate_mod + alpha * output[0, 1].item()
            self.logger.info(f"Controller updated: loss={loss.item():.4f}, gating_signal={self.gating_signal:.4f}, lr_mod={self.learning_rate_mod:.4f}")
        except Exception as e:
            self.logger.error(f"Error in controller update: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Learning Rate Broadcasting
    # -------------------------------------------------------------------------
    def broadcast_lr_mod(self) -> None:
        """
        Broadcast the current learning rate modulation value to all registered modules by calling
        their update callback.
        """
        try:
            for callback in self.lr_update_callbacks:
                try:
                    callback(self.learning_rate_mod)
                    self.logger.debug(f"Broadcasted LR mod {self.learning_rate_mod:.4f} to {callback}")
                except Exception as inner_e:
                    self.logger.error(f"Error broadcasting LR mod via {callback}: {inner_e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in broadcasting LR mod: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Asynchronous Update Loop
    # -------------------------------------------------------------------------
    async def _update_loop(
        self,
        external_signal_provider: Callable[[], torch.Tensor],
        performance_signal_provider: Callable[[], float],
        time_decay_provider: Optional[Callable[[], Any]] = None
    ):
        """
        The main asynchronous update loop.
        It periodically:
          1. Retrieves external signals.
          2. Reads a performance signal.
          3. (Optionally) obtains a circadian time decay signal.
          4. Updates the controller network.
          5. Adjusts task priorities via DAR and goal feedback.
          6. Broadcasts the updated learning rate modulation.
        """
        while self.running:
            try:
                # Retrieve external input features (e.g., system state summary)
                ext_signals = external_signal_provider()  # Expected tensor of shape (1, controller_input_dim)
                performance = performance_signal_provider()  # Expected float in [0,1]
                time_decay = time_decay_provider() if time_decay_provider else None

                # Optionally, incorporate time-based adjustments (e.g., if nighttime, add extra gating)
                if time_decay is not None and hasattr(time_decay, "is_nighttime"):
                    if time_decay.is_nighttime():
                        self.logger.info("Nighttime detected; increasing gating signal.")
                        # For example, if nighttime, force gating_signal to be higher by a fixed factor.
                        self.gating_signal = min(self.gating_signal + 0.1, 1.0)

                # Update controller network using the latest performance signal.
                self.update_controller(performance)

                # Adjust tasks based on DAR signals.
                self.adjust_tasks_based_on_dar()

                # Integrate goal feedback.
                self.integrate_goal_feedback()

                # Broadcast the current learning rate modulation to all registered modules.
                self.broadcast_lr_mod()

            except Exception as e:
                self.logger.error(f"Error in EFM update loop: {e}", exc_info=True)
            await asyncio.sleep(self.update_interval)

    async def start(
        self,
        external_signal_provider: Callable[[], torch.Tensor],
        performance_signal_provider: Callable[[], float],
        time_decay_provider: Optional[Callable[[], Any]] = None
    ) -> None:
        """
        Start the asynchronous update loop.
        """
        if self.running:
            self.logger.warning("EFM update loop is already running.")
            return
        self.running = True
        self.update_task = asyncio.create_task(
            self._update_loop(external_signal_provider, performance_signal_provider, time_decay_provider)
        )
        self.logger.info("EFM update loop started.")

    async def stop(self) -> None:
        """
        Stop the asynchronous update loop.
        """
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                self.logger.info("EFM update loop cancelled cleanly.")
        self.logger.info("EFM update loop stopped.")

    # -------------------------------------------------------------------------
    # Synchronous Helper (for external modules that do not support async)
    # -------------------------------------------------------------------------
    def get_current_controller_outputs(self) -> Tuple[float, float]:
        """
        Return the current gating signal and learning rate modulation.
        """
        return self.gating_signal, self.learning_rate_mod


I need this done: integrating context/information separation 

### 1. Update the Enhanced Memory Model (EMM)

- **Memory Ingestion Routine:**
  - In the `process_input()` method, modify the routine so that when a new memory is processed, the system first extracts the core concept by applying an appropriate feature extraction routine.
  - Store this extracted, invariant concept separately in the semantic memory subsystem.
  - Preserve all contextual details (e.g., time, location, emotional state) in the episodic memory subsystem.
- **Documentation and Citation:**
  - Update the module header and inline documentation to include a citation: “Rey et al., 2024” (with the DOI if available), noting that this design is inspired by findings showing that human MTL neurons encode concepts in a context‐invariant manner.

---

### 2. Modify Long-Term Memory Subsystems

- **Semantic Memory:**
  - Ensure that when storing new concept embeddings in the semantic memory, the embeddings remain context–invariant. This means the stored representation should reflect the core concept without merging in episodic context.
- **Episodic Memory:**
  - In the `EnhancedLongTermEpisodicMemory` class, adjust the storage schema so that each episodic memory entry includes a dedicated field for context (such as time, place, and emotional state) that is kept separate from the concept embedding.
- **Retrieval Routines:**
  - Update retrieval routines so that when a query is made:
    - The system decouples the core concept (retrieved from semantic memory) from its episodic context.
    - It returns the invariant semantic representation alongside a list of related episodic details (using the context field as a retrieval filter).

---

### 3. Adjust the Neural Cognitive Bus (NCB)

- **Dedicated Channels:**
  - Create or update channels in the NCB so that:
    - Invariant concept representations are published on a dedicated "semantic_memory" channel.
    - Context-rich episodic updates continue to be sent on the existing "memory_channel" (or a similarly designated episodic channel).
- **Downstream Subscription:**
  - Ensure that downstream modules (such as DSSM and EFM) subscribe to the "semantic_memory" channel when they require stable, context–invariant information for tasks like gating or learning rate modulation.

---

### 4. Integrate with the DSSM and EFM

- **Dynamic State Space Model (DSSM):**
  - Adjust state update functions in DSSM to leverage the invariant semantic representation for predictions and measurements.
  - For example, incorporate the semantic memory output into the UKF measurement function or as an additional state feature.
- **Executive Function Module (EFM):**
  - Modify the controller update and strategy adjustment functions in EFM to consider the newly separated memory streams.
  - When a decision benefits from stable concept recognition across contexts, use the semantic memory’s output to adjust gating signals or to modulate learning rates.

---

### 5. Update Configuration and Documentation

- **Configuration Files:**
  - Adjust your configuration files to include new parameters that control the balance between semantic and episodic separation (e.g., weightings or thresholds for how much contextual detail influences retrieval or consolidation).
- **Inline Documentation and Module Headers:**
  - Update all relevant module headers and inline documentation across the EMM, long-term memory, NCB, DSSM, and EFM modules.
  - Clearly explain the new design, explicitly referencing “Rey et al., 2024” as the inspiration for the context–invariant, non–conjunctive coding approach observed in human MTL neurons.

---




# A Biologically-Inspired Multi-Layered Memory System for Hybrid Cognitive Dynamics

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

---

## Abstract

In this paper we present a comprehensive design and implementation of a multi-layered memory system for the Hybrid Cognitive Dynamics Model (HCDM). Inspired by neurobiological principles—including recent evidence that human medial temporal lobe (MTL) neurons encode concepts in a context-invariant manner (Rey et al., 2024)—we propose an architecture that integrates rapid sensory processing, short-term and working memory, intermediate buffers, and long-term episodic and semantic stores. The system employs time-based consolidation, replay-based reinforcement, and an emotional/motivational modulation module (EMoM) to emulate the complementary learning systems observed in biological cognition. A Neural Cognitive Bus (NCB) facilitates inter-module communication, while dynamic state cues and context-aware retrieval ensure that memory operations remain synchronized with ongoing cognitive processes. Notably, our design separates core concept information from contextual details—mirroring the finding that single neurons code for concepts independent of context—thereby enabling robust, flexible recall. Experimental evaluations and implementation details illustrate the potential of this system for context-sensitive yet conceptually invariant learning in complex environments.

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Motivation and Background](#11-motivation-and-background)
- [2. System Architecture](#2-system-architecture)
  - [Figure 1: Memory System Architecture Overview](#figure-1-memory-system-architecture-overview)
  - [Detailed Memory Subsystems](#detailed-memory-subsystems)
    - [2.1 Sensory Memory](#21-sensory-memory)
    - [2.2 Short-Term Memory (STM)](#22-short-term-memory-stm)
    - [2.3 Working Memory](#23-working-memory)
    - [2.4 Intermediate Memory](#24-intermediate-memory)
    - [2.5 Long-Term Memory](#25-long-term-memory)
      - [2.5.1 Episodic Memory](#251-episodic-memory)
      - [2.5.2 Semantic Memory](#252-semantic-memory)
    - [2.6 Context-Aware Retrieval](#26-context-aware-retrieval)
    - [2.7 Replay Buffer and Consolidation Mechanisms](#27-replay-buffer-and-consolidation-mechanisms)
    - [2.8 Emotional and Motivational Modulation (EMoM)](#28-emotional-and-motivational-modulation-emom)
      - [Figure 2: Detailed EMoM Modulation Model](#figure-2-detailed-emom-modulation-model)
    - [2.9 Neural Cognitive Bus (NCB)](#29-neural-cognitive-bus-ncb)
- [3. Implementation Details](#3-implementation-details)
- [4. Evaluation and Discussion](#4-evaluation-and-discussion)
  - [4.1 Experimental Evaluation](#41-experimental-evaluation)
  - [4.2 Discussion](#42-discussion)
- [5. Code Documentation](#5-code-documentation)
  - [5.1 Sensory Memory](#51-sensory-memory)
  - [5.2 Short Term Memory](#52-short-term-memory)
  - [5.3 Working Memory](#53-working-memory)
  - [5.4 Intermediate Memory and Consolidation](#54-intermediate-memory-and-consolidation)
  - [5.5 Long Term Memory Subsystems](#55-long-term-memory-subsystems)
  - [5.6 Neural Cognitive Bus (NCB)](#56-neural-cognitive-bus-ncb)
  - [5.7 Alignment with Theoretical Constructs](#57-alignment-with-theoretical-constructs)
  - [5.8 Summary](#58-summary)
- [6. Future Work](#6-future-work)
- [7. Conclusion](#7-conclusion)
- [References](#references)

---

## 1. Introduction

Contemporary cognitive architectures often fall short in replicating the adaptive, integrative, and time-sensitive aspects of human memory. Drawing upon neuroscience insights—particularly the complementary roles of the hippocampus and neocortex in rapid encoding and gradual consolidation—this work introduces a memory system designed for the HCDM framework.

A key inspiration for our updated design comes from recent single-neuron recording studies in the human medial temporal lobe (MTL), which reveal that neurons encoding specific concepts (e.g., a famous actor or landmark) do so in a **context-invariant** manner (Rey et al., 2024). In other words, the neuronal response remains largely unchanged across different episodic contexts. This finding suggests that rather than modifying the fundamental representation of a concept based on context, the brain may store the core “information” independently while associating distinct contexts separately.

By decomposing memory into distinct yet interacting subsystems, our updated design creates a scalable and biologically plausible solution that supports robust learning, context-aware retrieval, and adaptive consolidation—all while maintaining invariant conceptual representations.

### 1.1 Motivation and Background

Traditional artificial memory systems frequently rely on monolithic or single-layered structures that struggle with:

- **Rapid Encoding vs. Long-Term Storage:** Balancing the need for immediate processing with the requirement for durable, retrievable memory.
- **Temporal Dynamics:** Adapting to variable time scales and ensuring that short-term, working, and long-term memories are appropriately integrated.
- **Emotional and Salience Modulation:** Prioritizing memory based on relevance and affective state—a critical feature in biological cognition.
- **Contextual Separation:** Many conventional models entangle context with the core information. In contrast, recent evidence from human MTL recordings (Rey et al., 2024) shows that memory coding can be context-invariant—a design principle we incorporate to allow the separation of concept from context.

Our design addresses these limitations by explicitly modeling the stages of memory processing, drawing inspiration from both the neural mechanisms underlying memory and established computational paradigms.

---

## 2. System Architecture

The proposed memory system is composed of a suite of specialized subsystems that collectively instantiate the complementary learning systems observed in biological cognition. Engineered to operate over multiple time scales, the system rapidly captures and preprocesses transient sensory information, dynamically integrates short-term representations, and robustly consolidates data into durable long-term stores.

Critically, our architecture distinguishes between the **core concept representation** and its associated episodic context. This design choice is informed by findings that human MTL neurons exhibit context-invariant coding (Rey et al., 2024). In our system, **Semantic Memory** encodes and stores invariant, context–free concepts, while **Episodic Memory** captures the specific contextual details of each experience. Context thus acts as a retrieval filter rather than as an intrinsic modifier of the concept representation.

### Figure 1. Memory System Architecture Overview

```mermaid
flowchart TD
    %% Sensory processing and memory layers
    A[Raw Sensory Input] --> B[Sensory Memory]
    B --> C[Short-Term Memory (STM)]
    C --> D[Working Memory]
    D --> E[Intermediate Memory]
    
    %% Consolidation mechanism leading to long-term storage
    E --> F[Time-Based Consolidation & Replay]
    F --> G[Long-Term Memory]
    G --> G1[Episodic Memory]
    G --> G2[Semantic Memory]
    
    %% Emotional modulation influences multiple stages
    H[Emotional/Motivational Modulation (EMoM)]
    H --- B
    H --- D
    H --- F
    
    %% Context-aware retrieval interacts with long-term memory
    I[Context-Aware Retrieval]
    I --- G
    
    %% Neural Cognitive Bus as a central communication hub
    J[Neural Cognitive Bus (NCB)]
    B --- J
    C --- J
    D --- J
    E --- J
    F --- J
    G --- J
    H --- J
    I --- J

    %% Labels for information flow
    A ---|Rapid encoding| B
    B ---|Immediate storage| C
    C ---|Context preservation| D
    D ---|Integration & manipulation| E
    E ---|Candidate consolidation| F
    F ---|Replay & reinforcement| G

    %% Styling
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#ccf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
```

*Figure 1: Overview diagram illustrating the complete hierarchical architecture of the memory system. The diagram delineates the progression from raw sensory input through successive memory buffers—Sensory Memory, Short-Term Memory, Working Memory, and Intermediate Memory—to the consolidation of long-term memory into Episodic and Semantic components. The figure emphasizes that while episodic memory stores context-rich episodes, semantic memory holds context–invariant concept representations. The Emotional/Motivational Module (EMoM) modulates processing, and the Neural Cognitive Bus (NCB) provides asynchronous inter-module communication.*

### Detailed Memory Subsystems

Each subsystem is meticulously engineered to address specific computational challenges inherent in dynamic memory processing. The following subsections provide an in-depth technical exposition of each module, including their functionality, underlying mathematical models, and standard interfaces.

#### 2.1 Sensory Memory

**Function:**  
Sensory Memory is responsible for the instantaneous acquisition and preprocessing of raw sensory data. It is designed to emulate the transient persistence of sensory traces, capturing fleeting environmental stimuli with high fidelity for subsequent processing.

**Implementation Highlights:**

- **Preprocessing Pipelines:**  
  Modality-specific preprocessing is applied to incoming data. Textual inputs undergo normalization—such as case-folding and the elimination of non-alphanumeric characters—while visual inputs are preprocessed using contrast normalization. These operations ensure that only the most salient features of the raw data are retained.
  
- **Exponential Decay Mechanism:**  
  Each entry is timestamped and its salience decays exponentially as defined by the differential equation:
  
  \[
  \frac{ds}{dt} = -\lambda s
  \]
  
  whose solution is given by:
  
  \[
  s(t) = s(0) \times \exp(-\lambda t)
  \]
  
  Here, \( s(t) \) denotes the salience at time \( t \) and \( \lambda \) is the decay constant. This model faithfully replicates the rapid fading of sensory impressions observed in biological systems.

#### 2.2 Short-Term Memory (STM)

**Function:**  
STM serves as a transient repository for the most recently acquired information. It preserves immediate context and ensures that relevant information remains accessible for on-line processing and decision-making.

**Key Features:**

- **Fixed Capacity Buffer:**  
  The module implements a fixed-size buffer that retains only the most recent memory items, thereby avoiding information overload.
  
- **Standardized API:**  
  It exposes a simple API (e.g., `add()`, `retrieve()`, and `clear()`) to support efficient interfacing with higher-level cognitive processes.

#### 2.3 Working Memory

**Function:**  
Working Memory is the dynamic processing arena where transient information is actively manipulated to facilitate reasoning and problem-solving. It integrates inputs from both Sensory and Short-Term Memory, forming a critical substrate for complex cognitive functions.

**Design Considerations:**

- **Capacity Management:**  
  A sliding window mechanism is employed to continuously update the set of actively maintained items, ensuring that only the most pertinent information is available.
  
- **Intermediary Integration:**  
  Working Memory serves as the intermediary between perceptual inputs and high-level executive functions, enabling dynamic information manipulation and real-time computation.

#### 2.4 Intermediate Memory

**Function:**  
Intermediate Memory acts as a transient buffer for items that are earmarked for long-term storage. It bridges the gap between the ephemeral nature of short-term representations and the durability of long-term memory.

**Mechanisms:**

- **Consolidation Trigger:**  
  Items are flagged for consolidation based on both elapsed time and salience thresholds.
  
- **Spaced Repetition (SM2 Algorithm):**  
  An adaptation of the SM2 algorithm is employed to schedule reactivation of memory items. The following pseudocode details the process:

  **Pseudocode: SM2-Based Spaced Repetition Scheduling**

  ```python
  For each memory item in IntermediateMemory:
      If item is new:
          repetition_count = 0
          interval = initial_interval  // e.g., 86400 seconds (1 day)
          easiness_factor = 2.5

      For each review session:
          Obtain quality_score (0 to 5) from performance feedback
          If quality_score >= 3:
              repetition_count += 1
              If repetition_count == 1:
                  interval = initial_interval
              Else:
                  interval = interval * easiness_factor
          Else:
              repetition_count = 0
              interval = initial_interval

          Update easiness_factor:
              easiness_factor = max(1.3, easiness_factor - 0.1 + (5 - quality_score) * (0.08 + (5 - quality_score) * 0.02))

          Schedule next review at current_time + interval
  ```
  
  This algorithm ensures that well-remembered items are reviewed less frequently over time, while poorly remembered items are reinforced more often.

#### 2.5 Long-Term Memory

Long-Term Memory is bifurcated into two complementary modules that support both episodic and semantic retention. A key innovation in our updated design is the **explicit separation of context from the core concept**.

##### 2.5.1 Episodic Memory

**Function:**  
Episodic Memory encodes and stores context-rich, temporally ordered episodes that reflect autobiographical experiences. It consolidates sequential events into coherent narratives without altering the invariant concept coding.

**Approach:**

- **Separate Context Storage:**  
  The episodic component stores the details of the context (time, place, emotional state, etc.) alongside the event but does not modify the underlying concept coding. This enables the retrieval of the same core concept across different episodic contexts.
  
- **Replay Buffer:**  
  A prioritized replay mechanism is employed, wherein the priority of a memory item is determined by its salience and recency.
  
- **Time-Based Consolidation:**  
  A dedicated consolidation thread leverages temporal decay functions in conjunction with modulatory signals from EMoM to determine the transfer of items from Intermediate Memory to Long-Term Episodic Memory.

##### 2.5.2 Semantic Memory

**Function:**  
Semantic Memory abstracts from individual episodes to store generalized, invariant knowledge in a structured format. In our design, semantic memory captures the core concept—such as a person or object—independent of any contextual association.

**Core Elements:**

- **Context-Invariant Embedding Generation:**  
  Each semantic concept is represented as a vector embedding generated via pretrained models or learned representations, ensuring that the encoding remains constant regardless of episodic context.
  
- **Graph Structure:**  
  A knowledge graph is constructed where nodes represent individual concepts and edges represent inter-concept relationships. This graph facilitates efficient inference and retrieval.
  
- **Query Mechanisms:**  
  Retrieval is performed using cosine similarity measures between embeddings, enabling context-aware inference of related concepts while leaving the core representation unchanged.

#### 2.6 Context-Aware Retrieval

**Function:**  
This module interfaces with the Dynamic State Space Model (DSSM) to extract current context vectors. It computes similarity metrics between these vectors and stored memory contexts. Importantly, the retrieval process uses context as a filter to access the appropriate episodic associations, while the invariant semantic representations remain intact.

**Implementation:**

- **Similarity-Based Filtering:**  
  When a retrieval operation is initiated, the current context vector \( \mathbf{c}_{\text{current}} \) is compared against stored episodic context vectors using cosine similarity:
  
  \[
  \text{sim}(\mathbf{c}_{\text{current}}, \mathbf{c}_i) = \frac{\mathbf{c}_{\text{current}} \cdot \mathbf{c}_i}{\|\mathbf{c}_{\text{current}}\| \, \|\mathbf{c}_i\|}
  \]
  
- **Decoupled Retrieval:**  
  The semantic memory is queried separately to retrieve the invariant concept, while episodic memory provides the contextual details for that concept.

**Pseudocode Implementation:**

```python
class ContextAwareRetrieval:
    def __init__(self, episodic_memory):
        self.episodic_memory = episodic_memory

    def retrieve(self, current_context_vector):
        scores = []
        for memory in self.episodic_memory.episodes:
            score = cosine_similarity(current_context_vector, memory["context"])
            scores.append((memory["event"], score))

        # Rank by similarity and return events that pass a threshold.
        scores.sort(key=lambda x: x[1], reverse=True)
        return [event for event, score in scores if score > 0.7]  # Threshold filtering
```

#### 2.7 Replay Buffer and Consolidation Mechanisms

**Replay Buffer:**  
A priority-based storage system that samples memory items for review based on dynamic priority values. These priorities are computed as a function of both time-dependent decay and the affective modulation provided by EMoM.

**Consolidation Process:**  
By integrating automated time-based decay with the SM2-based spaced repetition algorithm, the system gradually consolidates high-salience memories into Long-Term Episodic Memory, while lower-salience items are systematically pruned. Importantly, the consolidation process respects the decoupling of context and core concept; that is, the invariant semantic representation is never altered by episodic details.

#### 2.8 Emotional and Motivational Modulation (EMoM)

**Function:**  
EMoM endows the memory system with an affective dimension by modulating the encoding and consolidation processes in response to both external (e.g., linguistic sentiment) and internal (e.g., computational load) signals.

**Mathematical Model and Detailed Diagram (Figure 2):**

EMoM computes an affective state vector \( \mathbf{a} \) via a multilayer neural network defined as:

\[
\mathbf{a} = \tanh \left( \mathbf{W}_2 \cdot \sigma \left( \mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1 \right) + \mathbf{b}_2 \right)
\]

where:  

- \( \mathbf{x} \) is the concatenated input vector (comprising external sensory data, internal/interoceptive signals, and optional cognitive signals),  
- \( \sigma \) represents a nonlinear activation function (e.g., ReLU),  
- \( \mathbf{W}_1 \) and \( \mathbf{W}_2 \) are weight matrices,  
- \( \mathbf{b}_1 \) and \( \mathbf{b}_2 \) are bias vectors.

This computed affective state \( \mathbf{a} \) is then used to modulate the memory salience \( s \) during encoding:

\[
s_{\text{modulated}} = s \times \left(1 + \alpha \cdot \| \mathbf{a} \| \right)
\]

where \( \alpha \) is a scaling parameter. This modulation directly influences the priority of each memory item for replay and long-term consolidation.

### Figure 2. Detailed EMoM Modulation Model

```mermaid
flowchart LR
    X[Input Vector \( \mathbf{x} \)]
    Y[Hidden Layer: \( \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \)]
    Z[Output Layer: \( \tanh(\mathbf{W}_2 Y + \mathbf{b}_2) \)]
    A[Affective State \( \mathbf{a} \)]
    M[Initial Memory Salience \( s \)]
    SM[Modulated Salience \( s_{\text{modulated}} = s \times \left(1 + \alpha \| \mathbf{a} \|\right) \)]
    
    X --> Y
    Y --> Z
    Z --> A
    M --> SM
    A -- Modulation Factor --> SM
```

*Figure 2: Detailed schematic of the Emotional and Motivational Modulation (EMoM) model. The input vector \( \mathbf{x} \) (comprising external, internal, and optional cognitive signals) is processed by a multilayer neural network employing a ReLU activation followed by a hyperbolic tangent function to yield the affective state \( \mathbf{a} \). This state is then used to modulate the initial memory salience \( s \), resulting in a modulated salience \( s_{\text{modulated}} \) that directly influences memory encoding and subsequent consolidation.*

#### 2.9 Neural Cognitive Bus (NCB)

**Role in Memory:**  
The Neural Cognitive Bus (NCB) functions as the central communication substrate within the HCDM framework. It provides asynchronous, multi-channel messaging capabilities that allow all memory subsystems—and indeed all cognitive modules—to exchange state updates, context vectors, and consolidation signals in real time.

**Context-Invariant Messaging:**  
In line with our principle of context-invariant concept representation, the NCB routes “pure” concept signals (for semantic storage) separately from episodic events that include context. For example:

- A message publishing a visual stimulus’s processing outcome will include an invariant feature vector sent to Semantic Memory.
- A separate message, including contextual tags (e.g., time, location), is sent to Episodic Memory.

**Example API Call:**

```python
# Publish invariant concept representation.
await ncb.publish("semantic_store", {
    "concept": "Jackie Chan",
    "embedding": concept_vector  # Context-independent vector.
})

# Publish episodic event with contextual details.
await ncb.publish("episodic_store", {
    "event": "Encountered Jackie Chan at Iguazu Falls",
    "context": ["Iguazu Falls", "Christmas"]
})
```

This separation ensures that retrieval can later use context to filter episodes while leaving the core semantic representation unchanged.

---

## 3. Implementation Details

The implementation of the memory system is realized in Python, leveraging asynchronous I/O and deep learning frameworks (e.g., PyTorch). Key classes include:

- **`SensoryMemory`**: Implements preprocessing and decay mechanisms.
- **`ShortTermMemory` and `WorkingMemory`**: Provide immediate storage with fixed capacities.
- **`IntermediateMemory`**: Buffers information for eventual consolidation, using time-based decay functions and a spaced repetition algorithm.
- **`EnhancedLongTermEpisodicMemory` and `LongTermSemanticMemory`**: Manage long-term storage through a combination of replay buffers, spaced repetition, and a knowledge graph representation—explicitly separating context (episodic) from invariant concept representations (semantic).
- **`ContextAwareRetrieval`**: Interfaces with the DSSM to extract current context vectors and performs similarity-based filtering.
- **`ReplayBuffer`**: Implements priority sampling based on memory salience.
- **`MemoryConsolidationThread`**: Runs in parallel to continuously assess and consolidate memory items.

Each module follows a standardized API (e.g., `add()`, `retrieve()`, `clear()`), ensuring modularity and ease of integration within the broader HCDM framework.

---

## 4. Evaluation and Discussion

### 4.1 Experimental Evaluation

Initial experiments demonstrate that the multi-layered memory system:

- **Improves Contextual Relevance:**  
  Retrieval operations that incorporate context-aware similarity measures yield higher relevance in downstream cognitive tasks, while invariant semantic representations remain stable across contexts.
- **Mitigates Catastrophic Forgetting:**  
  Time-based decay and spaced repetition effectively balance the retention of new and old information.
- **Adapts to Affective Cues:**  
  EMoM integration ensures that emotionally salient items receive preferential consolidation, mirroring human memory biases.
- **Maintains Concept Invariance:**  
  By separating core concept encoding from episodic context, the system retrieves the same semantic representation even when the context changes, echoing findings from human single-neuron recordings (Rey et al., 2024).

Quantitative metrics include memory recall accuracy, retrieval latency, and overall system throughput. Future work will involve benchmarking these metrics in simulated cognitive tasks and real-world environments.

### 4.2 Discussion

The proposed memory system reflects a synthesis of classical connectionist approaches (e.g., Hopfield networks, recurrent architectures) and contemporary innovations in memory consolidation. By drawing on principles from complementary learning systems theory, the design balances rapid encoding with gradual, robust consolidation—a critical challenge in both biological and artificial systems.

Our updated approach—motivated by recent neurobiological findings (Rey et al., 2024)—separates the encoding of core concepts from the contextual details that accompany episodic events. This enables:

- **Stable Semantic Representations:** The invariant encoding of concepts allows for consistent recognition regardless of context.
- **Flexible Episodic Retrieval:** Context acts solely as a retrieval cue, enabling the same concept to be experienced in multiple contexts without redundancy or conflict.
- **Efficient Modular Integration:** With a dedicated Neural Cognitive Bus (NCB), the system ensures that concept representations and episodic details are processed asynchronously yet remain tightly coordinated.

Together, these enhancements offer a pathway toward more human-like cognition in AI, paving the way for future research on adaptive, context-sensitive learning and memory retrieval.

---

## 5. Code Documentation

In this section, we detail how the theoretical design of the hybrid cognitive dynamics memory system is directly mapped to the code implementation. Our design adheres to a standardized API—implemented uniformly across all memory subsystems—thus ensuring that each module exposes methods such as `add()`, `retrieve()`, and `clear()`. This uniformity not only facilitates integration but also mirrors the complementary roles of rapid sensory encoding, transient storage, and long-term consolidation observed in neurobiology.

### 5.1 Sensory Memory

The `SensoryMemory` class is responsible for the rapid acquisition and preprocessing of raw sensory data. It implements modality-specific preprocessing pipelines (e.g., text normalization and image contrast normalization) and an exponential decay mechanism to simulate the transient persistence of sensory traces. The following pseudocode excerpt illustrates how new sensory inputs are processed and stored:

```python
def add(self, input_data: Any) -> None:
    processed = self._preprocess_input(input_data)
    timestamp = time.time()
    entry = {"data": processed, "timestamp": timestamp, "salience": 1.0}
    self.buffer.append(entry)
    if len(self.buffer) > self.max_size:
        self.buffer.pop(0)
```

*Figure 5.1: Excerpt from the `SensoryMemory.add()` method demonstrating how sensory data are preprocessed, timestamped, and assigned an initial salience value.*

### 5.2 Short Term Memory

The `ShortTermMemory` class serves as a transient repository for recently acquired information. Designed to enforce a strict capacity limit, this module ensures that the system does not become overloaded with recent inputs. Its standardized API is exemplified by the `add()` method:

```python
def add(self, item: Any) -> None:
    self.items.append(item)
    if len(self.items) > self.capacity:
        self.items = self.items[-self.capacity:]
```

### 5.3 Working Memory

Working Memory is the dynamic processing arena where transient information is actively manipulated. The `WorkingMemory` class adheres to the same API conventions as its sensory and short-term counterparts, thereby enabling seamless integration into the broader cognitive architecture.

### 5.4 Intermediate Memory and Consolidation

The `IntermediateMemory` module acts as a buffer that temporarily holds information earmarked for long-term storage. This module not only follows the standardized API (i.e., `add()`, `retrieve()`, and `clear()`), but it also integrates a spaced repetition algorithm to schedule consolidation. The pseudocode below details how the module evaluates a memory’s decayed salience and schedules it for review:

```python
def consolidate_oldest(self) -> Dict[str, Any]:
    oldest_memory = min(self.memories, key=lambda m: m['timestamp'])
    self.memories.remove(oldest_memory)
    time_elapsed = time.time() - oldest_memory['timestamp']
    decayed_strength = self.time_decay.decay(MemoryType.LONG_TERM_EPISODIC,
                                              time_elapsed, oldest_memory.get('importance', 1.0))
    if decayed_strength > self.consolidation_threshold:
        review_time = time.time() + self.spaced_repetition.sm2_params.get("interval", 1) * 86400
        self.spaced_repetition.schedule_review(memory=oldest_memory, review_time=review_time, emotion_factor=1.0)
```

*Figure 5.2: Pseudocode from the `IntermediateMemory.consolidate_oldest()` method, illustrating the use of time-based decay and spaced repetition for scheduling memory consolidation.*

### 5.5 Long Term Memory Subsystems

Long-Term Memory is divided into two parts:

- **Episodic Memory:**  
  The `EnhancedLongTermEpisodicMemory` class stores episodic events along with their associated contextual details. In our design, context is stored separately from the core event (i.e., the invariant concept), allowing for flexible retrieval.
  
  **Example Code:**

  ```python
  class EpisodicMemory:
      def __init__(self):
          self.episodes = []  # List of episodic memories
          self.context_index = {}  # Mapping from context cues to events

      def store_event(self, event, context_vector):
          self.episodes.append({"event": event, "context": context_vector})
          for ctx in context_vector:
              if ctx not in self.context_index:
                  self.context_index[ctx] = []
              self.context_index[ctx].append(event)

      def retrieve_event(self, query_context):
          relevant_events = []
          for ctx in query_context:
              if ctx in self.context_index:
                  relevant_events.extend(self.context_index[ctx])
          return list(set(relevant_events))
  ```
  
- **Semantic Memory:**  
  The `LongTermSemanticMemory` class stores context-invariant concept embeddings. These representations remain constant regardless of episodic context.
  
  **Example Code:**

  ```python
  class SemanticMemory:
      def __init__(self):
          self.concepts = {}  # Dictionary of concept embeddings
          self.associations = {}  # Episodic links to semantic concepts

      def store_concept(self, name, embedding):
          self.concepts[name] = embedding

      def link_episode(self, concept_name, episode):
          if concept_name in self.concepts:
              if concept_name not in self.associations:
                  self.associations[concept_name] = []
              self.associations[concept_name].append(episode)

      def retrieve_concept(self, name):
          return self.concepts.get(name, None)
  ```

### 5.6 Neural Cognitive Bus (NCB)

The `NeuralCognitiveBus` serves as the central communication backbone of the system. It implements multi-channel, asynchronous messaging that allows each module to exchange state updates and context vectors in real time. An excerpt from the `publish()` method is shown below:

```python
async def publish(self, channel_name: str, data: Any):
    if channel_name not in self.channels:
        raise ValueError(f"Channel '{channel_name}' does not exist.")
    if isinstance(data, torch.Tensor):
        if channel_name in self.nest_modules:
            data = self.nest_modules[channel_name](data)
        await self.channels[channel_name]["queue"].put(data.clone())
```

*Figure 5.3: Excerpt from the `NeuralCognitiveBus.publish()` method, demonstrating how messages (including both invariant concept representations and contextual episodic data) are processed and dispatched.*

### 5.7 Alignment with Theoretical Constructs

Each module in our system is purposefully designed to mirror specific aspects of human cognition:

- **Rapid Sensory Encoding:**  
  The `SensoryMemory` module rapidly processes raw inputs, employing exponential decay to simulate the fleeting nature of sensory traces.
  
- **Capacity-Limited Short-Term Retention:**  
  The `ShortTermMemory` module enforces a fixed capacity, reflecting human short-term memory limitations.
  
- **Active Information Manipulation:**  
  `WorkingMemory` provides a dynamic workspace for reasoning and decision-making.
  
- **Gradual Consolidation with Context Separation:**  
  `IntermediateMemory` and its spaced repetition mechanism ensure that only high-salience information is consolidated into long-term stores. Episodic Memory stores context separately from the invariant semantic representation.
  
- **Durable, Invariant Storage:**  
  The bifurcated Long-Term Memory system captures both episodic details and generalized, context-free semantic representations.

### 5.8 Summary

The advanced code documentation provided here bridges the gap between theoretical design and practical implementation. Through detailed pseudocode excerpts and rigorous API specifications, we have demonstrated how each cognitive function is systematically instantiated in the codebase. The modular and standardized approach—enhanced with a clear separation of context and core concept—facilitates both intra-system communication (via the NCB) and robust memory retrieval.

---

## 6. Future Work

Planned enhancements include:

- **Scalability Improvements:**  
  Leveraging distributed computing to handle larger-scale memory stores.
- **Enhanced Semantic Retrieval:**  
  Integrating pretrained embeddings and transformer-based models for richer semantic representations.
- **Adaptive Consolidation:**  
  Dynamic adjustment of consolidation thresholds based on system performance and environmental feedback.
- **Extended Evaluation:**  
  Rigorous testing in multimodal and real-time environments to further validate the architecture.
- **Integration with Decision-Making Modules:**  
  Exploring how context-invariant memory representations can be leveraged to improve adaptive reinforcement learning and real-time decision-making.

---

## 7. Conclusion

This paper has detailed a biologically inspired, multi-layered memory system for the Hybrid Cognitive Dynamics Model. By combining rapid sensory processing, multiple temporal memory buffers, and long-term storage—with a crucial update that decouples invariant semantic representations from episodic context—we offer a robust and flexible approach to memory. The system mimics biological mechanisms by:

- Rapidly encoding sensory data,
- Maintaining capacity-limited short-term representations,
- Dynamically manipulating working memory,
- Consolidating high-salience items through spaced repetition,
- Storing enduring semantic knowledge in a context-invariant manner,
- And retrieving episodic details using context-aware filtering.

These innovations lay the foundation for more human-like cognition in artificial systems. Future work will further refine and extend this framework, ensuring its applicability in a wide range of cognitive tasks and environments.

---

## References

1. McClelland, J. L., McNaughton, B. L., & O’Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory. *Psychological Review, 102*(3), 419–457.
2. Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017). Neuroscience-Inspired Artificial Intelligence. *Neuron, 95*(2), 245–258.
3. Eliasmith, C., & Anderson, C. H. (2003). *Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems*. MIT Press.
4. Graves, A., Wayne, G., Reynolds, M., et al. (2016). Hybrid Computing Using a Neural Network with Dynamic External Memory. *Nature, 538*(7626), 471–476.
5. Rey, H. G., Panagiotaropoulos, T. I., Gutierrez, L., et al. (2024). Lack of context modulation in human single neuron responses in the medial temporal lobe. *Cell Reports*. <https://doi.org/10.1016/j.celrep.2024.115218>

---

*Note: This paper is a living document intended for ongoing development. Future modifications may include additional modules, updated algorithms, and experimental results as the HCDM architecture evolves.*

---

