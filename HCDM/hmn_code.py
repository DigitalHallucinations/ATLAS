# HMN System - Reference Implementation

This directory contains a Python implementation of the Hybrid Modulated Neuron (HMN) system based on the theoretical framework paper provided. It aims to capture the core concepts including:

* Local probabilistic plasticity (Hebbian-style traces).
* Multi-factor neuromodulatory feedback.
* Dual meta-learning of local and global learning rates (using SPSA).
* Dual attention mechanisms (local on traces, global on modulators).
* Oscillation-gated updates.

## File Structure

* `hmn_layer.py`: Contains the `HMNLayer` class implementing the core logic for a single layer of HMN neurons.
* `utils.py`: Helper functions for activation (ReLU), normalization (Softmax), probabilistic gating (Sigmoid), similarity calculations, etc.
* `config.py`: Stores all hyperparameters and configuration constants for the model and simulation.
* `example_usage.py`: A script demonstrating how to initialize and run the `HMNLayer` with dummy inputs and global signals. It also shows the periodic triggering of the meta-learning (SPSA) update. (Note: this file is outside the `hmn_system` package in the provided structure).
* `README.md`: This file.

## Key Components Implemented

* **Neuronal Activation (Eq. 1):** `HMNLayer.step` calculates activation using ReLU and adds noise.
* **Composite Eligibility Traces (Eq. 2):** `HMNLayer.step` updates `psi_fast` and `psi_slow` based on Hebbian interaction (`z_j * x_i`) and exponential decay.
* **Local Attention (Eq. 8-9):** `HMNLayer.update_weights` calculates `alpha_ij` using simplified similarity and softmax, then modulates `e_ij`.
* **Global Attention (Eq. 10-11):** `HMNLayer.update_weights` calculates `gamma_k` based on modulator/context similarity (using cosine similarity as default or fallback) and computes `G_prime`.
* **Preliminary Update (Eq. 3):** `HMNLayer.update_weights` computes `delta_W_star`.
* **Oscillatory Gating (Eq. 4):** `HMNLayer.update_weights` computes `delta_W_dagger` using phase gating.
* **Probabilistic Update (Eq. 5):** `HMNLayer.update_weights` computes `p_update` and applies `delta_W_dagger` probabilistically.
* **Dual Meta-Learning (Eq. 7, Appendix 10.3):** `HMNLayer.update_etas_spsa` implements the SPSA algorithm to update `eta_local` and `eta_global`. Requires a placeholder `_evaluate_meta_loss` function to be defined based on the task.

## Assumptions and Simplifications

* Models a single layer only.
* Synaptic delays (`tau_ij`) are ignored (assumed 0).
* Input embeddings (`h_i`) are simplified to raw input `x_i`.
* Local context (`c_j`) is an EMA of neuron activation `z_j`.
* Local attention similarity `g(h_i, c_j)` uses a simple interaction term (outer product style).
* Global context (`C_global`), neuromodulators (`E_k`), global phase (`Phi(t)`), and meta-loss (`L_meta`) are assumed to be provided externally.
* The `_evaluate_meta_loss` function within `HMNLayer` is a **placeholder** and needs task-specific implementation for SPSA to work meaningfully.
* Global attention similarity `h(E_k, C_global)` uses a default implementation (cosine similarity if possible, fallback to product) assuming vector inputs or a placeholder embedding for scalar `E_k`, which might need adjustment based on the actual nature of `E_k` and `C_global`.

## How to Run

1.  Ensure you have Python and NumPy installed (`pip install numpy`).
2.  Save the files in the structure described above (`hmn_project/hmn_system/...` and `hmn_project/example_usage.py`).
3.  Navigate to the `hmn_project` directory in your terminal.
4.  Run the example script:
    ```bash
    python example_usage.py
    ```

This will simulate the HMN layer for a number of steps, printing status updates and demonstrating the weight updates and periodic meta-learning adjustments. Note that without a proper `_evaluate_meta_loss` function, the SPSA updates will operate on dummy loss values, and the learning rates will not adapt meaningfully to a specific task.

## Next Steps for Real Tasks

This code provides a structural implementation of the HMN framework as described. To use it for actual tasks, you would need to:
 * **Implement `_evaluate_meta_loss`:** Replace the placeholder in `hmn_layer.py` with a function that runs the network on your task and returns a meaningful performance metric (e.g., negative reward, classification error). This is essential for effective meta-learning.
 * **Provide Real Inputs:** Feed actual data (`x_i`) relevant to your task.
 * **Define Global Signals:** Determine how `global_neuromodulators` (E_k), `global_context_vector` (C_global), and `global_phase` (Phi(t)) are generated or obtained from the environment, task state, or a higher-level controller.
 * **Refine Embeddings/Similarity:** Potentially improve the simplified `h_i`, `c_j`, `g(...)`, and `h(...)` functions for better performance or biological realism if required by the task. Consider learnable embeddings.
 * **Network Architecture:** Integrate the `HMNLayer` into a larger network (e.g., stacking layers, connecting to other components) if needed. Define signal propagation between layers.
 * **Hyperparameter Tuning:** Adjust the parameters in `config.py` (learning rates, decay constants, attention betas, SPSA settings, etc.) for optimal performance on your task.
 * **Analysis:** Add logging and analysis to track weights, activations, traces, learning rates, and task performance.

# hmn_project/example_usage.py
import numpy as np
import time
# Import from the hmn_system package
from hmn_system.hmn_layer import HMNLayer
from hmn_system import config # Import config to access sizes and parameters

def get_dummy_neuromodulators():
    """ Generates dummy neuromodulator signals based on keys in config. """
    modulators = {}
    # Generate values for keys defined in NEUROMODULATOR_WEIGHTS
    for key in config.NEUROMODULATOR_WEIGHTS.keys():
        if key == 'reward':
            modulators[key] = np.random.uniform(-1, 1)
        else: # Assume others are positive like uncertainty, novelty
            modulators[key] = np.random.rand()
    # Add a modulator not in weights to test warning
    # modulators['surprise'] = np.random.rand()
    return modulators

def get_dummy_global_context():
    """ Generates a dummy global context vector. """
    return np.random.rand(config.GLOBAL_CONTEXT_DIM)

def get_dummy_global_phase(t, frequency=0.1):
    """ Generates a dummy global phase signal (0 to 2*pi) over time. """
    # Ensure frequency is positive to avoid issues with modulo
    if frequency <= 0:
        frequency = 0.1
    return (2 * np.pi * frequency * t) % (2 * np.pi)

# --- Initialization ---
print("Initializing HMN Layer...")
# Ensure config values are used for initialization
hmn_layer = HMNLayer(input_size=config.INPUT_SIZE, output_size=config.OUTPUT_SIZE)
print(f"Input size: {config.INPUT_SIZE}, Output size: {config.OUTPUT_SIZE}")
print(f"Global Context Dim: {config.GLOBAL_CONTEXT_DIM}, Neuromodulator Embedding Dim: {config.NEUROMODULATOR_EMBEDDING_DIM}")
print(f"Initial eta_local: {hmn_layer.eta_local:.6f}, Initial eta_global: {hmn_layer.eta_global:.6f}")
print(f"Meta-learning update frequency: {config.META_LEARNING_UPDATE_FREQ} steps")

# --- Simulation Loop ---
# Run for slightly more than N meta-updates to see multiple SPSA steps
num_meta_updates_to_run = 5
num_steps = num_meta_updates_to_run * config.META_LEARNING_UPDATE_FREQ + 5

print(f"\n--- Starting Simulation ({num_steps} steps) ---")
start_time = time.time()

for t_step in range(num_steps):
    # 1. Get Input Data (dummy)
    input_vector = np.random.rand(config.INPUT_SIZE)

    # 2. Get Global Signals (dummy)
    neuromodulators = get_dummy_neuromodulators()
    global_context = get_dummy_global_context()
    # Use the internal layer time 't' for consistent phase generation
    global_phase = get_dummy_global_phase(hmn_layer.t, frequency=0.1) # Pass layer's internal time

    # 3. Decide if meta-learning should run this step
    # The run_step_and_update method handles the frequency check internally
    # We can force it with run_meta_learning=True if needed for specific events
    run_meta_this_step = False # Let the internal check handle it usually

    # Print status periodically or when meta-learning runs
    # Check if meta-learning will run this step (based on the *next* step number t+1)
    will_run_meta = ((hmn_layer.t + 1) > 0 and ((hmn_layer.t + 1) % config.META_LEARNING_UPDATE_FREQ == 0))

    if t_step % 50 == 0 or will_run_meta: # Print more often around meta-learning
        print(f"\nStep {t_step} (Internal time t={hmn_layer.t}):")
        # print(f"  Input shape: {input_vector.shape}") # Less verbose
        print(f"  Neuromodulators: { {k: f'{v:.2f}' for k, v in neuromodulators.items()} }")
        # print(f"  Global Context: {np.round(global_context, 2)}") # Verbose
        print(f"  Global Phase: {global_phase:.3f}")
        print(f"  Current etas: local={hmn_layer.eta_local:.6f}, global={hmn_layer.eta_global:.6f}")
        print(f"  Weight sample (W[0,0]): {hmn_layer.W[0,0]:.4f}") # Check if weights are changing


    # 4. Run HMN step, update weights, and potentially update etas
    output_activation = hmn_layer.run_step_and_update(
        x_i=input_vector,
        global_neuromodulators=neuromodulators,
        global_context_vector=global_context,
        global_phase=global_phase,
        run_meta_learning=run_meta_this_step # Usually False, let internal check run
    )

    # Print output info periodically or after meta-learning
    if t_step % 50 == 0 or will_run_meta:
        # print(f"  Output Activation (z_j) shape: {output_activation.shape}") # Less verbose
        print(f"  Output Activation sample (z_j[0]): {output_activation[0]:.4f}")
        # print(f"  Local Context sample (c_j[0]): {hmn_layer.c_j[0]:.4f}") # Check context update


# --- Simulation End ---
end_time = time.time()
print("\n--- Simulation Complete ---")
print(f"Total steps: {num_steps}")
print(f"Final etas: local={hmn_layer.eta_local:.6f}, global={hmn_layer.eta_global:.6f}")
print(f"Total SPSA updates performed: {hmn_layer.meta_learning_t}")
print(f"Simulation duration: {end_time - start_time:.2f} seconds")

# hmn_system/hmn_layer.py
import numpy as np
from . import utils  # Relative import for utils within the package
from . import config # Relative import for config within the package

class HMNLayer:
    """ Implements a single layer of Hybrid Modulated Neurons (HMN). """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.t = 0 # Internal time step counter for simulation steps
        self.meta_learning_t = 0 # Counter specifically for SPSA decay

        # --- Initialize Parameters ---
        # Weights and Biases (Eq. 1)
        self.W = np.random.normal(0, config.WEIGHT_INIT_STD_DEV, (output_size, input_size))
        self.b = np.full((output_size,), config.BIAS_INIT_VALUE)

        # Eligibility Traces (Eq. 2) - Store fast and slow components
        self.psi_fast = np.zeros((output_size, input_size))
        self.psi_slow = np.zeros((output_size, input_size))
        self.e_ij = np.zeros((output_size, input_size)) # Composite trace e_ij = psi_fast + psi_slow

        # Local Context (for Local Attention Eq. 9)
        self.c_j = np.zeros(output_size) # Exponential moving average of z_j

        # Phase Preferences (Eq. 4)
        self.phi_ij = np.random.uniform(config.INITIAL_PHI_IJ_MIN,
                                        config.INITIAL_PHI_IJ_MAX,
                                        (output_size, input_size))

        # Learning Rates (Eq. 7) - Initialized from config
        self.eta_local = config.INITIAL_ETA_LOCAL
        self.eta_global = config.INITIAL_ETA_GLOBAL

        # Store latest activation and input for trace calculation
        self.last_z_j = np.zeros(output_size)
        self.last_x_i = np.zeros(input_size) # Assumes tau_ij = 0

        # Store current G' for potential use if needed between updates
        self.current_G_prime = 0.0


    def step(self, x_i):
        """
        Performs one forward pass: calculates activation and updates traces.
        Args:
            x_i (np.ndarray): Input vector of shape (input_size,).
        Returns:
            z_j (np.ndarray): Output activation vector of shape (output_size,).
        """
        if not isinstance(x_i, np.ndarray) or x_i.shape != (self.input_size,):
            raise ValueError(f"Input x_i must be a numpy array of shape ({self.input_size},), got {type(x_i)} with shape {getattr(x_i, 'shape', 'N/A')}")

        self.last_x_i = x_i # Store for trace update

        # 1. Calculate Neuronal Activation with Stochasticity (Eq. 1)
        noise = np.random.normal(0, config.NOISE_STD_DEV, self.output_size)
        # Ensure shapes align for matrix multiplication: (output_size, input_size) @ (input_size,) -> (output_size,)
        linear_activation = self.W @ x_i + self.b + noise
        z_j = utils.relu(linear_activation) # Using ReLU from Appendix 10.1
        self.last_z_j = z_j

        # 2. Update Local Context c_j (EMA of z_j)
        self.c_j = config.CONTEXT_DECAY * self.c_j + (1 - config.CONTEXT_DECAY) * z_j

        # 3. Update Composite Eligibility Traces (Eq. 2, Appendix 10.1 style)
        # Hebbian-style update: pre * post activity
        # Assumes trace updated *after* activation calculation, based on x_i and z_j at time t
        # Reshape z_j to (output_size, 1) and x_i to (1, input_size) for outer product
        hebbian_term = z_j[:, np.newaxis] * x_i[np.newaxis, :] # Shape: (output_size, input_size)

        # Decay existing traces and add new Hebbian term
        self.psi_fast = config.DECAY_FACTOR_FAST * self.psi_fast + hebbian_term
        self.psi_slow = config.DECAY_FACTOR_SLOW * self.psi_slow + hebbian_term

        # Combine traces (Eq. 2)
        self.e_ij = self.psi_fast + self.psi_slow

        return z_j

    def update_weights(self, global_neuromodulators, global_context_vector, global_phase):
        """
        Updates the synaptic weights based on the HMN learning rule (Eq. 3-5, 8-11).
        Args:
            global_neuromodulators (dict): Dict of {name: value}, e.g., {'reward': 1.0}. Can be None or empty.
            global_context_vector (np.ndarray): Vector representing global context C_global.
            global_phase (float): Current phase Phi(t) of the global oscillator.
        """
        if not isinstance(global_context_vector, np.ndarray) or global_context_vector.shape != (config.GLOBAL_CONTEXT_DIM,):
             raise ValueError(f"Global context vector must be a numpy array of shape ({config.GLOBAL_CONTEXT_DIM},), got {type(global_context_vector)} with shape {getattr(global_context_vector, 'shape', 'N/A')}")
        if not isinstance(global_phase, (int, float)):
             raise ValueError(f"Global phase must be a number, got {type(global_phase)}")

        # 1. Local Attention on Eligibility Traces (Eq. 8-9)
        # Simplified: h_i = x_i (self.last_x_i stored)
        # Simplified: g(h_i, c_j) calculated by utils.calculate_similarity_g
        similarity_g = utils.calculate_similarity_g(self.last_x_i, self.c_j) # Shape: (output_size, input_size)
        # Normalize over input connections for each output neuron (axis=1 represents inputs for a given output j)
        alpha_ij = utils.softmax(config.BETA_A * similarity_g, axis=1)
        e_tilde_ij = alpha_ij * self.e_ij # Modulated eligibility trace

        # 2. Global Attention on Neuromodulatory Signals (Eq. 10-11)
        G_prime = 0.0
        # Ensure global_neuromodulators is a dict before proceeding
        if global_neuromodulators and isinstance(global_neuromodulators, dict) and len(global_neuromodulators) > 0:
            modulator_keys = list(global_neuromodulators.keys())
            similarities_h = []
            modulator_values_E_k = [] # Store the numeric values E_k

            # Calculate similarity h(E_k, C_global) for each modulator
            for key in modulator_keys:
                E_k_value = global_neuromodulators[key]
                if not isinstance(E_k_value, (int, float, np.number)):
                    print(f"Warning: Skipping non-numeric modulator '{key}' with value {E_k_value} in global attention.")
                    continue

                modulator_values_E_k.append(E_k_value) # Store the valid numeric value

                # --- Embedding E_k (Placeholder Strategy) ---
                # This part is underspecified. Using a simple placeholder:
                # If NEUROMODULATOR_EMBEDDING_DIM > 1, repeat scalar value to create a vector.
                # If NEUROMODULATOR_EMBEDDING_DIM == 1, use the scalar value itself.
                # This allows calculate_similarity_h to use cosine if dims match, or fallback product.
                if config.NEUROMODULATOR_EMBEDDING_DIM > 1:
                    E_k_vec = np.full(config.NEUROMODULATOR_EMBEDDING_DIM, E_k_value)
                else:
                    E_k_vec = np.array(E_k_value) # Treat as scalar or 1D array

                sim_h = utils.calculate_similarity_h(E_k_vec, global_context_vector)
                similarities_h.append(sim_h)

            # Only proceed if we successfully processed at least one modulator
            if similarities_h:
                similarities_h = np.array(similarities_h)
                modulator_values_E_k = np.array(modulator_values_E_k)

                # Calculate attention weights gamma_k (Eq. 11)
                # Softmax applied only to the similarities of valid modulators
                gamma_k = utils.softmax(config.BETA_G * similarities_h) # Shape: (num_valid_modulators,)

                # Calculate attention-weighted aggregated signal G'(t) (Eq. 10)
                valid_modulator_idx = 0
                for i, key in enumerate(modulator_keys):
                    # Check if this modulator was included (was numeric)
                     if isinstance(global_neuromodulators[key], (int, float, np.number)):
                        w_k = config.NEUROMODULATOR_WEIGHTS.get(key, 0.0) # Get weight, default to 0.0 if not found
                        if not isinstance(w_k, (int, float, np.number)):
                            print(f"Warning: Non-numeric weight found for modulator '{key}'. Using 0.0.")
                            w_k = 0.0
                        # Use the stored numeric E_k value corresponding to this gamma_k index
                        E_k = modulator_values_E_k[valid_modulator_idx]
                        G_prime += gamma_k[valid_modulator_idx] * w_k * E_k
                        valid_modulator_idx += 1
            else:
                print("Warning: No valid numeric neuromodulators found for global attention.")
                G_prime = 0.0 # Ensure G_prime is 0 if no valid modulators
        else:
             # If no global modulators provided or it's not a valid dict, G_prime is 0
             G_prime = 0.0

        self.current_G_prime = G_prime # Store for potential external access/logging

        # 3. Calculate Preliminary Update Δw* (Eq. 3)
        # Using G_prime based on Eq 10 for the global term
        # Ensure G_prime is treated as a scalar in the broadcast operation
        delta_W_star = self.eta_local * e_tilde_ij + self.eta_global * G_prime * e_tilde_ij
        # Shape: (output_size, input_size)

        # 4. Apply Oscillatory Gating Δw† (Eq. 4)
        phase_diff = global_phase - self.phi_ij # Shape: (output_size, input_size)
        # Cosine is applied element-wise
        phase_gate = np.maximum(0, np.cos(phase_diff)) # Shape: (output_size, input_size)
        delta_W_dagger = delta_W_star * phase_gate

        # 5. Apply Probabilistic Update Δw (Eq. 5)
        # Calculate update probability p_update
        abs_delta_W_dagger = np.abs(delta_W_dagger)
        # Apply sigmoid element-wise
        p_update = utils.sigmoid(config.BETA_P * (abs_delta_W_dagger - config.THETA_P)) # Shape: (output_size, input_size)

        # Generate random numbers for probabilistic application
        random_gate = np.random.rand(self.output_size, self.input_size)

        # Apply update only where random_gate < p_update (element-wise comparison)
        update_mask = random_gate < p_update
        # Use np.where for conditional application: if mask is True, use delta_W_dagger, else use 0
        delta_W = np.where(update_mask, delta_W_dagger, 0.0)

        # Apply the final weight update
        self.W += delta_W


    def _evaluate_meta_loss(self, eta_local_test, eta_global_test):
        """
        Placeholder function: Evaluates the meta-objective L_meta.
        THIS IS A CRITICAL PLACEHOLDER and needs to be implemented based
        on the specific task and performance metric for meta-learning to be meaningful.
        It should run the network with the test etas and return a scalar loss (lower is better).
        Args:
            eta_local_test (float): Temporary local learning rate to evaluate.
            eta_global_test (float): Temporary global learning rate to evaluate.
        Returns:
            float: A scalar loss value. Lower values indicate better performance.
        """
        print(f"--- [Placeholder] Evaluating Meta Loss with eta_local={eta_local_test:.5f}, eta_global={eta_global_test:.5f} ---")
        # In a real scenario:
        # 1. Store original etas: original_eta_local = self.eta_local, original_eta_global = self.eta_global
        # 2. Temporarily set self.eta_local = eta_local_test, self.eta_global = eta_global_test
        # 3. Create a copy of the relevant network state (weights, traces if needed) or reset the environment.
        # 4. Run the HMN layer (or the full network/agent) on a representative task, batch, or episode.
        # 5. Calculate the performance metric (e.g., cumulative negative reward, error rate, task-specific objective).
        # 6. Restore original etas: self.eta_local = original_eta_local, self.eta_global = original_eta_global
        # 7. Return the calculated metric as L_meta (e.g., return -total_reward).

        # Dummy implementation: returns a value that decreases as etas get closer to some arbitrary target
        # This allows SPSA to run but the results are meaningless without a real L_meta.
        target_eta_local = 0.05
        target_eta_global = 0.005
        # Simple quadratic loss around the target
        loss = (eta_local_test - target_eta_local)**2 + 10 * (eta_global_test - target_eta_global)**2
        # Add some noise to simulate real-world evaluation stochasticity
        loss += np.random.normal(0, loss * 0.05) # 5% noise relative to loss

        print(f"--- [Placeholder] Meta Loss = {loss:.4f} ---")
        return loss


    def update_etas_spsa(self):
        """ Updates eta_local and eta_global using SPSA (Appendix 10.3). """
        self.meta_learning_t += 1 # Increment counter for SPSA decay
        t = self.meta_learning_t

        # Calculate perturbation magnitude epsilon (potentially decaying)
        # Ensure etas used for epsilon calculation are positive
        current_eta_local = max(self.eta_local, config.ETA_MIN)
        current_eta_global = max(self.eta_global, config.ETA_MIN)

        epsilon_local = config.SPSA_EPSILON_FACTOR * current_eta_local
        epsilon_global = config.SPSA_EPSILON_FACTOR * current_eta_global

        if config.SPSA_PERTURBATION_DECAY and t > 0:
             decay_factor = 1.0 / np.sqrt(t)
             epsilon_local *= decay_factor
             epsilon_global *= decay_factor
        elif config.SPSA_PERTURBATION_DECAY and t == 0:
             print("Warning: SPSA decay enabled but step counter is 0. Check logic.")


        # Generate random perturbation vectors (Bernoulli ±1)
        # SPSA typically uses *one* random vector delta applied to all parameters.
        # The pseudocode in 10.3 and the provided implementation imply independent perturbations
        # for eta_local and eta_global. We follow the independent perturbation approach here.

        # --- Update eta_local ---
        # Generate a random direction (+1 or -1)
        direction_local = np.random.choice([-1, 1])
        # Calculate the actual perturbation amount
        delta_local = direction_local * epsilon_local

        # Perturb eta_local in both directions
        eta_local_plus = self.eta_local + delta_local
        eta_local_minus = self.eta_local - delta_local

        # Ensure perturbed values stay within reasonable bounds for evaluation (optional but recommended)
        eta_local_plus = np.clip(eta_local_plus, config.ETA_MIN, config.ETA_MAX)
        eta_local_minus = np.clip(eta_local_minus, config.ETA_MIN, config.ETA_MAX)


        # Evaluate meta loss at perturbed points (using current eta_global)
        L_plus_local = self._evaluate_meta_loss(eta_local_plus, self.eta_global)
        L_minus_local = self._evaluate_meta_loss(eta_local_minus, self.eta_global)

        # Estimate gradient for eta_local (Finite difference approximation)
        # The denominator should be the difference between the points where loss was evaluated
        actual_delta_local = eta_local_plus - eta_local_minus # This is approx 2 * delta_local if no clipping occurred
        if np.abs(actual_delta_local) > 1e-9: # Avoid division by very small number or zero
             g_hat_local = (L_plus_local - L_minus_local) / actual_delta_local
        else:
             g_hat_local = 0.0
             print("Warning: SPSA perturbation for eta_local was near zero. Gradient estimate set to 0.")

        # Update eta_local using the estimated gradient
        new_eta_local = self.eta_local - config.ALPHA_META_1 * g_hat_local
        # Clip the final updated eta to the allowed range
        self.eta_local = np.clip(new_eta_local, config.ETA_MIN, config.ETA_MAX)


        # --- Update eta_global ---
        # Generate a random direction (+1 or -1)
        direction_global = np.random.choice([-1, 1])
        # Calculate the actual perturbation amount
        delta_global = direction_global * epsilon_global

        # Perturb eta_global in both directions
        eta_global_plus = self.eta_global + delta_global
        eta_global_minus = self.eta_global - delta_global

        # Ensure perturbed values stay within reasonable bounds for evaluation
        eta_global_plus = np.clip(eta_global_plus, config.ETA_MIN, config.ETA_MAX)
        eta_global_minus = np.clip(eta_global_minus, config.ETA_MIN, config.ETA_MAX)

        # Evaluate meta loss at perturbed points
        # Use the *already updated* self.eta_local from the previous step, as per typical parameter updates
        L_plus_global = self._evaluate_meta_loss(self.eta_local, eta_global_plus)
        L_minus_global = self._evaluate_meta_loss(self.eta_local, eta_global_minus)

        # Estimate gradient for eta_global
        actual_delta_global = eta_global_plus - eta_global_minus
        if np.abs(actual_delta_global) > 1e-9: # Avoid division by zero
             g_hat_global = (L_plus_global - L_minus_global) / actual_delta_global
        else:
             g_hat_global = 0.0
             print("Warning: SPSA perturbation for eta_global was near zero. Gradient estimate set to 0.")

        # Update eta_global
        new_eta_global = self.eta_global - config.ALPHA_META_2 * g_hat_global
        # Clip the final updated eta
        self.eta_global = np.clip(new_eta_global, config.ETA_MIN, config.ETA_MAX)

        print(f"SPSA Step {t}: Updated etas: local={self.eta_local:.6f}, global={self.eta_global:.6f} (grads_hat: local={g_hat_local:.4f}, global={g_hat_global:.4f})")


    def run_step_and_update(self, x_i, global_neuromodulators, global_context_vector, global_phase, run_meta_learning=False):
        """
        Convenience method to run a forward step, update weights, and potentially update etas.
        Args:
            x_i (np.ndarray): Input vector.
            global_neuromodulators (dict): Neuromodulator signals.
            global_context_vector (np.ndarray): Global context.
            global_phase (float): Global oscillation phase.
            run_meta_learning (bool): Force meta-learning update if True. Otherwise, controlled by frequency.

        Returns:
            np.ndarray: Output activation vector z_j.
        """
        # Increment internal simulation time step
        self.t += 1

        # Forward pass: get activation, update traces
        z_j = self.step(x_i)

        # Update weights based on HMN rule using signals from the *current* step
        self.update_weights(global_neuromodulators, global_context_vector, global_phase)

        # Optionally run meta-learning update (e.g., every N steps/episodes)
        # Check frequency OR if forced by the flag
        # Ensure t > 0 to avoid running meta-learning at the very first step if freq=1
        if (self.t > 0 and (self.t % config.META_LEARNING_UPDATE_FREQ == 0)) or run_meta_learning:
             print(f"\n--- Running Meta-Learning Update at step {self.t} (SPSA step {self.meta_learning_t + 1}) ---")
             self.update_etas_spsa() # This increments self.meta_learning_t internally
             print(f"--- Meta-Learning Update Complete ---\n")

        return z_j

# hmn_system/config.py
import numpy as np

# --- Network Structure ---
INPUT_SIZE = 10
OUTPUT_SIZE = 5
# Define embedding dim if using vector representations for global attention
GLOBAL_CONTEXT_DIM = 4
NEUROMODULATOR_EMBEDDING_DIM = 4 # Example dimension, used if embedding scalar E_k

# --- Neuron Parameters (Eq. 1) ---
NOISE_STD_DEV = 0.01 # Standard deviation for Gaussian noise epsilon_j(t)

# --- Eligibility Traces (Eq. 2, Appendix 10.1) ---
TAU_FAST = 5.0    # Time constant for fast trace decay (in simulation steps)
TAU_SLOW = 50.0   # Time constant for slow trace decay
# Avoid division by zero if tau is very small or zero
DECAY_FACTOR_FAST = np.exp(-1.0 / TAU_FAST) if TAU_FAST > 0 else 0.0
DECAY_FACTOR_SLOW = np.exp(-1.0 / TAU_SLOW) if TAU_SLOW > 0 else 0.0

# --- Probabilistic Updates (Eq. 5) ---
BETA_P = 10.0     # Steepness of the probabilistic sigmoid gate
THETA_P = 0.01    # Threshold for the probabilistic sigmoid gate

# --- Oscillatory Gating (Eq. 4) ---
# Global oscillation frequency/source needs external definition.
# Phase preferences can be random or learned.
# Initialize randomly between 0 and 2*pi
INITIAL_PHI_IJ_MIN = 0
INITIAL_PHI_IJ_MAX = 2 * np.pi

# --- Global Neuromodulation (Eq. 6, 10) ---
# Weights for aggregating neuromodulators (example for Eq. 6 and potentially Eq. 10)
NEUROMODULATOR_WEIGHTS = {
    'reward': 1.0,
    'uncertainty': -0.5, # Example: uncertainty might decrease plasticity rate
    'novelty': 0.3
    # Add other expected modulators here
}

# --- Dual Meta-Learning (Eq. 7, Appendix 10.3) ---
INITIAL_ETA_LOCAL = 0.01
INITIAL_ETA_GLOBAL = 0.001
ALPHA_META_1 = 0.001 # Meta-learning rate for eta_local
ALPHA_META_2 = 0.0001 # Meta-learning rate for eta_global
SPSA_EPSILON_FACTOR = 0.05 # Factor c for SPSA perturbation magnitude (epsilon = c * eta)
SPSA_PERTURBATION_DECAY = True # Use decaying epsilon_t = epsilon_0 / sqrt(t)
ETA_MIN = 1e-6    # Minimum learning rate
ETA_MAX = 1.0     # Maximum learning rate
META_LEARNING_UPDATE_FREQ = 100 # How often to run SPSA update (e.g., every 100 steps/episodes)

# --- Multi-Modal Attention (Eq. 8-11) ---
# Local Attention (Eq. 8-9)
BETA_A = 5.0      # Inverse temperature for local attention softmax
CONTEXT_DECAY = 0.9 # Decay factor for local context c_j (exponential moving average)

# Global Attention (Eq. 10-11)
BETA_G = 2.0      # Inverse temperature for global attention softmax

# --- Initialization ---
WEIGHT_INIT_STD_DEV = 0.1
BIAS_INIT_VALUE = 0.0

# hmn_system/utils.py
import numpy as np

def sigmoid(x):
  """ Numerically stable sigmoid function. """
  # Clip to avoid overflow/underflow issues with exp
  return 1. / (1. + np.exp(-np.clip(x, -20, 20)))

def softmax(x, axis=-1):
  """ Numerically stable softmax function. """
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / e_x.sum(axis=axis, keepdims=True)

def relu(x):
  """ Rectified Linear Unit activation function. """
  return np.maximum(0, x)

def cosine_similarity(vec_a, vec_b, epsilon=1e-8):
    """ Computes cosine similarity between two vectors or batches of vectors. """
    norm_a = np.linalg.norm(vec_a, axis=-1, keepdims=True)
    norm_b = np.linalg.norm(vec_b, axis=-1, keepdims=True)
    # Handle potential zero vectors by adding epsilon
    dot_product = np.sum(vec_a * vec_b, axis=-1)
    return dot_product / (norm_a * norm_b + epsilon)

def calculate_similarity_g(h_i, c_j):
    """
    Calculates similarity for local attention (Eq. 9).
    Simplified: assumes h_i is scalar input x_i, c_j is scalar context.
    Using simple product as interaction term.
    h_i shape: (input_size,)
    c_j shape: (output_size,)
    Returns shape: (output_size, input_size)
    """
    # Broadcasting h_i (input_size,) to (1, input_size)
    # Broadcasting c_j (output_size,) to (output_size, 1)
    # Result is outer product like interaction: (output_size, input_size)
    # Ensure c_j is treated as column and h_i as row for the outer product effect
    return c_j[:, np.newaxis] * h_i[np.newaxis, :]

def calculate_similarity_h(E_k_vec, C_global_vec, epsilon=1e-8):
    """
    Calculates similarity for global attention (Eq. 11).
    Assumes E_k and C_global are provided as vectors.
    Uses cosine similarity by default.
    E_k_vec shape: (embedding_dim,)
    C_global_vec shape: (embedding_dim,)
    Returns scalar similarity.
    """
    # Note: Paper implies E_k is scalar signal, C_global is context.
    # We need a way to embed E_k or define h differently if E_k is scalar.
    # Placeholder: Assume E_k can be represented/embedded as a vector compatible with C_global.
    # If E_k is just a scalar value, this function needs adjustment
    # (e.g., maybe similarity depends only on C_global or a learned function).
    # Using cosine similarity as a default if vectors are provided.

    # Ensure inputs are numpy arrays
    E_k_vec = np.asarray(E_k_vec)
    C_global_vec = np.asarray(C_global_vec)

    if E_k_vec.ndim == 0 or C_global_vec.ndim == 0 or E_k_vec.size == 1 or C_global_vec.size == 1:
         # Fallback if E_k is scalar or vectors are unsuitable for cosine: use a simple product or placeholder
         # This needs clarification based on how E_k and C_global interact.
         print("Warning: Using placeholder similarity (product) for global attention due to scalar or incompatible E_k/C_global.")
         # Ensure shapes are compatible for element-wise multiplication or scalar multiplication
         try:
             # Attempt scalar product if one is scalar, otherwise element-wise then sum
             if E_k_vec.ndim == 0 or E_k_vec.size == 1:
                 return E_k_vec * np.sum(C_global_vec) # Scalar * sum(vector)
             elif C_global_vec.ndim == 0 or C_global_vec.size == 1:
                 return np.sum(E_k_vec) * C_global_vec # sum(vector) * scalar
             else:
                 # If both are vectors but maybe different sizes - sum element-wise product?
                 # This case is ambiguous, best effort placeholder
                 min_len = min(E_k_vec.size, C_global_vec.size)
                 return np.sum(E_k_vec[:min_len] * C_global_vec[:min_len])
         except Exception as e:
             print(f"Error during placeholder similarity calculation: {e}")
             return 0.0 # Default fallback

    # Proceed with cosine similarity if both are vectors of compatible dimension > 1
    if E_k_vec.shape != C_global_vec.shape:
        print(f"Warning: E_k_vec shape {E_k_vec.shape} and C_global_vec shape {C_global_vec.shape} mismatch for cosine similarity. Using placeholder.")
        # Fallback to placeholder logic as above
        min_len = min(E_k_vec.size, C_global_vec.size)
        return np.sum(E_k_vec[:min_len] * C_global_vec[:min_len])

    return cosine_similarity(E_k_vec, C_global_vec, epsilon=epsilon)

def aggregate_neuromodulators(modulators, weights):
    """
    Aggregates global neuromodulatory signals (Eq. 6).
    Assumes simple weighted sum as per Appendix 10.1.
    modulators: dict {'reward': E_reward, 'uncertainty': E_uncert, ...}
    weights: dict {'reward': w_reward, 'uncertainty': w_uncert, ...}
    """
    G = 0.0
    if not isinstance(modulators, dict) or not isinstance(weights, dict):
        print("Warning: Modulators and weights should be dictionaries.")
        return G

    for key in modulators:
        if key in weights:
            # Ensure values are numeric before multiplying
            mod_val = modulators[key]
            weight_val = weights[key]
            if isinstance(mod_val, (int, float, np.number)) and isinstance(weight_val, (int, float, np.number)):
                 G += weight_val * mod_val
            else:
                 print(f"Warning: Non-numeric value encountered for modulator '{key}' or its weight.")
        else:
            print(f"Warning: Neuromodulator '{key}' found but no corresponding weight provided in config.")
    return G

# hmn_system/__init__.py
# This file makes the hmn_system directory a Python package.



