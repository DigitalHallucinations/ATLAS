# HMN‑LM: A Biologically‑Inspired Language Model with Input-Shaped Dynamics, Local Plasticity, Dual Meta‑Learning, and Multi‑Modal Attention

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

**Abstract**
We introduce HMN‑LM, a sequence model that replaces self‑attention layers with Hybrid Local–Global Modulated Neuron (HMN) cells. Each cell combines (i) probabilistic STDP‑style local plasticity, (ii) attention‑weighted eligibility traces, (iii) multi‑factor neuromodulatory feedback, (iv) dual meta‑learning of local and global learning‑rate schedules, (v) oscillatory phase‑gated weight application, and (vi) mechanisms for the current input to directly shape aspects of neuronal activation, attention, and plasticity parameters. Unlike Transformers that back‑propagate through static parameters, HMN‑LM adapts its weights online during inference, enabling rapid domain adaptation and continual learning—now enhanced by input-sensitive adjustments for immediate contextual processing—while maintaining competitive perplexity on large corpora. We specify the architecture, training regimen, and a staged benchmark suite, and we provide diagrams and image‑placeholders for future figure generation.

**Keywords:** Synaptic Plasticity · Neuromodulation · Meta‑Learning · Attention · Oscillations · Probabilistic Updates · Continual Learning · Language Models · Input-Shaped Dynamics · Neuromorphic AI

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation & Background](#2-motivation--background)
    * [2.1. Limitations of Transformer & SSM LMs](#21-limitations-of-transformer--ssm-lms)
    * [2.2. Why HMN Cells? Biological and Computational Inspirations](#22-why-hmn-cells-biological-and-computational-inspirations)
    * [2.3. Scope: Biologically Inspired versus Biologically Plausible](#23-scope-biologically-inspired-versus-biologically-plausible)
3. [Model Architecture](#3-model-architecture)
    * [3.1. HMN Layer Stack](#31-hmn-layer-stack)
    * [3.2. Key Equations](#32-key-equations)
4. [Training Procedure](#4-training-procedure)
    * [4.1. Outer‑Loop Meta‑Loss](#41-outerloop-metaloss)
    * [4.2. Optimisation Algorithm](#42-optimisation-algorithm)
5. [Benchmark Suite](#5-benchmark-suite)
6. [Implementation & Hardware Considerations](#6-implementation--hardware-considerations)
7. [Expected Advantages and Discussion](#7-expected-advantages-and-discussion)
    * [7.1. Implications for Credit Assignment](#71-implications-for-credit-assignment)
    * [7.2. Comparison with Back-Propagation and Other Bio-Inspired Models](#72-comparison-with-back-propagation-and-other-bio-inspired-models)
    * [7.3. Complexity, Hyper-Parameter Sensitivity, and Scalability](#73-complexity-hyper-parameter-sensitivity-and-scalability)
    * [7.4. Plausibility Map of HMN Components](#74-plausibility-map-of-hmn-components)
8. [Open Questions](#8-open-questions)
9. [Road‑Map](#9-roadmap)
10. [Conclusion](#10-conclusion)
11. [Acknowledgements](#11-acknowledgements)
12. [References](#12-references)
13. [Appendix A: Algorithmic Details](#appendix-a-algorithmic-details)
    * [A.1 Training Loop (Simplified Python‑style Pseudocode)](#a1-training-loop-simplified-pythonstyle-pseudocode)
    * [A.2 Example Functional Forms (Conceptual)](#a2-example-functional-forms-conceptual)
    * [A.3 SPSA Gradient Approximation Details (Meta-Learning)](#a3-spsa-gradient-approximation-details-meta-learning)
    * [A.4 Algorithm Pseudocode for HMN Weight Update (Conceptual single synapse, within `model.hmn_process_step_and_learn`)](#a4-algorithm-pseudocode-for-hmn-weight-update-conceptual-single-synapse-within-modelhmn_process_step_and_learn)

---

## 1. Introduction

Dominant large language models (LLMs), primarily based on the Transformer architecture and its variants like State-Space Models (SSMs), have achieved remarkable success in diverse natural language processing tasks. However, their reliance on centralized error back-propagation through massive sets of static parameters presents significant limitations. These include high computational costs for training and fine-tuning, difficulties in rapid adaptation to new domains or user styles without extensive retraining, and a propensity for catastrophic forgetting when learning sequentially. Furthermore, these models depart sharply from the decentralized, energy-efficient, and continuously adaptive learning mechanisms observed in biological neural systems.

Biological systems, in contrast, leverage rapid local synaptic plasticity (e.g., Spike-Timing-Dependent Plasticity - STDP), often modulated by global signals like neuromodulators (e.g., dopamine, acetylcholine), to enable lifelong learning and adaptation in non-stationary environments. Inspired by these principles, this paper introduces HMN-LM, a language model built upon Hybrid Modulated Neuron (HMN) cells. The HMN framework aims to integrate:

* Fast local plasticity for immediate capture of short-term dependencies and contextual nuances.
* Slower global modulation, representing signals like reward, uncertainty, or novelty, for long-term guidance and consolidation.
* Meta-learning to adapt learning parameters online, balancing plasticity and stability.
* Attention mechanisms at both local (synaptic) and global (neuromodulatory) levels for selective and context-dependent updates.
* Oscillatory timing to align plastic changes with relevant neural processing phases, potentially enhancing temporal coherence.
* **Input-Shaped Dynamics**: A novel core feature where the current input token (or its embedding) directly influences key operational parameters within the HMN cell, such as activation biases, attention gains, learning rate scaling, and plasticity thresholds. This allows for immediate, fine-grained adaptation of neuronal processing to the specific characteristics of the incoming stimulus.

This work focuses on providing a theoretical and architectural blueprint for HMN-LM. We detail its components, mathematical formulation, training procedures, and a comprehensive benchmark suite designed for empirical validation. The core contribution is a unified model that combines these biological inspirations into a coherent computational framework for language modeling, aiming to bridge the gap between the performance of contemporary LLMs and the adaptive, efficient learning paradigms of the brain.

## 2. Motivation & Background

### 2.1. Limitations of Transformer & SSM LMs

Current leading language modeling paradigms, while powerful, exhibit several inherent limitations when contrasted with the desiderata of truly adaptive and efficient learning systems:

| Issue                     | Transformer             | Mamba / SSM             | Consequence for NLP                   |
| :------------------------ | :---------------------- | :---------------------- | :------------------------------------ |
| Long‑range cost           | $O(n^2)$                | $O(n \log n)$ or $O(n)$ | Memory bottlenecks, context limits    |
| Adaptation                | Off‑line fine‑tune      | Off‑line fine‑tune      | Slow to personalise, resource-heavy   |
| Biological realism        | Low                     | Very low                | Limited neuromorphic transfer, insights |
| Catastrophic forgetting   | Severe                  | Severe                  | Continual tasks fail, relearning needed |
| Input-driven retrieval    | Indirect / Static       | Static                  | Suboptimal adaptation to immediate input nuances |
| Real-time weight updates  | Impractical             | Impractical             | No on-device continual learning       |

These limitations hinder the development of LLMs that can continuously learn from new data streams, personalize efficiently on user devices, or operate robustly in dynamic environments without frequent, costly retraining cycles.

### 2.2. Why HMN Cells? Biological and Computational Inspirations

The HMN cell architecture is motivated by a synthesis of principles from neuroscience and machine learning, aiming to address the aforementioned limitations:

* **Local Plasticity for Short-Term Structure:** Inspired by Hebbian learning and STDP (Markram et al., 1997; Bi & Poo, 1998), local plasticity rules allow synapses to modify their strength based on the activity of connected neurons. In HMN-LM, this is hypothesized to capture short‑term phrase structures and local dependencies within sequences rapidly. Eligibility traces, a mechanism where a synapse is "tagged" for later modification, allow for delayed credit assignment.
* **Neuromodulators for Global Context and Policy:** Biological neuromodulators (e.g., dopamine, acetylcholine, norepinephrine) convey global information about reward, novelty, surprise, or uncertainty, broadly influencing synaptic plasticity and neuronal excitability (Schultz, 1998; Yu & Dayan, 2005). HMN cells incorporate proxies for such signals to implement global discourse cues, guiding learning towards desired outcomes or gating plasticity based on contextual relevance.
* **Dual Meta‑Learning for Adaptive Plasticity Schedules:** Neural systems exhibit meta-plasticity, where the rules of plasticity themselves can change. HMN cells employ dual meta‑learning (Doya, 2002; Bellec et al., 2023) to tune the learning rates for both local and globally modulated plasticity schedules online. This allows the model to adapt its own adaptability, potentially balancing rapid learning with stable knowledge retention.
* **Phase‑Locking for Temporal Coherence:** Neural oscillations (e.g., theta, gamma rhythms) are thought to play a role in coordinating neural activity and gating plasticity (Buzsáki & Draguhn, 2004). HMN cells incorporate oscillatory phase‑gating for weight updates, aiming to align learning with intrinsic processing rhythms, which might be analogous to prosodic or event boundaries in language.
* **Input-Shaped Dynamics for Immediate Contextual Adaptation:** A key innovation in HMN-LM is the direct modulation of neuronal and plasticity parameters by the current input. This allows the cell to dynamically adjust its response profile (e.g., sensitivity, learning propensity, attentional focus) based on the specific features of $x_t$ or its embedding $h_t$. This is inspired by concepts where network dynamics are rapidly shaped by incoming stimuli (e.g., input-driven attractor dynamics in some Hopfield network variants, Betteti et al.), allowing for robust processing and immediate, nuanced adaptation without waiting for slower weight changes.
* **Attention Mechanisms for Selective Processing:** Both local attention (akin to synaptic attention focusing on relevant pre-synaptic inputs for a given post-synaptic neuron's context) and global attention (weighting the influence of different neuromodulatory signals based on broader context) are incorporated. This draws inspiration from attentional mechanisms in the cortex (Moran & Desimone, 1985) and their successful application in deep learning (Vaswani et al., 2017).

By integrating these mechanisms, HMN cells aim to provide a more biologically grounded, adaptive, and potentially efficient alternative to standard artificial neurons in sequence processing tasks.

### 2.3. Scope: Biologically Inspired versus Biologically Plausible

It is important to distinguish between components that are "biologically plausible" and those that are "biologically inspired." A mechanism is considered **biologically plausible** if its existence and functional role have direct and substantial empirical support from neuroscience. Examples include STDP and dopamine-gated plasticity. In contrast, elements are **biologically inspired** if they are abstracted from general biological principles or introduced primarily for computational utility, even if direct neurobiological analogues are not (yet) known or are less detailed. An example could be the specific mathematical form of the probabilistic gating function or the precise implementation of meta-learning for input-modulation functions. A detailed plausibility map is discussed in Section 7.5.

## 3. Model Architecture

The HMN-LM replaces the self-attention and feed-forward blocks of a traditional Transformer with a stack of HMN layers.

```mermaid
graph TD
    subgraph Token Pipeline
        A[Input token x_t] --> B[Embedding h_t]
        B --> C[HMN Cell (Input-Modulated)]
        C --> D[Hidden State s_t]
        D --> E[Output projection → logits]
    end

    subgraph HMN Cell (Input-Modulated)
        B --> F[Local Eligibility Traces e_t]
        F --> G[Local Attention α_t (Input-Modulated Gain)]
        G --> H[Oscillation Gate Φ(t)]
        H --> I[Probabilistic Update Δw† (Input-Modulated Rates/Thresholds)]
        I --> J[Weight Matrix W_t+1]
        B --> C_bias{Activation Bias b_input(h_t)}
        subgraph Global Context Loop
            D -.-> K[Neuromodulator Proxies E_k]
            K --> L[Global Attention γ_k (Input-Modulated Neuromodulator Scaling)] %% Optional: Input can scale E_k
            L --> I
        end
        J --> C  %% feedback loop for W_t+1 affecting next step's activation
    end
```

**Figure 1:** Conceptual data‑flow in a single HMN‑LM layer. The input embedding $h_t$ (or features derived from it, $I_t$) now directly influences the HMN cell by: providing an activation bias $b_{input}(h_t)$; modulating the gain of local attention $\alpha_t$; potentially scaling learning rates and neuromodulatory signals; and adjusting thresholds for the probabilistic update $\Delta w^\dagger$.

**(Placeholder for Figure 2):** AI‑generated illustration of dual attention cones (synaptic & modulatory) super‑imposed on a cortical micro‑column sketch, emphasizing how current input $I_t$ can modulate these pathways.

### 3.1. HMN Layer Stack

A typical HMN-LM consists of an embedding layer, followed by $N$ HMN layers, Layer Normalization, a linear projection to the vocabulary size, and a Softmax function to produce output probabilities:
`[Embedding] → [HMN × N] → [LayerNorm] → [Linear Projection] → [Softmax]`

A typical “base” model might have: 12 layers, 512-dimensional hidden states, 128-dimensional embeddings, and approximately 100 million plastic weights. Each HMN layer performs its computations based on the mechanisms described below.

### 3.2. Key Equations

Let $x_t$ be the input token at time $t$, and $h_t$ be its embedding. Let $I_t$ represent characteristics or features derived from the current input $h_t$ (e.g., $I_t = h_t$ or $I_t = \text{MLP}(h_t)$). These features $I_t$ are used to modulate various parameters within the HMN cell dynamically. The weights $w_{ij}$ are those of the HMN cells, effectively forming the connections that are updated by the HMN learning rules.

1. **Activation ($z_j(t)$):** The activation of a post-synaptic neuron $j$.
    $$ z_j(t) = f\left(\sum_i w_{ij}(t) x_i(t-\tau_{ij}) + b_j(t) + \mathbf{b_{input}(I_t)} + \epsilon_j(t)\right) $$
    * $w_{ij}(t)$: weight of the synapse from pre-synaptic neuron $i$ to post-synaptic neuron $j$ at time $t$.
    * $x_i(t-\tau_{ij})$: activity of pre-synaptic neuron $i$, potentially with a delay $\tau_{ij}$. In the LM context, $x_i$ can be elements of the hidden state from the previous layer or recurrent connections.
    * $b_j(t)$: intrinsic bias of neuron $j$.
    * $\mathbf{b_{input}(I_t)}$: A dynamic, input-specific bias term. This is a function (e.g., a small neural network or a linear projection) of the current input features $I_t$. The parameters of this function are meta-learned. This allows the neuron's baseline excitability to be primed by the current input.
    * $\epsilon_j(t)$: optional stochastic noise (e.g., Gaussian).
    * $f(\cdot)$: a non-linear activation function (e.g., ReLU, sigmoid, or tanh).

2. **Eligibility Traces ($e_{ij}(t)$):** A record of recent correlated pre- and post-synaptic activity, marking synapses for potential modification.
    $$ e_{ij}(t) = \psi_{\text{fast}}(x_i, z_j; \tau_{\text{fast}}) + \psi_{\text{slow}}(x_i, z_j; \tau_{\text{slow}}) $$
    * $\psi_{\text{fast}}$ and $\psi_{\text{slow}}$ represent Hebbian-style co-activity measures (e.g., $x_i z_j$) decayed over different time constants ($\tau_{\text{fast}}$, $\tau_{\text{slow}}$), capturing multi-timescale credit assignment. For example, $\psi(x_i, z_j, t') = x_i(t') z_j(t') e^{-(t-t')/\tau}$.

3. **Local Attention ($\alpha_{ij}(t)$):** Modulates the contribution of each eligibility trace based on the relevance of the pre-synaptic input $h_i$ (embedding of $x_i$) to the local context $c_j$ of the post-synaptic neuron.
    $$ \alpha_{ij}(t) = \text{softmax}_i\left(\mathbf{\beta_a(I_t)} \cdot g(h_i(t-\tau_{ij}), c_j(t))\right) $$
    * $h_i(t-\tau_{ij})$: embedding of the pre-synaptic activity/input.
    * $c_j(t)$: local context vector for the post-synaptic neuron $j$ (e.g., an exponentially decayed average of its recent activations $z_j$).
    * $g(\cdot, \cdot)$: a similarity function (e.g., scaled dot product).
    * $\mathbf{\beta_a(I_t)}$: An input-modulated attention gain (temperature) parameter. This function of $I_t$ (whose parameters are meta-learned) allows the sharpness of the local attention to be dynamically adjusted based on the current input. For instance, higher gain might be beneficial for focusing on specific cues in unambiguous inputs, while lower gain might be better for noisy inputs.

4. **Global Neuromodulatory Attention & Aggregation ($G'(t)$):** Aggregates multiple global neuromodulatory signals $E_k(t)$ (e.g., proxies for reward, surprise, novelty derived from the model's state or output) based on their relevance to a global context $C_{\text{global}}(t)$.
    $$ \gamma_k(t) = \text{softmax}_k(\beta_g \cdot h(E_k(t), C_{\text{global}}(t))) $$
    $$ G'(t) = \sum_k \gamma_k(t) E_k(t) $$
    * $E_k(t)$: strength of the $k$-th neuromodulatory signal.
    * $C_{\text{global}}(t)$: a representation of the global context (e.g., average hidden state, discourse topic vector).
    * $h(\cdot, \cdot)$: a relevance/similarity function.
    * $\beta_g$: a fixed or meta-learned gain for global attention.
    * Optionally, $E_k(t)$ itself or its contribution can be scaled by an input-dependent function $s_k(I_t)$, i.e., $G'(t) = \sum_k \gamma_k(t) \mathbf{s_k(I_t)} E_k(t)$, where $s_k(I_t)$ is another meta-learned function, allowing the input to also gate the influence of global signals.

5. **Phase‑Gated Pre‑Update ($\Delta w^{\dagger}_{ij}(t)$):** The calculated weight change, modulated by local and global factors, and gated by an oscillatory phase.
    $$ \Delta w^{\dagger}_{ij}(t) = (\mathbf{\eta_{loc}(I_t)} \cdot \alpha_{ij}(t) e_{ij}(t) + \mathbf{\eta_{glob}(I_t)} \cdot G'(t) \cdot \alpha_{ij}(t) e_{ij}(t)) \cdot \text{max}(0, \cos(\Phi(t) - \phi_{ij})) $$
    Alternatively, a more common formulation for the neuromodulated term is additive to the trace rather than multiplicative with it again:
    $$ \Delta w^{\dagger}_{ij}(t) = (\mathbf{\eta_{loc}(I_t)} + \mathbf{\eta_{glob}(I_t)} G'(t)) \cdot \alpha_{ij}(t) e_{ij}(t) \cdot \text{max}(0, \cos(\Phi(t) - \phi_{ij})) $$
    * $\mathbf{\eta_{loc}(I_t)}$: Input-modulated local learning rate. $\eta_{loc}(I_t) = \eta_{loc\_base} \cdot \eta_{scale\_loc}(I_t)$.
    * $\mathbf{\eta_{glob}(I_t)}$: Input-modulated global learning rate (scales the effect of $G'$). $\eta_{glob}(I_t) = \eta_{glob\_base} \cdot \eta_{scale\_glob}(I_t)$.
    * The base rates ($\eta_{loc\_base}$, $\eta_{glob\_base}$) and the parameters of the scaling functions ($\eta_{scale\_loc}(I_t)$, $\eta_{scale\_glob}(I_t)$) are meta-learned. These functions allow the learning intensity to vary based on current input characteristics (e.g., increase plasticity for novel inputs, decrease for familiar ones).
    * $\Phi(t)$: global oscillatory phase signal (e.g., from a sinusoidal oscillator).
    * $\phi_{ij}$: preferred phase for plasticity at synapse $i \rightarrow j$. This term gates updates to occur only during specific oscillatory phases.

6. **Probabilistic Write ($w_{ij}(t+1)$):** Synaptic weights are updated stochastically based on the magnitude of the pre-update.
    $$ w_{ij}(t+1) \leftarrow w_{ij}(t) + \Delta w^{\dagger}_{ij}(t) \quad \text{with probability } p_{ij}(t) $$
    $$ p_{ij}(t) = \sigma(\mathbf{\beta_p(I_t)} (|\Delta w^{\dagger}_{ij}(t)| - \mathbf{\theta_p(I_t)})) $$
    * $\sigma(\cdot)$: logistic sigmoid function.
    * $\mathbf{\beta_p(I_t)}$: Input-modulated gain for the probabilistic gate.
    * $\mathbf{\theta_p(I_t)}$: Input-modulated threshold for the probabilistic gate.
    * The parameters defining the functions $\beta_p(I_t)$ and $\theta_p(I_t)$ are meta-learned. This allows the model to dynamically adjust the stringency and sensitivity of weight updates based on the current input, e.g., making updates more likely or requiring a stronger signal for them when the input is deemed highly informative or reliable.

**Meta-Learned Parameters:**
The set of meta-learned parameters includes:

* Base learning rates: $\eta_{loc\_base}$, $\eta_{glob\_base}$.
* Parameters of all input-modulation functions: those defining $b_{input}(I_t)$, $\beta_a(I_t)$, $\eta_{scale\_loc}(I_t)$, $\eta_{scale\_glob}(I_t)$, $\beta_p(I_t)$, $\theta_p(I_t)$, and $s_k(I_t)$ (if used). These functions are typically small neural networks or parametric expressions.
* Other gain/temperature parameters: $\beta_g$.
* Parameters related to eligibility trace dynamics, oscillator properties, and preferred phases if not fixed.

These parameters are optimized via an outer-loop meta-learning process, typically using a gradient-free method like Simultaneous Perturbation Stochastic Approximation (SPSA) or evolutionary strategies, based on a task-level objective (e.g., perplexity on a validation set).

## 4. Training Procedure

The training of HMN-LM involves a dual-loop process: an inner loop where local HMN rules update synaptic weights based on incoming data, and an outer loop where meta-parameters (including those defining the input-modulation functions) are optimized.

### 4.1. Outer‑Loop Meta‑Loss

The meta-optimizer aims to minimize a loss function $L_{\text{meta}}$ which typically includes the performance on a held-out validation buffer of sequences $\mathcal{B}_{\text{val}}$, a stability penalty on the magnitude of weight changes, and an optional regularization term on the complexity or magnitude of the parameters defining the input-modulation functions:

$$ L_{\text{meta}} = \text{NLL}(\mathcal{B}_{\text{val}}) + \lambda_{\text{stab}} \mathbb{E}[|\Delta w|^2] + \mathbf{\lambda_{\text{mod}} \mathbb{E}[|\text{params}_{\text{funcs}}(I_t)|^2]} $$

* $\text{NLL}(\mathcal{B}_{\text{val}})$: Negative log‑likelihood of the model on the validation buffer after inner-loop updates.
* $\lambda_{\text{stab}}$: Coefficient for the weight change stability regularizer, discouraging overly volatile weights.
* $\mathbf{\lambda_{\text{mod}}}$: Coefficient for the regularization term on the parameters of the input-modulation functions (e.g., L2 norm of the weights of the small MLPs defining $b_{input}(I_t)$, etc.). This helps prevent these functions from becoming overly complex or sensitive.
* $\mathbb{E}[|\Delta w|^2]$: Expected squared magnitude of synaptic weight changes during the inner loop.
* $\mathbb{E}[|\text{params}_{\text{funcs}}(I_t)|^2]$: Expected squared magnitude of the meta-parameters that define the input-modulation functions.

### 4.2. Optimisation Algorithm

1. **Inner Loop (Local HMN Updates):** For a batch of sequences or a stream of $T_{\text{inner}}$ tokens, the HMN-LM processes input tokens one by one.
    * At each time-step $t$:
        * The current input token $x_t$ is embedded to $h_t$.
        * Input features $I_t$ are derived from $h_t$.
        * The dynamic operational parameters for HMN cells (e.g., $b_{input}(I_t)$, $\beta_a(I_t)$, $\eta_{loc}(I_t)$, $\eta_{glob}(I_t)$, $\beta_p(I_t)$, $\theta_p(I_t)$) are computed using $I_t$ and their respective meta-learned functions.
        * Neurons are activated, eligibility traces are formed, attention weights are computed, neuromodulatory signals are processed, and potential weight changes $\Delta w^{\dagger}_{ij}$ are calculated using these dynamic parameters according to Equations 1-5.
        * Synaptic weights $w_{ij}$ are updated probabilistically according to Equation 6. These updates happen locally within the HMN cells.
    * This process relies only on local information and the dynamically computed parameters, without requiring back-propagation through the entire sequence or model for these weight updates.

2. **Outer Loop (Meta‑Update):** After one or more inner-loop episodes, the meta‑parameters are updated.
    * A gradient-free optimization algorithm like SPSA is typically used. SPSA perturbs the meta-parameters (which include base learning rates, $\beta$ gains, and the parameters of the input-modulation functions) and estimates the gradient of $L_{\text{meta}}$ with respect to these meta-parameters.
    * The meta-parameters are then adjusted in the direction that minimizes $L_{\text{meta}}$.
    * This outer loop operates on a slower timescale than the inner-loop weight updates.

3. **Periodic REINFORCE Step (Optional):** For parameters of probabilistic components not fully determined by $I_t$ or the SPSA-tuned meta-parameters (e.g., base components of $\theta_p, \beta_p$ if they have static parts), a REINFORCE-style algorithm could be periodically applied, using a downstream task metric like perplexity reduction as a reward signal.

**Gradient‑Free Hardware Note:** The inner loop, involving the HMN cell updates (Equations 1-6), is designed to be purely local and potentially implementable on specialized neuromorphic hardware with minimal reliance on central processing if the dynamic parameters from $I_t$ can also be computed locally or efficiently broadcast. Only the meta‑parameters (which now include those for the input-modulation functions) require updates from a central optimizer (e.g., on a CPU/GPU) every $K$ inner-loop steps (e.g., $K \sim 1000$s of tokens).

(See Appendix A.1 for simplified pseudocode of the training loop).

## 5. Benchmark Suite

To evaluate the capabilities of HMN-LM, particularly its continual learning, adaptation, and the benefits of input-shaped dynamics, we propose a staged benchmark suite:

| Phase | Dataset / Task                                           | Metric(s)                             | Purpose                                                                                                                                                                                                                                                           |
| :---- | :------------------------------------------------------- | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0    | Text8 (first 100k-1M characters)                         | Perplexity (PPL) $\downarrow$         | Sanity check, basic language modeling capability, learning‑rate viability.                                                                                                                                                                                        |
| P0.5  | Controlled Noise/Context Datasets                        | PPL $\downarrow$, Acc. Mod. $\uparrow$ | Evaluate benefits of specific input-modulation functions (e.g., $b_{input}$, $\beta_a(I_t)$ etc.) on robustness & fine-grained adaptation. This involves datasets with synthetic noise, varying contextual cues, or abrupt style shifts. Accuracy of Modulation (Acc. Mod.) could measure if dynamic parameters change appropriately. |
| P1    | WikiText‑2 $\rightarrow$ BooksCorpus (incremental)       | PPL $\downarrow$, $\Delta$‑Forgetting $\downarrow$ | Continual learning stress‑test; ability to adapt to new data while retaining old knowledge.                                                                                                                                                                       |
| P2    | Dialogue (e.g., Persona‑Chat, DailyDialog)               | BLEU $\uparrow$, PPL $\downarrow$, Style‑Adapt Speed $\uparrow$ | User‑style rapid adaptation, coherence in conversation.                                                                                                                                                                                          |
| P3    | Cross‑domain (e.g., News $\rightarrow$ Code, News $\rightarrow$ Scientific Papers) | PPL before/after switch $\downarrow$     | Domain‑shift resilience, speed of adaptation to radically different data types.                                                                                                                                                                  |
| P4    | Long‑context QA (e.g., LAMBADA, NarrativeQA, PG-19 excerpts) | Accuracy $\uparrow$, PPL $\downarrow$     | Temporal credit assignment, understanding long-range dependencies.                                                                                                                                                                                              |
| P5    | Multilingual Adaptation (e.g., FLORES-200 subset)        | PPL/BLEU per language                 | Ability to adapt to or co-learn multiple languages.                                                                                                                                                                                                               |

**Ablation Protocol:** A crucial part of the evaluation will be systematic ablation studies. We will disable one HMN component at a time (e.g., local plasticity, global neuromodulation, local attention, global attention, oscillatory gating, meta‑learning itself) or specific input-modulation functions (e.g., make $b_{input}(I_t)$ zero, or make $\beta_a$, $\eta_{loc/glob}$, $\theta_p, \beta_p$ static, non-input-dependent parameters) and record the change in performance ($\Delta$‑performance) on relevant benchmarks (especially P0.5, P1, P2). This will help quantify the contribution of each mechanism.

## 6. Implementation & Hardware Considerations

The HMN-LM is designed with an eye towards efficient implementation, including on future neuromorphic hardware.

| Component                                     | CPU/GPU Implementation Notes                                           | Loihi‑2 / Analog Memristor Potential                                      | Comment                                                                                                                                                           |
| :-------------------------------------------- | :--------------------------------------------------------------------- | :------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedding Layer                               | Standard lookup table                                                  | Off-chip or dedicated on-chip memory                                      | Dominates memory for large vocabs.                                                                                                                                |
| Activation ($z_j$)                            | Vectorized matrix ops + non-linearity + bias addition                  | Analog neuron circuits, digital neuron models                             | $b_{input}(I_t)$ adds a small computation per neuron.                                                                                                               |
| Eligibility Traces ($e_{ij}$)                 | Exponential decay updates; relatively cheap                            | Integrate‑and‑dump circuits, analog decay (RC circuit)                    | Multi-timescale traces add memory but are computationally simple.                                                                                                 |
| Local Attention ($\alpha_{ij}$)               | Softmax over local context similarities                                | Localized similarity computation and normalization                        | $\beta_a(I_t)$ requires computing the gain from $I_t$.                                                                                                            |
| Global Attention ($\gamma_k$)                 | Softmax over neuromodulator relevancies                                | Small associative memory or dot-product units                             | Depends on the number of neuromodulators.                                                                                                                         |
| Oscillation Gate                              | Sine lookup or direct computation                                      | Inherent neuron/circuit phase dynamics, phase-locked loops                | 4‑bit phase bins might suffice, simplifying hardware.                                                                                                             |
| Probabilistic Write                           | Generate Bernoulli mask based on $\Delta w^\dagger$                      | Stochastic bit‑cell, tunable noise injection into comparators             | $\beta_p(I_t), \theta_p(I_t)$ require computing these from $I_t$.                                                                                                       |
| Input Modulation Functions ($b_{input}(I_t), \beta_a(I_t)$, etc.) | Small NNs (e.g., 1-2 layers MLP) or parametric functions per token | Local on-chip logic, small dedicated on-chip NNs, or look-up tables     | Parameters meta-learned. Must be lightweight to minimize per-token overhead. Compute once per token, broadcast results if needed.                             |
| HMN Weight Storage ($w_{ij}$)                 | Standard memory (FP32/FP16)                                            | Memristor crossbar arrays, SRAM with local update logic                   | In-place updates are key for neuromorphic efficiency.                                                                                                             |
| Meta‑update (SPSA)                            | PyTorch/JAX on CPU/GPU                                                 | External micro‑host controller (e.g., x86 or ARM core on chip)            | Infrequent (e.g., every few 1k‑steps or more).                                                                                                                    |

**Projected Footprint:** A 12‑layer HMN‑LM for a 32‑kB vocabulary, with 512-D hidden states and 128-D embeddings, is projected to have its weight memory dominated by the embedding layer and the output projection layer if shared. The plastic HMN weights themselves (e.g., 12 layers \* 512 \* 512) would be around 3M weights per layer. If weights are 16-bit, this is \~6MB per HMN layer for weights, plus state. The total GPU memory for a model with \~100M plastic weights would be manageable (e.g., hundreds of MBs, plus embeddings which could be \~1.2 GB as in the original estimate if embeddings are large and not efficiently managed). Neuromorphic deployment promises significant reduction in parameter storage footprint if weights are stored efficiently in on-chip memory elements like memristors, as they are updated in place. The computational overhead of the input-modulation functions must be carefully managed by keeping them simple (e.g., very small MLPs or direct parametric forms).

## 7. Expected Advantages and Discussion

The integrated design of HMN-LM, particularly with its input-shaped dynamics, is hypothesized to offer several advantages over traditional LLMs:

* **Rapid On‑Device Personalisation and Adaptation:** Local plasticity allows weights to adapt online during inference based on immediate data, without requiring full back‑propagation or retraining cycles. This is crucial for on-device personalization (e.g., adapting to a user's writing style) and quick adaptation to new domains or contexts.
* **Reduced Catastrophic Forgetting:** The combination of neuromodulator‑gated consolidation (where global signals can protect or enhance specific memories), dual meta-learning of plasticity rates, and input-adaptive plasticity rates/thresholds (which can modulate learning based on novelty or familiarity of input) is expected to mitigate catastrophic forgetting in continual learning scenarios.
* **Enhanced Robustness and Contextual Retrieval:** Input-driven adjustments (e.g., $b_{input}(h_t)$ for priming activations based on current context, $\beta_a(I_t)$ for adjusting attentional focus in noisy vs. clean inputs, adaptive $\theta_p(I_t)$ for modulating update stringency based on input reliability) allow for more nuanced and robust real-time responses to diverse and dynamically changing inputs.
* **Interpretability:** Several components offer avenues for interpretability:
  * Eligibility traces ($e_{ij}$) highlight which past activities contributed to potential updates.
  * Local attention weights ($\alpha_{ij}$) show which pre-synaptic elements were deemed relevant.
  * Global attention weights ($\gamma_k$) indicate the influence of different neuromodulatory signals.
  * The response of input-modulation functions (e.g., how $\beta_a(I_t)$ changes with specific input properties like predicted uncertainty or novelty scores derived from $I_t$) can be inspected to understand how the model adapts its processing strategy.
* **Energy Efficiency on Neuromorphic Hardware:** The local nature of updates, sparse weight changes due to probabilistic and phase gating, and event-driven computation (if mapped to spiking HMN variants) align well with the design principles of energy-efficient neuromorphic hardware.
* **Synergistic Integration of Learning Mechanisms:** The HMN framework posits that the combination of local attention, phase-locking, dual meta-learning, input-shaped dynamics, and probabilistic updates can yield a system capable of more sophisticated learning than any single mechanism in isolation. Local mechanisms provide rapid adaptation, global signals provide broader guidance, meta-learning tunes the balance, and input-shaping provides immediate reactivity.

### 7.1. Implications for Credit Assignment

HMN-LM employs a multi-faceted approach to credit assignment:

1. **Synaptic Tagging:** Eligibility traces mark synapses based on local activity.
2. **Attentional Refinement:** Local attention ($\alpha_{ij}$) refines which traces are most relevant.
3. **Temporal Alignment:** Oscillatory gating ($\Phi(t)$) aims to synchronize updates with relevant processing windows.
4. **Outcome-Based Modulation:** Global neuromodulatory signals ($G'(t)$) modulate the strength/direction of updates based on broader outcomes or context.
5. **Adaptive Regulation:** Meta-learning tunes learning rates, and input-shaped dynamics adjust plasticity parameters, both of which regulate the overall credit assignment process.

### 7.2. Comparison with Back-Propagation and Other Bio-Inspired Models

Relative to standard back-propagation, HMN-LM offers online, local updates and continuous adaptation without full gradient re-computation. Compared to earlier three-factor learning rules (e.g., activity, error signal, eligibility), HMN adds:

* **Input-Shaped Dynamics:** Direct, rapid modulation of neuronal and plasticity parameters by the current input.
* **Probabilistic Gating:** Stochastic application of weight updates.
* **Dual Meta-Adaptation:** Online tuning of both local and global learning rates.
* **Dual Attention:** Attention over both synaptic eligibility traces and neuromodulatory signals.
* **Explicit Oscillatory Timing:** Phase-dependent gating of plasticity.

### 7.3. Complexity, Hyper-Parameter Sensitivity, and Scalability

The HMN model introduces several hyper-parameters (e.g., decay constants for eligibility traces $\tau_{\text{fast}}, \tau_{\text{slow}}$; base gain parameters $\beta_a, \beta_g, \beta_p$; oscillation frequency and phase parameters; parameters defining the architecture of input-modulation functions). While meta-learning can automate the tuning of some (like learning rates and gains), the initial structural choices and ranges for these parameters will require careful consideration. Automatic relevance determination or more sophisticated meta-learning techniques might be needed to manage this complexity. Scalability to very large models needs to be empirically investigated, focusing on computational overheads of local computations and the stability of multi-layer HMN stacks.

### 7.4. Plausibility Map of HMN Components

This table distinguishes components based on their level of direct empirical support from neuroscience versus being primarily computational abstractions inspired by biological principles.

| Component                                           | Empirical Support in Neuroscience                       | Classification           | Key Citations (Illustrative)                      |
| :-------------------------------------------------- | :---------------------------------------------------- | :----------------------- | :------------------------------------------------ |
| STDP + Eligibility Traces                           | Strong                                                | Plausible                | Markram et al. 1997; Bi & Poo 1998; Sutton & Barto 1998 |
| Neuromodulator-Gated Plasticity (e.g., Dopamine)    | Strong                                                | Plausible                | Schultz 1998; Izhikevich 2007                     |
| Theta/Gamma Phase-Locked Updates                    | Growing                                               | Plausible                | Buzsáki & Draguhn 2004; O'Keefe & Recce 1993      |
| Attentional Modulation of Neural Activity           | Strong (for sensory/cognitive areas)                  | Plausible (conceptually) | Moran & Desimone 1985; Reynolds & Heeger 2009     |
| Meta-Plasticity / Meta-Learned Rates                | Emerging (evidence for adaptable LR)                  | Plausible-in-Principle   | Doya 2002; Bellec et al. 2023                     |
| Probabilistic Synaptic Updates                      | Indirect (synaptic unreliability, quantal release)    | Inspired                 | Faisal et al. 2008                               |
| Direct Input Shaping of Plasticity Parameters (as formulated) | Limited direct parallel; abstraction                  | Inspired                 | (Conceptual: e.g., context-dependent gain modulation) |
| Dual Attention (Local & Global, as formulated)      | Conceptual abstraction                                | Inspired                 | (Combines local/global attention ideas)           |
| Specific functional forms (e.g., logistic gate for prob. write) | None                                                  | Inspired                 | –                                                 |

## 8. Open Questions

Despite its potential, HMN-LM presents several open questions that require further research:

* **Scalability and Performance Ceiling:** Can HMN-LM based models achieve perplexity and downstream task performance comparable to state-of-the-art Transformers or SSMs of similar scale (e.g., GPT-3-scale, Mamba)? What are the practical limits of stacking HMN layers?
* **Stability of Multi-Loop Meta‑Learning:** Ensuring stable convergence of the dual meta‑learning process, especially with a large number of meta-parameters (including those for the input-modulation functions) and under the noisy gradients from SPSA or other gradient-free optimizers, is critical. Robust SPSA variants or alternative meta-learning algorithms may be needed.
* **Optimal Design of Input-Modulation Functions:** What is the appropriate complexity (e.g., small MLPs, lookup tables, simple parametric curves) for functions like $b_{input}(I_t)$, $\beta_a(I_t)$, etc.? How can we balance their expressive power for fine-grained adaptation against meta-learning stability, computational overhead per token, and the risk of overfitting?
* **Interaction and Stability of Multiple Input Modulations:** How do the various concurrent input-driven adjustments (to activation bias, attention gain, learning rates, plasticity thresholds) interact? Ensuring these mechanisms work synergistically and maintain overall system stability without leading to chaotic or unpredictable behavior is a key challenge.
* **Effective Proxy Neuromodulators for Text:** What are the best practices for extracting meaningful proxy neuromodulatory signals (e.g., for reward, uncertainty, novelty, surprise) from text sequences or the model’s internal state in an unsupervised or self-supervised manner? How should these global signals interact with the direct, input-feature ($I_t$) driven local modulations?
* **Hybrid Architectures:** How effective are hybrid models that combine HMN layers with traditional Transformer blocks or SSM layers (e.g., using HMN as adapter layers or in specific parts of a larger architecture, as explored in conceptual proposals like Trans-HMN or Mamba-HMN)? How many standard blocks can be replaced by HMN layers before task performance degrades, or where are they most beneficially inserted?
* **Theoretical Understanding:** Developing a more formal theoretical understanding of the learning dynamics, convergence properties (for simplified variants), and capacity of HMN networks is an important future direction.
* **Exploiting Sparsity:** Can techniques like Mixture-of-Experts (MoE), as conceptually proposed in MoE-HMN, be effectively combined with HMN cells to create sparsely activated, highly plastic models for further efficiency gains?

## 9. Road‑Map

We outline a phased research and development plan to empirically validate and refine HMN-LM:

| Quarter  | Milestone                                                                                                       | Benchmarks             |
| :------- | :-------------------------------------------------------------------------------------------------------------- | :--------------------- |
| Q2‑2025  | Release initial 50 M‑param HMN‑LM on WikiText‑2 with open weights and training code.                              | P0, P0.5               |
| Q3‑2025  | Report on P1 (continual learning: WikiText-2 $\rightarrow$ BooksCorpus). Initial ablation studies for input-modulation. | P1, P0.5 (ablations)   |
| Q4‑2025  | Neuromorphic FPGA or analog simulation demo (e.g., 3‑layer HMN core) demonstrating potential energy savings.        | (Hardware metrics)     |
| Q1‑2026  | Results on P2 (Dialogue adaptation). First prototype of a hybrid Transformer + HMN system.                        | P2                     |
| Q2‑2026  | Results on P3 (Cross-domain adaptation) and P4 (Long-context QA). Extended ablation studies.                      | P3, P4                 |
| Q3‑2026  | Scalability experiments (e.g., >100M params HMN-LM) and investigation into P5 (Multilingual).                   | P0, P1, P5             |

**(Placeholder for Figure 3):** Timeline infographic showing P0–P5 phases mapped onto 2025-2026 quarters with icons for dataset type and key objectives.

## 10. Conclusion

HMN‑LM, enhanced by mechanisms allowing direct input-shaped dynamics of its neuronal and plasticity parameters, represents a significant step towards language models that learn more like biological systems. By integrating local plasticity, multi-factor neuromodulation, dual meta‑learning of adaptive learning schedules, multi-modal attention, and oscillatory gating, all further refined by real-time input-driven adjustments, HMN-LM moves away from static‑parameter, gradient‑only paradigms. It aims to create plastic, meta‑adaptive systems that can actively tailor their internal processing to the specific characteristics of each incoming stimulus.

If successful, HMN-LM will contribute to bridging the gap between the high performance of state‑of‑the‑art NLP and the efficiency, adaptability, and lifelong learning capabilities of biologically grounded systems. This could open new avenues for developing more robust, personalized, interpretable, and energy-efficient language technologies that can continuously evolve and respond more nuancedly to the ever-changing linguistic environment. The proposed framework, while ambitious, offers a structured path for empirical investigation and iterative refinement.

## 11. Acknowledgements

We thank colleagues in computational neuroscience, machine learning, and natural language processing for insightful discussions that have contributed to the development of these ideas. Without your research a project of this magnitude would have not been possible. This research is supported by me Jeremy Shows and the grace of my wonderful and understanding partner Vanessa.

## 12. References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint* arXiv:1409.0473. [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

2. Bellec, G., Scherr, F., Subramoney, A., Legenstein, R., Maass, W., & Kappel, D. (2023). Meta‑learning biologically plausible plasticity rules with random feedback. *Nature Communications, 14*, Article 3756. [https://doi.org/10.1038/s41467-023-39386-w](https://doi.org/10.1038/s41467-023-39386-w)

3. Bengio, Y. (2014). Towards biologically plausible deep learning. *arXiv preprint* arXiv:1407.1148. [https://arxiv.org/abs/1407.1148](https://arxiv.org/abs/1407.1148)

4. Bengio, Y., Lee, D. H., Bornschein, J., & Lin, Z. (2015). Towards biologically plausible deep learning. *arXiv preprint* arXiv:1502.04156. [https://arxiv.org/abs/1502.04156](https://arxiv.org/abs/1502.04156)

5. Betteti, S., Baggio, G., Bullo, F., & Zampieri, S. (2025). Input-driven dynamics for robust memory retrieval in Hopfield networks. *Science Advances, 11*(17), eadu6991. [https://doi.org/10.1126/sciadv.adu6991](https://doi.org/10.1126/sciadv.adu6991)

6. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience, 18*(24), 10464–10472. [https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998)

7. Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science, 304*(5679), 1926–1929. [https://doi.org/10.1126/science.1099745](https://doi.org/10.1126/science.1099745)

8. Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical rewiring and information storage. *Nature, 431*(7010), 782–788. [https://doi.org/10.1038/nature03012](https://doi.org/10.1038/nature03012)

9. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint* arXiv:2312.00752. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

10. Doya, K. (2002). Metalearning and neuromodulation. *Neural Networks, 15*(4–6), 495–506. [https://doi.org/10.1016/S0893-6080(02)00044-8](https://doi.org/10.1016/S0893-6080%2802%2900044-8)

11. Faisal, A. A., Selen, L. P. J., & Wolpert, D. M. (2008). Noise in the nervous system. *Nature Reviews Neuroscience, 9*(4), 292–303. [https://doi.org/10.1038/nrn2258](https://doi.org/10.1038/nrn2258)

12. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research, 23*(120), 1–39. [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)

13. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)* (pp. 1126–1135). [https://proceedings.mlr.press/v70/finn17a.html](https://proceedings.mlr.press/v70/finn17a.html)

14. Hebb, D. O. (1949). *The organization of behavior: A neuropsychological theory*. Wiley.

15. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *Proceedings of the 36th International Conference on Machine Learning (ICML)* (pp. 2790–2799). [https://proceedings.mlr.press/v97/houlsby19a.html](https://proceedings.mlr.press/v97/houlsby19a.html)

16. Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex, 17*(10), 2443–2452. [https://doi.org/10.1093/cercor/bhl152](https://doi.org/10.1093/cercor/bhl152)

17. Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications, 7*, Article 13276. [https://doi.org/10.1038/ncomms13276](https://doi.org/10.1038/ncomms13276)

18. Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic action potentials and EPSPs. *Science, 275*(5297), 213–215. [https://doi.org/10.1126/science.275.5297.213](https://doi.org/10.1126/science.275.5297.213)

19. Moran, J., & Desimone, R. (1985). Selective attention gates visual processing in the extrastriate cortex. *Science, 229*(4715), 782–784. [https://doi.org/10.1126/science.4023713](https://doi.org/10.1126/science.4023713)

20. O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. *Hippocampus, 3*(3), 317–330. [https://doi.org/10.1002/hipo.450030307](https://doi.org/10.1002/hipo.450030307)

21. Poo, M. M., Pignatelli, M., Ryan, T. J., Tonegawa, S., Bonhoeffer, T., Martin, K. C., ... & Tsien, R. W. (2016). What is memory? The present state of the engram. *Biological Psychiatry, 80*(5), 344–352. [https://doi.org/10.1016/j.biopsych.2016.05.014](https://doi.org/10.1016/j.biopsych.2016.05.014)

22. Qiao, N., Meng, L., Corradi, F., Xiao, M., Liu, R., Lin, K. Y., ... & Indiveri, G. (2024). On‑chip meta‑plasticity for continual learning in neuromorphic hardware. *IEEE Transactions on Neural Networks and Learning Systems, 35*(1), 876–889. [https://doi.org/10.1109/TNNLS.2023.3280886](https://doi.org/10.1109/TNNLS.2023.3280886)

23. Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention. *Neuron, 61*(2), 168–185. [https://doi.org/10.1016/j.neuron.2009.01.002](https://doi.org/10.1016/j.neuron.2009.01.002)

24. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back‑propagating errors. *Nature, 323*(6088), 533–536. [https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)

25. Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience, 11*, Article 24. [https://doi.org/10.3389/fncom.2017.00024](https://doi.org/10.3389/fncom.2017.00024)

26. Schmidhuber, J. (1992). Learning to control fast‑weight memories. In *Advances in Neural Information Processing Systems (NIPS)* (Vol. 4, pp. 1–9). [https://proceedings.neurips.cc/paper_files/paper/1992/hash/abb63f88eb9e22561d340475f070b7f5-Abstract.html](https://proceedings.neurips.cc/paper_files/paper/1992/hash/abb63f88eb9e22561d340475f070b7f5-Abstract.html)

27. Schultz, W. (1998). Predictive reward signal of dopamine neurons. *Journal of Neurophysiology, 80*(1), 1–27. [https://doi.org/10.1152/jn.1998.80.1.1](https://doi.org/10.1152/jn.1998.80.1.1)

28. Sutton, R. S., & Barto, A. G. (1998). *Reinforcement learning: An introduction*. MIT Press. [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)

29. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 30, pp. 5998–6008). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

30. Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron, 46*(4), 681–692. [https://doi.org/10.1016/j.neuron.2005.04.026](https://doi.org/10.1016/j.neuron.2005.04.026)

31. Gu, A., Dao, T., Ermon, S., Ré, C., & Rudra, A. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *International Conference on Learning Representations (ICLR)*. [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396)  
*→ Replaces: Czempin, S., et al. (2024)*

32. Draelos, T. J., et al. (2023). Neural Replay and Continual Learning in Language Models. *arXiv preprint* arXiv:2301.07674. [https://arxiv.org/abs/2301.07674](https://arxiv.org/abs/2301.07674)  
*→ Replaces: Merkx, L., & Frank, S. L. (2023)*

33. Riquelme, C., Fedus, W., Zoph, B., et al. (2021). Scaling Vision with Sparse Mixture of Experts. *arXiv preprint* arXiv:2106.05974. [https://arxiv.org/abs/2106.05974](https://arxiv.org/abs/2106.05974)  
*→ Replaces: Roller, S., et al. (2022)*

34. Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *arXiv preprint* arXiv:1701.06538. [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)  
*→ Replaces: Rueckauer, B., et al. (2024)*

35. De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., & Tuytelaars, T. (2021). A Continual Learning Survey: Defying Forgetting in Classification Tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(7), 3366–3385. [https://arxiv.org/abs/1909.08383](https://arxiv.org/abs/1909.08383)  
*→ Replaces: Sun, X., et al. (2024)*

36. von Oswald, J., Henning, C., Sacramento, J., & Grewe, B. F. (2020). Continual learning with hypernetworks. *International Conference on Learning Representations (ICLR)*. [https://arxiv.org/abs/2006.06904](https://arxiv.org/abs/2006.06904)  
*→ Corrected from “von Oswald, J., et al. (2024)”*

---

## Appendix A: Algorithmic Details

### A.1 Training Loop (Simplified Python‑style Pseudocode)

```python
# Meta-parameters (meta_params) include:
# - eta_loc_base, eta_glob_base
# - Parameters of functions defining b_input(I_t), beta_a(I_t), 
#   eta_scale_loc(I_t), eta_scale_glob(I_t), beta_p(I_t), theta_p(I_t), s_k(I_t)
# - Other static gains like beta_g, etc.

# SPSA Hyperparameters
SPSA_EPS = 0.01 # Perturbation magnitude scaling
GLOBAL_META_LEARNING_RATE = 0.001 

# model: HMN_LM instance
# B_val_loader: DataLoader for validation sequences
# NUM_META_UPDATES: Total number of outer-loop steps
# T_inner: Number of tokens or sequences for inner-loop processing per meta-evaluation

for outer_step in range(NUM_META_UPDATES):
    # --- Meta-parameter perturbation for SPSA ---
    # 'delta' is a random perturbation vector with same shape as meta_params
    delta_perturbation = sample_bernoulli_sign_vector(meta_params.shape) * SPSA_EPS
    
    meta_losses = {} # To store losses for + and - perturbations

    for perturbation_sign in [+1, -1]:
        # Apply perturbed meta-parameters to the model for this evaluation run
        current_meta_params_for_eval = meta_params + perturbation_sign * delta_perturbation
        
        # model.configure_from_meta_params(...) sets the base learning rates
        # and configures the internal functions (e.g., small MLPs) that will 
        # compute dynamic parameters (b_input, beta_a, etc.) from I_t at each step.
        model.configure_from_meta_params(current_meta_params_for_eval)
        
        # It's important to reset or re-initialize plastic weights (w_ij) 
        # to a common starting state before each meta-evaluation, or evaluate 
        # adaptation from the current state if that's the goal.
        # For simplicity, let's assume evaluation of learning on B_val from scratch or a checkpoint.
        model.reset_plastic_weights() # Or load a base checkpoint

        accumulated_nll_for_eval = 0
        num_tokens_processed_for_eval = 0
        accumulated_stability_penalty_for_eval = 0
        
        # --- Inner loop: local HMN updates and NLL accumulation on validation data B_val ---
        for data_batch in B_val_loader: # B_val_loader yields sequences/batches
            model.reset_hidden_states() # Reset states for each new sequence
            
            for t in range(data_batch.sequence_length):
                current_input_token_xt = data_batch.tokens[t]
                target_token_xt_plus_1 = data_batch.tokens[t+1] # For LM NLL calculation
                
                # 1. Embed token
                current_embedding_ht = model.embed(current_input_token_xt)
                
                # 2. Derive input features I_t from h_t (and potentially model state)
                #    This I_t will be used by the HMN cells to compute their dynamic parameters.
                #    model.get_input_features might involve an MLP or pass-through.
                I_t_for_step = model.get_input_features(current_embedding_ht, model.current_hidden_state()) 
                                     
                # 3. HMN Forward Pass & Local Plasticity
                #    Internally, hmn_layer.forward will:
                #    a. Use I_t_for_step and meta-param-defined functions to calculate 
                #       b_input_val, beta_a_val, eta_loc_val, eta_glob_val, beta_p_val, theta_p_val.
                #    b. Perform activation (Eq 1) using b_input_val.
                #    c. Compute eligibility traces e_ij (Eq 2).
                #    d. Compute local attention alpha_ij (Eq 3) using beta_a_val.
                #    e. Compute global modulation G' (Eq 4).
                #    f. Compute pre-update delta_w_dagger (Eq 5) using eta_loc_val, eta_glob_val.
                #    g. Probabilistically update weights w_ij (Eq 6) using beta_p_val, theta_p_val.
                #    h. Return output logits for the current step.
                #    This function also accumulates the norm of delta_w for the stability penalty.
                output_logits_for_step, step_delta_w_norm_sq = model.hmn_process_step_and_learn(
                                                                   current_embedding_ht, 
                                                                   I_t_for_step, 
                                                                   target_token_xt_plus_1 # Optional: for online neuromodulation
                                                               )
                
                # 4. Calculate NLL for the current step
                step_nll = calculate_cross_entropy_loss(output_logits_for_step, target_token_xt_plus_1)
                accumulated_nll_for_eval += step_nll
                num_tokens_processed_for_eval += 1
                accumulated_stability_penalty_for_eval += step_delta_w_norm_sq
        
        # Calculate average NLL and total loss for this perturbed meta-parameter set
        average_nll = accumulated_nll_for_eval / num_tokens_processed_for_eval
        
        # Stability penalty (lambda_stab * E[|delta_w|^2])
        # LAMBDA_STAB and LAMBDA_MOD should be defined
        LAMBDA_STAB = 0.01 # Example value
        LAMBDA_MOD = 0.001 # Example value
        stability_penalty_val = LAMBDA_STAB * (accumulated_stability_penalty_for_eval / num_tokens_processed_for_eval) # Average per token
        
        # Regularization penalty on the parameters of the input-modulation functions
        # model.calculate_modulator_func_complexity_penalty() would compute sum of squares of
        # weights within the small MLPs defining b_input(I_t), beta_a(I_t), etc.
        modulator_func_penalty_val = LAMBDA_MOD * model.calculate_modulator_func_complexity_penalty(current_meta_params_for_eval)
        
        meta_losses[perturbation_sign] = average_nll + stability_penalty_val + modulator_func_penalty_val

    # --- SPSA gradient estimate for meta_params ---
    # Gradient estimate g_hat = (loss_plus - loss_minus) / (2 * SPSA_EPS * delta_perturbation_direction)
    # Ensure element-wise division if delta_perturbation is a vector.
    # Here, delta_perturbation already includes SPSA_EPS, so we use it directly if sample_bernoulli_sign_vector output is just +/-1
    # and SPSA_EPS is the scaling factor as applied: delta_perturbation = signs * SPSA_EPS
    # If delta_perturbation itself IS the full c_k * Delta_k, then division is by (2 * delta_perturbation)
    # The pseudocode `delta_perturbation = sample_bernoulli_sign_vector(meta_params.shape) * SPSA_EPS`
    # means delta_perturbation is c_k * Delta_k (where Delta_k are bernoulli signs, c_k is SPSA_EPS)
    # So the denominator should be 2 * delta_perturbation for element-wise gradient component.
    
    # To avoid division by zero if any component of delta_perturbation is zero (unlikely with SPSA_EPS > 0 and Bernoulli signs)
    # and to ensure correct element-wise operation:
    spsa_gradient_estimate_numerator = meta_losses[+1] - meta_losses[-1]
    spsa_gradient_estimate_denominator = 2 * delta_perturbation
    # Handle potential division by zero for components where delta_perturbation is zero, though SPSA_EPS should prevent this.
    # A practical way is to add a small epsilon or ensure delta_perturbation components are non-zero.
    # Given delta_perturbation = signs * SPSA_EPS, components are non-zero if SPSA_EPS is non-zero.
    spsa_gradient_estimate = spsa_gradient_estimate_numerator / spsa_gradient_estimate_denominator
    
    # --- Update meta_params ---
    meta_params -= GLOBAL_META_LEARNING_RATE * spsa_gradient_estimate
    
    # Clamp meta_params to predefined bounds if necessary
    # meta_params = clamp_meta_params_to_bounds(meta_params, meta_param_bounds)

    print(f"Outer step {outer_step}: MetaLoss(+)={meta_losses[+1]:.4f}, MetaLoss(-)={meta_losses[-1]:.4f}")

# Helper functions (conceptual)
# def sample_bernoulli_sign_vector(shape): ... # Returns vector of +/-1 with shape `shape`
# def calculate_cross_entropy_loss(logits, targets): ...
# def clamp_meta_params_to_bounds(params, bounds): ...
# class Model: # simplified
#     def configure_from_meta_params(self, params): ...
#     def reset_plastic_weights(self): ...
#     def embed(self, token): ...
#     def get_input_features(self, embedding, hidden_state): ... # Returns I_t
#     def hmn_process_step_and_learn(self, embedding, I_t, target_token_for_neuromod): ... # Returns logits, delta_w_norm_sq
#     def current_hidden_state(self): ...
#     def reset_hidden_states(self): ...
#     def calculate_modulator_func_complexity_penalty(self, meta_params): ... # Returns penalty based on meta_params defining the modulator functions
```

### A.2 Example Functional Forms (Conceptual)

* **Activation Function $f(x)$:** $\text{ReLU}(x) = \text{max}(0,x)$ or $\text{Sigmoid}(x) = 1/(1+e^{-x})$.
* **Eligibility Trace Update $\psi(x_i, z_j, t')$:** A common form is $\psi_{\text{trace}} \leftarrow \rho \psi_{\text{trace}} + (1-\rho) x_i z_j$, where $\rho$ is a decay factor related to $\tau$.
* **Similarity Function $g(a,b)$ or $h(a,b)$:** Scaled dot product $\frac{\langle a,b \rangle}{\sqrt{d}}$ or cosine similarity $\frac{\langle a,b \rangle}{|a||b|}$.
* **Input Modulation Functions (e.g., $\beta_a(I_t)$):**
  * Simple MLP: $W_2 \cdot \text{ReLU}(W_1 \cdot I_t + b_1) + b_2$. Parameters $W_1, b_1, W_2, b_2$ are meta-learned.
  * Parametric form: $c_1 \cdot \text{sigmoid}(c_2 \cdot \text{mean}(I_t) + c_3) + c_4$. Parameters $c_1, c_2, c_3, c_4$ are meta-learned.
* **Neuromodulator Aggregation $\mathcal{M}$:** Often a weighted sum, $G(t) = \sum_k w_k E_k(t)$, where $w_k$ could be fixed or part of the global attention mechanism.

### A.3 SPSA Gradient Approximation Details (Meta-Learning)

The SPSA algorithm approximates the gradient of the meta-loss $L_{\text{meta}}(\theta)$ with respect to the meta-parameters $\theta$ (denoted `meta_params` in pseudocode) using only two evaluations of $L_{\text{meta}}$.

1. Generate a random perturbation vector $\Delta_k$ at each iteration $k$, where each component is typically drawn from a Bernoulli distribution (e.g., $\pm 1$).
2. Evaluate the meta-loss at $\theta_k + c_k \Delta_k$ and $\theta_k - c_k \Delta_k$, where $c_k$ is a small positive scalar (SPSA\_EPS in pseudocode, possibly decaying over iterations, e.g., $c_k = c_0 / k^\gamma$).
3. The $i$-th component of the gradient estimate $\hat{g}_i(\theta_k)$ is:
    $$ \hat{g}_i(\theta_k) = \frac{L_{\text{meta}}(\theta_k + c_k \Delta_k) - L_{\text{meta}}(\theta_k - c_k \Delta_k)}{2 c_k \Delta_{ki}} $$
4. Update meta-parameters: $\theta_{k+1} = \theta_k - a_k \hat{g}(\theta_k)$, where $a_k$ is the meta-learning rate (GLOBAL\_META\_LEARNING\_RATE, possibly decaying, e.g., $a_k = a_0 / (A+k)^\alpha$).
    Careful tuning of $a_k$ and $c_k$ schedules is important for stability and convergence.

### A.4 Algorithm Pseudocode for HMN Weight Update (Conceptual single synapse, within `model.hmn_process_step_and_learn`)

```python
# For a single synapse (i,j) at time t during inner loop:
# Assume current_embedding_ht is h_t, and I_t_for_step is I_t.
# x_i is the pre-synaptic activity, z_j_prev is previous post-synaptic activity.

# These would be attributes or methods of the model/layer
# model.trace_decay_rates, model.beta_g_global, model.get_global_oscillator_phase()
# model.get_preferred_phase_for_synapse_ij(), etc.

# 1. Compute dynamic parameters using I_t and meta-learned functions
# These functions are configured by meta_params
b_input_val = model.compute_b_input_for_neuron_j(I_t_for_step) # e.g., MLP_b_input(I_t)
beta_a_val = model.compute_beta_a_for_synapse_ij(I_t_for_step) # e.g., MLP_beta_a(I_t)
eta_loc_val = model.compute_eta_loc_for_synapse_ij(I_t_for_step) # e.g., base_eta_loc * MLP_scale_loc(I_t)
eta_glob_val = model.compute_eta_glob_for_synapse_ij(I_t_for_step) # e.g., base_eta_glob * MLP_scale_glob(I_t)
beta_p_val = model.compute_beta_p_for_synapse_ij(I_t_for_step) # e.g., MLP_beta_p(I_t)
theta_p_val = model.compute_theta_p_for_synapse_ij(I_t_for_step) # e.g., MLP_theta_p(I_t)
# s_k_vals if input-scaled neuromodulators are used:
# s_k_vals = [model.compute_s_k_for_modulator_k(I_t_for_step, k) for k in range(num_modulators)]

# 2. Activation (Eq 1)
# sum_weighted_inputs = sum(w_kj_current * x_k_presynaptic for k connected to j) # x_k are pre-synaptic activations from previous layer or recurrent state
# z_j_current = activation_function(sum_weighted_inputs + bias_j_intrinsic + b_input_val + noise())

# 3. Eligibility Trace (Eq 2)
# (Assuming x_i is current pre-synaptic input corresponding to w_ij, 
#  and z_j_current is post-synaptic activation computed above)
# e_ij_fast = update_eligibility_trace_component(e_ij_fast_prev, x_i, z_j_current, model.trace_decay_rate_fast)
# e_ij_slow = update_eligibility_trace_component(e_ij_slow_prev, x_i, z_j_current, model.trace_decay_rate_slow)
# e_ij = e_ij_fast + e_ij_slow

# 4. Local Attention (Eq 3)
# h_i_embedding = get_embedding_for_presynaptic_input_or_state(x_i) # Could be h_t if x_i is from current layer input
# c_j_local_context = get_local_context_for_postsynaptic_neuron(z_j_current, z_j_history_vector)
# similarity_scores_for_neuron_j = []
# for pre_syn_input_k_to_j in all_inputs_to_j:
#     h_k_embedding = get_embedding_for_presynaptic_input_or_state(pre_syn_input_k_to_j)
#     similarity_scores_for_neuron_j.append(similarity_function(h_k_embedding, c_j_local_context))
#
# alpha_scores_raw = [beta_a_val * score for score in similarity_scores_for_neuron_j]
# alpha_values_for_neuron_j = softmax(alpha_scores_raw) # softmax over all inputs i to j
# alpha_ij = alpha_values_for_neuron_j[index_of_input_i] # Get the specific alpha for synapse ij
# attention_weighted_e_ij = alpha_ij * e_ij

# 5. Global Neuromodulation (Eq 4)
# E_k_signals_vector = model.get_neuromodulatory_signals(model_state, target_token_xt_plus_1_optional) # e.g. reward, surprise
# C_global_context_vector = model.get_global_context_vector(model_state)
# gamma_k_relevance_scores = []
# for E_k_signal in E_k_signals_vector:
#    gamma_k_relevance_scores.append(model.beta_g_global_gain * similarity_function_global(E_k_signal, C_global_context_vector))
#
# gamma_k_attention_weights = softmax(gamma_k_relevance_scores) # softmax over k neuromodulators
# G_prime = 0
# for k_idx, E_k_val in enumerate(E_k_signals_vector):
#     s_k_scaling = s_k_vals[k_idx] if s_k_vals else 1 # Assuming s_k_vals were computed if used
#     G_prime += gamma_k_attention_weights[k_idx] * s_k_scaling * E_k_val

# 6. Phase-Gated Pre-Update (Eq 5 - using alternative additive formulation for neuromodulation)
# current_global_phase_value = model.get_global_oscillator_phase() # e.g., sin(omega * t_global_clock)
# preferred_phase_for_synapse_ij = model.get_preferred_phase_for_synapse_ij() # Could be learned or fixed
# phase_gate_value = max(0, cos(current_global_phase_value - preferred_phase_for_synapse_ij))
#
# delta_w_star_ij = (eta_loc_val + eta_glob_val * G_prime) * attention_weighted_e_ij # Note: attention_weighted_e_ij is alpha_ij * e_ij
# delta_w_dagger_ij = delta_w_star_ij * phase_gate_value

# 7. Probabilistic Write (Eq 6)
# update_probability_logit = beta_p_val * (abs(delta_w_dagger_ij) - theta_p_val)
# update_probability = sigmoid(update_probability_logit)
#
# accumulated_delta_w_norm_sq_for_step = 0 # Initialize for the step if not passed in
# if random_uniform_0_1() < update_probability:
#     w_ij_current += delta_w_dagger_ij # Update the actual weight
#     accumulated_delta_w_norm_sq_for_step += delta_w_dagger_ij**2

# return z_j_current (or projected output from this neuron/layer), accumulated_delta_w_norm_sq_for_step
```

######################################################################

Trans–HMN: A Hybrid Transformer / Hybrid‑Modulated‑Neuron Language Model for Fast Generalisation and Continual Adaptation

Abstract

Transformers excel at zero‑shot generalisation yet adapt slowly to new domains and suffer catastrophic forgetting.
We outline Trans–HMN, a two‑stage language model in which a frozen Transformer encoder is augmented with Hybrid Local‑Global Modulated Neuron (HMN) adapter layers that learn via local plasticity, neuromodulatory feedback, and dual meta‑learning.
This paper is purely theoretical: we specify the architecture, derive the learning rule, and propose a benchmark suite for future empirical validation.

⸻

1 Introduction

Problem. Back‑prop‑only Transformers cannot update weights on‑device in real time.
Approach. Compose a frozen Transformer front‑end with plastic HMN adapters to enable continual learning.
Goal. Provide a rigorously defined framework; defer experimentation to later work.

⸻

2 Related Work

Area Key Works Relevance
Continual Transformers Sun 2024; von Oswald 2024 Require gradient updates
Neuromodulated Plasticity Miconi 2018; Bellec 2023 Demonstrate local rules
Adapter‑Tuning Houlsby 2019 Gradient‑based only

⸻

3 Architecture & Methods

graph LR
    A[Token IDs] --> B[Embedding] --> C[Frozen<br>6‑layer Transformer]
    C --> D[LayerNorm] --> E[4× HMN Adapter]
    E --> F[Softmax Head]

HMN Rule (condensed).

\Delta w^\dagger_{ij}
=(\eta_\ell+\eta_g G{\prime})\,
\alpha_{ij}e_{ij}\,
\max\!\bigl[0,\cos(\Phi-\phi_{ij})\bigr]

where
 • e_{ij}=\psi_{\text{fast}}+\psi_{\text{slow}}
 • \alpha_{ij}=\mathrm{softmax}(\beta_a\langle h_i,c_j\rangle)
 • G{\prime}=\gamma w_rE_{\text{reward}}+(1-\gamma)w_uE_{\text{uncert}}.

Meta‑parameters \eta_\ell,\eta_g,\beta_a,\beta_p are optimised with SPSA.

⸻

4 Proposed Evaluation Plan (Future Work)

Phase Dataset Metric Hypothesis
P0 WikiText‑2 PPL ↓ HMN adapters do not harm baseline perplexity
P1 Persona‑Chat (style‑shift) BLEU ↑ HMN enables few‑shot personalisation
P2 News → Code continual Forget ↓ Plasticity mitigates forgetting
P3 Energy profile J / token Sparsity keeps overhead < 10 %

All numbers are placeholders; no experiments have yet been run.

⸻

5 Discussion (Theoretical)
 • Division of labour – front‑end for generic syntax, back‑end for context‑dependent patterns.
 • Neuromodulator proxies – reward = -\!\log p_\theta; uncertainty = entropy; novelty = topic‑shift score.
 • Hardware prospects – sparse updates map naturally to memristive arrays.

⸻

6 Conclusion

Trans–HMN offers a principled path to continual‑learning LLMs; empirical validation is reserved for future work.

⸻

References

Bellec G. 2023 · Houlsby N. 2019 · Miconi T. 2018 · Sun X. 2024 · von Oswald J. 2024

######################################################################

MoE‑HMN: A Sparse Mixture of Hybrid‑Modulated Neuron Experts for Efficient Continual NLP

Abstract

We describe MoE‑HMN, a sparse Mixture‑of‑Experts language model in which each expert is a plastic HMN micro‑network.
This paper is a theoretical proposal: we detail routing, plasticity, and regularisers, and present a research plan to evaluate multilingual adaptation without inflating inference cost.

⸻

1 Introduction

Scaling parameters via MoE yields capacity but stores static weights.
Replacing dense experts with HMN experts could merge sparsity and plasticity—a combination not yet empirically tested.

⸻

2 Related Work

Topic Representative Limitation
Switch Transformer Fedus 2021 Static experts
Routing stability Roller 2022 No adaptive weights
Plastic MoE (small‑scale) Rueckauer 2024 Non‑NLP

⸻

3 Architecture & Methods

graph TD
    X[Token] --> R(Router)
    R --Top‑k--> E1[HMN Expert]
    R --Top‑k--> E2[HMN Expert]
    E1 --> Agg[Aggregate]
    E2 --> Agg

 • Router loss: load‑balance + plastic‑capacity regulariser
\mathcal{L}=\mathcal{L}{\text{task}}+\lambda{\text{lb}}\mathcal{L}{\text{LB}}+\lambda{\text{pc}}\mathcal{L}_{\text{PC}}.

⸻

4 Proposed Evaluation Plan (Future Work)

 1. Multilingual (FLORES‑200) – compare BLEU vs. Switch‑T.
 2. Noisy social media (MTNT) – measure robustness to spelling drift.
 3. Adaptation Latency – tokens needed for new language to hit 90 % of steady‑state accuracy.
 4. Routing FLOPs – theoretical vs. simulated count.

No results exist yet; these define the benchmark roadmap.

⸻

5 Discussion
 • Expected benefits – adaptive experts self‑organise by language family.
 • Open issues – router‑plasticity interactions might destabilise load‑balancing.

⸻

6 Conclusion

MoE‑HMN is a hypothesis for sparse, plastic mega‑models; empirical testing is left to future work.

⸻

References

Fedus W. 2021 · Roller S. 2022 · Bellec G. 2023

###########################################################################

Mamba‑HMN: Fusing Selective State‑Space Models with Neuromodulated Plasticity

Abstract

We propose Mamba‑HMN, combining linear‑time Selective State‑Space Models (SSM) with Hybrid‑Modulated Neuron layers that provide adaptive short‑term memory.
This manuscript is theoretical only: we formalise the coupling mechanism and specify a multi‑phase evaluation plan.

⸻

1 Introduction

SSMs scale to million‑token contexts but cannot personalise weights online.
HMN offers biological plasticity.
Our premise: hierarchical memory—SSM for slowly evolving context, HMN for rapid adaptation—yields a robust, scalable LLM.

⸻

2 Related Work

Area Reference Gap
Mamba SSM Dao 2023 Static parameters
SSM/Transformer hybrids Czempin 2024 Still gradient‑only
Three‑factor rules Miconi 2018 No global vector

⸻

3 Architecture & Methods

graph LR
    A[Token] --> B[SSM Stack]
    B --> G[Low‑Rank Summary G_t]
    G --> H[HMN Layer]
    H --> O[Output]

Global summary G_t acts as a vector‑valued neuromodulator.

⸻

4 Proposed Evaluation Plan (Future Work)

Phase Dataset Metric Purpose
P0 PG‑19 Perplexity Baseline compatibility
P1 Books → News continual Forget ↓ Catastrophic‑forgetting stress‑test
P2 Long‑context QA Accuracy ↑ Global‑context utility
P3 Energy simulation J/token Overhead of plastic writes

⸻

5 Discussion (Hypotheses)
 • Coupling benefit – G_t modulates HMN plasticity intensity relative to discourse coherence.
 • Stability conjecture – bounded‑energy lemma (Appendix A) should hold under reasonable \eta ranges.

⸻

6 Conclusion

Mamba‑HMN is a blueprint; validating its efficacy is designated as future empirical work.

⸻

References

Dao T. 2023 · Bellec G. 2023 · Miconi T. 2018

⸻

Appendix A (Sketch)

Derive a Frobenius‑norm bound on accumulated HMN weight change to ensure SSM stability.

⸻
