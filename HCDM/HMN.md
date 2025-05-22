# HMN – Hybrid Local–Global Modulated Neuron with Multi-Modal Attention

## A Theoretical Framework

-----

## Table of Contents

* [Abstract](https://www.google.com/search?q=%231-abstract)
* [Keywords](https://www.google.com/search?q=%23keywords)
* [1. Introduction](https://www.google.com/search?q=%232-introduction)
  * [2.1 Motivation and Problem Statement](https://www.google.com/search?q=%2321-motivation-and-problem-statement)
  * [2.2 Biological and Computational Inspirations](https://www.google.com/search?q=%2322-biological-and-computational-inspirations)
  * [2.3 Scope: Biologically Inspired versus Biologically Plausible](https://www.google.com/search?q=%2323-scope-biologically-inspired-versus-biologically-plausible)
  * [2.4 Core Contributions and Paper Organization](https://www.google.com/search?q=%2324-core-contributions-and-paper-organization)
* [3. Background and Related Work](https://www.google.com/search?q=%233-background-and-related-work)
  * [3.1 Local Synaptic Plasticity: Hebbian Rules, STDP, Probabilistic Dynamics](https://www.google.com/search?q=%2331-local-synaptic-plasticity-hebbian-rules-stdp-probabilistic-dynamics)
  * [3.2 Global Neuromodulation: Reinforcement, Attention, Meta-Learning](https://www.google.com/search?q=%2332-global-neuromodulation-reinforcement-attention-meta-learning)
  * [3.3 Attention Mechanisms in Neural and Deep Systems](https://www.google.com/search?q=%2333-attention-mechanisms-in-neural-and-deep-systems)
  * [3.4 Limits of Centralized Learning and Alternative Approaches](https://www.google.com/search?q=%2334-limits-of-centralized-learning-and-alternative-approaches)
* [4. Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)](https://www.google.com/search?q=%234-model-architecture-and-methods-the-hybrid-modulated-neuron-hmn)
  * [4.1 Theoretical Framework and Key Assumptions](https://www.google.com/search?q=%2341-theoretical-framework-and-key-assumptions)
  * [4.2 Local Processing Unit: Probabilistic Synaptic Dynamics](https://www.google.com/search?q=%2342-local-processing-unit-probabilistic-synaptic-dynamics)
    * [4.2.1 Neuronal Activation with Stochasticity](https://www.google.com/search?q=%23421-neuronal-activation-with-stochasticity)
    * [4.2.2 Composite Eligibility Traces (Multi-Timescale)](https://www.google.com/search?q=%23422-composite-eligibility-traces-multi-timescale)
    * [4.2.3 Preliminary Weight Update](https://www.google.com/search?q=%23423-preliminary-weight-update)
    * [4.2.4 Phase-Locked Gating](https://www.google.com/search?q=%23424-phase-locked-gating)
    * [4.2.5 Probabilistic Application of Updates](https://www.google.com/search?q=%23425-probabilistic-application-of-updates)
  * [4.3 Global Neuromodulatory Integration & Dual Meta-Learning](https://www.google.com/search?q=%2343-global-neuromodulatory-integration--dual-meta-learning)
    * [4.3.1 Aggregation and Attention on Neuromodulators](https://www.google.com/search?q=%23431-aggregation-and-attention-on-neuromodulators)
    * [4.3.2 Dual Meta-Learning of Learning Rates](https://www.google.com/search?q=%23432-dual-meta-learning-of-learning-rates)
  * [4.4 Multi-Modal Attention Mechanisms](https://www.google.com/search?q=%2344-multi-modal-attention-mechanisms)
    * [4.4.1 Local Attention on Eligibility Traces](https://www.google.com/search?q=%23441-local-attention-on-eligibility-traces)
  * [Figures](https://www.google.com/search?q=%23figures)
* [5. Hypothesized Capabilities and Applications](https://www.google.com/search?q=%235-hypothesized-capabilities-and-applications)
* [6. Discussion](https://www.google.com/search?q=%236-discussion)
  * [6.1 Synergistic Advantages of Integrated Dynamics](https://www.google.com/search?q=%2361-synergistic-advantages-of-integrated-dynamics)
  * [6.2 Comparison with Back-Propagation and Other Bio-Inspired Models](https://www.google.com/search?q=%2362-comparison-with-back-propagation-and-other-bio-inspired-models)
  * [6.3 Implications for Credit Assignment](https://www.google.com/search?q=%2363-implications-for-credit-assignment)
  * [6.4 Complexity, Hyper-Parameter Sensitivity, and Scalability](https://www.google.com/search?q=%2364-complexity-hyper-parameter-sensitivity-and-scalability)
  * [6.5 Plausibility Map of HMN Components](https://www.google.com/search?q=%2365-plausibility-map-of-hmn-components)
* [7. Conclusion and Future Work](https://www.google.com/search?q=%237-conclusion-and-future-work)
  * [7.1 Summary of Contributions and Impact](https://www.google.com/search?q=%2371-summary-of-contributions-and-impact)
  * [7.2 Empirical Road-Map and Benchmark Suite](https://www.google.com/search?q=%2372-empirical-road-map-and-benchmark-suite)
  * [7.3 Theoretical Analysis and Hardware Directions](https://www.google.com/search?q=%2373-theoretical-analysis-and-hardware-directions)
* [8. Acknowledgements](https://www.google.com/search?q=%238-acknowledgements)
* [9. References](https://www.google.com/search?q=%239-references)
* [10. Appendix](https://www.google.com/search?q=%2310-appendix)
  * [10.1 Example Functional Forms](https://www.google.com/search?q=%23101-example-functional-forms)
  * [10.2 Meta-Learning Gradient Approximation Details](https://www.google.com/search?q=%23102-meta-learning-gradient-approximation-details)
  * [10.3 SPSA Pseudocode for $\\eta$-Updates](https://www.google.com/search?q=%23103-spsa-pseudocode-for-eta-updates)
  * [10.4 Algorithm Pseudocode for HMN Weight Update](https://www.google.com/search?q=%23104-algorithm-pseudocode-for-hmn-weight-update)

-----

## 1\. Abstract

Biological neural systems exhibit remarkable adaptability by combining rapid, local synaptic plasticity with slower, context-dependent global neuromodulation, enabling lifelong learning in dynamic environments. Current artificial neural networks, often relying on centralized error back-propagation, lack this nuanced, decentralized adaptability. The Hybrid Modulated Neuron (HMN) is presented as a theoretical framework aiming to bridge this gap. It unifies probabilistic local plasticity rules (inspired by STDP and Hebbian learning), multi-factor neuromodulatory feedback, dual meta-learning for adapting local and global learning rates, dual attention mechanisms operating on both synaptic eligibility and neuromodulatory signals, and plasticity gating synchronized with neural oscillations. The framework provides concrete definitions for input representations and local context signals, utilizes a consistent notation for weight updates ($\\Delta w^\*, \\Delta w^\\dagger, \\Delta w$), and explicitly maps its components to biological plausibility versus computational inspiration. A staged empirical validation roadmap is proposed. We hypothesize that integrating these mechanisms synergistically equips HMN networks for enhanced continual learning, adaptive reinforcement learning, and robust credit assignment compared to conventional models.

-----

## Keywords

Synaptic Plasticity · Neuromodulation · Meta-Learning · Attention · Neural Oscillations · Probabilistic Updates · Continual Learning · Bio-Inspired AI · Neuromorphic Computing

-----

## 2\. Introduction

### 2.1 Motivation and Problem Statement

The success of deep learning largely stems from supervised training using error back-propagation. However, this paradigm faces challenges related to biological plausibility (e.g., weight transport problem, non-local error signals) and struggles with continual learning in non-stationary environments (catastrophic forgetting). Biological systems, in contrast, employ decentralized learning rules modulated by global context, achieving robust lifelong adaptation. The HMN framework seeks to develop a computationally effective neuronal model incorporating principles observed in biology, specifically aiming to integrate:

* Fast, local synaptic plasticity (e.g., Hebbian, STDP variants) for rapid adaptation based on correlated activity.
* Slower, global neuromodulation (e.g., mimicking dopamine for reward, acetylcholine for attention/uncertainty, norepinephrine for novelty) providing context-dependent guidance.
* Meta-learning capabilities to dynamically adjust the balance between local and global influences (learning rates) based on performance or environmental statistics.
* Attention mechanisms to selectively weigh inputs and neuromodulatory signals based on relevance.
* Oscillatory gating to potentially enhance temporal credit assignment and coordinate plasticity across neuronal assemblies.

### 2.2 Biological and Computational Inspirations

HMN draws inspiration from multiple domains:
(i) **Neurobiology**: well-documented Hebbian learning and Spike-Timing-Dependent Plasticity (STDP), the widespread influence of neuromodulators (dopamine, acetylcholine, etc.) on synaptic gain and plasticity, observed meta-plasticity (activity-dependent changes in plasticity itself), and the role of neural oscillations (e.g., theta, gamma) in coordinating neural activity and synaptic modification.
(ii) **Machine Learning**: attention mechanisms popularized by Transformer models, meta-learning algorithms for learning to learn (e.g., MAML, REINFORCE), and reinforcement learning principles for reward-based adaptation.

### 2.3 Scope: Biologically Inspired versus Biologically Plausible

We differentiate between components with direct empirical support for both their existence and functional role in biological learning (biologically plausible) and components introduced primarily for computational benefit while maintaining compatibility with biological constraints (biologically inspired). For example, dopamine-gated STDP is considered plausible, while the specific mathematical form of probabilistic gating used here (logistic function) is inspired. This distinction is summarized in the Plausibility Map (Discussion §6.5).

### 2.4 Core Contributions and Paper Organization

This paper introduces the HMN framework with the following core contributions:

* **Unified HMN Learning Rule**: A single theoretical rule integrating probabilistic local eligibility traces, globally modulated feedback, dual meta-learning of rates, dual attention mechanisms, and oscillatory gating.
* **Multi-Timescale Plasticity**: Achieved through composite eligibility traces and phase-locked updates potentially aligning plasticity with different behavioral or cognitive states.
* **Dual Meta-Learning**: A mechanism to jointly tune local ($\\eta\_{\\text{local}}$) and global ($\\eta\_{\\text{global}}$) learning rates online, adapting the learning dynamics to the environment.
* **Dual Attention**: Context-dependent modulation operating locally on synaptic eligibility traces and globally on incoming neuromodulatory signals.
* **Plausibility Mapping**: A systematic classification of HMN components based on biological evidence versus computational design.
* **Empirical Roadmap**: A proposed sequence of experiments for validating HMN capabilities.

The paper is structured as follows: Section 3 reviews related work. Section 4 details the HMN model architecture and mathematical formulation. Section 5 outlines hypothesized capabilities. Section 6 discusses advantages, comparisons, limitations, and plausibility. Section 7 concludes and proposes future directions. Appendices provide supplementary details.

-----

## 3\. Background and Related Work

### 3.1 Local Synaptic Plasticity: Hebbian Rules, STDP, Probabilistic Dynamics

Hebbian learning ("neurons that fire together, wire together") and its temporally precise extension, STDP, provide foundational mechanisms for associative learning based on local activity. Eligibility traces extend these ideas by creating a short-term memory of synaptic activity correlation, which can later be consolidated into a weight change by a third factor, such as a neuromodulator (forming "three-factor rules").

### 3.2 Global Neuromodulation: Reinforcement, Attention, Meta-Learning

Neuromodulatory systems (e.g., dopamine, acetylcholine, serotonin, norepinephrine) exert broad influence, altering neuronal firing properties and synaptic plasticity rules across large brain regions. They convey global state information related to reward prediction error, uncertainty, novelty, arousal, and attention, thereby shaping learning and behavior based on context and outcomes.

### 3.3 Attention Mechanisms in Neural and Deep Systems

Biological attention allows organisms to prioritize processing of relevant stimuli. Computationally, attention mechanisms (e.g., in Transformers) dynamically weight information based on context, enabling models to focus on salient features. HMN incorporates attention at both the synaptic (local) and neuromodulatory (global) levels.

### 3.4 Limits of Centralized Learning and Alternative Approaches

Back-propagation requires symmetric feedback weights (weight transport problem) and instantaneous global error propagation, deviating from known biological constraints. This has spurred research into more biologically plausible alternatives, including feedback alignment, equilibrium propagation, predictive coding, and various three-factor learning rules. HMN builds upon the lineage of three-factor rules while integrating additional mechanisms like meta-learning, advanced attention, and oscillatory gating.

-----

## 4\. Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)

### 4.1 Theoretical Framework and Key Assumptions

We consider a neuron $j$ receiving inputs $x\_i(t - \\tau\_{ij})$ from presynaptic neurons $i$ arriving with delays $\\tau\_{ij}$. The neuron computes an activation $z\_j(t)$. It also receives multiple global neuromodulatory signals $E\_k(t)$ (e.g., $E\_{\\text{reward}}, E\_{\\text{novel}}, E\_{\\text{uncertainty}}$) broadcast from external sources or specialized network populations, and a global context signal $C\_{\\text{global}}(t)$ representing the broader network or task state. A background oscillatory signal $\\Phi(t)$ (e.g., theta or gamma rhythm), potentially global or regional, provides a temporal reference frame.

**Key Assumptions:**

* Neurons operate stochastically.
* Synaptic plasticity depends on local activity (via eligibility traces) and global modulatory signals.
* Learning rates ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$) are adaptable via meta-learning.
* Attention mechanisms can modulate both local trace effectiveness and global signal integration.
* Plasticity can be gated by oscillatory phase.
* Synaptic delays $\\tau\_{ij}$ and preferred phases $\\phi\_{ij}$ are assumed fixed parameters for simplicity, though they could potentially be learned.
* The bias term $b\_j(t)$ is assumed to be adapted slowly via a separate homeostatic or simpler learning mechanism (not detailed here) to maintain neuronal responsiveness.

### 4.2 Local Processing Unit: Probabilistic Synaptic Dynamics

#### 4.2.1 Neuronal Activation with Stochasticity

The activation $z\_j(t)$ of neuron $j$ is computed via a non-linear activation function $f$ applied to the weighted sum of inputs, bias, and noise:
$$z_j(t) = f\left(\sum_i w_{ij}(t) x_i(t-\tau_{ij}) + b_j(t) + \epsilon_j(t)\right) \quad (1)$$
where $w\_{ij}(t)$ is the synaptic weight, $b\_j(t)$ is the bias term, and $\\epsilon\_j(t)$ represents neural noise (e.g., sampled from $\\mathcal{N}(0, \\sigma^2\_{\\text{noise}})$).

#### 4.2.2 Composite Eligibility Traces (Multi-Timescale)

Synaptic eligibility $e\_{ij}(t)$ captures the potential for plasticity based on recent correlated pre- and post-synaptic activity. It is composed of components with different time constants:
$$e_{ij}(t) = \psi_{\text{fast}}(t) + \psi_{\text{slow}}(t) \quad (2)$$
where $\\psi\_{\\text{fast}}$ and $\\psi\_{\\text{slow}}$ represent eligibility components decaying over short ($\\tau\_{\\text{fast}}$) and long ($\\tau\_{\\text{slow}}$) timescales, respectively. These are functions of pre-synaptic input $x\_i$ and post-synaptic activation $z\_j$ (and potentially their timing), implementing a form of Hebbian or STDP-like correlation detection. (See Appendix §10.1 for example functional forms).

#### 4.2.3 Preliminary Weight Update

Before gating and probabilistic application, a preliminary weight change $\\Delta w^{\*}*{ij}(t)$ is calculated. It combines a purely local term and a globally modulated term, both acting on the attended eligibility trace $\\tilde{e}*{ij}(t)$ (defined in §4.4.1):
$$\Delta w^{*}_{ij}(t) = \left(\eta_{\text{local}} + \eta_{\text{global}} G_{\text{eff}}(t)\right) \tilde{e}_{ij}(t) \quad (3)$$
Here, $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ are the meta-learned local and global learning rates (see §4.3.2). $G\_{\\text{eff}}(t)$ is the effective, attention-modulated global neuromodulatory signal (defined in §4.3.1). This formulation implies that the global signal $G\_{\\text{eff}}$ acts as a dynamic, context-dependent scaling factor for the influence of the global learning rate $\\eta\_{\\text{global}}$.

#### 4.2.4 Phase-Locked Gating

The preliminary update is then gated by the phase of the background oscillation $\\Phi(t)$ relative to a synapse-specific preferred phase $\\phi\_{ij}$:
$$\Delta w^{\dagger}_{ij}(t) = \Delta w^{*}_{ij}(t) \max\left(0, \cos(\Phi(t) - \phi_{ij})\right) \quad (4)$$
This mechanism restricts significant plasticity to specific oscillatory phases, potentially corresponding to optimal windows for information encoding or consolidation. The $\\max(0, \\cdot)$ ensures gating is multiplicative and non-negative. $\\phi\_{ij}$ is assumed to be a fixed parameter for each synapse.

#### 4.2.5 Probabilistic Application of Updates

Finally, the phase-gated update $\\Delta w^{\\dagger}*{ij}(t)$ is applied probabilistically, based on its magnitude relative to a threshold $\\theta\_p$:
$$ \\Delta w*{ij}(t) =
\\begin{cases}
\\Delta w^{\\dagger}*{ij}(t), & \\text{with probability } p*{\\text{update}} = \\sigma(\\beta\_p(|\\Delta w^{\\dagger}*{ij}(t)| - \\theta\_p)) \\
0, & \\text{otherwise}
\\end{cases} \\quad (5) $$
where $\\sigma(x) = 1 / (1 + e^{-x})$ is the logistic sigmoid function, $\\beta\_p$ controls the steepness of the probability transition, and $\\theta\_p$ is the magnitude threshold. This probabilistic step is biologically inspired by the stochastic nature of synaptic vesicle release and receptor dynamics, and computationally may introduce sparsity, improve robustness to noise, or aid exploration. The final weight is updated as $w*{ij}(t+1) = w\_{ij}(t) + \\Delta w\_{ij}(t)$.

### 4.3 Global Neuromodulatory Integration & Dual Meta-Learning

#### 4.3.1 Aggregation and Attention on Neuromodulators

Multiple neuromodulatory factors $E\_k(t)$ (e.g., reward, novelty, uncertainty signals) contribute to learning. Their baseline influence is determined by weights $w\_k$. We define the base contribution of each modulator as $M\_k(t) = w\_k E\_k(t)$. These contributions are then dynamically re-weighted by a global attention mechanism based on the current global context $C\_{\\text{global}}(t)$:
$$G_{\text{eff}}(t) = \sum_k \gamma_k(t) M_k(t) = \sum_k \gamma_k(t) w_k E_k(t) \quad (6)$$
The attention weights $\\gamma\_k(t)$ are computed similarly to local attention (§4.4.1), allowing the system to prioritize specific neuromodulatory signals based on the global state:
$$\gamma_k(t) = \frac{\exp(\beta_g h(E_k(t), C_{\text{global}}(t)))}{\sum_m \exp(\beta_g h(E_m(t), C_{\text{global}}(t)))} \quad (7)$$
where $h(\\cdot, \\cdot)$ is a similarity or relevance function (e.g., cosine similarity, see Appendix §10.1) and $\\beta\_g$ is an inverse temperature parameter controlling attention sharpness. The weights $w\_k$ determining the baseline importance of each $E\_k$ are assumed fixed or slowly adapted.

#### 4.3.2 Dual Meta-Learning of Learning Rates

The local ($\\eta\_{\\text{local}}$) and global ($\\eta\_{\\text{global}}$) learning rates are not fixed hyperparameters but are adapted online via meta-learning. This allows the HMN neuron to adjust its own learning dynamics based on experience. The updates follow a gradient descent rule on a meta-objective function $L\_{\\text{meta}}$:
$$\eta_{\text{local}} \leftarrow \eta_{\text{local}} - \alpha_{\text{meta},1} \nabla_{\eta_{\text{local}}} L_{\text{meta}} \quad (8)$$
$$\eta_{\text{global}} \leftarrow \eta_{\text{global}} - \alpha_{\text{meta},2} \nabla_{\eta_{\text{global}}} L_{\text{meta}} \quad (9)$$
where $\\alpha\_{\\text{meta},1}$ and $\\alpha\_{\\text{meta},2}$ are meta-learning rates. $L\_{\\text{meta}}$ represents a higher-level objective, such as maximizing long-term task reward, minimizing prediction error on a validation set, or achieving a balance between learning speed and stability. Since $L\_{\\text{meta}}$ might be non-differentiable or depend on long-term outcomes, its gradient is typically approximated using techniques like REINFORCE or Simultaneous Perturbation Stochastic Approximation (SPSA), as detailed in Appendix §10.2 and §10.3.

### 4.4 Multi-Modal Attention Mechanisms

HMN employs two distinct attention mechanisms:

#### 4.4.1 Local Attention on Eligibility Traces

This mechanism modulates the effective strength of each synapse's eligibility trace $e\_{ij}(t)$ based on the relevance of the presynaptic input $x\_i$ in the context of the postsynaptic neuron's recent activity $c\_j(t)$.
$$\tilde{e}_{ij}(t) = \alpha_{ij}(t) e_{ij}(t) \quad (10)$$
The attention weight $\\alpha\_{ij}(t)$ depends on the similarity between an embedding $h\_i(t)$ of the presynaptic input and a context representation $c\_j(t)$ derived from the postsynaptic neuron's state:
$$\alpha_{ij}(t) = \frac{\exp(\beta_a g(h_i(t), c_j(t)))}{\sum_l \exp(\beta_a g(h_l(t), c_j(t)))} \quad (11)$$

  * $h\_i(t)$: An embedding representing the features of input $x\_i(t - \\tau\_{ij})$. This could be generated by an upstream network layer, a fixed feature extractor, or potentially learned locally.
  * $c\_j(t)$: A representation of the postsynaptic neuron's local context, e.g., an exponentially weighted moving average of its recent activation $z\_j(t)$.
  * $g(\\cdot, \\cdot)$: A similarity function (e.g., dot product or cosine similarity, see Appendix §10.1).
  * $\\beta\_a$: An inverse temperature parameter controlling the focus of the attention.

This allows the neuron to dynamically prioritize plasticity contributions from inputs deemed most relevant to its current processing state.

(§4.4.2 Global Attention on Neuromodulatory Signals was integrated into §4.3.1 for better flow)

-----

### Figures

(Descriptions remain, assuming figures would be generated)

  * **Figure 1 (Conceptual Overview)**: Mermaid diagram illustrating the flow of information: inputs $x\_i$ contribute to activation $z\_j$ and eligibility traces $e\_{ij}$. Local attention ($\\alpha\_{ij}$) generates $\\tilde{e}*{ij}$. Global signals $E\_k$ are weighted ($\\gamma\_k$) to form $G*{\\text{eff}}$. $\\tilde{e}*{ij}$ and $G*{\\text{eff}}$ drive the preliminary update $\\Delta w^\*$, which is phase-gated ($\\Phi(t)$) to $\\Delta w^\\dagger$, and finally applied probabilistically to update $w\_{ij}$. Meta-learning adjusts $\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$.
  * **Figure 2 (Detailed Schematic)**: Layered block diagram showing signals ($x\_i, z\_j, e\_{ij}, E\_k, C\_{\\text{global}}, \\Phi(t)$) and transformations (activation, eligibility, attention $\\alpha\_{ij}$, attention $\\gamma\_k$, combination into $\\Delta w^\*$, phase gating $\\Delta w^\\dagger$, probabilistic application $\\Delta w\_{ij}$, meta-learning updates for $\\eta$).

-----

## 5\. Hypothesized Capabilities and Applications

The integrated mechanisms within HMN are hypothesized to confer several advantages:

  * **Adaptive Reinforcement Learning Agents**: The interplay between fast local plasticity (driven by $\\eta\_{\\text{local}}$) and slower, outcome-driven global modulation (driven by $\\eta\_{\\text{global}} G\_{\\text{eff}}$) could enable agents to balance exploration and exploitation effectively. Attention and meta-learning further refine this balance based on context and experience.
  * **Continual/Lifelong Learning Systems**: Dual meta-learning ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$) can potentially adjust the plasticity-stability balance dynamically, allowing networks to acquire new knowledge while mitigating catastrophic forgetting of previously learned tasks. Oscillatory gating might help segregate learning across different contexts or time scales.
  * **Complex Decision-Making Models**: The layered credit assignment mechanism—combining local correlations, attention-based filtering, temporal gating, and outcome-based modulation—provides a richer, more biologically grounded substrate for modeling complex reasoning and decision processes compared to simpler learning rules.
  * **Neuromorphic Hardware Implementations**: The focus on local computations, event-driven updates (potentially via probabilistic gating), and parallelism makes HMN potentially suitable for implementation on next-generation neuromorphic hardware (e.g., Intel Loihi, SpiNNaker, or memristive systems).

-----

## 6\. Discussion

### 6.1 Synergistic Advantages of Integrated Dynamics

The novelty of HMN lies not just in individual components but in their proposed synergy. Local attention ($\\alpha\_{ij}$) focuses plasticity on relevant inputs. Global attention ($\\gamma\_k$) prioritizes relevant neuromodulators. Phase-locking ($\\Phi(t)$) aligns updates temporally. Probabilistic application ($\\sigma(\\cdot)$) adds robustness and sparsity. Dual meta-learning ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$) adapts the core learning dynamics. Together, these mechanisms could create a system capable of rapid adaptation in familiar contexts, cautious exploration in novel situations, robust long-term retention, noise resilience, and context-aware credit assignment.

### 6.2 Comparison with Back-Propagation and Other Bio-Inspired Models

Compared to standard back-propagation, HMN offers a decentralized, temporally continuous learning process without requiring symmetric weights or explicit error derivatives. Relative to simpler three-factor rules (e.g., basic dopamine-modulated STDP), HMN adds: (i) probabilistic update application, (ii) dual meta-learning of learning rates, (iii) dual attention mechanisms (local/synaptic and global/modulatory), and (iv) explicit oscillatory phase gating. These additions aim to provide greater flexibility, adaptability, and context-sensitivity.

### 6.3 Implications for Credit Assignment

HMN employs a multi-faceted approach to credit assignment:

  * **Synaptic Tagging**: Eligibility traces $e\_{ij}$ mark synapses based on local causal activity.
  * **Attention Refinement**: Local attention $\\alpha\_{ij}$ refines credit based on input relevance; global attention $\\gamma\_k$ refines credit based on modulator relevance.
  * **Temporal Alignment**: Oscillatory gating $\\Delta w^{\\dagger}$ potentially aligns plasticity with relevant network states or behavioral epochs.
  * **Outcome Modulation**: Neuromodulatory signal $G\_{\\text{eff}}$ scales updates based on global outcomes (reward, novelty, etc.).
  * **Adaptive Regulation**: Meta-learning adjusts the overall learning sensitivity ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$).

### 6.4 Complexity, Hyper-Parameter Sensitivity, and Scalability

The HMN framework introduces several hyper-parameters beyond basic neural models, including decay constants ($\\tau\_{\\text{fast}}, \\tau\_{\\text{slow}}$), attention parameters ($\\beta\_a, \\beta\_g$), probabilistic gating parameters ($\\beta\_p, \\theta\_p$), oscillation parameters ($\\phi\_{ij}$ and the frequency of $\\Phi(t)$), meta-learning rates ($\\alpha\_{\\text{meta}}$), and modulator weights ($w\_k$). Tuning these could be challenging. However, the meta-learning of $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ aims to automate the tuning of two crucial parameters. Techniques like automatic relevance determination (ARD) could potentially be adapted to prune less relevant modulators or connections. Scalability to large networks depends on the computational cost of local updates and the communication overhead for global signals ($E\_k, C\_{\\text{global}}, \\Phi(t)$), which are assumed to be broadcast efficiently.

### 6.5 Plausibility Map of HMN Components

| Component                                           | Empirical Support Level                            | Classification           | Key Supporting Concepts/Citations                 |
| :-------------------------------------------------- | :------------------------------------------------- | :----------------------- | :------------------------------------------------ |
| STDP & Eligibility Traces                           | Strong                                             | Plausible                | Hebb 1949; Markram 1997; Bi & Poo 1998; Sutton & Barto 1998 |
| Neuromodulator-Gated Plasticity                     | Strong                                             | Plausible                | Schultz 1998; Izhikevich 2007; Yu & Dayan 2005    |
| Oscillatory Phase-Locked Updates                    | Growing                                            | Plausible                | Buzsáki & Draguhn 2004; Fries 2005; Lisman & Jensen 2013 |
| Attentional Modulation (General)                    | Strong                                             | Plausible                | Moran & Desimone 1985; Posner 1990               |
| Dual Attention (Synaptic/Modulatory)                | Indirect / Conceptual                            | Inspired                 | Extension of general attention principles         |
| Probabilistic Synaptic Updates                      | Indirect                                           | Inspired / Plausible     | Stochastic vesicle release, channel noise; Computationally motivated |
| Meta-Plasticity / Rate Adaptation                   | Emerging                                           | Plausible-in-Principle   | Doya 2002; Bellec et al. 2023; Abraham & Bear 1996 |
| Dual Meta-Learning ($\\eta\_{local}/\\eta\_{global}$)     | Conceptual                                         | Inspired                 | Computational mechanism based on meta-plasticity concepts |
| Specific Logistic Gate Form                         | None                                               | Inspired                 | Computational choice for probabilistic gating     |

*(Note: "Inspired" components leverage biological principles but their specific implementation here is primarily for computational function. "Plausible-in-Principle" suggests biological mechanisms exist that could implement such a function, even if the exact form isn't confirmed.)*

-----

## 7\. Conclusion and Future Work

### 7.1 Summary of Contributions and Impact

The HMN framework offers a novel, unified theoretical model of neuronal learning that integrates multiple biologically-inspired mechanisms: probabilistic local plasticity, multi-factor global neuromodulation, dual meta-learning of learning rates, dual attention (synaptic and modulatory), and oscillatory gating. By combining these elements, HMN aims to provide a more adaptive, robust, and biologically grounded alternative to conventional learning algorithms like back-propagation, potentially offering advantages in continual learning, reinforcement learning, and neuromorphic applications.

### 7.2 Empirical Road-Map and Benchmark Suite

Validating HMN requires systematic empirical evaluation. We propose a staged approach:

  * **Phase 0 (Sanity Check)**: Implement HMN in simple tasks (e.g., contextual bandits) to verify basic functionality and the viability of meta-learning $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ to optimize reward. Metrics: Reward accumulation, convergence of $\\eta$ values.
  * **Phase 1 (Continual Learning)**: Test HMN on sequential learning benchmarks (e.g., permuted MNIST, sequential CIFAR-10/100) against baseline models. Metrics: Accuracy on current task, forgetting of previous tasks, forward transfer. Insight: Assess stability-plasticity balance provided by meta-learning and other components.
  * **Phase 2 (Temporal Credit Assignment / RL)**: Evaluate HMN in reinforcement learning tasks requiring longer-term credit assignment (e.g., DeepMind Control Suite, Meta-World). Metrics: Sample efficiency, final performance, reward curve dynamics. Insight: Assess benefits of composite traces, phase-locking, and modulated global signals.
  * **Phase 3 (Ablation Studies)**: Systematically disable individual HMN components (e.g., attention mechanisms, probabilistic gating, phase-locking, meta-learning) across various tasks. Metrics: Performance difference ($\\Delta$-score) compared to full HMN. Insight: Quantify the synergistic contribution of each component.

### 7.3 Theoretical Analysis and Hardware Directions

Future theoretical work should focus on analyzing the learning dynamics and convergence properties of simplified HMN variants under specific assumptions. Understanding the theoretical interplay between local Hebbian forces, global modulatory guidance, and meta-learned rate adaptation is crucial. Furthermore, the decentralized and event-driven potential of HMN motivates prototyping on neuromorphic hardware platforms (e.g., spiking networks on Loihi 2, analog implementations using memristors) to explore potential efficiency gains and real-time learning capabilities (cf. Qiao et al. 2024).

-----

## 8\. Acknowledgements

We thank colleagues in computational neuroscience and machine learning for insightful discussions that helped shape this framework. [If applicable: Support from Funding Agency Grant XXX is gratefully acknowledged.]

-----

## 9\. References

1.  Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of synaptic plasticity. *Trends in Neurosciences, 19*(4), 126–130.
2.  Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint* arXiv:1409.0473.
3.  Bellec, G., Scherr, F., Subramoney, A., Legenstein, R., Maass, W., & Kappel, D. (2023). Meta‑learning biologically plausible plasticity rules with random feedback. *Nature Communications, 14*, Article 3756.
4.  Bengio, Y. (2014). Towards biologically plausible deep learning. *arXiv preprint* arXiv:1407.1148.
5.  Bengio, Y., Lee, D. H., Bornschein, J., & Lin, Z. (2015). Towards biologically plausible deep learning. *arXiv preprint* arXiv:1502.04156.
6.  Betteti, S., Baggio, G., Bullo, F., & Zampieri, S. (2025). Input-driven dynamics for robust memory retrieval in Hopfield networks. *Science Advances, 11*(17), eadu6991.
7.  Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience, 18*(24), 10464–10472.
8.  Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science, 304*(5679), 1926–1929.
9.  Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical rewiring and information storage. *Nature, 431*(7010), 782–788.
10. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint* arXiv:2312.00752.
11. De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., & Tuytelaars, T. (2021). A Continual Learning Survey: Defying Forgetting in Classification Tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(7), 3366–3385.
12. Doya, K. (2002). Metalearning and neuromodulation. *Neural Networks, 15*(4–6), 495–506.
13. Draelos, T. J., et al. (2023). Neural Replay and Continual Learning in Language Models. *arXiv preprint* arXiv:2301.07674.
14. Faisal, A. A., Selen, L. P. J., & Wolpert, D. M. (2008). Noise in the nervous system. *Nature Reviews Neuroscience, 9*(4), 292–303.
15. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research, 23*(120), 1–39.
16. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)* (pp. 1126–1135).
17. Fries, P. (2005). A mechanism for cognitive dynamics: neuronal communication through neuronal coherence. *Trends in Cognitive Sciences, 9*(10), 474-480.
18. Gu, A., Dao, T., Ermon, S., Ré, C., & Rudra, A. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *International Conference on Learning Representations (ICLR)*.
19. Hebb, D. O. (1949). *The organization of behavior: A neuropsychological theory*. Wiley.
20. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *Proceedings of the 36th International Conference on Machine Learning (ICML)* (pp. 2790–2799).
21. Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex, 17*(10), 2443–2452.
22. Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications, 7*, Article 13276.
23. Lisman, J. E., & Jensen, O. (2013). The theta-gamma neural code. *Neuron, 77*(6), 1002-1016.
24. Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic action potentials and EPSPs. *Science, 275*(5297), 213–215.
25. Moran, J., & Desimone, R. (1985). Selective attention gates visual processing in the extrastriate cortex. *Science, 229*(4715), 782–784.
26. O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. *Hippocampus, 3*(3), 317–330.
27. Poo, M. M., Pignatelli, M., Ryan, T. J., Tonegawa, S., Bonhoeffer, T., Martin, K. C., ... & Tsien, R. W. (2016). What is memory? The present state of the engram. *Biological Psychiatry, 80*(5), 344–352.
28. Posner, M. I. (1990). Hierarchical model of attentive operations. *Cognitive Psychology, 22*(2), 233-239. *(Note: A more canonical citation might be Posner, M. I. (1980). Orienting of attention. Quarterly journal of experimental psychology, 32(1), 3-25.)*
29. Qiao, N., Meng, L., Corradi, F., Xiao, M., Liu, R., Lin, K. Y., ... & Indiveri, G. (2024). On‑chip meta‑plasticity for continual learning in neuromorphic hardware. *IEEE Transactions on Neural Networks and Learning Systems, 35*(1), 876–889.
30. Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention. *Neuron, 61*(2), 168–185.
31. Riquelme, C., Fedus, W., Zoph, B., et al. (2021). Scaling Vision with Sparse Mixture of Experts. *arXiv preprint* arXiv:2106.05974.
32. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back‑propagating errors. *Nature, 323*(6088), 533–536.
33. Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience, 11*, Article 24.
34. Schmidhuber, J. (1992). Learning to control fast‑weight memories. In *Advances in Neural Information Processing Systems (NIPS)* (Vol. 4, pp. 1–9).
35. Schultz, W. (1998). Predictive reward signal of dopamine neurons. *Journal of Neurophysiology, 80*(1), 1–27.
36. Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *arXiv preprint* arXiv:1701.06538.
37. Sutton, R. S., & Barto, A. G. (1998). *Reinforcement learning: An introduction*. MIT Press.
38. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 30, pp. 5998–6008).
39. von Oswald, J., Henning, C., Sacramento, J., & Grewe, B. F. (2020). Continual learning with hypernetworks. *International Conference on Learning Representations (ICLR)*.
40. Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron, 46*(4), 681–692.

-----

## 10\. Appendix

### 10.1 Example Functional Forms

  * **Activation Function $f(x)$**: Rectified Linear Unit (ReLU) $f(x) = \\max(0, x)$ or Sigmoid $f(x) = \\sigma(x) = 1 / (1 + e^{-x})$.
  * **Eligibility Trace Components $\\psi$**: Example Hebbian trace: $\\psi\_{\\text{decay}}(t) = \\text{correlation}(x\_i(t-\\tau\_{ij}), z\_j(t)) \\times e^{-(t-t\_{\\text{event}})/\\tau\_{\\text{decay}}}$, where $\\text{correlation}(\\cdot)$ could be $x\_i z\_j$ or based on spike timing differences for STDP, and $t\_{\\text{event}}$ is the time of the relevant pre/post activity pairing. $\\psi\_{\\text{fast}}$ uses $\\tau\_{\\text{fast}}$, $\\psi\_{\\text{slow}}$ uses $\\tau\_{\\text{slow}}$.
  * **Neuromodulator Aggregation Baseline Weights $w\_k$**: These represent the default influence of each modulator $E\_k$. Could be fixed based on prior knowledge or slowly adapted.
  * **Similarity Functions $g(a, b)$ and $h(a, b)$**: Cosine similarity $g(a, b) = \\frac{a \\cdot b}{|a| |b|}$ or scaled dot product $g(a, b) = \\frac{a \\cdot b}{\\sqrt{d}}$, where $d$ is the dimension of embeddings $a, b$.
  * **Local Context $c\_j(t)$**: Exponential Moving Average (EMA) of activation: $c\_j(t) = (1 - \\delta) c\_j(t-1) + \\delta z\_j(t)$ for some small decay rate $\\delta$.
  * **Input Embedding $h\_i(t)$**: Could be an EMA of recent $x\_i(t-\\tau\_{ij})$, features extracted by a fixed function, or embeddings learned by an upstream layer/network.

### 10.2 Meta-Learning Gradient Approximation

For updating $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ (Eqs 8-9), where $L\_{\\text{meta}}$ might not be directly differentiable w.r.t. $\\eta$, stochastic approximation methods are needed.

  * **SPSA (Simultaneous Perturbation Stochastic Approximation)**: Efficient for high-dimensional parameter spaces (though here only 2D: $\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$). Perturbs all parameters simultaneously using a random direction vector. See §10.3.
  * **REINFORCE (Policy Gradient)**: Applicable if $L\_{\\text{meta}}$ can be framed as an expected reward in a stochastic system. Requires estimating the gradient $\\nabla\_{\\eta} \\log p(\\text{trajectory}; \\eta) \\times \\text{Reward}$.
  * **Finite Differences**: Perturb each $\\eta$ individually to estimate gradient, less efficient than SPSA for more parameters.

A common practice is to use decaying perturbation sizes or learning rates for stability, e.g., $\\epsilon\_t = \\epsilon\_0 / t^{0.5}$ or $\\alpha\_{\\text{meta},t} = \\alpha\_{\\text{meta},0} / t^{0.602}$.

### 10.3 SPSA Pseudocode for $\\eta$-Updates

```python
# Update eta = [eta_local, eta_global] using SPSA
# alpha_meta: meta-learning rate
# perturbation_scale (epsilon): controls size of perturbation

# Generate random perturbation vector (delta)
# Typically uses Rademacher distribution (random +1 or -1 for each component)
delta = perturbation_scale * bernoulli_plus_minus(dimension=2) # dim=2 for [eta_local, eta_global]

# Evaluate meta-loss at perturbed points
eta_plus = eta + delta
eta_minus = eta - delta
# Ensure rates stay within bounds [eta_min, eta_max] if needed before evaluation
L_plus = evaluate_meta_loss(eta_plus)
L_minus = evaluate_meta_loss(eta_minus)

# Estimate gradient component-wise
# Element-wise division: gradient_estimate_i = (L_plus - L_minus) / (2 * delta_i)
gradient_estimate = (L_plus - L_minus) / (2 * delta)

# Update eta using gradient descent
eta = eta - alpha_meta * gradient_estimate

# Clip eta to maintain stability / enforce bounds
eta = clip(eta, eta_min, eta_max)
```

### 10.4 Algorithm Pseudocode for HMN Weight Update (Single Neuron j)

```python
# --- Inputs at time t ---
# x_i(t-tau_ij): Delayed inputs from presynaptic neurons i
# E_k(t): Global neuromodulatory signals (e.g., reward, novelty)
# C_global(t): Global context signal
# Phi_t: Global/regional oscillation phase (Using Phi_t to avoid conflict with Phi function symbol)
# eta_local, eta_global: Current learning rates (from meta-learning)
# w_ij_current: Current synaptic weights (Using w_ij_current for clarity)
# b_j_current: Current bias (updated separately) (Using b_j_current for clarity)
# Parameters: beta_a, beta_g, beta_p, theta_p, w_k_modulator (neuromodulator baseline weights),
#             phi_ij_preferred (preferred phase per synapse), tau_fast, tau_slow

# --- Neuron Activation ---
summed_input = sum(w_ij_current[i] * x_i(t - tau_ij) for i in presynaptic_neurons) + b_j_current
noise = sample_gaussian_noise()
z_j = activation_function(summed_input + noise) # Eq (1)

# --- Update Synaptic Weights w_ij for all inputs i ---
for i in presynaptic_neurons:
    # 1. Calculate Eligibility Trace
    psi_f = calculate_psi_fast(x_i(t - tau_ij), z_j, t, tau_fast) # See Appendix 10.1
    psi_s = calculate_psi_slow(x_i(t - tau_ij), z_j, t, tau_slow) # See Appendix 10.1
    e_ij = psi_f + psi_s # Eq (2)

    # 2. Calculate Local Attention
    h_i_embedding = get_input_embedding(x_i(t - tau_ij)) # Assume function available (Appendix 10.1)
    c_j_context = update_local_context(z_j)             # Assume function available (Appendix 10.1)

    # Calculate similarities for all inputs l to neuron j
    similarities_g_all_inputs = []
    for l in presynaptic_neurons: # Iterate over all presynaptic inputs to neuron j
        h_l_embedding = get_input_embedding(x_l(t - tau_lj)) # x_l, tau_lj for input l
        similarities_g_all_inputs.append(similarity_g(h_l_embedding, c_j_context))

    # Softmax applied over all inputs to neuron j
    alpha_ij_all_inputs = softmax(beta_a * np.array(similarities_g_all_inputs)) # Assuming numpy for array ops
    alpha_ij = alpha_ij_all_inputs[presynaptic_neurons.index(i)] # Get alpha for specific input i
    e_tilde_ij = alpha_ij * e_ij # Eq (10)

    # 3. Calculate Effective Global Modulation
    M_k_contributions = [w_k_modulator[k_idx] * E_k[t](k_idx) for k_idx in range(len(E_k(t)))] # Iterate by index for E_k and w_k_modulator

    similarities_h_all_modulators = []
    for m_idx in range(len(E_k(t))): # Iterate by index for E_k
        similarities_h_all_modulators.append(similarity_h(E_k[t](m_idx), C_global(t)))

    gamma_k_weights = softmax(beta_g * np.array(similarities_h_all_modulators)) # Softmax over all modulators
    G_eff = sum(gamma_k_weights[k_idx] * M_k_contributions[k_idx] for k_idx in range(len(M_k_contributions))) # Eq (6)

    # 4. Calculate Preliminary Update
    delta_w_star = (eta_local + eta_global * G_eff) * e_tilde_ij # Eq (3)

    # 5. Apply Phase Gating
    phase_diff = Phi_t - phi_ij_preferred[i] # phi_ij_preferred assumed fixed per synapse i
    gate = max(0, cos(phase_diff))
    delta_w_dagger = delta_w_star * gate # Eq (4)

    # 6. Apply Probabilistic Update
    update_prob = sigmoid(beta_p * (abs(delta_w_dagger) - theta_p)) # Eq (5)
    delta_w_ij_final = 0 # Initialize final update for this synapse
    if random_uniform(0, 1) < update_prob:
        delta_w_ij_final = delta_w_dagger

    # Store update or apply immediately
    w_ij_current[i] = w_ij_current[i] + delta_w_ij_final # Update weight for synapse i

# --- Update Learning Rates (Periodically or Online) ---
# (Requires evaluating L_meta and using SPSA/REINFORCE)
# update_eta_via_meta_learning(eta_local, eta_global, L_meta) # Using Eq (8, 9) and Appendix 10.3

# --- Update Bias Term (Separate mechanism) ---
# update_bias(b_j_current) # Assumed separate process
```
