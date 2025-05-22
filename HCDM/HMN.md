Okay, I will integrate the detailed figure descriptions and image generation guidance into your paper, placing them in the most relevant sections. I will also update the Table of Contents accordingly.

Here is the updated paper:

# HMN – Hybrid Local–Global Modulated Neuron with Multi-Modal Attention

## A Theoretical Framework

-----

## Table of Contents

  * [Abstract](https://www.google.com/search?q=abstract)
  * [1. Introduction](https://www.google.com/search?q=introduction)
      * [2.1 Motivation and Problem Statement](https://www.google.com/search?q=motivation-and-problem-statement)
      * [2.2 Biological and Computational Inspirations](https://www.google.com/search?q=biological-and-computational-inspirations)
      * [2.3 Scope: Biologically Inspired versus Biologically Plausible](https://www.google.com/search?q=scope-biologically-inspired-versus-biologically-plausible)
      * [2.4 Core Contributions and Paper Organization](https://www.google.com/search?q=core-contributions-and-paper-organization)
  * [3. Background and Related Work](https://www.google.com/search?q=background-and-related-work)
      * [3.1 Local Synaptic Plasticity: Hebbian Rules, STDP, Probabilistic Dynamics](https://www.google.com/search?q=local-synaptic-plasticity-hebbian-rules-stdp-probabilistic-dynamics)
      * [3.2 Global Neuromodulation: Reinforcement, Attention, Meta-Learning](https://www.google.com/search?q=global-neuromodulation-reinforcement-attention-meta-learning)
      * [3.3 Attention Mechanisms in Neural and Deep Systems](https://www.google.com/search?q=attention-mechanisms-in-neural-and-deep-systems)
      * [3.4 Limits of Centralized Learning and Alternative Approaches](https://www.google.com/search?q=limits-of-centralized-learning-and-alternative-approaches)
  * [4. Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)](https://www.google.com/search?q=model-architecture-and-methods-the-hybrid-modulated-neuron-hmn)
      * [4.1 Theoretical Framework and Key Assumptions](https://www.google.com/search?q=theoretical-framework-and-key-assumptions)
      * [4.2 Local Processing Unit: Probabilistic Synaptic Dynamics](https://www.google.com/search?q=local-processing-unit-probabilistic-synaptic-dynamics)
          * [4.2.1 Neuronal Activation with Stochasticity](https://www.google.com/search?q=neuronal-activation-with-stochasticity)
          * [4.2.2 Composite Eligibility Traces (Multi-Timescale)](https://www.google.com/search?q=composite-eligibility-traces-multi-timescale)
          * [4.2.3 Preliminary Weight Update](https://www.google.com/search?q=preliminary-weight-update)
          * [4.2.4 Phase-Locked Gating](https://www.google.com/search?q=phase-locked-gating)
          * [4.2.5 Probabilistic Application of Updates](https://www.google.com/search?q=probabilistic-application-of-updates)
      * [4.3 Global Neuromodulatory Integration & Dual Meta-Learning](https://www.google.com/search?q=global-neuromodulatory-integration--dual-meta-learning)
          * [4.3.1 Aggregation and Attention on Neuromodulators](https://www.google.com/search?q=aggregation-and-attention-on-neuromodulators)
          * [4.3.2 Dual Meta-Learning of Learning Rates](https://www.google.com/search?q=dual-meta-learning-of-learning-rates)
      * [4.4 Multi-Modal Attention Mechanisms](https://www.google.com/search?q=multi-modal-attention-mechanisms)
          * [4.4.1 Local Attention on Eligibility Traces](https://www.google.com/search?q=local-attention-on-eligibility-traces)
      * [4.5 Model and Component Illustrations](https://www.google.com/search?q=%23model-and-component-illustrations)
          * [Figure 1: Conceptual Overview of HMN](https://www.google.com/search?q=figure-1-conceptual-overview-of-hmn)
          * [Figure 2: Detailed Schematic of HMN Operations](https://www.google.com/search?q=figure-2-detailed-schematic-of-hmn-operations)
          * [Figure 3: Multi-Timescale Eligibility Traces](https://www.google.com/search?q=figure-3-multi-timescale-eligibility-traces)
          * [Figure 4: Phase-Locked Gating Mechanism](https://www.google.com/search?q=figure-4-phase-locked-gating-mechanism)
          * [Figure 5: Probabilistic Application of Updates](https://www.google.com/search?q=figure-5-probabilistic-application-of-updates)
          * [Figure 6: Dual Attention Mechanisms in HMN](https://www.google.com/search?q=figure-6-dual-attention-mechanisms-in-hmn)
          * [Figure 7: Conceptual Dynamics of Dual Meta-Learning](https://www.google.com/search?q=figure-7-conceptual-dynamics-of-dual-meta-learning)
  * [5. Hypothesized Capabilities and Applications](https://www.google.com/search?q=hypothesized-capabilities-and-applications)
      * [Figure 9: HMN Components Enabling Hypothesized Capabilities (Conceptual)](https://www.google.com/search?q=figure-9-hmn-components-enabling-hypothesized-capabilities-conceptual)
  * [6. Discussion](https://www.google.com/search?q=discussion)
      * [6.1 Synergistic Advantages of Integrated Dynamics](https://www.google.com/search?q=synergistic-advantages-of-integrated-dynamics)
      * [6.2 Comparison with Back-Propagation and Other Bio-Inspired Models](https://www.google.com/search?q=comparison-with-back-propagation-and-other-bio-inspired-models)
          * [Figure 8: Comparison of HMN with Other Learning Paradigms](https://www.google.com/search?q=figure-8-comparison-of-hmn-with-other-learning-paradigms)
      * [6.3 Implications for Credit Assignment](https://www.google.com/search?q=implications-for-credit-assignment)
      * [6.4 Complexity, Hyper-Parameter Sensitivity, and Scalability](https://www.google.com/search?q=complexity-hyper-parameter-sensitivity-and-scalability)
      * [6.5 Plausibility Map of HMN Components](https://www.google.com/search?q=plausibility-map-of-hmn-components)
  * [7. Conclusion and Future Work](https://www.google.com/search?q=conclusion-and-future-work)
      * [7.1 Summary of Contributions and Impact](https://www.google.com/search?q=summary-of-contributions-and-impact)
      * [7.2 Empirical Road-Map and Benchmark Suite](https://www.google.com/search?q=empirical-road-map-and-benchmark-suite)
          * [Figure 14: Visual Roadmap for HMN Empirical Validation](https://www.google.com/search?q=figure-14-visual-roadmap-for-hmn-empirical-validation)
      * [7.3 Theoretical Analysis and Hardware Directions](https://www.google.com/search?q=theoretical-analysis-and-hardware-directions)
      * [7.4 Future Work: Integrating Astrocyte-Mediated Neuromodulation](https://www.google.com/search?q=future-work-integrating-astrocyte-mediated-neuromodulation)
          * [7.4.1 Modeling an Astrocyte-Intermediary Layer for Norepinephrine](https://www.google.com/search?q=modeling-an-astrocyte-intermediary-layer-for-norepinephrine)
              * [Figure 10: Astrocyte-Mediated Neuromodulation Pathway in HMN](https://www.google.com/search?q=figure-10-astrocyte-mediated-neuromodulation-pathway-in-hmn)
          * [7.4.2 Investigating Computational Consequences](https://www.google.com/search?q=investigating-computational-consequences)
          * [7.4.3 Empirical Validation and Plausibility](https://www.google.com/search?q=empirical-validation-and-plausibility)
      * [7.5 Future Work: Hybrid Architectures – Trans-HMN](https://www.google.com/search?q=future-work-hybrid-architectures--trans-hmn)
          * [Figure 11: Trans-HMN Hybrid Architecture](https://www.google.com/search?q=figure-11-trans-hmn-hybrid-architecture)
      * [7.6 Future Work: Sparse Activation Models – MoE-HMN](https://www.google.com/search?q=future-work-sparse-activation-models--moe-hmn)
          * [Figure 12: MoE-HMN Sparse Activation Architecture](https://www.google.com/search?q=figure-12-moe-hmn-sparse-activation-architecture)
      * [7.7 Future Work: Efficient Long-Context Models – Mamba-HMN](https://www.google.com/search?q=future-work-efficient-long-context-models--mamba-hmn)
          * [Figure 13: Mamba-HMN Architecture for Long-Context Processing](https://www.google.com/search?q=figure-13-mamba-hmn-architecture-for-long-context-processing)
  * [8. Acknowledgements](https://www.google.com/search?q=acknowledgements)
  * [9. References](https://www.google.com/search?q=references)
  * [10. Appendix](https://www.google.com/search?q=appendix)
      * [10.1 Example Functional Forms](https://www.google.com/search?q=example-functional-forms)
      * [10.2 Meta-Learning Gradient Approximation Details](https://www.google.com/search?q=meta-learning-gradient-approximation-details)
      * [10.3 SPSA Pseudocode for $\\eta$-Updates](https://www.google.com/search?q=spsa-pseudocode-for-eta-updates)
      * [10.4 Algorithm Pseudocode for HMN Weight Update](https://www.google.com/search?q=algorithm-pseudocode-for-hmn-weight-update)

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

The paper is structured as follows: Section 3 reviews related work. Section 4 details the HMN model architecture and mathematical formulation, including illustrative figures. Section 5 outlines hypothesized capabilities. Section 6 discusses advantages, comparisons, limitations, and plausibility. Section 7 concludes and proposes future directions. Appendices provide supplementary details.

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
where $\\psi\_{\\text{fast}}$ and $\\psi\_{\\text{slow}}$ represent eligibility components decaying over short ($\\tau\_{\\text{fast}}$) and long ($\\tau\_{\\text{slow}}$) timescales, respectively. These are functions of pre-synaptic input $x\_i$ and post-synaptic activation $z\_j$ (and potentially their timing), implementing a form of Hebbian or STDP-like correlation detection. (See Appendix §10.1 for example functional forms and Figure 3 for an illustration).

#### 4.2.3 Preliminary Weight Update

Before gating and probabilistic application, a preliminary weight change $\\Delta w^{\*}*{ij}(t)$ is calculated. It combines a purely local term and a globally modulated term, both acting on the attended eligibility trace $\\tilde{e}*{ij}(t)$ (defined in §4.4.1):
$$\Delta w^{*}_{ij}(t) = \left(\eta_{\text{local}} + \eta_{\text{global}} G_{\text{eff}}(t)\right) \tilde{e}_{ij}(t) \quad (3)$$
Here, $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ are the meta-learned local and global learning rates (see §4.3.2). $G\_{\\text{eff}}(t)$ is the effective, attention-modulated global neuromodulatory signal (defined in §4.3.1). This formulation implies that the global signal $G\_{\\text{eff}}$ acts as a dynamic, context-dependent scaling factor for the influence of the global learning rate $\\eta\_{\\text{global}}$.

#### 4.2.4 Phase-Locked Gating

The preliminary update is then gated by the phase of the background oscillation $\\Phi(t)$ relative to a synapse-specific preferred phase $\\phi\_{ij}$:
$$\Delta w^{\dagger}_{ij}(t) = \Delta w^{*}_{ij}(t) \max\left(0, \cos(\Phi(t) - \phi_{ij})\right) \quad (4)$$
This mechanism restricts significant plasticity to specific oscillatory phases, potentially corresponding to optimal windows for information encoding or consolidation. The $\\max(0, \\cdot)$ ensures gating is multiplicative and non-negative. $\\phi\_{ij}$ is assumed to be a fixed parameter for each synapse. (See Figure 4 for an illustration).

#### 4.2.5 Probabilistic Application of Updates

Finally, the phase-gated update $\\Delta w^{\\dagger}*{ij}(t)$ is applied probabilistically, based on its magnitude relative to a threshold $\\theta\_p$:
$$ \\Delta w*{ij}(t) =
\\begin{cases}
\\Delta w^{\\dagger}*{ij}(t), & \\text{with probability } p*{\\text{update}} = \\sigma(\\beta\_p(|\\Delta w^{\\dagger}*{ij}(t)| - \\theta\_p)) \\
0, & \\text{otherwise}
\\end{cases} \\quad (5) $$
where $\\sigma(x) = 1 / (1 + e^{-x})$ is the logistic sigmoid function, $\\beta\_p$ controls the steepness of the probability transition, and $\\theta\_p$ is the magnitude threshold. This probabilistic step is biologically inspired by the stochastic nature of synaptic vesicle release and receptor dynamics, and computationally may introduce sparsity, improve robustness to noise, or aid exploration. The final weight is updated as $w*{ij}(t+1) = w\_{ij}(t) + \\Delta w\_{ij}(t)$. (See Figure 5 for an illustration).

### 4.3 Global Neuromodulatory Integration & Dual Meta-Learning

#### 4.3.1 Aggregation and Attention on Neuromodulators

Multiple neuromodulatory factors $E\_k(t)$ (e.g., reward, novelty, uncertainty signals) contribute to learning. Their baseline influence is determined by weights $w\_k$. We define the base contribution of each modulator as $M\_k(t) = w\_k E\_k(t)$. These contributions are then dynamically re-weighted by a global attention mechanism based on the current global context $C\_{\\text{global}}(t)$:
$$G_{\text{eff}}(t) = \sum_k \gamma_k(t) M_k(t) = \sum_k \gamma_k(t) w_k E_k(t) \quad (6)$$The attention weights $\\gamma\_k(t)$ are computed similarly to local attention (§4.4.1), allowing the system to prioritize specific neuromodulatory signals based on the global state:$$\gamma_k(t) = \frac{\exp(\beta_g h(E_k(t), C_{\text{global}}(t)))}{\sum_m \exp(\beta_g h(E_m(t), C_{\text{global}}(t)))} \quad (7)$$
where $h(\\cdot, \\cdot)$ is a similarity or relevance function (e.g., cosine similarity, see Appendix §10.1) and $\\beta\_g$ is an inverse temperature parameter controlling attention sharpness. The weights $w\_k$ determining the baseline importance of each $E\_k$ are assumed fixed or slowly adapted. (The global attention component is also illustrated in Figure 6).

#### 4.3.2 Dual Meta-Learning of Learning Rates

The local ($\\eta\_{\\text{local}}$) and global ($\\eta\_{\\text{global}}$) learning rates are not fixed hyperparameters but are adapted online via meta-learning. This allows the HMN neuron to adjust its own learning dynamics based on experience. The updates follow a gradient descent rule on a meta-objective function $L\_{\\text{meta}}$:
$$\eta_{\text{local}} \leftarrow \eta_{\text{local}} - \alpha_{\text{meta},1} \nabla_{\eta_{\text{local}}} L_{\text{meta}} \quad (8)$$
$$\eta_{\text{global}} \leftarrow \eta_{\text{global}} - \alpha_{\text{meta},2} \nabla_{\eta_{\text{global}}} L_{\text{meta}} \quad (9)$$
where $\\alpha\_{\\text{meta},1}$ and $\\alpha\_{\\text{meta},2}$ are meta-learning rates. $L\_{\\text{meta}}$ represents a higher-level objective, such as maximizing long-term task reward, minimizing prediction error on a validation set, or achieving a balance between learning speed and stability. Since $L\_{\\text{meta}}$ might be non-differentiable or depend on long-term outcomes, its gradient is typically approximated using techniques like REINFORCE or Simultaneous Perturbation Stochastic Approximation (SPSA), as detailed in Appendix §10.2 and §10.3. (See Figure 7 for a conceptual illustration of these dynamics).

### 4.4 Multi-Modal Attention Mechanisms

HMN employs two distinct attention mechanisms, illustrated in Figure 6.

#### 4.4.1 Local Attention on Eligibility Traces

This mechanism modulates the effective strength of each synapse's eligibility trace $e\_{ij}(t)$ based on the relevance of the presynaptic input $x\_i$ in the context of the postsynaptic neuron's recent activity $c\_j(t)$.
$$\tilde{e}_{ij}(t) = \alpha_{ij}(t) e_{ij}(t) \quad (10)$$The attention weight $\\alpha\_{ij}(t)$ depends on the similarity between an embedding $h\_i(t)$ of the presynaptic input and a context representation $c\_j(t)$ derived from the postsynaptic neuron's state:$$\alpha_{ij}(t) = \frac{\exp(\beta_a g(h_i(t), c_j(t)))}{\sum_l \exp(\beta_a g(h_l(t), c_j(t)))} \quad (11)$$

  * $h\_i(t)$: An embedding representing the features of input $x\_i(t - \\tau\_{ij})$. This could be generated by an upstream network layer, a fixed feature extractor, or potentially learned locally.
  * $c\_j(t)$: A representation of the postsynaptic neuron's local context, e.g., an exponentially weighted moving average of its recent activation $z\_j(t)$.
  * $g(\\cdot, \\cdot)$: A similarity function (e.g., dot product or cosine similarity, see Appendix §10.1).
  * $\\beta\_a$: An inverse temperature parameter controlling the focus of the attention.

This allows the neuron to dynamically prioritize plasticity contributions from inputs deemed most relevant to its current processing state.

(§4.4.2 Global Attention on Neuromodulatory Signals was integrated into §4.3.1 for better flow and is also depicted in Figure 6).

-----

### 4.5 Model and Component Illustrations

This section provides descriptions for figures that illustrate the core HMN architecture and its key components.

#### Figure 1: Conceptual Overview of HMN

  * **Content**: A Mermaid diagram illustrating the flow of information: inputs $x\_i$ contribute to activation $z\_j$ and eligibility traces $e\_{ij}$. Local attention ($\\alpha\_{ij}$) generates $\\tilde{e}*{ij}$. Global signals $E\_k$ are weighted ($\\gamma\_k$) to form $G*{\\text{eff}}$. $\\tilde{e}*{ij}$ and $G*{\\text{eff}}$ drive the preliminary update $\\Delta w^\*$, which is phase-gated ($\\Phi(t)$) to $\\Delta w^\\dagger$, and finally applied probabilistically to update $w\_{ij}$. Meta-learning adjusts $\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$.
  * **Benefit**: Provides a high-level, easily digestible summary of the entire HMN processing pipeline within a single neuron.
  * **Creation Guidance**: Can be generated using Mermaid syntax or a similar diagramming tool (e.g., Lucidchart, draw.io) focusing on clear flow and component labeling. AI-assisted diagram generation could also be used with a detailed prompt.

#### Figure 2: Detailed Schematic of HMN Operations

  * **Content**: A layered block diagram showing signals ($x\_i, z\_j, e\_{ij}, E\_k, C\_{\\text{global}}, \\Phi(t)$) and transformations (activation, eligibility calculation, local attention $\\alpha\_{ij}$, global neuromodulator attention $\\gamma\_k$, combination into $\\Delta w^\*$, phase gating to $\\Delta w^\\dagger$, probabilistic application to $\\Delta w\_{ij}$, and meta-learning updates for $\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$). Each block would represent a mathematical operation or module described in Section 4.
  * **Benefit**: Offers a more granular view than Figure 1, mapping specific equations and sub-mechanisms to visual blocks, aiding in understanding the detailed interactions.
  * **Creation Guidance**: Best created with vector graphics software (e.g., Inkscape, Adobe Illustrator) or specialized diagramming tools that allow precise control over layout and connections. AI generation with a very structured prompt describing each block and connection is also feasible.

#### Figure 3: Multi-Timescale Eligibility Traces

  * **Content**: A plot showing example time courses of $\\psi\_{\\text{fast}}(t)$, $\\psi\_{\\text{slow}}(t)$, and the composite $e\_{ij}(t)$ in response to a hypothetical pre-synaptic input spike (or brief activity burst) and a subsequent or coincident post-synaptic activation. X-axis represents time; Y-axis represents eligibility strength. Different colors and line styles should distinguish $\\psi\_{\\text{fast}}$, $\\psi\_{\\text{slow}}$, and $e\_{ij}$. Key parameters like $\\tau\_{\\text{fast}}$ and $\\tau\_{\\text{slow}}$ could be indicated.
  * **Benefit**: Makes the "multi-timescale" aspect tangible by visually demonstrating the different decay rates and their additive combination, clarifying how short-term and longer-term correlations are captured.
  * **Creation Guidance**: Can be generated using graphing software (e.g., Matplotlib in Python, Gnuplot, MATLAB) by plotting example exponential decay functions.

#### Figure 4: Phase-Locked Gating Mechanism

  * **Content**: A diagram composed of two subplots sharing a time axis.
      * Top subplot: A sinusoidal wave representing the background oscillation $\\Phi(t)$. A vertical line or shaded region indicating a synapse-specific preferred phase $\\phi\_{ij}$.
      * Bottom subplot: The resulting gating function $g(t) = \\max(0, \\cos(\\Phi(t) - \\phi\_{ij}))$. This will show periods where the gate is open (positive values) and closed (zero values), synchronized with the oscillation.
      * Optionally, an example $\\Delta w^{\*}*{ij}(t)$ could be shown, and then the resulting $\\Delta w^{\\dagger}*{ij}(t)$ after gating.
  * **Benefit**: Clearly visualizes how neural oscillations are proposed to temporally gate plasticity, restricting weight updates to specific phases of the oscillation.
  * **Creation Guidance**: Can be generated using graphing software (e.g., Matplotlib, MATLAB) by plotting cosine functions and the max operation.

#### Figure 5: Probabilistic Application of Updates

  * **Content**: A graph with the magnitude of the phase-gated update $|\\Delta w^{\\dagger}*{ij}(t)|$ on the X-axis and the probability of update $p*{\\text{update}}$ on the Y-axis. The plot will show the logistic sigmoid curve $\\sigma(\\beta\_p(|\\Delta w^{\\dagger}\_{ij}(t)| - \\theta\_p))$. The threshold $\\theta\_p$ should be clearly marked on the X-axis, showing where the probability starts to rise significantly. The effect of different $\\beta\_p$ values (steepness) could be shown with multiple curves.
  * **Benefit**: Visually explains the stochastic nature of the final weight update, illustrating how update likelihood depends on the strength of the proposed change relative to a threshold.
  * **Creation Guidance**: Can be generated using graphing software by plotting the sigmoid function with varying parameters.

#### Figure 6: Dual Attention Mechanisms in HMN

  * **Content**: Two small, parallel, or connected conceptual diagrams.
      * **Local Attention (Left/Top)**: Shows presynaptic inputs $x\_i$ (or their embeddings $h\_i(t)$) and the postsynaptic context $c\_j(t)$ feeding into an "Attention Module $\\alpha$". This module outputs attention weights $\\alpha\_{ij}(t)$ that multiply the raw eligibility traces $e\_{ij}(t)$ to produce attended traces $\\tilde{e}\_{ij}(t)$.
      * **Global Attention (Right/Bottom)**: Shows various neuromodulatory signals $E\_k(t)$ and the global context $C\_{\\text{global}}(t)$ feeding into an "Attention Module $\\gamma$". This module outputs attention weights $\\gamma\_k(t)$ that multiply the baseline modulator contributions $M\_k(t)$ (or $w\_k E\_k(t)$) to produce the effective global neuromodulatory signal $G\_{\\text{eff}}(t)$.
  * **Benefit**: Clearly differentiates and visually explains the operation and inputs/outputs of the two distinct attention mechanisms operating at different levels of the HMN.
  * **Creation Guidance**: Best created with diagramming software (e.g., draw.io, Lucidchart) or vector graphics tools to create clear, iconographic representations of signals and modules. AI-assisted diagram generation could also be prompted.

#### Figure 7: Conceptual Dynamics of Dual Meta-Learning

  * **Content**: A conceptual plot, possibly 2D or 3D.
      * Axes could represent $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$. A third dimension or contour lines could represent the meta-objective $L\_{\\text{meta}}$ (e.g., performance).
      * Arrows or a trajectory could illustrate how $(\\eta\_{\\text{local}}, \\eta\_{\\text{global}})$ might evolve over time (e.g., across different task phases like initial learning, adaptation to a new task, stabilization) as they are updated to optimize $L\_{\\text{meta}}$.
      * Alternatively, two separate plots showing $\\eta\_{\\text{local}}(t)$ and $\\eta\_{\\text{global}}(t)$ over time during a hypothetical learning scenario.
  * **Benefit**: Helps visualize the adaptive nature of the learning rates, showing how they are not static but dynamically tuned by the meta-learning process to suit different learning contexts or objectives.
  * **Creation Guidance**: For a trajectory plot, this could be a conceptual drawing made with vector graphics software. For time-series plots, graphing software could simulate example evolutions.

-----

## 5\. Hypothesized Capabilities and Applications

The integrated mechanisms within HMN are hypothesized to confer several advantages, potentially enabled by specific combinations of its components (as conceptualized in Figure 9).

  * **Adaptive Reinforcement Learning Agents**: The interplay between fast local plasticity (driven by $\\eta\_{\\text{local}}$) and slower, outcome-driven global modulation (driven by $\\eta\_{\\text{global}} G\_{\\text{eff}}$) could enable agents to balance exploration and exploitation effectively. Attention and meta-learning further refine this balance based on context and experience.
  * **Continual/Lifelong Learning Systems**: Dual meta-learning ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$) can potentially adjust the plasticity-stability balance dynamically, allowing networks to acquire new knowledge while mitigating catastrophic forgetting of previously learned tasks. Oscillatory gating might help segregate learning across different contexts or time scales.
  * **Complex Decision-Making Models**: The layered credit assignment mechanism—combining local correlations, attention-based filtering, temporal gating, and outcome-based modulation—provides a richer, more biologically grounded substrate for modeling complex reasoning and decision processes compared to simpler learning rules.
  * **Neuromorphic Hardware Implementations**: The focus on local computations, event-driven updates (potentially via probabilistic gating), and parallelism makes HMN potentially suitable for implementation on next-generation neuromorphic hardware (e.g., Intel Loihi, SpiNNaker, or memristive systems).

#### Figure 9: HMN Components Enabling Hypothesized Capabilities (Conceptual)

  * **Content**: A set of four mini-diagrams or a table. Each mini-diagram (or row) corresponds to one hypothesized capability (Adaptive RL, Continual Learning, Complex Decision-Making, Neuromorphic Suitability). Within each, icons or short text labels would highlight the primary HMN components theorized to enable that capability. For example:
      * Adaptive RL: $\\eta\_{\\text{local}}$, $\\eta\_{\\text{global}} G\_{\\text{eff}}$, Dual Meta-Learning, Attention.
      * Continual Learning: Dual Meta-Learning ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$ adapt balance), Oscillatory Gating (context segregation), Composite Traces.
      * Complex Decision-Making: Full stack – $e\_{ij}$, $\\alpha\_{ij}$, $\\gamma\_k$, $\\Phi(t)$ gating, $G\_{\\text{eff}}$.
      * Neuromorphic Suitability: Local computations ($e\_{ij}$), Probabilistic gating (event-driven), Parallelism.
  * **Benefit**: Makes the link between HMN's specific mechanisms and its potential broad applications clearer and more direct.
  * **Creation Guidance**: Can be created using presentation software (e.g., PowerPoint, Google Slides) with icons and text, or a simple table in a word processor. AI could generate icons based on component names.

-----

## 6\. Discussion

### 6.1 Synergistic Advantages of Integrated Dynamics

The novelty of HMN lies not just in individual components but in their proposed synergy. Local attention ($\\alpha\_{ij}$) focuses plasticity on relevant inputs. Global attention ($\\gamma\_k$) prioritizes relevant neuromodulators. Phase-locking ($\\Phi(t)$) aligns updates temporally. Probabilistic application ($\\sigma(\\cdot)$) adds robustness and sparsity. Dual meta-learning ($\\eta\_{\\text{local}}, \\eta\_{\\text{global}}$) adapts the core learning dynamics. Together, these mechanisms could create a system capable of rapid adaptation in familiar contexts, cautious exploration in novel situations, robust long-term retention, noise resilience, and context-aware credit assignment.

### 6.2 Comparison with Back-Propagation and Other Bio-Inspired Models

Compared to standard back-propagation, HMN offers a decentralized, temporally continuous learning process without requiring symmetric weights or explicit error derivatives. Relative to simpler three-factor rules (e.g., basic dopamine-modulated STDP), HMN adds: (i) probabilistic update application, (ii) dual meta-learning of learning rates, (iii) dual attention mechanisms (local/synaptic and global/modulatory), and (iv) explicit oscillatory phase gating. These additions aim to provide greater flexibility, adaptability, and context-sensitivity. Figure 8 provides a visual comparison.

#### Figure 8: Comparison of HMN with Other Learning Paradigms

  * **Content**: A comparative block diagram with three panels:
      * **(a) Back-propagation**: Shows layers, forward pass, global error calculation, and backward error propagation requiring symmetric weights (indicated visually). Highlights centralized error.
      * **(b) Basic Three-Factor Rule**: Shows local pre/post activity forming an eligibility trace, and a single global neuromodulatory signal gating the update. Simpler than HMN.
      * **(c) HMN**: Shows local activity, eligibility traces, local attention ($\\alpha\_{ij}$), multiple global neuromodulators with global attention ($\\gamma\_k$) forming $G\_{\\text{eff}}$, meta-learned $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$, phase-gating ($\\Phi(t)$), and probabilistic updates. Highlights decentralized nature and additional modulatory mechanisms.
  * **Benefit**: Provides a strong visual argument for HMN's unique position by directly contrasting its information flow and key components against established and simpler bio-inspired learning rules.
  * **Creation Guidance**: Can be created using diagramming software (e.g., Lucidchart, draw.io) or vector graphics tools. Consistency in representing signals and processing units across panels is key. AI-assisted diagram generation could be used for each panel based on detailed descriptions.

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

| Component                                     | Empirical Support Level                     | Classification           | Key Supporting Concepts/Citations                      |
| :-------------------------------------------------- | :------------------------------------------------- | :----------------------- | :------------------------------------------------ |
| STDP & Eligibility Traces                         | Strong                                             | Plausible                | Hebb 1949; Markram 1997; Bi & Poo 1998; Sutton & Barto 1998 |
| Neuromodulator-Gated Plasticity                     | Strong                                             | Plausible                | Schultz 1998; Izhikevich 2007; Yu & Dayan 2005      |
| Oscillatory Phase-Locked Updates                  | Growing                                            | Plausible                | Buzsáki & Draguhn 2004; Fries 2005; Lisman & Jensen 2013 |
| Attentional Modulation (General)                    | Strong                                             | Plausible                | Moran & Desimone 1985; Posner 1990                  |
| Dual Attention (Synaptic/Modulatory)                | Indirect / Conceptual                              | Inspired                 | Extension of general attention principles           |
| Probabilistic Synaptic Updates                      | Indirect                                           | Inspired / Plausible     | Stochastic vesicle release, channel noise; Computationally motivated |
| Meta-Plasticity / Rate Adaptation                   | Emerging                                           | Plausible-in-Principle   | Doya 2002; Bellec et al. 2023; Abraham & Bear 1996 |
| Dual Meta-Learning ($\\eta\_{local}/\\eta\_{global}$) | Conceptual                                         | Inspired                 | Computational mechanism based on meta-plasticity concepts |
| Specific Logistic Gate Form                         | None                                               | Inspired                 | Computational choice for probabilistic gating       |

*(Note: "Inspired" components leverage biological principles but their specific implementation here is primarily for computational function. "Plausible-in-Principle" suggests biological mechanisms exist that could implement such a function, even if the exact form isn't confirmed.)*

-----

## 7\. Conclusion and Future Work

### 7.1 Summary of Contributions and Impact

The HMN framework offers a novel, unified theoretical model of neuronal learning that integrates multiple biologically-inspired mechanisms: probabilistic local plasticity, multi-factor global neuromodulation, dual meta‑learning of learning rates, dual attention (synaptic and modulatory), and oscillatory gating. By combining these elements, HMN aims to provide a more adaptive, robust, and biologically grounded alternative to conventional learning algorithms like back-propagation, potentially offering advantages in continual learning, reinforcement learning, and neuromorphic applications.

-----

### 7.2 Empirical Road-Map and Benchmark Suite

Validating HMN requires systematic empirical evaluation. We propose a staged approach (visualized in Figure 14):

  * **Phase 0 (Sanity Check)**: Implement HMN in simple tasks (e.g., contextual bandits) to verify basic functionality and the viability of meta-learning $\\eta\_{\\text{local}}$ and $\\eta\_{\\text{global}}$ to optimize reward. Metrics: Reward accumulation, convergence of $\\eta$ values.
  * **Phase 1 (Continual Learning)**: Test HMN on sequential learning benchmarks (e.g., permuted MNIST, sequential CIFAR-10/100) against baseline models. Metrics: Accuracy on current task, forgetting of previous tasks, forward transfer. Insight: Assess stability-plasticity balance provided by meta-learning and other components.
  * **Phase 2 (Temporal Credit Assignment / RL)**: Evaluate HMN in reinforcement learning tasks requiring longer-term credit assignment (e.g., DeepMind Control Suite, Meta-World). Metrics: Sample efficiency, final performance, reward curve dynamics. Insight: Assess benefits of composite traces, phase-locking, and modulated global signals.
  * **Phase 3 (Ablation Studies)**: Systematically disable individual HMN components (e.g., attention mechanisms, probabilistic gating, phase-locking, meta-learning) across various tasks. Metrics: Performance difference ($\\Delta$-score) compared to full HMN. Insight: Quantify the synergistic contribution of each component.

#### Figure 14: Visual Roadmap for HMN Empirical Validation

  * **Content**: A visual timeline or flowchart. The main flow would represent time or progression through research phases.
      * Each phase (Phase 0, Phase 1, Phase 2, Phase 3) would be a distinct block or stage.
      * Within each phase block, brief descriptions or icons representing the task type (e.g., bandit icon for Phase 0, MNIST/CIFAR logos for Phase 1, RL agent icon for Phase 2) and key metrics (e.g., graph icon for reward, brain icon for forgetting).
      * Arrows would connect the phases, indicating the progression.
  * **Benefit**: Makes the experimental plan more accessible, engaging, and easy to grasp at a glance.
  * **Creation Guidance**: Can be created using presentation software (e.g., PowerPoint, Google Slides with SmartArt or custom shapes), diagramming tools (e.g., Lucidchart), or even timeline generation tools. AI could help design icons for tasks/metrics.

-----

### 7.3 Theoretical Analysis and Hardware Directions

Future theoretical work should focus on analyzing the learning dynamics and convergence properties of simplified HMN variants under specific assumptions. Understanding the theoretical interplay between local Hebbian forces, global modulatory guidance, and meta-learned rate adaptation is crucial. Furthermore, the decentralized and event-driven potential of HMN motivates prototyping on neuromorphic hardware platforms (e.g., spiking networks on Loihi 2, analog implementations using memristors) to explore potential efficiency gains and real-time learning capabilities (cf. Qiao et al., 2024).

-----

### 7.4 Future Work: Integrating Astrocyte-Mediated Neuromodulation 🧠

The HMN framework aims to capture key biological learning principles. Recent advances in neuroscience, particularly the discovery that neuromodulators like norepinephrine (NE) can exert their influence on synapses indirectly via astrocytes, offer a compelling avenue for enhancing the biological realism and potentially the computational capabilities of HMN. Future work should focus on incorporating this astrocyte-mediated pathway for NE-like signals within the HMN, as detailed in the research by Lefton et al. (2025).

#### 7.4.1 Modeling an Astrocyte-Intermediary Layer for Norepinephrine

The current HMN model (Section 4.3.1) aggregates various neuromodulatory signals $E\_k(t)$ into an effective global signal $G\_{\\text{eff}}(t)$. For a neuromodulator like norepinephrine, $E\_{\\text{NE}}(t)$, which is implicated in arousal, novelty, and attention, the findings from Lefton et al. (2025) suggest a more nuanced pathway, as illustrated in Figure 10.

**Proposed Integration Steps:**

1.  **Astrocyte State Representation**: Introduce a new set of state variables, $A\_j(t)$, representing the activation level or signaling state of an astrocyte (or a population of astrocytes) associated with neuron $j$ or a local group of neurons.
2.  **NE Reception by Astrocytes**: The NE signal, $E\_{\\text{NE}}(t)$, instead of directly contributing to $G\_{\\text{eff}}(t)$ for synaptic modulation, would primarily act as an input to these astrocyte units:
    $$A_j(t+1) = f_{\text{astro}}(A_j(t), E_{\text{NE}}(t), \text{params}_{\text{astro}})$$
    where $f\_{\\text{astro}}$ describes the astrocyte's response dynamics, potentially including slower integration timescales and non-linearities reflecting calcium dynamics or other internal processes. `params_astro` would include sensitivity to NE and decay rates.
3.  **Astrocyte Output Generation**: Activated astrocytes would then release their own secondary modulatory signals, $M\_{\\text{astro},j}(t)$ (e.g., mimicking adenosine release, as suggested by the research):
    $$M_{\text{astro},j}(t) = g_{\text{astro}}(A_j(t))$$
    This $M\_{\\text{astro},j}(t)$ could then contribute to the global effective modulation $G\_{\\text{eff}}(t)$ alongside other direct-acting neuromodulators, or it could have a more localized effect.
4.  **Modulation of Synaptic Plasticity**: The astrocyte-derived signal $M\_{\\text{astro},j}(t)$ would then influence synaptic plasticity. Based on the research, this is often a **dampening effect on synaptic strength** or a modulation of **presynaptic efficacy**. This could be implemented by:
      * Having $M\_{\\text{astro},j}(t)$ scale the global learning rate component in the preliminary weight update (Equation 3):
        $$\Delta w^{*}_{ij}(t) = (\eta_{\text{local}} + \eta_{\text{global}} (G'_{\text{eff}}(t) + \lambda_{\text{astro}} M_{\text{astro},j}(t))) \tilde{e}_{ij}(t)$$
        where $G'*{\\text{eff}}(t)$ is the effect of other neuromodulators, and $\\lambda*{\\text{astro}}$ scales the astrocyte influence (which could be negative if dampening).
      * Alternatively, $M\_{\\text{astro},j}(t)$ could directly modulate the eligibility trace $\\tilde{e}*{ij}(t)$ or a specific aspect of neuronal activation $z\_j(t)$ related to presynaptic release probability if the model detail allows. The research highlights NE's effect via astrocytes can involve control of presynaptic efficacy (e.g., through ATP-derived adenosine A1 receptor signaling), which might be modeled as a modulation of $x\_i$ before it contributes to the weighted sum, or as a specific factor affecting $e*{ij}$.

##### Figure 10: Astrocyte-Mediated Neuromodulation Pathway in HMN

  * **Content**: A schematic diagram illustrating the flow of norepinephrine (NE) signaling through an astrocyte intermediary.
      * Show $E\_{\\text{NE}}(t)$ (Norepinephrine signal) as an input to an "Astrocyte Unit" block, representing $A\_j(t)$.
      * The Astrocyte Unit block then outputs $M\_{\\text{astro},j}(t)$ (Astrocyte-derived modulator).
      * This $M\_{\\text{astro},j}(t)$ is shown influencing the synaptic plasticity process, for example, by feeding into the calculation of $G\_{\\text{eff}}(t)$ (alongside other direct neuromodulators $E\_k(t)$) or by directly modulating $\\tilde{e}*{ij}(t)$ or the synaptic weight update $\\Delta w^{\*}*{ij}(t)$.
      * Clearly label all signals and components.
  * **Benefit**: Essential for explaining this novel and somewhat complex proposed integration, making the indirect modulatory pathway visually clear.
  * **Creation Guidance**: Can be created using diagramming software (e.g., draw.io, Lucidchart) or vector graphics, focusing on clear signal flow and component interaction. AI-assisted diagramming tools could be used with a detailed prompt.

#### 7.4.2 Investigating Computational Consequences 💡

Integrating this indirect pathway would allow for:

  * **Differential Timescales**: Exploring how the potentially slower dynamics of astrocyte mediation affect the overall learning process, particularly in response to sustained periods of novelty or arousal (signaled by NE).
  * **Refined Credit Assignment**: Investigating if this intermediary step allows for more nuanced spatial or temporal credit assignment by NE-like signals, as astrocytes can integrate signals over local domains.
  * **Homeostatic Regulation**: The dampening effect of NE-astrocyte signaling on synaptic activity could contribute to homeostatic mechanisms within the network, preventing runaway excitation during periods of high arousal.
  * **Interaction with Other HMN Components**: Studying how astrocyte-mediated NE modulation interacts with the HMN's existing mechanisms like local/global attention ($\\alpha\_{ij}$, $\\gamma\_k$) and meta-learning ($\\eta\_{\\text{local}}$, $\\eta\_{\\text{global}}$). For instance, should the astrocyte pathway itself be subject to attentional modulation or meta-learning?

#### 7.4.3 Empirical Validation and Plausibility ✅

  * The empirical roadmap (Section 7.2) should be extended to include benchmarks that specifically probe the effects of sustained novelty or arousal, where such an astrocyte-mediated pathway might show distinct advantages.
  * Ablation studies would be crucial to compare the HMN with and without the explicit astrocyte layer for NE processing.
  * The Plausibility Map (Section 6.5) should be updated to reflect this more detailed and biologically supported mechanism for NE action.

**Key Reference for Astrocyte Integration:**

  * Lefton, K. B., Wu, Y., Dai, Y., Okuda, T., Zhang, Y., Yen, A., Rurak, G. M., Walsh, S., Manno, R., Myagmar, B.-E., Dougherty, J. D., Samineni, V. K., Simpson, P. C., & Papouin, T. (2025). Norepinephrine signals through astrocytes to modulate synapses. *Science*. (DOI: 10.1126/science.adq5480. Preprint available at bioRxiv).

-----

### 7.5 Future Work: Hybrid Architectures – Trans-HMN 🔄

A promising direction is the development of **Trans-HMN**, a hybrid model combining the strengths of pre-trained Transformers with the adaptive capabilities of HMN cells (visualized in Figure 11).

**Concept:**
The core idea is to use a frozen, pre-trained Transformer (e.g., first few layers) as a powerful feature encoder to capture generic syntactic and semantic representations from input text. These rich representations would then be fed into subsequent HMN layers. The HMN layers, acting as adaptive "adapter" modules, would learn online via their inherent local plasticity, neuromodulatory feedback, and meta-learned learning rates.

**Motivation:**
This approach aims to:

  * Leverage the strong zero-shot generalization capabilities of large pre-trained Transformers.
  * Enable rapid on-device adaptation and continual learning for specific domains, styles, or users through the plastic HMN layers, without costly retraining of the entire Transformer backbone.
  * Potentially mitigate catastrophic forgetting by localizing adaptive changes within the HMN layers while preserving the stable knowledge in the frozen Transformer.

**Research & Evaluation:**
Future empirical work should:

  * Determine the optimal interface and number of HMN layers to augment the Transformer.
  * Investigate how neuromodulatory signals can be derived or propagated from the Transformer's state to guide HMN plasticity.
  * Evaluate Trans-HMN on benchmarks assessing few-shot personalization (e.g., Persona-Chat), continual domain adaptation (e.g., News → Code), and its ability to maintain baseline perplexity while enabling adaptation (cf. Houlsby et al., 2019; Sun et al., 2024; von Oswald et al., 2020).
  * Analyze the division of labor between the frozen Transformer front-end (handling generic structure) and the plastic HMN back-end (capturing context-dependent, rapidly changing patterns).

#### Figure 11: Trans-HMN Hybrid Architecture

  * **Content**: A block diagram showing the Trans-HMN architecture.
      * An initial block labeled "Pre-trained Transformer (Frozen Layers)" takes raw input (e.g., text).
      * The output of this block (feature embeddings) is shown as input to a subsequent series of blocks labeled "Adaptive HMN Layers."
      * Indicate that the HMN layers are plastic (e.g., with a synapse icon or learning symbol) and receive neuromodulatory signals.
      * The final output comes from the HMN layers.
  * **Benefit**: Visually represents the proposed hybrid architecture, clearly distinguishing between the static pre-trained component and the adaptive HMN component.
  * **Creation Guidance**: Can be created using diagramming software. Using different visual styles or colors for the Transformer block versus the HMN blocks can enhance clarity.

-----

### 7.6 Future Work: Sparse Activation Models – MoE-HMN 🧩

Exploring sparsity through a **Mixture-of-Experts (MoE)** architecture, termed **MoE-HMN**, could significantly enhance the scalability and efficiency of HMN-based language models (visualized in Figure 12).

**Concept:**
In MoE-HMN, the model would consist of multiple HMN "expert" sub-networks. For each input token or sequence segment, a learnable router mechanism would select a sparse subset of these HMN experts (e.g., top-k) to process the input. Each HMN expert would be a smaller, fully plastic HMN network as described in this paper.

**Motivation:**
This architecture aims to:

  * Increase model capacity significantly without a proportional increase in inference cost, as only a fraction of experts are activated per input (cf. Fedus et al., 2022; Shazeer et al., 2017).
  * Allow different HMN experts to specialize in different aspects of language, data domains, or even specific languages in multilingual settings.
  * Combine the benefits of sparse activation with the continual learning and adaptive properties of HMN cells.

**Research & Evaluation:**
Key research questions include:

  * Designing effective routing mechanisms that account for the plastic nature of HMN experts and ensure load balancing. This might involve new regularization terms for the router that consider the "plastic capacity" or learning state of experts.
  * Investigating how neuromodulation and meta-learning should operate within an MoE framework (e.g., global neuromodulators for all active experts vs. expert-specific modulation).
  * Evaluating MoE-HMN on tasks like multilingual adaptation (e.g., FLORES-200), robustness to noisy or diverse inputs (e.g., social media text), and measuring adaptation latency for new data types or languages.
  * Analyzing the interaction between router stability and the ongoing plasticity within experts (cf. Roller et al., 2022; Riquelme et al., 2021).

#### Figure 12: MoE-HMN Sparse Activation Architecture

  * **Content**: A diagram illustrating the Mixture-of-Experts HMN model.
      * Input is shown feeding into a "Router" mechanism (e.g., a small neural network).
      * The Router is connected to multiple blocks, each labeled "HMN Expert Sub-network." Arrows from the router should indicate that it selects a subset of these experts.
      * Each HMN Expert block should imply it's a full HMN (perhaps with a mini HMN icon).
      * Outputs from the selected experts are then shown being combined (e.g., weighted sum) to produce the final output.
  * **Benefit**: Clarifies the Mixture-of-Experts concept as applied to HMN, highlighting the sparse selection of expert networks.
  * **Creation Guidance**: Diagramming software is suitable. Visual cues like faded-out unselected experts or highlighted pathways for selected experts can enhance understanding.

-----

### 7.7 Future Work: Efficient Long-Context Models – Mamba-HMN 🐍

Fusing HMN cells with efficient sequence modeling backbones like **Selective State-Space Models (SSMs)**, such as Mamba (Dao et al., 2023), could lead to **Mamba-HMN** models capable of handling very long contexts while retaining online adaptability (visualized in Figure 13).

**Concept:**
Mamba-HMN would utilize an SSM stack (e.g., Mamba) as its primary mechanism for processing long sequences and capturing long-range dependencies with linear or near-linear time complexity. The output or internal states of the SSM layers could then be processed by HMN layers. A key idea is that the SSM could provide a compressed, low-rank summary of the long-term context ($G\_t$), which then acts as a powerful, vector-valued neuromodulatory signal for the HMN layers.

**Motivation:**
This hybrid approach seeks to:

  * Combine the linear-time scaling and impressive long-context capabilities of SSMs with the online weight adaptation and personalization offered by HMN cells.
  * Create a hierarchical memory system: SSMs for efficiently managing and compressing slowly evolving global context, and HMNs for rapid adaptation to local nuances and immediate history.
  * Modulate the plasticity of HMN layers based on the broader discourse coherence or contextual information derived from the SSM.

**Research & Evaluation:**
Future investigations should:

  * Formalize the coupling mechanism, particularly how the SSM's state ($G\_t$) influences HMN plasticity (e.g., as a direct neuromodulator, by gating learning rates, or by shaping attention within HMN).
  * Evaluate Mamba-HMN on long-context language modeling (e.g., PG-19) and question-answering tasks (e.g., NarrativeQA) to assess if the combination preserves long-range understanding while adding adaptability.
  * Test its performance in continual learning scenarios involving domain shifts, to see if the HMN component can adapt effectively while the SSM maintains stable long-term context representation.
  * Theoretically analyze the stability of such coupled systems, particularly how the influence of $G\_t$ on HMN learning rates ($\\eta$) impacts overall model behavior.

#### Figure 13: Mamba-HMN Architecture for Long-Context Processing

  * **Content**: A block diagram showing the Mamba-HMN architecture.
      * Input sequence feeds into a block labeled "SSM Backbone (e.g., Mamba)" which processes long contexts efficiently.
      * The output of the SSM, or its internal states (labeled as $G\_t$ - long-term context summary), is shown feeding into subsequent "HMN Layers."
      * The connection from SSM to HMN layers might be specifically depicted as providing neuromodulatory input or context for plasticity.
      * HMN layers are shown as adaptive and producing the final output.
  * **Benefit**: Illustrates the proposed fusion of efficient long-context SSMs with adaptive HMNs, highlighting the role of SSM output as a contextual modulator for HMN plasticity.
  * **Creation Guidance**: Diagramming software is appropriate. Emphasize the flow of information and the distinct roles of the SSM and HMN components.

These future directions aim to push the HMN-LM framework towards greater scalability, efficiency, and even deeper integration with state-of-the-art architectures, further exploring the potential of biologically-inspired learning in advanced AI systems.

-----

## 8\. Acknowledgements

We thank colleagues in computational neuroscience and machine learning for insightful https://www.google.com/search?q=discussions that helped shape this framework. Without your research a project of this magnitude would have not been possible. This research is supported by me Jeremy Shows and the grace of my wonderful and understanding partner Vanessa.

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
37. Sutton, R. S., & Barto, A. G. (1998). *Reinforcement learning: An https://www.google.com/search?q=introduction*. MIT Press.
38. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 30, pp. 5998–6008).
39. von Oswald, J., Henning, C., Sacramento, J., & Grewe, B. F. (2020). Continual learning with hypernetworks. *International Conference on Learning Representations (ICLR)*.
40. Yu, A. J., & Dayan, P. (2025). Uncertainty, neuromodulation, and attention. *Neuron, 46*(4), 681–692.

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
    c_j_context = update_local_context(z_j)          # Assume function available (Appendix 10.1)

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