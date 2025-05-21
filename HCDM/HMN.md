HMN – Hybrid Local–Global Modulated Neuron with Multi-Modal Attention
A Theoretical Framework
Table of Contents
 * Abstract
 * Introduction
   2.1 Motivation and Problem Statement
   2.2 Biological and Computational Inspirations
   2.3 Scope: Biologically Inspired versus Biologically Plausible
   2.4 Core Contributions and Paper Organization
 * Background and Related Work
   3.1 Local Synaptic Plasticity: Hebbian Rules, STDP, Probabilistic Dynamics
   3.2 Global Neuromodulation: Reinforcement, Attention, Meta-Learning
   3.3 Attention Mechanisms in Neural and Deep Systems
   3.4 Limits of Centralized Learning and Alternative Approaches
 * Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)
   4.1 Theoretical Framework and Key Assumptions
   4.2 Local Processing Unit: Probabilistic Synaptic Dynamics
   4.2.1 Neuronal Activation with Stochasticity
   4.2.2 Composite Eligibility Traces (Multi-Timescale)
   4.2.3 Preliminary Weight Update
   4.2.4 Phase-Locked Gating
   4.2.5 Probabilistic Application of Updates
   4.3 Global Neuromodulatory Integration & Dual Meta-Learning
   4.3.1 Aggregation and Attention on Neuromodulators
   4.3.2 Dual Meta-Learning of Learning Rates
   4.4 Multi-Modal Attention Mechanisms
   4.4.1 Local Attention on Eligibility Traces
   Figures
 * Hypothesized Capabilities and Applications
 * Discussion
   6.1 Synergistic Advantages of Integrated Dynamics
   6.2 Comparison with Back-Propagation and Other Bio-Inspired Models
   6.3 Implications for Credit Assignment
   6.4 Complexity, Hyper-Parameter Sensitivity, and Scalability
   6.5 Plausibility Map of HMN Components
 * Conclusion and Future Work
   7.1 Summary of Contributions and Impact
   7.2 Empirical Road-Map and Benchmark Suite
   7.3 Theoretical Analysis and Hardware Directions
 * Acknowledgements
 * References
 * Appendix
   10.1 Example Functional Forms
   10.2 Meta-Learning Gradient Approximation Details
   10.3 SPSA Pseudocode for η-Updates
   10.4 Algorithm Pseudocode for HMN Weight Update
1 Abstract
Biological neural systems exhibit remarkable adaptability by combining rapid, local synaptic plasticity with slower, context-dependent global neuromodulation, enabling lifelong learning in dynamic environments. Current artificial neural networks, often relying on centralized error back-propagation, lack this nuanced, decentralized adaptability. The Hybrid Modulated Neuron (HMN) is presented as a theoretical framework aiming to bridge this gap. It unifies probabilistic local plasticity rules (inspired by STDP and Hebbian learning), multi-factor neuromodulatory feedback, dual meta-learning for adapting local and global learning rates, dual attention mechanisms operating on both synaptic eligibility and neuromodulatory signals, and plasticity gating synchronized with neural oscillations. The framework provides concrete definitions for input representations and local context signals, utilizes a consistent notation for weight updates (\Delta w^*, \Delta w^\dagger, \Delta w), and explicitly maps its components to biological plausibility versus computational inspiration. A staged empirical validation roadmap is proposed. We hypothesize that integrating these mechanisms synergistically equips HMN networks for enhanced continual learning, adaptive reinforcement learning, and robust credit assignment compared to conventional models.
Keywords: Synaptic Plasticity · Neuromodulation · Meta-Learning · Attention · Neural Oscillations · Probabilistic Updates · Continual Learning · Bio-Inspired AI · Neuromorphic Computing
2 Introduction
2.1 Motivation and Problem Statement
The success of deep learning largely stems from supervised training using error back-propagation. However, this paradigm faces challenges related to biological plausibility (e.g., weight transport problem, non-local error signals) and struggles with continual learning in non-stationary environments (catastrophic forgetting). Biological systems, in contrast, employ decentralized learning rules modulated by global context, achieving robust lifelong adaptation. The HMN framework seeks to develop a computationally effective neuronal model incorporating principles observed in biology, specifically aiming to integrate:
 * Fast, local synaptic plasticity (e.g., Hebbian, STDP variants) for rapid adaptation based on correlated activity.
 * Slower, global neuromodulation (e.g., mimicking dopamine for reward, acetylcholine for attention/uncertainty, norepinephrine for novelty) providing context-dependent guidance.
 * Meta-learning capabilities to dynamically adjust the balance between local and global influences (learning rates) based on performance or environmental statistics.
 * Attention mechanisms to selectively weigh inputs and neuromodulatory signals based on relevance.
 * Oscillatory gating to potentially enhance temporal credit assignment and coordinate plasticity across neuronal assemblies.
2.2 Biological and Computational Inspirations
HMN draws inspiration from multiple domains: (i) Neurobiology: well-documented Hebbian learning and Spike-Timing-Dependent Plasticity (STDP), the widespread influence of neuromodulators (dopamine, acetylcholine, etc.) on synaptic gain and plasticity, observed meta-plasticity (activity-dependent changes in plasticity itself), and the role of neural oscillations (e.g., theta, gamma) in coordinating neural activity and synaptic modification. (ii) Machine Learning: attention mechanisms popularized by Transformer models, meta-learning algorithms for learning to learn (e.g., MAML, REINFORCE), and reinforcement learning principles for reward-based adaptation.
2.3 Scope: Biologically Inspired versus Biologically Plausible
We differentiate between components with direct empirical support for both their existence and functional role in biological learning (biologically plausible) and components introduced primarily for computational benefit while maintaining compatibility with biological constraints (biologically inspired). For example, dopamine-gated STDP is considered plausible, while the specific mathematical form of probabilistic gating used here (logistic function) is inspired. This distinction is summarized in the Plausibility Map (Discussion §6.5).
2.4 Core Contributions and Paper Organization
This paper introduces the HMN framework with the following core contributions:
 * Unified HMN Learning Rule: A single theoretical rule integrating probabilistic local eligibility traces, globally modulated feedback, dual meta-learning of rates, dual attention mechanisms, and oscillatory gating.
 * Multi-Timescale Plasticity: Achieved through composite eligibility traces and phase-locked updates potentially aligning plasticity with different behavioral or cognitive states.
 * Dual Meta-Learning: A mechanism to jointly tune local (\eta_{\text{local}}) and global (\eta_{\text{global}}) learning rates online, adapting the learning dynamics to the environment.
 * Dual Attention: Context-dependent modulation operating locally on synaptic eligibility traces and globally on incoming neuromodulatory signals.
 * Plausibility Mapping: A systematic classification of HMN components based on biological evidence versus computational design.
 * Empirical Roadmap: A proposed sequence of experiments for validating HMN capabilities.
The paper is structured as follows: Section 3 reviews related work. Section 4 details the HMN model architecture and mathematical formulation. Section 5 outlines hypothesized capabilities. Section 6 discusses advantages, comparisons, limitations, and plausibility. Section 7 concludes and proposes future directions. Appendices provide supplementary details.
3 Background and Related Work
3.1 Local Synaptic Plasticity
Hebbian learning ("neurons that fire together, wire together") and its temporally precise extension, STDP, provide foundational mechanisms for associative learning based on local activity. Eligibility traces extend these ideas by creating a short-term memory of synaptic activity correlation, which can later be consolidated into a weight change by a third factor, such as a neuromodulator (forming "three-factor rules").
3.2 Global Neuromodulation
Neuromodulatory systems (e.g., dopamine, acetylcholine, serotonin, norepinephrine) exert broad influence, altering neuronal firing properties and synaptic plasticity rules across large brain regions. They convey global state information related to reward prediction error, uncertainty, novelty, arousal, and attention, thereby shaping learning and behavior based on context and outcomes.
3.3 Attention Mechanisms
Biological attention allows organisms to prioritize processing of relevant stimuli. Computationally, attention mechanisms (e.g., in Transformers) dynamically weight information based on context, enabling models to focus on salient features. HMN incorporates attention at both the synaptic (local) and neuromodulatory (global) levels.
3.4 Limits of Centralized Learning and Alternative Approaches
Back-propagation requires symmetric feedback weights (weight transport problem) and instantaneous global error propagation, deviating from known biological constraints. This has spurred research into more biologically plausible alternatives, including feedback alignment, equilibrium propagation, predictive coding, and various three-factor learning rules. HMN builds upon the lineage of three-factor rules while integrating additional mechanisms like meta-learning, advanced attention, and oscillatory gating.
4 Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)
4.1 Theoretical Framework and Key Assumptions
We consider a neuron j receiving inputs x_i(t - \tau_{ij}) from presynaptic neurons i arriving with delays \tau_{ij}. The neuron computes an activation z_j(t). It also receives multiple global neuromodulatory signals E_k(t) (e.g., E_{\text{reward}}, E_{\text{novel}}, E_{\text{uncertainty}}) broadcast from external sources or specialized network populations, and a global context signal C_{\text{global}}(t) representing the broader network or task state. A background oscillatory signal \Phi(t) (e.g., theta or gamma rhythm), potentially global or regional, provides a temporal reference frame.
Key Assumptions:
 * Neurons operate stochastically.
 * Synaptic plasticity depends on local activity (via eligibility traces) and global modulatory signals.
 * Learning rates (\eta_{\text{local}}, \eta_{\text{global}}) are adaptable via meta-learning.
 * Attention mechanisms can modulate both local trace effectiveness and global signal integration.
 * Plasticity can be gated by oscillatory phase.
 * Synaptic delays \tau_{ij} and preferred phases \phi_{ij} are assumed fixed parameters for simplicity, though they could potentially be learned.
 * The bias term b_j(t) is assumed to be adapted slowly via a separate homeostatic or simpler learning mechanism (not detailed here) to maintain neuronal responsiveness.
4.2 Local Processing Unit: Probabilistic Synaptic Dynamics
4.2.1 Neuronal Activation with Stochasticity
The activation z_j(t) of neuron j is computed via a non-linear activation function f applied to the weighted sum of inputs, bias, and noise:
z_j(t) = f\left(\sum_i w_{ij}(t) x_i(t-\tau_{ij}) + b_j(t) + \epsilon_j(t)\right) 
where w_{ij}(t) is the synaptic weight, b_j(t) is the bias term, and \epsilon_j(t) represents neural noise (e.g., sampled from \mathcal{N}(0, \sigma^2_{\text{noise}})).
4.2.2 Composite Eligibility Traces (Multi-Timescale)
Synaptic eligibility e_{ij}(t) captures the potential for plasticity based on recent correlated pre- and post-synaptic activity. It is composed of components with different time constants:
e_{ij}(t) = \psi_{\text{fast}}(t) + \psi_{\text{slow}}(t) 
where \psi_{\text{fast}} and \psi_{\text{slow}} represent eligibility components decaying over short (\tau_{\text{fast}}) and long (\tau_{\text{slow}}) timescales, respectively. These are functions of pre-synaptic input x_i and post-synaptic activation z_j (and potentially their timing), implementing a form of Hebbian or STDP-like correlation detection. (See Appendix §10.1 for example functional forms).
4.2.3 Preliminary Weight Update
Before gating and probabilistic application, a preliminary weight change \Delta w^{*}_{ij}(t) is calculated. It combines a purely local term and a globally modulated term, both acting on the attended eligibility trace \tilde{e}_{ij}(t) (defined in §4.4.1):
\Delta w^{*}_{ij}(t) = \left(\eta_{\text{local}} + \eta_{\text{global}} G_{\text{eff}}(t)\right) \tilde{e}_{ij}(t) 
Here, \eta_{\text{local}} and \eta_{\text{global}} are the meta-learned local and global learning rates (see §4.3.2). G_{\text{eff}}(t) is the effective, attention-modulated global neuromodulatory signal (defined in §4.3.1). This formulation implies that the global signal G_{\text{eff}} acts as a dynamic, context-dependent scaling factor for the influence of the global learning rate \eta_{\text{global}}.
4.2.4 Phase-Locked Gating
The preliminary update is then gated by the phase of the background oscillation \Phi(t) relative to a synapse-specific preferred phase \phi_{ij}:
\Delta w^{\dagger}_{ij}(t) = \Delta w^{*}_{ij}(t) \max\left(0, \cos(\Phi(t) - \phi_{ij})\right) 
This mechanism restricts significant plasticity to specific oscillatory phases, potentially corresponding to optimal windows for information encoding or consolidation. The \max(0, \cdot) ensures gating is multiplicative and non-negative. \phi_{ij} is assumed to be a fixed parameter for each synapse.
4.2.5 Probabilistic Application of Updates
Finally, the phase-gated update \Delta w^{\dagger}_{ij}(t) is applied probabilistically, based on its magnitude relative to a threshold \theta_p:
\Delta w_{ij}(t) =
\begin{cases}
\Delta w^{\dagger}_{ij}(t), & \text{with probability } p_{\text{update}} = \sigma(\beta_p(|\Delta w^{\dagger}_{ij}(t)| - \theta_p)) \\
0, & \text{otherwise}
\end{cases} 

where \sigma(x) = 1 / (1 + e^{-x}) is the logistic sigmoid function, \beta_p controls the steepness of the probability transition, and \theta_p is the magnitude threshold. This probabilistic step is biologically inspired by the stochastic nature of synaptic vesicle release and receptor dynamics, and computationally may introduce sparsity, improve robustness to noise, or aid exploration. The final weight is updated as w_{ij}(t+1) = w_{ij}(t) + \Delta w_{ij}(t).
4.3 Global Neuromodulatory Integration & Dual Meta-Learning
4.3.1 Aggregation and Attention on Neuromodulators
Multiple neuromodulatory factors E_k(t) (e.g., reward, novelty, uncertainty signals) contribute to learning. Their baseline influence is determined by weights w_k. We define the base contribution of each modulator as M_k(t) = w_k E_k(t). These contributions are then dynamically re-weighted by a global attention mechanism based on the current global context C_{\text{global}}(t):
G_{\text{eff}}(t) = \sum_k \gamma_k(t) M_k(t) = \sum_k \gamma_k(t) w_k E_k(t) 
The attention weights \gamma_k(t) are computed similarly to local attention (§4.4.1), allowing the system to prioritize specific neuromodulatory signals based on the global state:
\gamma_k(t) = \frac{\exp(\beta_g h(E_k(t), C_{\text{global}}(t)))}{\sum_m \exp(\beta_g h(E_m(t), C_{\text{global}}(t)))} 
where h(\cdot, \cdot) is a similarity or relevance function (e.g., cosine similarity, see Appendix §10.1) and \beta_g is an inverse temperature parameter controlling attention sharpness. The weights w_k determining the baseline importance of each E_k are assumed fixed or slowly adapted.
4.3.2 Dual Meta-Learning of Learning Rates
The local (\eta_{\text{local}}) and global (\eta_{\text{global}}) learning rates are not fixed hyperparameters but are adapted online via meta-learning. This allows the HMN neuron to adjust its own learning dynamics based on experience. The updates follow a gradient descent rule on a meta-objective function L_{\text{meta}}:
\eta_{\text{local}} \leftarrow \eta_{\text{local}} - \alpha_{\text{meta},1} \nabla_{\eta_{\text{local}}} L_{\text{meta}} 
\eta_{\text{global}} \leftarrow \eta_{\text{global}} - \alpha_{\text{meta},2} \nabla_{\eta_{\text{global}}} L_{\text{meta}} 
where \alpha_{\text{meta},1} and \alpha_{\text{meta},2} are meta-learning rates. L_{\text{meta}} represents a higher-level objective, such as maximizing long-term task reward, minimizing prediction error on a validation set, or achieving a balance between learning speed and stability. Since L_{\text{meta}} might be non-differentiable or depend on long-term outcomes, its gradient is typically approximated using techniques like REINFORCE or Simultaneous Perturbation Stochastic Approximation (SPSA), as detailed in Appendix §10.2 and §10.3.
4.4 Multi-Modal Attention Mechanisms
HMN employs two distinct attention mechanisms:
4.4.1 Local Attention on Eligibility Traces
This mechanism modulates the effective strength of each synapse's eligibility trace e_{ij}(t) based on the relevance of the presynaptic input x_i in the context of the postsynaptic neuron's recent activity c_j(t).
\tilde{e}_{ij}(t) = \alpha_{ij}(t) e_{ij}(t) 
The attention weight \alpha_{ij}(t) depends on the similarity between an embedding h_i(t) of the presynaptic input and a context representation c_j(t) derived from the postsynaptic neuron's state:
\alpha_{ij}(t) = \frac{\exp(\beta_a g(h_i(t), c_j(t)))}{\sum_l \exp(\beta_a g(h_l(t), c_j(t)))} 
 * h_i(t): An embedding representing the features of input x_i(t - \tau_{ij}). This could be generated by an upstream network layer, a fixed feature extractor, or potentially learned locally.
 * c_j(t): A representation of the postsynaptic neuron's local context, e.g., an exponentially weighted moving average of its recent activation z_j(t).
 * g(\cdot, \cdot): A similarity function (e.g., dot product or cosine similarity, see Appendix §10.1).
 * \beta_a: An inverse temperature parameter controlling the focus of the attention.
This allows the neuron to dynamically prioritize plasticity contributions from inputs deemed most relevant to its current processing state.
(§4.4.2 Global Attention on Neuromodulatory Signals was integrated into §4.3.1 for better flow)
Figures
(Descriptions remain, assuming figures would be generated)
Figure 1 (Conceptual Overview): Mermaid diagram illustrating the flow of information: inputs x_i contribute to activation z_j and eligibility traces e_{ij}. Local attention (\alpha_{ij}) generates \tilde{e}_{ij}. Global signals E_k are weighted (\gamma_k) to form G_{\text{eff}}. \tilde{e}_{ij} and G_{\text{eff}} drive the preliminary update \Delta w^*, which is phase-gated (\Phi(t)) to \Delta w^\dagger, and finally applied probabilistically to update w_{ij}. Meta-learning adjusts \eta_{\text{local}}, \eta_{\text{global}}.
Figure 2 (Detailed Schematic): Layered block diagram showing signals (x_i, z_j, e_{ij}, E_k, C_{\text{global}}, \Phi(t)) and transformations (activation, eligibility, attention \alpha_{ij}, attention \gamma_k, combination into \Delta w^*, phase gating \Delta w^\dagger, probabilistic application \Delta w_{ij}, meta-learning updates for \eta).
5 Hypothesized Capabilities and Applications
The integrated mechanisms within HMN are hypothesized to confer several advantages:
 * Adaptive Reinforcement Learning Agents: The interplay between fast local plasticity (driven by \eta_{\text{local}}) and slower, outcome-driven global modulation (driven by \eta_{\text{global}} G_{\text{eff}}) could enable agents to balance exploration and exploitation effectively. Attention and meta-learning further refine this balance based on context and experience.
 * Continual/Lifelong Learning Systems: Dual meta-learning (\eta_{\text{local}}, \eta_{\text{global}}) can potentially adjust the plasticity-stability balance dynamically, allowing networks to acquire new knowledge while mitigating catastrophic forgetting of previously learned tasks. Oscillatory gating might help segregate learning across different contexts or time scales.
 * Complex Decision-Making Models: The layered credit assignment mechanism—combining local correlations, attention-based filtering, temporal gating, and outcome-based modulation—provides a richer, more biologically grounded substrate for modeling complex reasoning and decision processes compared to simpler learning rules.
 * Neuromorphic Hardware Implementations: The focus on local computations, event-driven updates (potentially via probabilistic gating), and parallelism makes HMN potentially suitable for implementation on next-generation neuromorphic hardware (e.g., Intel Loihi, SpiNNaker, or memristive systems).
6 Discussion
6.1 Synergistic Advantages of Integrated Dynamics
The novelty of HMN lies not just in individual components but in their proposed synergy. Local attention (\alpha_{ij}) focuses plasticity on relevant inputs. Global attention (\gamma_k) prioritizes relevant neuromodulators. Phase-locking (\Phi(t)) aligns updates temporally. Probabilistic application (\sigma(\cdot)) adds robustness and sparsity. Dual meta-learning (\eta_{\text{local}}, \eta_{\text{global}}) adapts the core learning dynamics. Together, these mechanisms could create a system capable of rapid adaptation in familiar contexts, cautious exploration in novel situations, robust long-term retention, noise resilience, and context-aware credit assignment.
6.2 Comparison with Back-Propagation and Other Bio-Inspired Models
Compared to standard back-propagation, HMN offers a decentralized, temporally continuous learning process without requiring symmetric weights or explicit error derivatives. Relative to simpler three-factor rules (e.g., basic dopamine-modulated STDP), HMN adds: (i) probabilistic update application, (ii) dual meta-learning of learning rates, (iii) dual attention mechanisms (local/synaptic and global/modulatory), and (iv) explicit oscillatory phase gating. These additions aim to provide greater flexibility, adaptability, and context-sensitivity.
6.3 Implications for Credit Assignment
HMN employs a multi-faceted approach to credit assignment:
 * Synaptic Tagging: Eligibility traces e_{ij} mark synapses based on local causal activity.
 * Attention Refinement: Local attention \alpha_{ij} refines credit based on input relevance; global attention \gamma_k refines credit based on modulator relevance.
 * Temporal Alignment: Oscillatory gating \Delta w^{\dagger} potentially aligns plasticity with relevant network states or behavioral epochs.
 * Outcome Modulation: Neuromodulatory signal G_{\text{eff}} scales updates based on global outcomes (reward, novelty, etc.).
 * Adaptive Regulation: Meta-learning adjusts the overall learning sensitivity (\eta_{\text{local}}, \eta_{\text{global}}).
6.4 Complexity, Hyper-Parameter Sensitivity, and Scalability
The HMN framework introduces several hyper-parameters beyond basic neural models, including decay constants (\tau_{\text{fast}}, \tau_{\text{slow}}), attention parameters (\beta_a, \beta_g), probabilistic gating parameters (\beta_p, \theta_p), oscillation parameters (\phi_{ij} and the frequency of \Phi(t)), meta-learning rates (\alpha_{\text{meta}}), and modulator weights (w_k). Tuning these could be challenging. However, the meta-learning of \eta_{\text{local}} and \eta_{\text{global}} aims to automate the tuning of two crucial parameters. Techniques like automatic relevance determination (ARD) could potentially be adapted to prune less relevant modulators or connections. Scalability to large networks depends on the computational cost of local updates and the communication overhead for global signals (E_k, C_{\text{global}}, \Phi(t)), which are assumed to be broadcast efficiently.
6.5 Plausibility Map of HMN Components
| Component | Empirical Support Level | Classification | Key Supporting Concepts/Citations |
|---|---|---|---|
| STDP & Eligibility Traces | Strong | Plausible | Hebb 1949; Markram 1997; Bi & Poo 1998; Sutton & Barto 1998 |
| Neuromodulator-Gated Plasticity | Strong | Plausible | Schultz 1998; Izhikevich 2007; Yu & Dayan 2005 |
| Oscillatory Phase-Locked Updates | Growing | Plausible | Buzsáki & Draguhn 2004; Fries 2005; Lisman & Jensen 2013 |
| Attentional Modulation (General) | Strong | Plausible | Moran & Desimone 1985; Posner 1990 |
| Dual Attention (Synaptic/Modulatory) | Indirect / Conceptual | Inspired | Extension of general attention principles |
| Probabilistic Synaptic Updates | Indirect | Inspired / Plausible | Stochastic vesicle release, channel noise; Computationally motivated |
| Meta-Plasticity / Rate Adaptation | Emerging | Plausible-in-Principle | Doya 2002; Bellec et al. 2023; Abraham & Bear 1996 |
| Dual Meta-Learning (η_local/η_global) | Conceptual | Inspired | Computational mechanism based on meta-plasticity concepts |
| Specific Logistic Gate Form | None | Inspired | Computational choice for probabilistic gating |
(Note: "Inspired" components leverage biological principles but their specific implementation here is primarily for computational function. "Plausible-in-Principle" suggests biological mechanisms exist that could implement such a function, even if the exact form isn't confirmed.)
7 Conclusion and Future Work
7.1 Summary of Contributions and Impact
The HMN framework offers a novel, unified theoretical model of neuronal learning that integrates multiple biologically-inspired mechanisms: probabilistic local plasticity, multi-factor global neuromodulation, dual meta-learning of learning rates, dual attention (synaptic and modulatory), and oscillatory gating. By combining these elements, HMN aims to provide a more adaptive, robust, and biologically grounded alternative to conventional learning algorithms like back-propagation, potentially offering advantages in continual learning, reinforcement learning, and neuromorphic applications.
7.2 Empirical Road-Map and Benchmark Suite
Validating HMN requires systematic empirical evaluation. We propose a staged approach:
 * Phase 0 (Sanity Check): Implement HMN in simple tasks (e.g., contextual bandits) to verify basic functionality and the viability of meta-learning \eta_{\text{local}} and \eta_{\text{global}} to optimize reward. Metrics: Reward accumulation, convergence of \eta values.
 * Phase 1 (Continual Learning): Test HMN on sequential learning benchmarks (e.g., permuted MNIST, sequential CIFAR-10/100) against baseline models. Metrics: Accuracy on current task, forgetting of previous tasks, forward transfer. Insight: Assess stability-plasticity balance provided by meta-learning and other components.
 * Phase 2 (Temporal Credit Assignment / RL): Evaluate HMN in reinforcement learning tasks requiring longer-term credit assignment (e.g., DeepMind Control Suite, Meta-World). Metrics: Sample efficiency, final performance, reward curve dynamics. Insight: Assess benefits of composite traces, phase-locking, and modulated global signals.
 * Phase 3 (Ablation Studies): Systematically disable individual HMN components (e.g., attention mechanisms, probabilistic gating, phase-locking, meta-learning) across various tasks. Metrics: Performance difference (\Delta-score) compared to full HMN. Insight: Quantify the synergistic contribution of each component.
7.3 Theoretical Analysis and Hardware Directions
Future theoretical work should focus on analyzing the learning dynamics and convergence properties of simplified HMN variants under specific assumptions. Understanding the theoretical interplay between local Hebbian forces, global modulatory guidance, and meta-learned rate adaptation is crucial. Furthermore, the decentralized and event-driven potential of HMN motivates prototyping on neuromorphic hardware platforms (e.g., spiking networks on Loihi 2, analog implementations using memristors) to explore potential efficiency gains and real-time learning capabilities (cf. Qiao et al. 2024).
8 Acknowledgements
We thank colleagues in computational neuroscience and machine learning for insightful discussions that helped shape this framework. [If applicable: Support from Funding Agency Grant XXX is gratefully acknowledged.]
9 References
(References list remains the same as provided in the original text, assuming it is complete and accurate.)
Bahdanau, D. et al. (2014). Neural machine translation by jointly learning to align and translate. arXiv:1409.0473.
Bellec, G. et al. (2023). Meta-learning biologically plausible plasticity rules with random feedback. Nat. Commun. 14, 37562.
... (rest of references) ...
Yu, A.J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. Neuron, 46, 681–692.
10 Appendix
10.1 Example Functional Forms
 * Activation Function f(x): Rectified Linear Unit (ReLU) f(x) = \max(0, x) or Sigmoid f(x) = \sigma(x) = 1 / (1 + e^{-x}).
 * Eligibility Trace Components \psi: Example Hebbian trace: \psi_{\text{decay}}(t) = \text{correlation}(x_i(t-\tau_{ij}), z_j(t)) \times e^{-(t-t_{\text{event}})/\tau_{\text{decay}}}, where \text{correlation}(\cdot) could be x_i z_j or based on spike timing differences for STDP, and t_{\text{event}} is the time of the relevant pre/post activity pairing. \psi_{\text{fast}} uses \tau_{\text{fast}}, \psi_{\text{slow}} uses \tau_{\text{slow}}.
 * Neuromodulator Aggregation Baseline Weights w_k: These represent the default influence of each modulator E_k. Could be fixed based on prior knowledge or slowly adapted.
 * Similarity Functions g(a, b) and h(a, b): Cosine similarity g(a, b) = \frac{a \cdot b}{\|a\| \|b\|} or scaled dot product g(a, b) = \frac{a \cdot b}{\sqrt{d}}, where d is the dimension of embeddings a, b.
 * Local Context c_j(t): Exponential Moving Average (EMA) of activation: c_j(t) = (1 - \delta) c_j(t-1) + \delta z_j(t) for some small decay rate \delta.
 * Input Embedding h_i(t): Could be an EMA of recent x_i(t-\tau_{ij}), features extracted by a fixed function, or embeddings learned by an upstream layer/network.
10.2 Meta-Learning Gradient Approximation
For updating \eta_{\text{local}} and \eta_{\text{global}} (Eqs 8-9), where L_{\text{meta}} might not be directly differentiable w.r.t. \eta, stochastic approximation methods are needed.
 * SPSA (Simultaneous Perturbation Stochastic Approximation): Efficient for high-dimensional parameter spaces (though here only 2D: \eta_{\text{local}}, \eta_{\text{global}}). Perturbs all parameters simultaneously using a random direction vector. See §10.3.
 * REINFORCE (Policy Gradient): Applicable if L_{\text{meta}} can be framed as an expected reward in a stochastic system. Requires estimating the gradient \nabla_{\eta} \log p(\text{trajectory}; \eta) \times \text{Reward}.
 * Finite Differences: Perturb each \eta individually to estimate gradient, less efficient than SPSA for more parameters.
   A common practice is to use decaying perturbation sizes or learning rates for stability, e.g., \epsilon_t = \epsilon_0 / t^{0.5} or \alpha_{\text{meta},t} = \alpha_{\text{meta},0} / t^{0.602}.
10.3 SPSA Pseudocode for \eta-Updates
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
gradient_estimate = (L_plus - L_minus) / (2 * delta) # Element-wise division

# Update eta using gradient descent
eta = eta - alpha_meta * gradient_estimate

# Clip eta to maintain stability / enforce bounds
eta = clip(eta, eta_min, eta_max)

10.4 Algorithm Pseudocode for HMN Weight Update (Single Neuron j)
# --- Inputs at time t ---
# x_i(t-tau_ij): Delayed inputs from presynaptic neurons i
# E_k(t): Global neuromodulatory signals (e.g., reward, novelty)
# C_global(t): Global context signal
# Phi(t): Global/regional oscillation phase
# eta_local, eta_global: Current learning rates (from meta-learning)
# w_ij(t): Current synaptic weights
# b_j(t): Current bias (updated separately)
# Parameters: beta_a, beta_g, beta_p, theta_p, w_k, phi_ij, tau_fast, tau_slow

# --- Neuron Activation ---
summed_input = sum(w_ij[t] * x_i(t - tau_ij) for i) + b_j[t]
noise = sample_gaussian_noise()
z_j = activation_function(summed_input + noise) # Eq (1)

# --- Update Synaptic Weights w_ij for all inputs i ---
for i in presynaptic_neurons:
    # 1. Calculate Eligibility Trace
    psi_f = calculate_psi_fast(x_i(t - tau_ij), z_j, t, tau_fast) # See Appendix 10.1
    psi_s = calculate_psi_slow(x_i(t - tau_ij), z_j, t, tau_slow) # See Appendix 10.1
    e_ij = psi_f + psi_s # Eq (2)

    # 2. Calculate Local Attention
    h_i = get_input_embedding(x_i(t - tau_ij)) # Assume function available (Appendix 10.1)
    c_j = update_local_context(z_j)          # Assume function available (Appendix 10.1)
    similarities_g = [similarity_g(h_l, c_j) for l in presynaptic_neurons]
    alpha_ij = softmax(beta_a * similarities_g)[i] # Eq (11)
    e_tilde_ij = alpha_ij * e_ij # Eq (10)

    # 3. Calculate Effective Global Modulation
    M_k = [w_k[k] * E_k(t)[k] for k in modulators]
    similarities_h = [similarity_h(E_k(t)[m], C_global(t)) for m in modulators]
    gamma_k = softmax(beta_g * similarities_h) # Eq (7)
    G_eff = sum(gamma_k[k] * M_k[k] for k in modulators) # Eq (6)

    # 4. Calculate Preliminary Update
    delta_w_star = (eta_local + eta_global * G_eff) * e_tilde_ij # Eq (3)

    # 5. Apply Phase Gating
    phase_diff = Phi(t) - phi_ij[i] # phi_ij assumed fixed per synapse
    gate = max(0, cos(phase_diff))
    delta_w_dagger = delta_w_star * gate # Eq (4)

    # 6. Apply Probabilistic Update
    update_prob = sigmoid(beta_p * (abs(delta_w_dagger) - theta_p)) # Eq (5)
    if random_uniform(0, 1) < update_prob:
        delta_w_ij = delta_w_dagger
    else:
        delta_w_ij = 0

    # Store update or apply immediately
    w_ij[t+1][i] = w_ij[t][i] + delta_w_ij # Update weight

# --- Update Learning Rates (Periodically or Online) ---
# (Requires evaluating L_meta and using SPSA/REINFORCE)
update_eta_via_meta_learning(eta_local, eta_global, L_meta) # Using Eq (8, 9) and Appendix 10.3

# --- Update Bias Term (Separate mechanism) ---
update_bias(b_j) # Assumed separate process

