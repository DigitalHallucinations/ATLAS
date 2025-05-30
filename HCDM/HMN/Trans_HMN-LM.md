Trans–HMN: A Hybrid Transformer / Hybrid‑Modulated‑Neuron Language Model for Fast Generalisation and Continual Adaptation

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

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