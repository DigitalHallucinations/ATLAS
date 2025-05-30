Mamba‑HMN: Fusing Selective State‑Space Models with Neuromodulated Plasticity

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

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
