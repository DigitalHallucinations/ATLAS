MoE‑HMN: A Sparse Mixture of Hybrid‑Modulated Neuron Experts for Efficient Continual NLP

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

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
