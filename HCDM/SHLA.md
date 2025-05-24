# Sparse Hierarchical Latent Attention (SHLA): A Novel Framework for Efficient and Scalable Attention in Deep Learning Models

**Author:**  
Jeremy Shows  
Digital Hallucinations  
<jeremyshws@digitalhallucinations.net>

**Abstract**
Modern deep learning models, particularly Transformers, rely heavily on attention mechanisms. However, the standard self-attention mechanism exhibits quadratic complexity with respect to sequence length, posing significant computational and memory challenges for long-sequence processing and multi-modal integration. This paper introduces Sparse Hierarchical Latent Attention (SHLA), a novel attention framework designed to mitigate these limitations. SHLA constructs a hierarchical structure of latent representations by employing adaptive sparsification and learned hierarchical compression functions to reduce sequence length. Crucially, it integrates Progressive Dimensional Vector Compression-Decompression (PDVCD) to dynamically modulate the feature dimensionality of these latent representations based on information-theoretic measures (e.g., vector entropy), allowing for selective expansion only when necessary. This paper details the formal architecture of SHLA, the mathematical underpinnings of its components—including sparsification, hierarchical sequence compression, and PDVCD-driven dynamic feature dimension expansion—and its mechanisms for efficient latent routing and memory management. We also delineate a comprehensive experimental protocol for rigorous empirical validation. SHLA aims to provide a significant advancement in attention efficiency, scalability, and interpretability for complex AI systems.

---

## 1. Introduction

The advent of the Transformer architecture (Vaswani et al., 2017) revolutionized sequence processing tasks, largely due to its self-attention mechanism. Self-attention enables models to capture long-range dependencies by computing pairwise interactions between all elements in a sequence. However, this strength comes at a cost: the computational and memory requirements of self-attention scale quadratically, $O(T^2 d)$ and $O(T^2 + Td)$ respectively, where $T$ is the sequence length and $d$ is the feature dimension. This complexity renders standard Transformers impractical for very long sequences (e.g., high-resolution images, entire documents, extended audio/video streams) and cumbersome for multi-modal systems requiring fusion of such sequences.

Several approaches have sought to alleviate this bottleneck. Sparse attention mechanisms (Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020) restrict the attention computation to a subset of token pairs, often using predefined patterns (e.g., local, strided, global) or learned sparsity. Architectures like Perceiver IO (Jaegle et al., 2022) introduce a fixed-size latent array to decouple model depth from input size, iteratively attending to inputs and latents. State-Space Models (SSMs) (Gu et al., 2022; Gu & Dao, 2023) offer an alternative with linear or near-linear scaling but may differ in expressive power for certain tasks compared to attention-centric models. While these methods offer improvements, they often involve trade-offs such as fixed sparsity patterns that might not be optimal across all data or tasks, potential information loss in fixed-size latents, or different inductive biases.

This paper introduces Sparse Hierarchical Latent Attention (SHLA), a novel framework that combines the strengths of sparse attention, hierarchical processing, and dynamic representational capacity. SHLA's core contributions are:

1. **Hierarchical Latent Representation:** SHLA organizes input information into a multi-resolution hierarchy. Higher levels (closer to input) capture fine-grained details with longer sequence lengths, while lower levels (deeper in hierarchy) store progressively summarized/compressed representations with shorter sequence lengths.
2. **Adaptive Sparsification:** Within each level of the hierarchy, SHLA employs adaptive sparsification (e.g., top-k selection of attention scores) to focus computation on the most salient interactions.
3. **Dynamic Feature Dimensionality via PDVCD:** SHLA integrates Progressive Dimensional Vector Compression-Decompression (PDVCD) to modulate the *feature dimensionality* ($d_h$) of latent representations dynamically. Based on criteria such as vector entropy, PDVCD allows SHLA to expand the feature dimensionality of a compressed latent vector for detailed processing or keep it compact, thus balancing fidelity and computational cost.
4. **Efficient Routing and Memory Management:** SHLA incorporates mechanisms for routing queries to appropriate hierarchical levels and managing the memory of latent representations.

SHLA aims to significantly reduce computational overhead while preserving critical contextual information, paving the way for more scalable and efficient deep learning models. This paper presents the formal architectural details of SHLA, its theoretical underpinnings, and a rigorous experimental protocol designed for its validation.

---

## 2. Formal Architecture of SHLA

SHLA operates by transforming an input sequence $X \in \mathbb{R}^{T \times d_{model}}$ through a hierarchy of $L$ levels. Each level $h \in \{0, \dots, L-1\}$ processes representations characterized by a sequence length $T_h$ and a feature dimension $d_h$. Typically, $T_0 = T$, $d_0 = d_{model}$, and for $h > 0$, $T_h < T_{h-1}$ (sequence compression), while $d_h$ can be modulated by PDVCD, potentially differing from $d_{h-1}$.

[FIGURE 1: Overall SHLA Architecture]
Description: A high-level diagram illustrating the SHLA framework. The figure should show an input sequence $X$ passing through multiple hierarchical levels ($h=0, 1, \dots, L-1$). Each level should indicate a reduction in sequence length ($T_0 > T_1 > \dots > T_{L-1}$) due to sequence compression and a potentially variable feature dimension ($d_0, d_1, \dots, d_{L-1}$) due to PDVCD modulation. Arrows should indicate information flow, including query routing possibilities between levels and the final aggregated output.

### 2.1. Level 0: Initial Processing and Sparsification**

Given an input sequence $X$, the initial representation at level 0 is $Z^{(0)} = X \in \mathbb{R}^{T_0 \times d_0}$.
Queries $Q^{(0)}$, Keys $K^{(0)}$, and Values $V^{(0)}$ are derived via linear projections:
$Q^{(0)} = Z^{(0)}W_Q^{(0)}$, $K^{(0)} = Z^{(0)}W_K^{(0)}$, $V^{(0)} = Z^{(0)}W_V^{(0)}$
where $W_Q^{(0)}, W_K^{(0)} \in \mathbb{R}^{d_0 \times (N_{heads} \cdot d_k)}$ and $W_V^{(0)} \in \mathbb{R}^{d_0 \times (N_{heads} \cdot d_v)}$. $N_{heads}$ is the number of attention heads, and $d_k, d_v$ are the per-head key and value dimensions. The feature dimension $d_0$ is often $N_{heads} \cdot d_k$ if $d_k=d_v$. For simplicity, we can consider the total dimension for keys as $d_{K\_tot} = N_{heads} \cdot d_k$ and for values as $d_{V\_tot} = N_{heads} \cdot d_v$.

The raw attention scores are computed (typically per head, then combined; shown here conceptually for $d_k$ as overall key dim for simplicity):
$S_{raw}^{(0)} = \frac{Q^{(0)}(K^{(0)})^T}{\sqrt{d_k}}$ (assuming $d_k$ is the effective dimension used in scaling)

**Sparsification Strategy:**
SHLA employs a dynamic sparsification mechanism. A common approach is top-k selection based on attention scores. For each query $q_i^{(0)}$ (representing the i-th query's projection, potentially across all heads or per head), only the top-$k_s$ attention scores (and corresponding keys/values) are retained. Let $A^{(0)} = \text{softmax}(S_{raw}^{(0)})$ (applied row-wise, per head). The sparsified attention scores $\hat{A}^{(0)}$ are defined for each query row $i$ (and each head):
$\hat{A}_{ij}^{(0)} = A_{ij}^{(0)} \cdot \mathbf{1}[A_{ij}^{(0)} \in \text{TopK}_{\text{scores}}(A_i^{(0)}, k_s)]$
where $\text{TopK}_{\text{scores}}(A_i^{(0)}, k_s)$ returns the set of the $k_s$ largest attention scores in the i-th row of $A^{(0)}$ (for a given head), and $\mathbf{1}[\cdot]$ is the indicator function. The value $k_s$ can be a fixed number or a fraction of $T_0$. The output of this level is:
$O^{(0)} = \hat{A}^{(0)}V^{(0)}$. If using multi-head attention, $O^{(0)}$ is typically $ \text{Concat}(head_1, \dots, head_{N_{heads}})W_O^{(0)} $, resulting in shape $\mathbb{R}^{T_0 \times d_{out}^{(0)}}$, where $d_{out}^{(0)}$ is often $d_0$.

### 2.2. Hierarchical Sequence Compression and Attention Propagation**

For subsequent levels $h > 0$, the input $Z_{in}^{(h)} \in \mathbb{R}^{T_{h-1} \times d_{out}^{(h-1)}}$ is typically the output $O^{(h-1)}$ from the previous level. This level will produce Key/Value representations $K^{(h)}, V^{(h)}$ with a reduced sequence length $T_h < T_{h-1}$ and a feature dimension $d_h$.

**Sequence Compression Functions $f_{\text{seq_compress}}^{(h)}$:**
These functions reduce the sequence length $T_{h-1} \to T_h$. The input $Z_{in}^{(h)}$ might first be projected to a suitable feature dimension $d_{feat\_for\_compress}$ if needed: $Z_{proj\_for\_seq\_compress}^{(h)} = Z_{in}^{(h)}W_{proj\_seq}^{(h)}$.
More simply, if $f_{\text{seq_compress}}$ operates on $Z_{in}^{(h)}$ directly:
$Z_{seq\_compressed}^{(h)} = f_{\text{seq_compress}}^{(h)}(Z_{in}^{(h)}) \in \mathbb{R}^{T_h \times d_{out}^{(h-1)}}$
Examples of $f_{\text{seq_compress}}^{(h)}: \mathbb{R}^{T_{h-1} \times d_{in}} \to \mathbb{R}^{T_h \times d_{in}}$ include:

**Pooling:** Max/average pooling over windows of tokens (e.g., a window of size $s$ reduces $T$ by factor $s$).
**Strided Convolution:** 1D convolution with stride $s > 1$, potentially changing $d_{in}$.
**Learned Token Merging:** A small attention layer or MLP that takes blocks of $s$ tokens and outputs a single summary token for each block.
**Clustering-based:** E.g., K-means clustering on token embeddings, where cluster centroids form the new, shorter sequence of vectors.
The output vectors from sequence compression, say $Z_{sc}^{(h)}$, have shape $\mathbb{R}^{T_h \times d_{sc}}$ (where $d_{sc}$ could be $d_{out}^{(h-1)}$ or a new dimension from conv).

**PDVCD for Feature Dimension Modulation:**
The sequence-compressed vectors $Z_{sc}^{(h)}$ are then subject to PDVCD to potentially adjust their feature dimension from $d_{sc}$ to $d_h$.
$Z_{pdvcd}^{(h)} = \text{PDVCD_Modulate}(Z_{sc}^{(h)}) \in \mathbb{R}^{T_h \times d_h}$
(PDVCD is detailed in Section 3).
From $Z_{pdvcd}^{(h)}$, we derive $K^{(h)}$ and $V^{(h)}$ via projections:
$K^{(h)} = Z_{pdvcd}^{(h)}W_K^{(h)}$
$V^{(h)} = Z_{pdvcd}^{(h)}W_V^{(h)}$

**Queries at Level $h$:**
Queries $Q^{(h)} \in \mathbb{R}^{T_{q_h} \times (N_{heads} \cdot d_k)}$ can be derived:

1. From $Z_{in}^{(h)}$ (i.e., $O^{(h-1)}$)), potentially after its own sequence compression (if $T_{q_h} < T_{h-1}$) and PDVCD modulation. If $T_{q_h} = T_h$, then $Q^{(h)}$ can be derived from $Z_{pdvcd}^{(h)}$ via $W_Q^{(h)}$.
2. As persistent "latent queries" that are fewer in number ($T_{q_h} \ll T_h$) and are refined through the hierarchy.
3. From an external source or task-specific signal.

Attention is then computed using $Q^{(h)}, K^{(h)}, V^{(h)}$ with sparsification, producing $O^{(h)} \in \mathbb{R}^{T_{q_h} \times d_{out}^{(h)}}$.

[FIGURE 2: Detailed View of a Single SHLA Level $h$]
Description: This figure should zoom into a single level $h > 0$ of the SHLA hierarchy. It should depict:

1. Input $Z_{in}^{(h)}$ (e.g., $O^{(h-1)}$) from the previous level.
2. The $f_{\text{seq_compress}}^{(h)}$ module processing $Z_{in}^{(h)}$ to produce $Z_{sc}^{(h)}$ with reduced sequence length $T_h$.
3. The $\text{PDVCD_Modulate}$ module processing $Z_{sc}^{(h)}$ to produce $Z_{pdvcd}^{(h)}$ with feature dimension $d_h$.
4. Derivation of $Q^{(h)}, K^{(h)}, V^{(h)}$ from $Z_{pdvcd}^{(h)}$ (or other sources for $Q^{(h)}$).
5. The sparsified attention mechanism ($\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ followed by TopK selection) operating on $Q^{(h)}, K^{(h)}, V^{(h)}$.
6. The output $O^{(h)}$ of the level.
Arrows should clearly show data flow and transformations.

### 2.3. PDVCD-Modulated Expansion for Cross-Level Interaction**

If a query from level $h'$ needs to interact with keys/values from a different level $h''$ that are stored in a PDVCD-compressed feature dimension, those keys/values might be temporarily expanded by PDVCD before attention computation to ensure sufficient representational capacity for the interaction.

**Caching and Hierarchical Structure:**
Each level $h$ maintains its set of (potentially sequence-compressed and PDVCD feature-dimension-modulated) keys $K^{(h)}$ and values $V^{(h)}$. These act as a cache or latent bank.

---

## 3. Progressive Dimensional Vector Compression-Decompression (PDVCD) Integration

PDVCD provides a mechanism for dynamically adjusting the *feature dimensionality* of latent vectors (the $d_h$ dimension of $Z_{pdvcd}^{(h)}$ from which Keys, Values, and potentially Queries are derived) within each SHLA level. This allows SHLA to allocate more representational capacity (more feature dimensions) to information-rich or critical vectors while conserving resources for others.

### 3.1. Vector Entropy Tracking**

For a given latent vector $z \in \mathbb{R}^{d_{current}}$ (e.g., a vector in $Z_{sc}^{(h)}$ with current feature dimension $d_{current}=d_{sc}$), its information content or "activity" can be approximated by its entropy. If features of $z$ are normalized (e.g., to $[0,1]$) and binned into $B$ bins, forming a discrete probability distribution $p_j$ for $j=1, \dots, B$:
$H(z) = -\sum_{j=1}^{B} p_j \log_2 p_j$
A simpler proxy could be the variance of the elements within $z$, or its $L_2$ norm, under the assumption that higher variance/norm correlates with significance. The choice of entropy measure or proxy is a design parameter.

### 3.2. Decision Rule for Decompression/Expansion**

Given a latent vector $z_i \in \mathbb{R}^{d_{current}}$ and an entropy threshold $\tau_H$, the PDVCD modulation function determines its new dimension $d_h$:
$$ \text{PDVCD_Modulate}(z_i, \tau_H, d_{expand\_target}, d_{compress\_target}) = \\ \begin{cases} f_{\text{dim_expand}}(z_i, d_{expand\_target}) & \text{if } H(z_i) < \tau_H \text{ and } d_{current} < d_{expand\_target} \\ f_{\text{dim_compress}}(z_i, d_{compress\_target}) & \text{if } H(z_i) \ge \tau_H \text{ (or other criteria) and } d_{current} > d_{compress\_target} \\ z_i & \text{otherwise (no change, or project to a default } d_h \text{ if } d_{current} \text{ is not suitable)} \end{cases} $$
The underlying hypothesis for expansion on low entropy is that a very low-entropy vector might be too summarized or "collapsed" (having lost distinguishing features), and expanding its feature dimensionality could restore necessary fidelity or allow it to capture more nuanced information. Conversely, if entropy is high (vector already information-rich or noisy) and its current dimension is large, it might be a candidate for dimensional compression if resources are constrained.

* $f_{\text{dim_expand}}(z_i, d_{target}): \mathbb{R}^{d_{current}} \to \mathbb{R}^{d_{target}}$ (where $d_{target} > d_{current}$) is a learned transformation (e.g., a linear layer with padding, or a small MLP).
* $f_{\text{dim_compress}}(z_i, d_{target}): \mathbb{R}^{d_{current}} \to \mathbb{R}^{d_{target}}$ (where $d_{target} < d_{current}$) is also a learned transformation (e.g., a linear layer).

The thresholds and target dimensions can be level-specific and potentially learned or scheduled. The output of PDVCD_Modulate is $Z_{pdvcd}^{(h)}$ with feature dimension $d_h$.

[FIGURE 3: PDVCD Mechanism]
Description: This figure should illustrate the PDVCD process for a single latent vector $z_i$.

1. Input vector $z_i$ with current dimension $d_{current}$.
2. Calculation of its entropy $H(z_i)$.
3. Comparison of $H(z_i)$ against a threshold $\tau_H$.
4. Decision logic:
    * If $H(z_i) < \tau_H$: Path to $f_{\text{dim_expand}}$ module, outputting $z_i'$ with dimension $d_{expand\_target}$.
    * If $H(z_i) \ge \tau_H$: Path to $f_{\text{dim_compress}}$ module (if applicable) or identity, outputting $z_i''$ with dimension $d_{compress\_target}$ or $d_{current}$.
The figure should visually represent the change in vector dimensionality.

---

## 4. Latent Routing & Memory Management

### 4.1. Routing Algorithm**

When a query $q$ (e.g., $Q^{(h')}_i$) needs to retrieve information, SHLA must decide which hierarchical level(s) of keys/values $(K^{(h)}, V^{(h)})$ to attend to.

1. **Default Routing:** A query $Q^{(h')}$ typically attends to keys/values at the same level, $K^{(h')}, V^{(h')}$.
2. **Cross-Level Attention:** Queries can be routed to attend to other levels.
    *Upward Probing (Finer Detail):* A query from a coarser level $h'$ (smaller $T_{h'}$) can be upsampled or broadcast to attend to a finer level $h'' < h'$ (larger $T_{h''}$).
    *Downward Summarization (Broader Context):* A query from a finer level $h'$ can attend to a coarser level $h'' > h'$ to gather summarized context.
    This requires projecting $q$ to match the key dimension of the target level and handling sequence length differences.
3. **Learned Routing:** A small router network $g_{route}(q, \text{state})$ could predict the optimal level(s) $h$ for query $q$.
4. **Budget-Aware Routing:** Prioritize coarser levels under tight computational budgets.

[FIGURE 4: Latent Query Routing]
Description: A diagram showing a query $q$ originating from, or targeted at, a specific level $h'$. Arrows should show how this query $q$ could be routed to:

1. Attend to Keys/Values at its own level $h'$.
2. Be routed "down" to attend to Keys/Values at a more compressed (coarser) level $h'' > h'$.
3. Be routed "up" to attend to Keys/Values at a less compressed (finer) level $h''' < h'$.
The latent banks for different levels ($K^{(h)}, V^{(h)}$) should be visually distinct.

### 4.2. Memory Management**

Each level $h$ stores its key-value pairs $(K^{(h)}, V^{(h)})$ of shape $\mathbb{R}^{T_h \times d_h}$ (post-PDVCD, per head dimensions would be $d_k, d_v$).

**Latent Memory Banks:** The collection $\{ (K^{(h)}, V^{(h)}) \}_{h=0}^{L-1}$ forms SHLA's distributed memory.
**Caching Strategy:** Finer levels (larger $T_h$, potentially larger $d_h$) are more expensive. Their cache persistence might be shorter. Coarser levels (smaller $T_h, d_h$) are cheaper and can be cached longer. PDVCD influences $d_h$, impacting memory.
**Update Mechanism:** For streaming data, updates to $X$ propagate through the hierarchy, potentially recomputing latent banks.

---

## 5. Multi-Modal Integration with SHLA

SHLA's architecture is well-suited for multi-modal integration.

1. **Modality-Specific Hierarchies:** Each modality (e.g., vision $\mathbf{X}_v$, text $\mathbf{X}_t$) can be initially processed by its own SHLA stack to generate $\{ (K_m^{(h)}, V_m^{(h)}) \}$ for each modality $m$.
2. **Cross-Modal Attention via Latent Banks:** Fusion occurs when queries from one modality attend to latent key/value banks of another (e.g., text query $Q_t^{(h')}$ attends to visual bank $K_v^{(h)}, V_v^{(h)}$). Queries are projected to match target key dimensions. SHLA's hierarchy allows flexible fusion granularity.
3. **Shared Latent Space:** Modality-specific outputs can be projected into a shared embedding space and processed by a joint SHLA hierarchy for deeper fusion.
SHLA's sequence compression and PDVCD-based feature dimension modulation are critical for managing the combined complexity.

[FIGURE 5: Multi-Modal Integration using SHLA]
Description: This figure should illustrate two primary strategies for multi-modal fusion with SHLA.
Strategy 1: Modality-specific SHLA stacks. Show an input image processed by a Vision-SHLA and input text by a Text-SHLA. Then, show cross-attention arrows where, for example, queries from a level in Text-SHLA attend to a latent bank (K,V) in Vision-SHLA, and vice-versa.
Strategy 2 (Optional): Tokens from different modalities are projected into a shared space and then processed by a single, joint SHLA stack.
The hierarchical nature and compressed latent banks should be emphasized as enablers of efficient fusion.

---

## 6. Interpretability Mechanics in SHLA

SHLA incorporates mechanisms for interpretability.

### 6.1. Internal Confidence Score Calculation**

For an attention output $o_i$ from query $q_i$, a confidence score $\text{Conf}(o_i)$ can be computed:
2. **Max Attention Score:** The magnitude of the maximum attention weight in $\hat{A}_i^{(h)}$.
3. **PDVCD State:** If key vectors contributing to $o_i$ were expanded by PDVCD to higher feature dimensions, it might indicate the model deemed them important for detailed processing, potentially correlating with confidence or criticality.

### 6.2. Stack Trace Encoding**

A stack trace for an output $O_{final}$ logs the DAG of information flow:

Sequence of queries $q^{(h_x)}$ contributing to $O_{final}$.

For each $q^{(h_i)}$: its source, operating level $h_i$, target key/value bank $(K^{(h_j)}, V^{(h_j)})$, indices of top-k selected keys, PDVCD decisions (entropy, dimension changes) for involved vectors, and the resulting $o^{(h_i)}$.

[FIGURE 6: Conceptual Stack Trace for Interpretability]
Description: An abstract representation of a decision pathway (stack trace) as a Directed Acyclic Graph (DAG). Nodes could represent queries, latent vectors, or attention outputs at different hierarchical levels. Edges represent information flow or attention operations. Annotations on nodes/edges could indicate:

* Hierarchical level $h$.
* Sparsification choices (e.g., selected top-k keys).
* PDVCD decisions (e.g., "vector $z_k$ expanded from $d_c$ to $d_e$ due to low entropy").
* Confidence scores at intermediate steps.
The graph should trace back from a final output to its contributing sources through the SHLA hierarchy.

### 6.3. Perturbation-Based Validation and Refinement

If $\text{Conf}(o_i) < \tau_{conf}$, a validation pathway is triggered:

1. **Identify Alternative Context:** Select alternative keys (e.g., next-best scoring keys not in original top-k) or route $q_i$ to a different hierarchical level or modality.

2. **Compute Alternative Output:** Generate $o_i^{alt}$ using this perturbed context.

3. **Compare/Flag:** If $o_i^{alt}$ leads to significantly different downstream outcomes (e.g., higher task-specific score or confidence), it flags the original pathway or suggests refinement.

---

## 7. Pseudocode for SHLA Forward Pass

(Conceptual, assumes multi-head attention where the overall feature dimension $d_{input\_level}$ or $d_{output\_level}$ is modulated by PDVCD before being split/projected into per-head dimensions $d_{head\_k}, d_{head\_v}$)

```python
import torch
import torch.nn as nn
import math

# Helper function (conceptual)
def calculate_vector_entropy(vector_batch): # vector_batch: Batch x SeqLen x Dim
    # This is a placeholder. A practical implementation needs a robust measure.
    # For this example, low variance is a proxy for low entropy.
    if vector_batch.numel() == 0: return torch.tensor(100.0) # High entropy if empty
    mean_variance_per_vector = torch.var(vector_batch, dim=-1).mean() # scalar
    return mean_variance_per_vector # Lower value means lower entropy for this proxy

class PDVCD_Modulator(nn.Module):
    def __init__(self, base_dim, expand_target_dim, compress_target_dim, entropy_threshold):
        super().__init__()
        self.base_dim = base_dim
        self.expand_target_dim = expand_target_dim
        self.compress_target_dim = compress_target_dim
        self.entropy_threshold = entropy_threshold # Threshold for *low* entropy to trigger expansion

        # These layers are defined with fixed target dimensions.
        # The input dimension to these layers will be `base_dim`.
        self.expander = nn.Linear(base_dim, expand_target_dim) if base_dim != expand_target_dim else nn.Identity()
        self.compressor = nn.Linear(base_dim, compress_target_dim) if base_dim != compress_target_dim else nn.Identity()
        self.identity_map = nn.Identity()

    def forward(self, z_vector_batch): # Expected input: B x T x base_dim
        # Assumes z_vector_batch already has feature dimension `base_dim`.
        avg_entropy_proxy = calculate_vector_entropy(z_vector_batch)

        if avg_entropy_proxy < self.entropy_threshold and self.base_dim < self.expand_target_dim:
            output_vectors = self.expander(z_vector_batch)
        elif avg_entropy_proxy >= self.entropy_threshold and self.base_dim > self.compress_target_dim:
            output_vectors = self.compressor(z_vector_batch)
        else: # Maintain base_dim
            output_vectors = self.identity_map(z_vector_batch)
            # If input z_vector_batch.shape[-1] is not self.base_dim, it should have been projected before.
            # Here, we assume it's already base_dim if identity_map is chosen.
            if output_vectors.shape[-1] != self.base_dim: # Ensure output is base_dim if no op
                 temp_projector = nn.Linear(output_vectors.shape[-1], self.base_dim).to(output_vectors.device)
                 output_vectors = temp_projector(output_vectors)

        return output_vectors # Output: B x T x (expanded_dim or compressed_dim or base_dim)


class SHLA_Level(nn.Module):
    def __init__(self, level_idx, d_input_level, d_output_level, n_heads, d_head_k, d_head_v, top_k_factor,
                 pdvcd_config_level, seq_compress_factor=1, seq_compress_method='none'):
        super().__init__()
        self.level_idx = level_idx
        self.n_heads = n_heads
        self.d_head_k = d_head_k
        self.d_head_v = d_head_v
        self.scale = d_head_k ** -0.5
        self.top_k_factor = top_k_factor

        # PDVCD modulator for the K,V source features (which have d_input_level dimension)
        self.pdvcd_modulator = PDVCD_Modulator(
            base_dim=d_input_level, 
            expand_target_dim=pdvcd_config_level.get('expand_dim', d_input_level * 2),
            compress_target_dim=pdvcd_config_level.get('compress_dim', d_input_level // 2),
            entropy_threshold=pdvcd_config_level.get('entropy_thresh', 0.1)
        )
        
        # Q projection: from d_input_level (query source) to head dimension
        self.W_q_proj = nn.Linear(d_input_level, n_heads * d_head_k, bias=False)
        # Output projection: from concatenated head dimension to d_output_level
        self.W_o_proj = nn.Linear(n_heads * d_head_v, d_output_level, bias=False)

        # K, V projections will be defined dynamically in forward based on PDVCD output dimension
        
        self.seq_compress_factor = seq_compress_factor
        if level_idx >= 0 and seq_compress_factor > 1: 
            if seq_compress_method == 'pool':
                self.seq_compressor = nn.AvgPool1d(kernel_size=seq_compress_factor, stride=seq_compress_factor)
            elif seq_compress_method == 'conv':
                self.seq_compressor = nn.Conv1d(d_input_level, d_input_level, 
                                                kernel_size=seq_compress_factor, stride=seq_compress_factor)
            # Add other seq compress methods here if needed
            else: self.seq_compressor = nn.Identity()
        else: self.seq_compressor = nn.Identity()

    def forward(self, Z_queries_source, Z_kv_source, attention_mask_kv=None):
        # Z_queries_source: B x T_q_orig x d_input_level (source for Q for this level)
        # Z_kv_source: B x T_kv_orig x d_input_level (K,V source from previous level/input)
        # attention_mask_kv: B x T_kv_orig (mask for padding in original K,V source)
        
        B, T_q_orig, C_q_src = Z_queries_source.shape
        _, T_kv_orig, C_kv_src = Z_kv_source.shape
        
        # 1. Sequence Compression for K, V source
        if isinstance(self.seq_compressor, nn.Identity):
            Z_kv_seq_compressed = Z_kv_source
            T_kv_compressed = T_kv_orig
            mask_for_compressed_kv = attention_mask_kv 
        else:
            Z_kv_seq_compressed = self.seq_compressor(Z_kv_source.transpose(1, 2)).transpose(1, 2)
            T_kv_compressed = Z_kv_seq_compressed.shape[1]
            if attention_mask_kv is not None:
                # Adapt mask. MaxPool1d is a common way for sequence reduction.
                # Ensure mask is float for pooling, then bool.
                mask_for_compressed_kv = nn.MaxPool1d(self.seq_compress_factor, self.seq_compress_factor)(attention_mask_kv.float().unsqueeze(1)).squeeze(1).bool()
            else: mask_for_compressed_kv = None
        
        # 2. PDVCD modulation on feature dimension of sequence-compressed K, V source
        # Input to pdvcd_modulator should have feature dim `d_input_level` (its `base_dim`)
        if Z_kv_seq_compressed.shape[-1] != self.pdvcd_modulator.base_dim:
            # This case implies seq_compressor changed feature dimension, project it.
            temp_kv_proj = nn.Linear(Z_kv_seq_compressed.shape[-1], self.pdvcd_modulator.base_dim).to(Z_kv_seq_compressed.device)
            Z_kv_for_pdvcd = temp_kv_proj(Z_kv_seq_compressed)
        else:
            Z_kv_for_pdvcd = Z_kv_seq_compressed
            
        Z_kv_pdvcd_modulated = self.pdvcd_modulator(Z_kv_for_pdvcd) # B x T_kv_compressed x d_modulated
        d_modulated = Z_kv_pdvcd_modulated.shape[-1]

        # 3. Project to Q, K, V for attention
        T_q = T_q_orig # Assuming queries are not sequence compressed at this stage for simplicity

        q_heads = self.W_q_proj(Z_queries_source).view(B, T_q, self.n_heads, self.d_head_k).transpose(1, 2) 
        
        _W_k_runtime = nn.Linear(d_modulated, self.n_heads * self.d_head_k, bias=False).to(Z_kv_pdvcd_modulated.device)
        _W_v_runtime = nn.Linear(d_modulated, self.n_heads * self.d_head_v, bias=False).to(Z_kv_pdvcd_modulated.device)
        
        k_heads = _W_k_runtime(Z_kv_pdvcd_modulated).view(B, T_kv_compressed, self.n_heads, self.d_head_k).transpose(1, 2)
        v_heads = _W_v_runtime(Z_kv_pdvcd_modulated).view(B, T_kv_compressed, self.n_heads, self.d_head_v).transpose(1, 2)

        # 4. Sparsified Attention
        attn_scores_raw = (q_heads @ k_heads.transpose(-2, -1)) * self.scale
        
        if mask_for_compressed_kv is not None:
            expanded_attention_mask = mask_for_compressed_kv.view(B, 1, 1, T_kv_compressed)
            attn_scores_raw = attn_scores_raw.masked_fill(expanded_attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores_raw, dim=-1)

        k_to_select = max(1, int(T_kv_compressed * self.top_k_factor))
        if k_to_select > T_kv_compressed : k_to_select = T_kv_compressed

        top_k_probs, top_k_indices = torch.topk(attn_probs, k=k_to_select, dim=-1)
        
        # Efficient gather operation for selected values
        expanded_top_k_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head_v)
        v_heads_expanded_for_gather = v_heads.unsqueeze(2).expand(-1, -1, T_q, -1, -1) # B x n_h x T_q x T_kv_comp x d_hv
        V_selected = torch.gather(v_heads_expanded_for_gather, 3, expanded_top_k_indices) # B x n_h x T_q x k_sel x d_hv
        
        O_level_heads = torch.einsum('bhqk,bhqkv->bhqv', top_k_probs, V_selected) # B x n_h x T_q x d_hv
        O_level_reshaped = O_level_heads.transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.d_head_v)
        O_level_final = self.W_o_proj(O_level_reshaped) # B x T_q x d_output_level

        return O_level_final, Z_kv_pdvcd_modulated # Return output and the K/V source for the *next* level

class SHLA_Model(nn.Module):
    def __init__(self, num_levels, initial_feature_dim, n_heads, d_head_k, d_head_v, 
                 top_k_factors_per_level, pdvcd_configs_per_level, 
                 seq_compress_factors_per_level, seq_compress_methods_per_level,
                 output_aggregation_strategy='sum'):
        super().__init__()
        self.num_levels = num_levels
        self.initial_feature_dim = initial_feature_dim # Dimension of input X
        # Input projection if X's feature_dim != initial_feature_dim for levels
        self.input_projection = nn.Linear(initial_feature_dim, initial_feature_dim) 

        self.levels = nn.ModuleList()
        # The input dimension for level 0 K/V source and Query source
        current_d_input_for_level = initial_feature_dim 
        for i in range(num_levels):
            # For simplicity, d_output_level matches d_input_level of next K/V source
            # This means W_o_proj output dim must match pdvcd_modulator base_dim of next level
            # Or pdvcd_modulator of next level must take output of W_o_proj of current.
            # Let's assume d_output_level = current_d_input_for_level
            level_module = SHLA_Level(
                level_idx=i,
                d_input_level=current_d_input_for_level, 
                d_output_level=current_d_input_for_level, # Output dim for this level's O
                n_heads=n_heads, d_head_k=d_head_k, d_head_v=d_head_v,
                top_k_factor=top_k_factors_per_level[i],
                pdvcd_config_level=pdvcd_configs_per_level[i],
                seq_compress_factor=seq_compress_factors_per_level[i],
                seq_compress_method=seq_compress_methods_per_level[i]
            )
            self.levels.append(level_module)
            # The input dim for the *next* level's K/V source is the output dim of current level's PDVCD modulator
            # This is complex if fixed nn.Linear is used. Conceptually, the next level adapts.
            # To make it chainable, we'd fix PDVCD_modulator output to current_d_input_for_level
            # or ensure W_o_proj output is what next level's PDVCD modulator expects as base_dim.
            # For now, assume current_d_input_for_level remains consistent.

        self.output_aggregation_strategy = output_aggregation_strategy

    def forward(self, X_input, attention_mask=None): # X_input: B x T_orig x initial_feature_dim
        projected_input = self.input_projection(X_input)

        kv_source_for_current_level = projected_input # K,V source for level 0
        
        # Query strategy: For this example, all levels use the initial projected_input as query source.
        # This means T_q for all levels will be T_orig (original sequence length).
        # Other strategies: Q_h derived from O_h-1 (hierarchical queries).
        queries_source = projected_input 
        
        all_level_outputs = []
        current_mask_for_kv = attention_mask # Mask for T_kv of current kv_source_for_current_level

        for i in range(self.num_levels):
            level_module = self.levels[i]
            level_output, kv_source_after_pdvcd_for_next = level_module(
                Z_queries_source=queries_source, # Using consistent query source
                Z_kv_source=kv_source_for_current_level,
                attention_mask_kv=current_mask_for_kv
            )
            all_level_outputs.append(level_output)
            
            # Update K/V source for the next level
            kv_source_for_current_level = kv_source_after_pdvcd_for_next
            
            # Update attention mask if sequence length of kv_source changed for the *next* iteration
            # This happens based on the sequence compressor of the *next* level using current output
            if i < self.num_levels - 1: # If there's a next level
                next_level_seq_compress_factor = self.levels[i+1].seq_compress_factor
                if next_level_seq_compress_factor > 1 and current_mask_for_kv is not None:
                     # Mask must be for the current T_kv (T_kv_compressed from previous level)
                     current_T_kv = kv_source_for_current_level.shape[1]
                     # Check if mask length matches current T_kv
                     if current_mask_for_kv.shape[-1] != current_T_kv:
                         # This can happen if previous mask was for T_kv_orig and it got compressed
                         # The mask adaptation should happen *inside* SHLA_Level based on its T_kv_compressed
                         # The mask passed to SHLA_Level should match its Z_kv_source *before* its own compression
                         # For this loop: current_mask_for_kv should be for kv_source_for_current_level
                         # Its length should be T_kv_compressed of *previous* level or T_orig for level 0.
                         # So, if level_module.seq_compressor ran, the mask for *its output* is needed by *next* level
                         if not isinstance(level_module.seq_compressor, nn.Identity) and current_mask_for_kv is not None:
                            current_mask_for_kv = nn.MaxPool1d(level_module.seq_compress_factor, 
                                                               level_module.seq_compress_factor)(current_mask_for_kv.float().unsqueeze(1)).squeeze(1).bool()

        # Aggregate outputs from all levels
        if self.output_aggregation_strategy == 'sum':
            # This assumes all level_outputs have same T_q and feature dimension
            final_output = torch.sum(torch.stack(all_level_outputs), dim=0)
        elif self.output_aggregation_strategy == 'last':
            final_output = all_level_outputs[-1]
        elif self.output_aggregation_strategy == 'concat_and_project':
            concatenated_outputs = torch.cat(all_level_outputs, dim=-1)
            final_output_proj = nn.Linear(concatenated_outputs.shape[-1], self.initial_feature_dim).to(X_input.device) # Project back
            final_output = final_output_proj(concatenated_outputs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.output_aggregation_strategy}")

        return final_output
```

---

## 8. Experimental Protocol

The empirical validation of SHLA will focus on quantifying its efficiency, effectiveness on long-sequence and multi-modal tasks, and the utility of its interpretability features.

### 8.1. Synthetic Benchmarks (Efficiency and Compression Loss)

* **Dataset:** Generate synthetic sequences with controlled properties (e.g., "needle-in-a-haystack" tasks, sequences with varying levels of redundancy or varying numbers of long-range dependencies).
* **Tasks:**
    1. Perfect Recall: Retrieve specific tokens or patterns from long sequences.
    2. Reconstruction: For autoencoder setups using SHLA, measure the reconstruction error of $K,V$ pairs after compression and PDVCD-expansion cycles to quantify information loss.
* **Metrics:**
  * FLOPs per forward pass vs. sequence length $T$.
  * Peak memory usage vs. $T$.
  * Latency vs. $T$ on target hardware (CPU, GPU).
  * Task-specific accuracy (e.g., recall rate).
  * Compression ratio achieved by $f_{\text{seq_compress}}^{(h)}$ and average feature dimensionality maintained by PDVCD.
  * Bits per effective parameter (considering model size and learned compression).

### 8.2. Language Modeling and Understanding Tasks

* **Datasets:**
  * Long-sequence Language Modeling: PG-19 (Rae et al., 2019), arXiv dataset.
  * Long-document Question Answering: NarrativeQA (Kočiský et al., 2018), QuALITY (Pang et al., 2021).
* **Metrics:**
  * Perplexity (for LM).
  * Exact Match (EM), F1 Score, ROUGE (for QA/Summarization).
  * Attention dropout vs. semantic accuracy: Measure how aggressively SHLA can sparsify/compress while maintaining performance.

### 8.3. Multi-Modal Tasks

* **Datasets:**
  * Visual Question Answering: VQAv2 (Goyal et al., 2017), GQA (Hudson & Manning, 2019).
  * Image/Video Captioning: COCO Captions (Chen et al., 2015), ActivityNet Captions (Krishna et al., 2017).
  * Audio-Visual Scene-Aware Dialog: AVSD (Hori et al., 2019).
* **Metrics:** Task-specific standard metrics (e.g., VQA score, BLEU, CIDEr, METEOR, SPICE).

### 8.4. Baseline Comparisons

SHLA will be benchmarked against:

* Standard Transformer (Vaswani et al., 2017).
* Sparse Transformers: E.g., Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020).
* Perceiver IO (Jaegle et al., 2022).
* Relevant State-Space Models: E.g., S4 (Gu et al., 2022), Mamba (Gu & Dao, 2023).

### 8.5. Ablation Studies

* Impact of number of hierarchical levels $L$.
* Efficacy of different $f_{\text{seq_compress}}^{(h)}$ strategies (pooling, convolution, learned merging).
* Contribution of PDVCD: SHLA with fixed-feature-dimension latents vs. PDVCD-enabled SHLA. Varying entropy thresholds $\tau_H$.
* Effect of sparsification strategies (top-k vs. other methods, different $k_s$ values).
* Routing strategies for queries (fixed vs. learned/dynamic).

### 8.6. Evaluation of Interpretability Mechanics

* **Confidence Score Utility:** Correlate SHLA's internal confidence scores with actual prediction accuracy on validation sets. Measure error detection rate (how often low confidence correlates with an error).
* **Stack Trace Clarity:** Qualitative user studies with domain experts to assess if the generated stack traces provide meaningful insight into model reasoning, especially for incorrect predictions.
* **Perturbation-Based Validation Efficacy:** Quantify how often the mechanism identifies or corrects potentially flawed reasoning pathways (e.g., leading to improved prediction or higher confidence in a corrected prediction).

### 8.7. Hardware and Software

* **Hardware:** Experiments will be run on a range of hardware, including multi-GPU setups (e.g., NVIDIA A100s, H100s) and potentially resource-constrained edge devices for scalability assessment.
* **Software:** Implemented in PyTorch. Utilize tools like PyTorch Profiler for performance analysis, DeepSpeed/Megatron-LM for distributed training if applicable.

---

## 9. Discussion

SHLA is hypothesized to outperform dense attention mechanisms significantly on tasks requiring long-context understanding and multi-modal fusion, primarily due to its reduced computational complexity and adaptive resource allocation.

* **Expected Advantages:**
  * Sub-quadratic scaling with sequence length, potentially approaching linearithmic or linear in favorable scenarios (effective sequence compression, aggressive sparsification, efficient PDVCD).
  * Lower memory footprint, enabling processing of much longer sequences or more modalities.
  * Graceful performance degradation under tight computational budgets via PDVCD and hierarchical routing.
  * Enhanced interpretability through structured logging and confidence assessment.
* **Potential Limitations and Challenges:**
  * **Information Bottleneck:** Aggressive sequence compression or PDVCD-based feature dimension reduction might lead to loss of crucial information. The balance between efficiency and fidelity is key.
  * **Architectural Complexity:** Designing optimal sequence compression functions, routing mechanisms, and PDVCD thresholds may require extensive tuning or meta-learning. The interaction between sequence compression and feature dimension modulation needs careful design to ensure stable training and effective information flow.
  * **Latency of Dynamic Operations:** On-demand expansion of latent vector feature dimensions by PDVCD, if not managed with efficient caching or optimized transformations, could introduce latency. The overhead of entropy calculation for PDVCD decisions also needs consideration.
  * **Training Stability:** Training a deep hierarchical system with dynamic components and multiple forms of compression can be challenging. Careful initialization, normalization, and learning schedules will be necessary.
* **Theoretical Complexity:** The effective complexity of SHLA is data and task-dependent. With $k_s^{(h)}$ being the number of selected keys per query after sparsification at level $h$ (e.g., $k_s^{(h)} = \text{top_k_factor}^{(h)} \cdot T_h$), $T_h$ the sequence length, and $d_h$ the PDVCD-modulated feature dimension from which head dimensions $d_k, d_v$ are derived. The attention cost at one level is roughly $O(T_{q_h} \cdot k_s^{(h)} \cdot d_k + T_{q_h} \cdot k_s^{(h)} \cdot d_v)$ if $Q$ has $T_{q_h}$ tokens. Summing over levels, and considering sequence compression costs $C_{\text{seq_comp}}$ and PDVCD costs $C_{\text{PDVCD}}$: Total $\approx \sum_{h=0}^{L-1} (O(T_{q_h}^{(h)} \cdot k_s^{(h)} \cdot (d_k + d_v)) + C_{\text{seq_comp}}^{(h)} + C_{\text{PDVCD}}^{(h)})$. If $T_{q_h}^{(h)}$ (query sequence length for level h) and $T_h$ (key/value sequence length for level h) reduce hierarchically and $k_s^{(h)}$ is small, this can be substantially better than $O(T^2 d)$.

---

## 10. Conclusion & Future Work

Sparse Hierarchical Latent Attention (SHLA) offers a promising path towards building more scalable, efficient, and interpretable attention mechanisms. By combining hierarchical sequence compression, adaptive sparsification, and dynamic feature dimensionality modulation via PDVCD, SHLA aims to overcome the quadratic bottlenecks of traditional self-attention. Its design principles are geared towards robust performance on long-sequence and multi-modal tasks while providing valuable metacognitive insights.

The proposed experimental protocol will serve as a roadmap for empirically validating these claims. Future work will extend beyond validation to:

* **Learned Compression and Routing:** Developing end-to-end learnable sequence compression functions $f_{\text{seq_compress}}^{(h)}$ and query routing policies $g_{route}$ within the SHLA framework.
* **Advanced PDVCD Triggers & Mechanisms:** Exploring more sophisticated triggers for PDVCD beyond simple entropy, potentially incorporating task-specific importance signals, uncertainty estimates, or learning the PDVCD policy itself. Optimizing the implementation of dynamic dimension changes in neural network layers.
* **Neuromorphic Adaptations:** Investigating the mapping of SHLA's principles to event-based processing and neuromorphic hardware, leveraging its inherent sparsity and dynamic nature.
* **SHLA in Causal Language Models:** Adapting SHLA for auto-regressive generation, where efficient caching of hierarchical K/V states and consistent representations are paramount.
* **Theoretical Analysis of Information Flow:** Formal bounds on information loss and preservation across the SHLA hierarchy under different compression and PDVCD strategies.

Successful development and validation of SHLA would mark a significant step towards creating AI systems capable of handling the complexity and scale of real-world data with greater efficiency and transparency.

---

## 11. References

1. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv preprint arXiv:2004.05150*.
2. Chen, X., Fang, H., Lin, T. Y., Vedantam, R., Gupta, S., Dollár, P., & Zitnick, C. L. (2015). Microsoft COCO Captions: Data Collection and Evaluation Server. *arXiv preprint arXiv:1504.00325*.
3. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating Long Sequences with Sparse Transformers. *arXiv preprint arXiv:1904.10509*.
4. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., & Parikh, D. (2017). Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
5. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.
6. Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *International Conference on Learning Representations (ICLR)*.
7. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., & Ré, C. (2022). On the Parameterization and Initialization of Diagonal State-Space Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
8. Hori, T., Hori, C., Lee, T. Y., Kumatani, K., Seltzer, M. L., Droppo, J., & Dolan, B. (2019). End-to-End Audio Visual Scene-Aware Dialog. *Proceedings of the AAAI Conference on Artificial Intelligence*.
9. Hudson, D. A., & Manning, C. D. (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
10. Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., & Carreira, J. (2022). Perceiver IO: A General Architecture for Structured Inputs & Outputs. *International Conference on Learning Representations (ICLR)*.
11. Kočiský, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., & Grefenstette, E. (2018). The NarrativeQA Reading Comprehension Challenge. *Transactions of the Association for Computational Linguistics*.
12. Krishna, R., Hata, K., Ren, F., Fei-Fei, L., & Carlos Niebles, J. (2017). Dense-Captioning Events in Videos. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
13. Pang, R. Y., Parrish, A., Puranik, N., Nangia, N., Htut, P. M., Wang, A., & Bowman, S. R. (2021). QuALITY: Question Answering with Long Input Texts, Yes!. *arXiv preprint arXiv:2112.01027*.
14. Rae, J. W., Potapenko, A., Jayakumar, S. M., & Lillicrap, T. P. (2019). Compressive Transformers for Long-Range Sequence Modelling. *arXiv preprint arXiv:1911.05507*.
15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
16. Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. *Advances in Neural Information Processing Systems (NeurIPS)*.
