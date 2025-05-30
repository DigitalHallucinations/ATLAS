Latent Circuit Models: Uncovering and Perturbing Structured Low-Dimensional Dynamics in High-Dimensional Neural Systems
Authors: Jeremy Shows, Digital Hallucinations
Abstract:
Understanding the dynamical principles governing neural computation remains a fundamental challenge, hindered by the high dimensionality and complexity of neural population recordings. While neural activity often evolves on lower-dimensional manifolds, existing methods may not explicitly capture the underlying dynamical structure or allow for targeted manipulation of these dynamics. We introduce the Latent Circuit Model (LCM), a framework for identifying structured, low-dimensional dynamics embedded within high-dimensional time series data, such as neural population activity. The LCM parameterizes the system using an orthonormal projection matrix (Q) into an n-dimensional latent space and models the latent dynamics via a structured connectivity matrix A. We place a particular emphasis on skew-symmetric parameterizations (A=B′−(B′)T) to capture rotational dynamics prevalent in cortical areas. This parameterization is learned by optimizing a reconstruction objective on the observed high-dimensional data. A key feature of the LCM is its ability to facilitate structured perturbation analysis: targeted manipulations (δ) within the learned latent space can be projected back into the observation space via δW=QδQT, allowing for precise, in silico probing of the functional consequences of altering specific dynamical motifs. We outline a series of proposed experimental studies designed to validate the LCM's ability to accurately recover known latent dynamics and subspaces in synthetic datasets with ground truth. Application of the LCM is proposed for analyzing simulated neural activity from complex cognitive architectures (like the Hybrid Cognitive Dynamics Model - HCDM) and benchmark neuroscience datasets (e.g., motor cortex recordings), with the expectation of revealing interpretable, low-dimensional circuits. Furthermore, the proposed perturbation analysis aims to demonstrate how targeted modulation of these latent circuits can predictably influence system-level behavior and internal state representations, offering a powerful tool for both analyzing complex neural systems and constructing more interpretable and manipulable artificial intelligence models.
Keywords: Neural Dynamics, Dimensionality Reduction, State-Space Models, Rotational Dynamics, Neural Manifolds, Computational Neuroscience, Recurrent Neural Networks, System Identification, Perturbation Analysis, Interpretability, Artificial Intelligence, HCDM, Skew-Symmetric Systems.

1. Introduction
1.1. The Challenge of High-Dimensional Neural Data
Modern neuroscience techniques, such as multi-electrode arrays and calcium imaging, provide unprecedented access to the simultaneous activity of large neural populations [Stevenson & Kording, 2011]. However, this deluge of high-dimensional data presents significant analytical challenges. Understanding how neural circuits perform computations requires moving beyond single-neuron characterization to decipher the collective dynamics of these populations [Saxena & Cunningham, 2019]. Identifying the underlying principles governing these dynamics is crucial for understanding brain function in health and disease, and for building AI systems inspired by these principles.
1.2. Evidence for Low-Dimensional Structure (Neural Manifolds)
Despite the high number of recorded neurons (N), neural activity during behavior often appears to occupy a much lower-dimensional subspace or manifold (n≪N) [Gao et al., 2017; Gallego et al., 2017]. This suggests that the collective activity is constrained and coordinated, reflecting shared computational goals or underlying circuit structure. Identifying and characterizing these neural manifolds is a key goal of computational neuroscience, as they provide a simplified yet potentially comprehensive description of the population's computational state and its temporal evolution.
1.3. Importance of Dynamical Systems Perspective
Neural computation is inherently dynamic. Understanding how neural states evolve over time in response to inputs and internal processes is paramount. A dynamical systems perspective provides a powerful mathematical framework for modeling these temporal evolutions [Eliasmith & Anderson, 2003; Sussillo & Barak, 2013]. Modeling neural populations as dynamical systems allows us to characterize phenomena like attractor states, oscillations, sequential activity, and complex transient responses, which are thought to underlie cognitive functions such as decision-making, memory, and motor control.
1.4. Limitations of Existing Models
Several methods exist for analyzing high-dimensional neural data.
Linear Methods: Principal Component Analysis (PCA) and Factor Analysis (FA) identify low-dimensional subspaces but often ignore temporal dynamics [Cunningham & Yu, 2014].
Non-linear Visualization: t-SNE or UMAP excel at visualization but do not typically yield an explicit dynamical model [Maaten & Hinton, 2008; McInnes et al., 2018].
Recurrent Neural Networks (RNNs): While powerful dynamical models, standard RNNs often result in "black box" representations where the learned connectivity lacks clear structure or interpretability [Sussillo & Abbott, 2009].
Latent State-Space Models: Gaussian Process Factor Analysis (GPFA) [Yu et al., 2009] and Latent Factor Analysis via Dynamical Systems (LFADS) [Pandarinath et al., 2018] explicitly model latent dynamics but may not enforce specific interpretable structures (like rotational dynamics) directly within the latent space connectivity. They often rely on complex internal generators (like LSTMs in LFADS) whose own dynamics can be hard to interpret.
Emerging Models: Recently, Langdon et al. [2025] introduced a "Latent Circuit Model" that infers interactions between task variables within a latent recurrent circuit, explaining heterogeneous responses and proposing a similar QδQT perturbation. However, it focuses on learned, general recurrent dynamics rather than a priori imposing specific structures like skew-symmetry.
1.5. Proposed Approach: Latent Circuit Models (LCM)
To address these limitations, we propose the Latent Circuit Model (LCM). The LCM combines dimensionality reduction with structured dynamical modeling. It assumes that high-dimensional observations (y∈RN) arise from dynamics occurring within a lower-dimensional latent space (z∈Rn). The mapping between spaces is handled by an orthonormal projection matrix Q. Crucially, the LCM parameterizes the latent dynamics, often modeled linearly as z˙=Az+Culatent​, with a structured matrix A. A primary focus is the use of a skew-symmetric matrix A=B′−(B′)T, which naturally generates rotational or oscillatory dynamics, consistent with observations in motor cortex and other areas [Churchland et al., 2012]. This explicit structure enhances interpretability. Furthermore, the LCM framework facilitates targeted in silico perturbations within the latent space (A→A+δ), allowing researchers to project these changes back (QδQT) and probe the functional role of specific dynamical components.
1.6. Key Contributions
The main contributions of this work are:
Formal definition of the Latent Circuit Model (LCM) parameterization, emphasizing the orthonormal projection Q and structured latent connectivity A (particularly skew-symmetric forms).
Derivation of an optimization procedure based on minimizing reconstruction error, addressing the QTQ=In​ orthonormality constraint.
A methodology for applying structured perturbations (δW=QδQT) to probe latent circuit function.
Proposal of experimental studies designed to demonstrate LCM's ability to recover ground truth dynamics in synthetic datasets (Section 6.1).
Proposal of LCM application to analyze real or simulated neural data, aiming to reveal interpretable low-dimensional dynamics (Section 6.2).
Proposal of studies illustrating LCM's utility in analyzing and modulating complex AI systems, such as the Hybrid Cognitive Dynamics Model (HCDM) (Section 6.3).
1.7. Paper Organization
Section 2 reviews related work. Section 3 details the mathematical formulation of the LCM. Section 4 discusses model fitting and optimization. Section 5 presents the structured perturbation methodology. Section 6 outlines proposed experimental studies. Section 7 discusses potential findings, implications, and limitations. Section 8 concludes.

2. Related Work
The LCM builds upon several lines of research:
Dimensionality Reduction: Techniques like PCA, FA [Cunningham & Yu, 2014], ICA, and their non-linear counterparts (Isomap, LLE, t-SNE, UMAP) [Van der Maaten & Hinton, 2008; McInnes et al., 2018] are standard. LCM integrates this with an explicit dynamical model.
State-Space Models: Methods like Kalman filters, GPFA [Yu et al., 2009], and LFADS [Pandarinath et al., 2018] explicitly model latent dynamics. GPFA uses Gaussian processes; LFADS uses RNNs. LCM distinguishes itself by imposing specific interpretable structure (e.g., skew-symmetry) directly onto the latent connectivity matrix A, rather than relying on emergent dynamics.
Recurrent Neural Networks: RNNs are widely used [Sussillo & Abbott, 2009; Michaels et al., 2020]. While techniques exist to analyze RNN dynamics [Sussillo & Barak, 2013], LCM aims to build interpretability directly into the model structure.
Rotational Dynamics: Seminal work [Churchland et al., 2012] demonstrated strong rotational patterns in motor cortex. This finding is a primary motivator for LCM's focus on skew-symmetric latent dynamics, which naturally generate such rotations.
System Identification: LCM falls under system identification [Ljung, 1999], specifically targeting systems with a low-dimensional latent structure.
Latent Circuit Inference: As mentioned, Langdon et al. [2025] introduced a "Latent Circuit Model" focusing on task-variable interactions and heterogeneous responses. Their model learns a general recurrent matrix (wrec​) and shares the QδQT perturbation idea. Our LCM differs primarily in its a priori emphasis on skew-symmetric (or otherwise structured) A matrices as a means to directly test hypotheses about underlying dynamical motifs like rotations.

3. The Latent Circuit Model (LCM): Formulation
We model the observed high-dimensional activity y(t)∈RN as arising from unobserved, lower-dimensional latent dynamics z(t)∈Rn, where n≪N.
3.1. General State-Space Framework
The model takes the general form:
z˙(t)=f(z(t),ulatent​(t);θdyn​)(Latent Dynamics)
y(t)=g(z(t);θobs​)+ϵ(t)(Observation Model)
where ulatent​(t) represents inputs, θdyn​ and θobs​ are parameters, and ϵ(t) is observation noise.
3.2. Observation Model: Orthonormal Projection (Q)
We assume a linear observation model:
y(t)≈Qz(t)
Here, Q∈RN×n is an observation matrix whose columns are orthonormal, i.e., QTQ=In​ (the n×n identity matrix). The columns of Q form an orthonormal basis for the n-dimensional latent subspace within the N-dimensional observation space. QT acts as the projection operator, mapping from the observation space onto the latent subspace axes: z(t)≈QTy(t). The orthonormality is key: it ensures that the latent dimensions are uncorrelated in their projection and preserves distances within the subspace (∣∣Qz1​−Qz2​∣∣2=∣∣z1​−z2​∣∣2).
3.3. Latent Dynamics Model (z˙=Az+Culatent​)
We focus primarily on linear (or near-linear) latent dynamics:
z˙(t)=Az(t)+Culatent​(t)
where A∈Rn×n is the latent connectivity matrix, and C∈Rn×m maps external inputs u(t)∈Rm.
3.3.1. Structured Connectivity Matrix (A): Instead of learning an unconstrained A, LCM imposes structure.
3.3.2. Skew-Symmetric Parameterization: A key structure is skew-symmetry (A=−AT). We parameterize A using an auxiliary matrix B′∈Rn×n such that: A=B′−(B′)T In our proposed implementation (Appendix B), B′ is derived from a larger, learnable matrix B∈RN×n by taking its top-left n×n block (B′=B[:n,:n]). This allows gradients w.r.t. A to update B.
3.3.3. Properties of Skew-Symmetric Dynamics: When A is skew-symmetric and C=0, the dynamics z˙=Az conserve the squared norm of the state: dtd​∥z∥2=0. This leads to purely rotational or oscillatory dynamics. The eigenvalues of a skew-symmetric matrix are purely imaginary (iω), corresponding to oscillations with frequencies ω. Adding a symmetric component S (A=S+(B′−(B′)T)) allows for expansion or contraction (damping), as eigenvalues gain real parts. This structure provides a highly interpretable way to capture key dynamical features.
3.3.4. Input Mapping (C): The matrix C determines how inputs influence latent dynamics. It can be fixed (C=I, C=QT) or learned.
3.4. Observation Noise Assumptions
Typically, the observation noise ϵ(t) is modeled as isotropic Gaussian noise: ϵ(t)∼N(0,σ2IN​).
3.5. Full Parameterization Summary
The learnable parameters are B (defining A), Q, and potentially C and σ2. Q must satisfy QTQ=In​.

4. Model Fitting and Optimization
4.1. Objective Function
Given observed trajectories {yk​(t)}, we minimize the Mean Squared Error (MSE) reconstruction loss:
L(Θ)=K1​k=1∑K​Tk​1​t=1∑Tk​​∥yk​(t)−Qzk​(t)∥2+R(Θ)
where zk​(t) is obtained by integrating z˙=Az+..., and R(Θ) represents regularization terms (e.g., L2 on B).
4.2. Numerical Integration
We use numerical methods like Runge-Kutta 4 (RK4) or adaptive-step methods (e.g., Dormand-Prince 5, dopri5) via libraries like torchdiffeq to integrate the dynamics z˙=Az and obtain z(t).
4.3. Optimization Procedure
We use gradient-based methods (e.g., Adam [Kingma & Ba, 2014]). This requires computing gradients through the integration process. This is achieved using:
Backpropagation Through Time (BPTT): For discrete approximations or fixed-step solvers.
Adjoint Sensitivity Method: [Chen et al., 2018] A more memory-efficient method for continuous dynamics, crucial for long time series. Modern libraries like torchdiffeq implement this, allowing automatic differentiation through ODE solvers.
4.4. Handling Orthonormality Constraint on Q
Maintaining QTQ=In​ is crucial. Options include:
Projection (SVD/QR): After each gradient step, project Q back onto the Stiefel manifold (Qproj​=UVTfrom Qnew​=USVT). This is often practical (see Appendix B).
Geodesic Updates: More complex, performs updates along manifold curves.
Penalty Term: Add λ∥QTQ−In​∥2 to the loss. Less strict.
4.5. Initialization Strategies
B: Small random values.
Q: Can be initialized via PCA on y(t) (projecting data onto top n PCs) followed by QR/SVD for orthonormality, or randomly + QR/SVD. PCA often speeds up convergence.
4.6. Regularization
L2 regularization on B (λ∥B∥F2​) is used to prevent overfitting and encourage smoother, simpler dynamics (see Appendix B).

5. Structured Perturbation Analysis
A key advantage of LCM is interpretable perturbation.
5.1. Defining Perturbations (δ) in Latent Space
We design a perturbation δ∈Rn×n to modify the latent dynamics A→A+δ. Examples:
Targeting Rotations: A skew-symmetric δ alters rotation frequencies/planes.
Targeting Stability: A symmetric δ adds/removes damping.
Targeting Dimensions: A sparse δ affects specific latent dimensions.
Simulating Lesions: Setting rows/columns to zero.
5.2. Projecting Perturbations to Observation Space
The change in effective high-dimensional connectivity is:
δW=QδQT
This projects the meaningful, structured latent change δ into the N-dimensional space using the learned basis Q. This δW is an N×N matrix representing how the latent intervention manifests across the entire "neural" population. It is inherently low-rank (rank(δW)≤n).
5.3. Simulating Perturbation Effects
We can simulate the system with perturbed dynamics:
Latent Simulation: Integrate z˙=(A+δ)z and observe ypert​(t)=Qz(t).
Effective Matrix (HCDM): Apply δW to the system's weight matrices (Wnew​=Wold​+αδW). This allows direct in silico testing within a larger architecture, as proposed for HCDM (Section 6.3).
5.4. Interpreting Perturbation Effects
By observing changes in behavior or internal states resulting from applying δW, we can infer the causal functional role of the latent dynamics captured by A and the subspace Q. This provides a bridge between observed dynamics, latent structure, and functional consequence.

6. Experimental Evaluation: Proposed Studies
6.1. Study 1: Validation on Synthetic Data
Objective: Quantitatively verify LCM's ability to recover known Qtrue​, Atrue​ (especially skew-symmetric), and ntrue​ under varying noise, and validate the QδQT perturbation framework.
Methodology: Generate data from z˙=Atrue​z, y=Qtrue​z+ϵ. Fit LCM, compare Qfit​ (subspace angles), Afit​ (eigenvalues, norm), and nfit​ (cross-validation). Apply known δtrue​, calculate Qfit​δfit​QfitT​, and compare its effect to the ground truth perturbation. Compare against PCA+VAR, FA, and perhaps a standard RNN.
Expected Outcomes: LCM should outperform baselines in recovering Atrue​ and Qtrue​, particularly rotational structures. Perturbation effects should show high fidelity.
6.2. Study 2: Application to Neuroscience Data
Objective: Identify interpretable, behaviorally relevant latent dynamics in real neural data, focusing on datasets where rotations are hypothesized (e.g., motor cortex).
Methodology: Use public datasets (e.g., CRCNS.org M1/PMd reaching data). Preprocess (smoothing, alignment). Fit LCM, select n. Analyze A's structure (skew vs. symmetric ratio, eigenvalues). Correlate z(t)with behavior (reach direction, speed) using decoding. Compare with jPCA, GPFA, LFADS in terms of reconstruction, decoding, and interpretability.
Expected Outcomes: LCM should find low-D rotational dynamics in M1/PMd, predictive of behavior. Ashould be more directly interpretable than LFADS generators, potentially revealing circuit-level hypotheses.
6.3. Study 3: Analysis and Modulation of a Complex AI System (HCDM)
Objective: Use LCM to analyze HCDM's internal dynamics (via NCB) and demonstrate targeted modulation using QδQT.
Methodology: Run HCDM on a complex task. Collect NCB activity vectors. Fit LCM to NCB data. Correlate z(t) with HCDM states (DSSM thoughts, AGM actions). Design δ based on analysis (e.g., stabilize a "planning" state). Apply δW=QδQT to HCDM modules (e.g., DSSM/AGM weights, or via EFM modulation). Measure changes in HCDM performance and internal states.
Expected Outcomes: LCM will reveal interpretable low-D dynamics on the NCB. QδQT perturbations will cause predictable changes in HCDM behavior, validating LCM as an analysis and control tool for complex AI.

7. Discussion
7.1. Summary
LCM offers an interpretable framework for low-D dynamics, emphasizing structure (like skew-symmetry) and enabling targeted perturbation analysis (QδQT). Proposed studies aim to validate its utility.
7.2. Interpretation of Latent Circuits
Q defines the "what" (the manifold), and A defines the "how" (the flow/dynamics). Analyzing A provides direct insight into computational motifs.
7.3. Advantages of LCM
Interpretability: Structured A.
Priors: Incorporates biological/computational hypotheses (rotation).
Perturbation: Direct in silico causal testing.
Generative: Models data generation.
7.4. Limitations and Challenges
Linearity Assumption: Real dynamics are often non-linear (though LCM can capture local linearizations).
Scalability: ODE solving + BPTT/Adjoint can be intensive.
Identifiability: Latent basis rotation needs careful handling.
Hyperparameter Choice: n selection is critical.
Non-Stationarity: Assumes fixed dynamics.
7.5. Implications for Neuroscience
Provides a tool to infer and test dynamical generators, bridging manifolds to mechanisms and simulating interventions.
7.6. Implications for AI (HCDM)
Offers a path to:
Interpretability: Understand core AI dynamics.
Structured Design: Build AIs with LCM-inspired modules.
Targeted Control: Enable high-level systems (EFM) to precisely modulate AI states/behaviors via QδQT, a more nuanced approach than global tuning.

8. Conclusion and Future Work
8.1. Conclusion
LCM presents a promising approach for extracting, understanding, and manipulating structured low-dimensional dynamics in high-dimensional systems. Its emphasis on interpretability and causal probing via structured perturbations positions it as a potentially valuable tool for neuroscience and AI.
8.2. Future Directions
Non-Linear LCMs: Incorporate structured non-linearities (e.g., f(z)=(B′−B′T)tanh(z)).
Hierarchical LCMs: Model dynamics at multiple timescales.
Adaptive/Online LCMs: Track non-stationary dynamics.
Control Theory Integration: Explore optimal control within the LCM framework.
Closed-Loop Applications: Real-time BCI or AI agent control.

References
Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems,1 31.
Churchland, M. M., Cunningham, J. P., Kaufman, M. T., et al. (2012). Structure of neural population dynamics during reaching. Nature, 487(7405), 51–56.
Cunningham, J. P., & Yu, B. M. (2014). Dimensionality reduction for large-scale neural recordings. Nature Neuroscience, 17(11), 1500–1509.2
Eliasmith, C., & Anderson, C. H. (2003). Neural Engineering: Computation, Representation and Dynamics in Neurobiological Systems. MIT3 Press.
Gallego, J. A., Perich, M. G., Miller, L. E., & Solla, S. A. (2017). Neural Manifolds for the Control of Movement. Neuron, 94(5), 978–984.
Gao, P., Trautmann, E., Yu, B., et al. (2017). A theory of multineuronal dimensionality, dynamics and measurement. bioRxiv, 214262.
Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.
Langdon, A. E., Engel, T. A., & Saxe, A. M. (2025). Latent circuit inference from heterogeneous neural responses during cognitive tasks. Nature Neuroscience, 28(1), 161-171. (Note: Published Jan 2025, check for availability).
Ljung, L. (1999). System Identification: Theory for the User (2nd ed.). Prentice Hall.
Maaten, L. V. D., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(11).
McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426.4
Michaels, A. J., et al. (2020). Recurrent switching linear dynamical systems model stimulus-dependent variability in population activity. bioRxiv.
Pandarinath, C., O’Shea, D. J., Collins, J., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders.5 Nature Methods, 15(10), 805–815.6
Saxena, S., & Cunningham, J. P. (2019). Towards the neural population doctrine. Current Opinion in Neurobiology, 55, 103–111.
Stevenson, I. H., & Kording, K. P. (2011). How advances in neural recording affect data analysis. Nature Neuroscience, 14(2), 139–142.
Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
Sussillo,7 D., & Barak, O. (2013). Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks. Neural Computation, 25(3),8 626–649.
Werbos, P. J. (1990). Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10), 1550–1560.
Yu, B. M., Cunningham, J. P., Santhanam, G., et al. (2009). Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity.9 Journal of Neurophysiology,10 102(1), 614–635.

Appendices
A. Derivation of Gradient for LCM with Orthonormality Constraints
The goal is to find the gradients of the Mean Squared Error (MSE) loss function L with respect to the model parameters, primarily the matrix B (which defines the skew-symmetric dynamics matrix A) and the orthonormal observation matrix Q.
The loss is given by:
$L(Θ)=K1​∑k=1K​Tk​1​∑t=1Tk​​∥yk​(t)−Qzk​(t)∥2$where the latent state zk​(t) is obtained by integrating the latent dynamics:z˙k​(t)=Azk​(t)+Culatent,k​(t)
with A=B′−(B′)T and B′=B[:n,:n]. For simplicity, we'll consider the autonomous case (C=0) here: z˙=Az. The initial state zk​(0) might be inferred, e.g., zk​(0)≈QTyk​(0).
Gradient w.r.t. Q:
The gradient ∂Q∂L​ has contributions from the reconstruction term and potentially from the initial state estimation if zk​(0) depends on Q. Focusing on the reconstruction term:
$∂Q∂∥yk​(t)−Qzk​(t)∥2​=−2(yk​(t)−Qzk​(t))zk​(t)TSummingovertimeandtrialsgivestheoverallgradientcontributionfromthereconstructionerror:∇Q​L=K−2​∑k=1K​Tk​1​∑t=1Tk​​(yk​(t)−Qzk​(t))zk​(t)T$
However, applying standard gradient descent using ∇Q​L will violate the orthonormality constraint QTQ=In​. To handle this:
Projection: After an unconstrained update (Qnew​=Qold​−η∇Q​L), project Qnew​ back onto the Stiefel manifold. A common method is using the Singular Value Decomposition (SVD): If Qnew​=USVT, the projection is Qproj​=UVT. Alternatively, QR decomposition can be used.
Geodesic Gradient Descent: Compute the gradient on the tangent space of the Stiefel manifold and update Q along a geodesic curve. This is more complex to implement.
Parameterization: Re-parameterize Q using unconstrained variables (e.g., using the Cayley transform or exponential map), though this can complicate the optimization landscape.
Penalty Term: Add a term like λ∥QTQ−In​∥2 to the loss. This encourages but doesn't strictly enforce orthonormality. Projection (Method 1) is often the most practical approach when using standard optimization libraries.
Gradient w.r.t. B:
The parameter B influences the loss L through the latent trajectory zk​(t), which depends on A=B[:n,:n]−B[:n,:n]T. Using the chain rule:
∂B∂L​=k,t∑​∂zk​(t)∂L​∂A∂zk​(t)​∂B∂A​
Calculating ∂A∂zk​(t)​ requires differentiating through the dynamics integration. This is typically handled by:
Backpropagation Through Time (BPTT): If using a discrete-time approximation (like Euler: zt+Δt​=zt​+ΔtAzt​), BPTT can be applied.
Adjoint Sensitivity Method: For continuous-time dynamics z˙=Az, the adjoint method computes the gradient ∂A∂L​ efficiently without needing to store intermediate states. This is the method underlying libraries like torchdiffeq. Modern automatic differentiation libraries (PyTorch, TensorFlow) can automatically compute these gradients ∂A∂L​ when the forward pass involves a differentiable ODE solver. The final step is calculating ∂B∂A​. Given A=B′−(B′)T where B′=B[:n,:n], the gradient ∂B′∂L​obtained from the autograd library needs to be mapped back to B. Let ∇B′​L=∂B′∂L​. The gradient w.r.t. the relevant block of B is: (∇B​L)[:n,:n]=∇B′​L−(∇B′​L)T The gradient for the rest of B (elements Bij​where i≥n or j≥n) is zero. In practice, defining A as B[:n,:n] - B[:n,:n].T within a PyTorch model allows the autograd engine to correctly compute ∇B​L automatically when loss.backward() is called.
B. Pseudocode for LCM Fitting Algorithm
This appendix provides Python code using PyTorch and the torchdiffeq library to implement the LCM fitting process. torchdiffeq provides differentiable ODE solvers necessary for backpropagation through the dynamics.
Python
# Required installations:
# pip install torch torchdiffeq numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint # Use odeint_adjoint for memory efficiency

class LatentCircuitModel(nn.Module):
    """
    Latent Circuit Model (LCM) implementation using PyTorch.

    Assumes dynamics: dz/dt = A z
    Observation: y = Q z
    where A is skew-symmetric: A = B_prime - B_prime.T
    and Q is an orthonormal matrix: Q.T @ Q = I
    B_prime is the top-left (n x n) block of a larger learnable matrix B (N x n).
    """
    def __init__(self, N: int, n: int):
        """
        Initializes the LCM.

        Args:
            N (int): Observation dimension (e.g., number of neurons).
            n (int): Latent dimension (n << N).
        """
        super().__init__()
        if n >= N:
            print("Warning: Latent dimension n should be smaller than observation dimension N.")

        self.N = N
        self.n = n

        # Initialize learnable matrix B (N x n).
        # A = B[:n, :n] - B[:n, :n].T will be derived from this.
        self.B = nn.Parameter(torch.randn(N, n) * 0.1) # Scaled random init

        # Initialize observation matrix Q (N x n) with orthonormal columns.
        q_init = torch.randn(N, n)
        q_ortho, _ = torch.linalg.qr(q_init)
        self.Q = nn.Parameter(q_ortho) # Q is learnable

    def _get_latent_dynamics_matrix(self) -> torch.Tensor:
        """Computes the skew-symmetric latent dynamics matrix A."""
        B_prime = self.B[:self.n, :self.n] # Top-left n x n block
        A = B_prime - B_prime.T
        return A

    def latent_dynamics_func(self, t: float, z: torch.Tensor) -> torch.Tensor:
        """
        Defines the differential equation dz/dt = A z.
        Required format for torchdiffeq.odeint.

        Args:
            t (float): Time (required by odeint, but not used in autonomous linear system).
            z (torch.Tensor): Latent state tensor (shape: batch_size, n).

        Returns:
            torch.Tensor: Time derivative dz/dt (shape: batch_size, n).
        """
        A = self._get_latent_dynamics_matrix()
        # Batched matrix multiplication: (batch_size, n) @ (n, n).T
        dzdt = z @ A.T # Equivalent to A @ z for each row vector in batch
        return dzdt

    def forward(self, z0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        """
        Integrates latent dynamics and computes observations.

        Args:
            z0 (torch.Tensor): Initial latent state (shape: batch_size, n).
            t_eval (torch.Tensor): Time points at which to evaluate the solution
                                   (shape: num_time_points).

        Returns:
            torch.Tensor: Predicted observations y_pred at t_eval points
                          (shape: batch_size, num_time_points, N).
        """
        # Integrate latent dynamics: z(t) shape: (num_time_points, batch_size, n)
        # Use odeint_adjoint for potentially better memory usage during training
        z_t = odeint_adjoint(self.latent_dynamics_func, z0, t_eval, method='dopri5')

        # Reshape z_t to (batch_size, num_time_points, n) for batch processing
        z_t = z_t.permute(1, 0, 2)

        # Project latent states to observation space: y = Q z
        # Q shape: (N, n) -> Q.T shape: (n, N)
        # y_pred = z_t @ Q.T results in (batch_size, num_time_points, N)
        y_pred = z_t @ self.Q.T

        return y_pred

    @torch.no_grad()
    def orthogonalize_Q(self):
        """
        Enforces orthonormality constraint on Q using SVD projection.
        Should be called after each optimizer step if Q is learnable.
        """
        U, _, Vh = torch.linalg.svd(self.Q.data, full_matrices=False)
        self.Q.data = U @ Vh

    def estimate_initial_state(self, y0: torch.Tensor) -> torch.Tensor:
        """
        Estimates initial latent state z0 from initial observation y0.
        Simple projection: z0 = y0 @ Q

        Args:
            y0 (torch.Tensor): Initial observation (shape: batch_size, N).

        Returns:
            torch.Tensor: Estimated initial latent state (shape: batch_size, n).
        """
        z0 = y0 @ self.Q # Project observation onto latent basis
        return z0

# --- Training Loop Example ---
def train_lcm(model: LatentCircuitModel,
              data_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: nn.Module,
              num_epochs: int,
              device: torch.device,
              l2_lambda: float = 0.0, # Weight decay strength for B matrix
              print_every: int = 10):
    """
    Trains the Latent Circuit Model.

    Args:
        model (LatentCircuitModel): The LCM model instance.
        data_loader (DataLoader): DataLoader providing batches of (y_true, t_eval).
        optimizer (Optimizer): PyTorch optimizer.
        loss_fn (nn.Module): Loss function (e.g., nn.MSELoss).
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on ('cpu' or 'cuda').
        l2_lambda (float): L2 regularization strength applied to B.
        print_every (int): Frequency of printing training progress.
    """
    model.to(device)
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (y_true_batch, t_eval_batch) in enumerate(data_loader):
            y_true_batch = y_true_batch.to(device)
            t_eval_batch = t_eval_batch.to(device)

            y0_batch = y_true_batch[:, 0, :]
            t_eval_for_ode = t_eval_batch

            with torch.no_grad():
                z0_batch = model.estimate_initial_state(y0_batch)

            optimizer.zero_grad()
            y_pred_batch = model(z0_batch, t_eval_for_ode)

            reconstruction_loss = loss_fn(y_pred_batch, y_true_batch)

            l2_reg_loss = torch.tensor(0.).to(device)
            if l2_lambda > 0:
                l2_reg_loss = l2_lambda * torch.norm(model.B, p=2)**2

            loss = reconstruction_loss + l2_reg_loss

            loss.backward()
            optimizer.step()

            # Enforce Orthonormality on Q
            model.orthogonalize_Q()

            total_loss += reconstruction_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Reconstruction Loss: {avg_loss:.6f}")

    print("Training finished.")


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    N_dim = 50
    n_dim = 10
    num_trials = 64
    seq_len = 100
    batch_size = 16
    learning_rate = 1e-3
    l2_reg = 1e-4
    epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Generate Synthetic Data ---
    print("Generating synthetic data...")
    true_Q = torch.linalg.qr(torch.randn(N_dim, n_dim))[0]
    true_A = torch.randn(n_dim, n_dim)
    true_A = true_A - true_A.T # Skew-symmetric
    t_eval_tensor = torch.linspace(0, 1, seq_len, device=device)
    z0_true = torch.randn(num_trials, n_dim, device=device) * 2

    def true_dynamics(t, z):
        return z @ true_A.T.to(device)

    with torch.no_grad():
        z_true = odeint(true_dynamics, z0_true, t_eval_tensor).permute(1, 0, 2)
        y_true = z_true @ true_Q.T.to(device)
        noise_std = 0.1
        y_true += torch.randn_like(y_true) * noise_std
    print("Synthetic data generated.")

    # --- Create DataLoader ---
    from torch.utils.data import TensorDataset, DataLoader

    class CustomDataLoader:
        def __init__(self, dataset, t_eval, batch_size, shuffle=True):
            self.dataset = dataset
            self.t_eval = t_eval
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sampler = torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
            self.batch_sampler = torch.utils.data.BatchSampler(self.sampler, batch_size, drop_last=False)

        def __iter__(self):
            for indices in self.batch_sampler:
                batch_y = self.dataset[indices][0]
                yield batch_y, self.t_eval
        def __len__(self):
            return len(self.batch_sampler)

    dataset = TensorDataset(y_true)
    data_loader = CustomDataLoader(dataset, t_eval_tensor, batch_size=batch_size, shuffle=True)

    # --- Model, Loss, Optimizer ---
    lcm_model = LatentCircuitModel(N=N_dim, n=n_dim).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': lcm_model.B, 'weight_decay': l2_reg},
        {'params': lcm_model.Q, 'weight_decay': 0.0}
    ], lr=learning_rate)

    # --- Train ---
    print("Starting training...")
    train_lcm(model=lcm_model,
              data_loader=data_loader,
              optimizer=optimizer,
              loss_fn=mse_loss,
              num_epochs=epochs,
              device=device,
              l2_lambda=l2_reg,
              print_every=10)

    # --- Post-Training Analysis ---
    learned_A = lcm_model._get_latent_dynamics_matrix().detach().cpu()
    print("\nLearned A matrix (skew-symmetric):")
    print(learned_A)

C. Implementation Details and Hyperparameter Settings
This section provides details relevant to the Python implementation in Appendix B and typical hyperparameter choices for the proposed studies in Section 6.
Software & Libraries:
Python (>= 3.8 recommended).
PyTorch (>= 1.8 recommended) for automatic differentiation and neural network components.
torchdiffeq library for efficient and differentiable ODE solvers.
NumPy for numerical operations (often used alongside PyTorch).
Model Implementation (Appendix B):
Parameterization: The latent dynamics matrix A is parameterized as A=B′−(B′)T, where B′ is the top n×n block of a learnable N×n matrix B. This ensures A is skew-symmetric. B is initialized with small random values.
Observation Matrix Q: Parameterized as an N×n nn.Parameter. Initialized by taking the QR decomposition of a random N×n matrix to ensure initial orthonormality. It is treated as learnable.
Orthonormality: The QTQ=In​ constraint is enforced after each optimizer step using SVD projection (orthogonalize_Q method).
Dynamics Integration: Continuous dynamics z˙=Az are integrated using torchdiffeq.odeint_adjoint. The dopri5 adaptive step-size solver is a common default choice. odeint_adjoint is preferred over odeint during training for memory efficiency.
Initial State Estimation: The initial latent state z0​ for each trial is estimated by projecting the initial observation y0​ onto the current estimate of the latent subspace: z0​=y0​Q. This is done without gradient tracking (torch.no_grad()) for simplicity during training, focusing learning on B and Q.
Optimization & Training:
Optimizer: Adam [Kingma & Ba, 2014] is used.
Loss Function: Mean Squared Error (MSE) between predicted observations Qz(t) and true observations y(t).
Regularization: L2 weight decay is applied to the parameter matrix B (either via the optimizer or as an explicit loss term). Typical λ values range from 10−5 to 10−3. No weight decay is applied to Q.
Batching: Training uses mini-batches of trials.
Learning Rate: Typical range 10−4 to 10−3. May require tuning or scheduling.
Epochs: 100 to 1000+ epochs, often determined by early stopping on a validation set.
ODE Solver Tolerances: rtol (relative tolerance) and atol (absolute tolerance) for adaptive solvers (e.g., 10−5, 10−6) affect accuracy vs. speed.
Hyperparameter Selection (Relevant to Section 6 Studies):
Latent Dimensionality n: Crucial. Chosen via cross-validation (reconstruction error) or information criteria (BIC/AIC). Range explored typically based on PCA (e.g., n∈[2,30]).
Learning Rate η: Selected from e.g., {10−4,5×10−4,10−3}.
Weight Decay λ (for B): Selected from e.g., {0,10−5,10−4,10−3}.
Batch Size: Depends on memory (e.g., 16, 32, 64).
Initialization: PCA-based Q initialization can speed up convergence.
Perturbation Application (Study 3 / HCDM):
The perturbation δW=QδQT requires the learned Q from the fitted LCM.
The latent perturbation δ∈Rn×n is designed based on analysis of the learned A.
Applying δW in HCDM involves adding it (scaled by α) to relevant weight matrices or state update rules.



