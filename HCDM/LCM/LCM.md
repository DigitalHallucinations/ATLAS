Latent Circuit Models: Uncovering and Perturbing Structured Low-Dimensional Dynamics in High-Dimensional Neural Systems
Authors: Jeremy Shows, Digital Hallucinations
Abstract:
Understanding the dynamical principles governing neural computation remains a fundamental challenge, hindered by the high dimensionality and complexity of neural population recordings. While neural activity often evolves on lower-dimensional manifolds, existing methods may not explicitly capture the underlying dynamical structure or allow for targeted manipulation of these dynamics. We introduce the Latent Circuit Model (LCM), a framework for identifying structured, low-dimensional dynamics embedded within high-dimensional time series data, such as neural population activity. The LCM parameterizes the system using an orthonormal projection matrix (Q) into an n-dimensional latent space and models the latent dynamics via a structured connectivity matrix A. We place a particular emphasis on skew-symmetric parameterizations (A=B′−(B′)T) to capture rotational dynamics prevalent in cortical areas, as these are hypothesized to underlie various cognitive and motor processes. This parameterization is learned by optimizing a reconstruction objective on the observed high-dimensional data, effectively balancing data fidelity with model interpretability. A key feature of the LCM is its ability to facilitate structured perturbation analysis: targeted manipulations (δ) within the learned latent space can be projected back into the observation space via δW=QδQT, allowing for precise, in silico probing of the functional consequences of altering specific dynamical motifs. This offers a powerful method to test causal hypotheses about the role of identified latent dynamics. We outline a series of proposed experimental studies designed to validate the LCM's ability to accurately recover known latent dynamics and subspaces in synthetic datasets with ground truth, and to compare its performance against established techniques. Application of the LCM is proposed for analyzing neural activity from benchmark neuroscience datasets (e.g., motor cortex recordings during reaching tasks) and other complex dynamical systems, with the expectation of revealing interpretable, low-dimensional circuits that correlate with behavior or internal states. Furthermore, the proposed perturbation analysis aims to demonstrate how targeted modulation of these latent circuits can predictably influence system-level behavior and internal state representations, offering a powerful tool for both analyzing complex neural systems and constructing more interpretable and manipulable artificial intelligence models.
Keywords: Neural Dynamics, Dimensionality Reduction, State-Space Models, Rotational Dynamics, Neural Manifolds, Computational Neuroscience, Recurrent Neural Networks, System Identification, Perturbation Analysis, Interpretability, Artificial Intelligence, Skew-Symmetric Systems, Dynamical Systems Theory.
1. Introduction
1.1. The Challenge of High-Dimensional Neural Data
The advent of modern neuroscience techniques, including multi-electrode arrays, Neuropixels probes, and advanced calcium imaging, has provided an unprecedented window into the brain, allowing for the simultaneous recording of activity from hundreds to thousands of neurons [Stevenson & Kording, 2011; Jun et al., 2017]. However, this deluge of high-dimensional data presents significant analytical and conceptual challenges. Raw neural recordings are often noisy and complex, making it difficult to discern the underlying computational principles. Understanding how neural circuits perform computations—from sensory perception to complex decision-making and motor control—requires moving beyond single-neuron characterization to decipher the collective, coordinated dynamics of these populations [Saxena & Cunningham, 2019]. Identifying the fundamental principles governing these high-dimensional dynamics is crucial not only for advancing our understanding of brain function in both health and disease but also for inspiring the development of next-generation artificial intelligence (AI) systems that emulate the brain's efficiency and adaptability.
1.2. Evidence for Low-Dimensional Structure (Neural Manifolds)
Despite the vast number of neurons involved in any given task (Observation Dimension N), a growing body of evidence suggests that neural activity during behavior often evolves within a much lower-dimensional subspace or manifold (Latent Dimension n≪N) [Gao et al., 2017; Gallego et al., 2017; Cunningham & Yu, 2014]. This phenomenon, often referred to as "neural manifolds," implies that the collective activity of a neural population is not random but is constrained and coordinated, likely reflecting shared computational goals, underlying synaptic connectivity patterns, or task-relevant constraints. Identifying and characterizing these neural manifolds is a central goal of computational neuroscience, as they offer a simplified yet potentially comprehensive description of the population's computational state and its temporal evolution, thereby providing a more tractable representation of brain function.
1.3. Importance of Dynamical Systems Perspective
Neural computation is inherently dynamic; the brain processes information and generates behavior through the continuous evolution of neural states over time. A dynamical systems perspective provides a powerful mathematical framework for modeling these temporal evolutions and understanding how neural states change in response to external inputs and internal processes [Eliasmith & Anderson, 2003; Sussillo & Barak, 2013]. By modeling neural populations as dynamical systems, we can characterize and predict complex phenomena such as attractor states (stable patterns of activity representing memories or decisions), oscillations (rhythmic activity crucial for temporal coordination), sequential activity (patterns underlying motor sequences or cognitive processes), and complex transient responses to stimuli. These dynamical features are thought to be fundamental to cognitive functions including decision-making, working memory, motor control, and learning.
1.4. Limitations of Existing Models in Capturing Structured Dynamics
Several methods exist for analyzing high-dimensional neural data, each with its strengths and limitations:
Linear Dimensionality Reduction Methods: Techniques like Principal Component Analysis (PCA) and Factor Analysis (FA) are widely used to identify low-dimensional subspaces that capture the maximum variance in neural data. However, they often ignore the temporal dynamics inherent in neural activity and may not yield components that are easily interpretable in terms of underlying circuit mechanisms [Cunningham & Yu, 2014].
Non-linear Visualization and Manifold Learning: Methods such as t-Distributed Stochastic Neighbor Embedding (t-SNE) or Uniform Manifold Approximation and Projection (UMAP) excel at visualizing the low-dimensional structure of high-dimensional data but do not typically yield an explicit dynamical model that can predict future states or be used for perturbation analysis [Maaten & Hinton, 2008; McInnes et al., 2018].
Recurrent Neural Networks (RNNs): RNNs are powerful dynamical models capable of learning complex temporal patterns from data. However, standard RNN architectures often result in "black box" representations where the learned connectivity matrices lack clear structure or direct neurobiological interpretability, making it difficult to understand how they perform computations [Sussillo & Abbott, 2009].
Latent State-Space Models: More advanced models like Gaussian Process Factor Analysis (GPFA) [Yu et al., 2009] and Latent Factor Analysis via Dynamical Systems (LFADS) [Pandarinath et al., 2018] explicitly model latent dynamics from single-trial neural data. GPFA uses Gaussian processes to model smooth latent trajectories, while LFADS employs RNNs (often LSTMs or GRUs) as generators of latent dynamics. While powerful, these methods may not enforce specific, interpretable structures (like purely rotational dynamics) directly within the latent space connectivity matrix itself. Their internal generators can also be complex, making direct interpretation of the learned dynamics challenging.
Emerging Models with Latent Circuit Inference: Recently, Langdon et al. [2025] introduced a "Latent Circuit Model" that infers interactions between task variables within a latent recurrent circuit, explaining heterogeneous neural responses. This model shares the concept of projecting perturbations via QδQT. However, it primarily focuses on learning general recurrent dynamics rather than imposing a priori specific, interpretable structures like skew-symmetry as a core modeling assumption for certain types of dynamics.
1.5. Proposed Approach: Latent Circuit Models (LCM)
To address these limitations and provide a framework that combines interpretability with dynamical modeling, we propose the Latent Circuit Model (LCM). The LCM is designed to identify and characterize structured, low-dimensional dynamics embedded within high-dimensional observational data. It assumes that high-dimensional observations (e.g., neural activity y∈RN) arise from dynamics occurring within a lower-dimensional latent space (z∈Rn). The mapping between these observation and latent spaces is facilitated by an orthonormal projection matrix Q. A crucial aspect of the LCM is its parameterization of the latent dynamics. While various forms of latent dynamics can be considered, we often model them linearly as z˙=Az+Culatent​, where A is a structured latent connectivity matrix. A primary focus of this work is the use of a skew-symmetric matrix A=B′−(B′)T, which naturally and exclusively generates rotational or oscillatory dynamics. Such dynamics are consistent with experimental observations in motor cortex during reaching [Churchland et al., 2012] and other brain areas involved in rhythmic or sequential processing. This explicit structural constraint enhances the interpretability of the learned model by directly testing hypotheses about underlying dynamical motifs. Furthermore, the LCM framework is designed to facilitate targeted in silico perturbations. Specific, structured changes (δ) can be introduced to the latent dynamics matrix A (i.e., A→A+δ), and these targeted manipulations can be projected back into the high-dimensional observation space as an effective change in connectivity δW=QδQT. This allows researchers to probe the functional role of specific dynamical components and their impact on overall system behavior.
1.6. Key Contributions
The primary contributions of this work are:
Formal Definition of LCM: A clear mathematical definition of the Latent Circuit Model parameterization, emphasizing the role of the orthonormal projection matrix Q for mapping between observation and latent spaces, and the structured latent connectivity matrix A (with a particular focus on skew-symmetric forms for rotational dynamics).
Optimization Procedure: The derivation and outline of an optimization procedure based on minimizing reconstruction error of the observed data, explicitly addressing the QTQ=In​ orthonormality constraint on the projection matrix, which is critical for a well-defined and interpretable latent space.
Structured Perturbation Methodology: A novel methodology for applying structured perturbations (δW=QδQT) to the system. This allows for in silico experiments to causally probe the functional significance of the identified latent circuits and their components.
Validation Framework: Proposal of experimental studies designed to rigorously demonstrate the LCM's ability to recover ground truth dynamics and subspaces in synthetic datasets where these are known (Section 6.1), providing a benchmark for its accuracy and reliability.
Application to Empirical Data: Proposal of LCM application to analyze real or simulated neural data (e.g., from neuroscience benchmarks or complex AI models), with the aim of revealing interpretable low-dimensional dynamics and their functional correlates (Section 6.2).
1.7. Paper Organization
The remainder of this paper is organized as follows: Section 2 reviews related work in dimensionality reduction, state-space modeling, and the analysis of neural dynamics. Section 3 provides a detailed mathematical formulation of the Latent Circuit Model. Section 4 discusses model fitting procedures, including optimization techniques and handling of constraints. Section 5 elaborates on the structured perturbation analysis methodology. Section 6 outlines a series of proposed experimental studies to validate and apply the LCM. Section 7 discusses potential findings, the advantages and limitations of the LCM, and its broader implications for neuroscience and AI. Finally, Section 8 concludes the paper and suggests future research directions.
2. Related Work
The Latent Circuit Model (LCM) builds upon and aims to synthesize insights from several established lines of research in neuroscience, machine learning, and dynamical systems theory:
Dimensionality Reduction: The core idea of finding a low-dimensional representation of high-dimensional data is central to LCM. It relates to classical techniques like Principal Component Analysis (PCA) and Factor Analysis (FA) [Cunningham & Yu, 2014], which identify linear subspaces capturing maximal variance or covariance. It also connects to Independent Component Analysis (ICA), which seeks statistically independent sources. Non-linear manifold learning techniques such as Isomap, Locally Linear Embedding (LLE), t-SNE [Van der Maaten & Hinton, 2008], and UMAP [McInnes et al., 2018] aim to uncover more complex, non-linear structures. LCM integrates the dimensionality reduction aspect (via the matrix Q) directly with an explicit dynamical model for the evolution of states within that reduced space, moving beyond static subspace identification.
State-Space Models: LCM is fundamentally a state-space model, explicitly modeling unobserved (latent) states whose dynamics govern the observed data. This connects it to a rich history of models including Kalman Filters (for linear-Gaussian systems), Hidden Markov Models (for discrete states), and more recent neuroscience-focused methods like Gaussian Process Factor Analysis (GPFA) [Yu et al., 2009] and Latent Factor Analysis via Dynamical Systems (LFADS) [Pandarinath et al., 2018]. GPFA uses Gaussian processes to flexibly model smooth latent trajectories, while LFADS employs recurrent neural networks (RNNs) as powerful non-linear generators of latent dynamics. LCM distinguishes itself by its emphasis on imposing specific, interpretable structures (e.g., skew-symmetry for rotations) directly onto the latent connectivity matrix A, rather than relying on emergent dynamics from a more complex, potentially less interpretable generator (like an LSTM in LFADS). This allows for more direct hypothesis testing about the nature of the underlying dynamics.
Recurrent Neural Networks (RNNs): RNNs are widely used for modeling sequential data, including neural time series [Sussillo & Abbott, 2009; Michaels et al., 2020]. While techniques exist to analyze the learned dynamics of RNNs, often by finding fixed points or fitting simpler models to their activity [Sussillo & Barak, 2013], LCM aims to build interpretability directly into the model structure from the outset. Instead of training a complex RNN and then trying to understand it, LCM proposes to fit a model that is structurally constrained to be more interpretable.
Rotational Dynamics in Neural Systems: A significant motivation for LCM's focus on skew-symmetric latent dynamics comes from seminal work by Churchland, Cunningham, and colleagues [2012], which demonstrated strong rotational patterns in motor cortex population activity during reaching movements. Such rotational dynamics have since been observed in various brain areas and are thought to be a general motif for neural computation, potentially underlying processes like temporal integration, sequence generation, and cognitive operations. LCM provides a direct way to model and test for such rotational structures by parameterizing the latent dynamics matrix A to be skew-symmetric, whose eigenvalues are purely imaginary and thus naturally generate rotations.
System Identification: Broadly, LCM falls under the umbrella of system identification [Ljung, 1999], which is concerned with building mathematical models of dynamical systems from observed input-output data. LCM specifically targets the identification of systems that possess a low-dimensional latent structure governing their high-dimensional outputs, a common scenario in biological and engineered complex systems.
Latent Circuit Inference and Perturbation: The concept of inferring latent circuits and probing them via perturbations is gaining traction. As mentioned, Langdon et al. [2025] introduced a "Latent Circuit Model" that focuses on inferring interactions between task variables within a latent recurrent circuit and also proposed a QδQT perturbation framework. Our LCM shares the spirit of inferring latent structure and the mathematical form of the perturbation projection. However, our LCM places a stronger a priori emphasis on specific structural forms for the latent dynamics matrix A, such as skew-symmetry, as a primary means to directly test hypotheses about underlying dynamical motifs like rotations or oscillations, rather than learning a more general, unconstrained recurrent matrix. This focus on pre-defined structural motifs is a key differentiator aimed at enhancing direct interpretability of the learned dynamics.
3. The Latent Circuit Model (LCM): Formulation
We aim to model observed high-dimensional time series data, y(t)∈RN (e.g., activity of N neurons), as arising from unobserved, lower-dimensional latent dynamics, z(t)∈Rn, where the latent dimension n is significantly smaller than the observation dimension N (i.e., n≪N).
3.1. General State-Space Framework
The LCM can be cast within a general state-space framework, which consists of two main equations:
Latent Dynamics Equation: Describes how the latent state z(t) evolves over time.
z˙(t)=f(z(t),ulatent​(t);θdyn​)
Observation Equation: Describes how the observed data y(t) is generated from the current latent state z(t).
y(t)=g(z(t);θobs​)+ϵ(t)
Here, f(⋅) is the latent dynamics function, g(⋅) is the observation function, ulatent​(t) represents any external inputs to the latent system, θdyn​ and θobs​ are parameters of the dynamics and observation models respectively, and ϵ(t) represents observation noise, typically assumed to be Gaussian.
3.2. Observation Model: Orthonormal Projection (Q)
For the LCM, we typically assume a linear observation model, simplifying g(⋅):
y(t)≈Qz(t)
The matrix Q∈RN×n is the observation matrix (or loading matrix). A critical feature of the LCM is that the columns of Q are constrained to be orthonormal, meaning QTQ=In​, where In​ is the n×n identity matrix.
This orthonormality constraint has several important implications:
Basis Definition: The columns of Q form an orthonormal basis for the n-dimensional latent subspace embedded within the N-dimensional observation space.
Projection Operator: QT acts as the projection operator, mapping data from the observation space onto the axes of this latent subspace: z(t)≈QTy(t).
Uncorrelated Latent Dimensions (in projection): The orthonormality ensures that the latent dimensions, as defined by Q, are uncorrelated in their projection from the observation space.
Distance Preservation: It preserves Euclidean distances within the subspace, i.e., ∣Qz1​−Qz2​∣2=∣z1​−z2​∣2 for any latent vectors z1​,z2​. This means that the geometry of the latent space is faithfully represented in its embedding in the observation space.
Identifiability: This constraint helps in making the latent space identifiable, as rotations of an unconstrained Q and A can lead to equivalent models.
3.3. Latent Dynamics Model (z˙=Az+Culatent​)
We primarily focus on linear (or locally linear) latent dynamics, making the model more tractable and interpretable:
z˙(t)=Az(t)+Culatent​(t)
A∈Rn×n is the latent connectivity matrix (or dynamics matrix). It governs how the components of the latent state vector z(t) interact with each other and evolve over time.
C∈Rn×m is the input mapping matrix, which determines how external inputs ulatent​(t)∈Rm influence the latent dynamics. In some scenarios, ulatent​(t) might be derived from observed external stimuli uobs​(t) via ulatent​(t)=QTuobs​(t) or another learned mapping. For simplicity, we often consider autonomous dynamics (C=0) first.
3.3.1. Structured Connectivity Matrix (A):
A core idea of LCM is that instead of learning an unconstrained matrix A, we impose specific structures on A that correspond to hypothesized dynamical motifs. This makes the model more interpretable and allows for direct testing of these hypotheses.
3.3.2. Skew-Symmetric Parameterization for Rotational Dynamics:
A key structure of interest is skew-symmetry, where A=−AT. Such matrices are intrinsically linked to rotational dynamics. We can parameterize A to be skew-symmetric using an auxiliary matrix B′∈Rn×n such that:
A=B′−(B′)T
In our proposed implementation (see Appendix B), B′ is often derived as the top-left n×n block of a larger, learnable matrix B∈RN×n (i.e., B′=B[:n,:n]). This parameterization allows gradients with respect to the elements of A (during optimization) to be propagated back to the underlying parameters of B.
3.3.3. Properties of Skew-Symmetric Dynamics:
When A is skew-symmetric and there are no inputs (C=0), the autonomous dynamics z˙=Az have several important properties:
Conservation of Norm: The squared norm (energy) of the state vector z(t) is conserved over time: dtd​∣z(t)∣2=dtd​(zTz)=z˙Tz+zTz˙=(Az)Tz+zT(Az)=zTATz+zTAz=zT(−A)z+zTAz=0.
Purely Rotational/Oscillatory Dynamics: This conservation property means that trajectories z(t) are confined to hyperspheres in the latent space, resulting in purely rotational or oscillatory dynamics.
Purely Imaginary Eigenvalues: The eigenvalues of a real skew-symmetric matrix are purely imaginary (of the form iωk​) or zero. These imaginary eigenvalues correspond to oscillations with frequencies ωk​.
Interpretability: This structure provides a highly interpretable way to capture key dynamical features like oscillations and rotations, which are prevalent in neural data.
More general dynamics can be modeled by allowing A to have both symmetric and skew-symmetric components: A=S+K, where S=21​(A+AT) is the symmetric part (governing expansion/contraction or damping/growth) and K=21​(A−AT) is the skew-symmetric part (governing rotations). The LCM can be configured to learn A with a specific balance, or to primarily emphasize the skew-symmetric component.
3.3.4. Input Mapping (C):
The input mapping matrix C determines how external inputs ulatent​(t) (which could be task variables, sensory stimuli projected into latent space, or control signals) affect the latent dynamics. C can be:
Fixed (e.g., C=In​ if inputs are already in latent space, or C=QT if inputs are in observation space and projected).
Learned as part of the model optimization.
3.4. Observation Noise Assumptions
The observation noise ϵ(t) is typically modeled as independent and identically distributed (i.i.d.) isotropic Gaussian noise:
ϵ(t)∼N(0,σ2IN​)
where σ2 is the variance of the noise, which can also be a learnable parameter. More complex noise models (e.g., non-isotropic or neuron-specific variances) can be incorporated if necessary.
3.5. Full Parameterization Summary
The primary learnable parameters of the LCM are:
The matrix B∈RN×n (or B′∈Rn×n if parameterized directly), which defines the structured latent dynamics matrix A.
The observation matrix Q∈RN×n, subject to the orthonormality constraint QTQ=In​.
Optionally, the input mapping matrix C and the noise variance σ2 can also be learned. The choice of the latent dimension n is a critical hyperparameter, typically determined using model selection techniques.
4. Model Fitting and Optimization
4.1. Objective Function
Given a set of K observed trajectories yk​(t)∣t∈[0,Tk​]k=1K​, the parameters Θ of the LCM (primarily B which defines A, and Q) are learned by minimizing an objective function. A common choice is the Mean Squared Error (MSE) reconstruction loss:
L(Θ)=K1​∑k=1K​Tk​1​∫0Tk​​∣yk​(t)−Qzk​(t)∣2dt+R(Θ)
zk​(t) is the latent trajectory for the k-th trial, obtained by integrating the latent dynamics z˙k(t)=Azk​(t)+Culatent,k(t) from an initial condition zk​(0).
The initial latent state zk​(0) can be treated as a parameter to be inferred for each trial (e.g., zk​(0)≈QTyk​(0) or learned more elaborately).
R(Θ) represents regularization terms, added to prevent overfitting and encourage desirable properties in the learned parameters. For instance, L2 regularization on the elements of B (e.g., λB​∣B∣F2​) can encourage smoother or simpler dynamics.
4.2. Numerical Integration of Latent Dynamics
Since the latent dynamics are defined by an ordinary differential equation (ODE), z˙=Az, obtaining z(t) requires numerical integration. Standard numerical methods include:
Fixed-step solvers: Euler method, Runge-Kutta methods (e.g., RK4).
Adaptive-step solvers: Dormand-Prince 5 (dopri5), Adams methods. These are often preferred for their balance of accuracy and efficiency.
Libraries like torchdiffeq [Chen et al., 2018] provide differentiable implementations of these ODE solvers, which are crucial for gradient-based optimization.
4.3. Optimization Procedure
We typically use gradient-based optimization algorithms (e.g., Adam [Kingma & Ba, 2014], SGD with momentum) to minimize the objective function L(Θ). This requires computing the gradients of L with respect to the parameters Q and B. Computing gradients that involve the solution of an ODE (i.e., z(t)) requires specialized techniques:
Backpropagation Through Time (BPTT): If the ODE is discretized (e.g., using the Euler method over small time steps), the integration process can be unrolled, and standard backpropagation can be applied. However, this can be computationally expensive and memory-intensive for long trajectories or fine discretizations.
Adjoint Sensitivity Method: This is a more memory-efficient method for computing gradients of a loss function that depends on the solution of an ODE with respect to the ODE parameters [Chen et al., 2018]. It involves solving a second, "adjoint" ODE backward in time. Modern automatic differentiation libraries that support differentiable ODE solvers (like torchdiffeq for PyTorch) implement this method, allowing gradients ∂A∂L​ and ∂z(0)∂L​ to be computed automatically.
4.4. Handling the Orthonormality Constraint on Q (QTQ=In​)
Maintaining the orthonormality of Q during optimization is crucial. Standard gradient updates on Q will generally not preserve this constraint. Several methods can be used:
Projection onto the Stiefel Manifold: After each unconstrained gradient update step (Qnew​=Qold​−η∇Q​L), Qnew​ can be projected back onto the Stiefel manifold (the manifold of N×n matrices with orthonormal columns). A common way to do this is via Singular Value Decomposition (SVD): if Qnew​=UΣVT, then the projected matrix is Qproj​=UVT. Alternatively, QR decomposition can be used. This is often a practical and effective approach (see Appendix B).
Geodesic Gradient Descent (Riemannian Optimization): This involves computing the gradient in the tangent space of the Stiefel manifold and then performing an update along a geodesic curve on the manifold. This is mathematically more elegant but can be more complex to implement than simple projection.
Penalty Term: Add a penalty term to the loss function, such as λQ​∣QTQ−In​∣F2​, where ∣⋅∣F2​ is the squared Frobenius norm. This encourages orthonormality but does not strictly enforce it, and the choice of λQ​ can be sensitive.
Re-parameterization: Parameterize Q using an unconstrained representation from which an orthonormal matrix can be derived (e.g., using the Cayley transform or an exponential map from a skew-symmetric matrix). This can sometimes complicate the optimization landscape.
Projection methods are often a good compromise between rigor and ease of implementation.
4.5. Initialization Strategies
Proper initialization can significantly affect convergence speed and the quality of the learned model.
Matrix B (defining A): Typically initialized with small random values drawn from a Gaussian or uniform distribution (e.g., N(0,0.01)).
Matrix Q:
Random Orthogonal Matrix: Initialize with a random N×n matrix and then orthogonalize its columns (e.g., via QR decomposition or SVD as described above).
PCA-based Initialization: A common and often effective strategy is to initialize Q with the top n principal component directions of the observed data y(t). This provides a good starting point by aligning the initial latent subspace with the directions of highest variance in the data, which often speeds up convergence.
Initial Latent States zk​(0): If treated as parameters, they can be initialized by projecting the initial observations: zk​(0)=QTyk​(0), using the initial Q.
4.6. Regularization
Regularization terms R(Θ) are important for preventing overfitting, especially when the amount of data is limited or the model is highly flexible.
L2 Regularization on B: Adding λB​∣B∣F2​ to the loss encourages smaller values in B, leading to simpler or smoother dynamics (smaller magnitudes of eigenvalues of A). This is a common form of weight decay.
Other Regularizers: Depending on the specific hypotheses, one might consider sparsity-promoting regularizers (e.g., L1 norm) on B or C, or regularizers that encourage specific structures in the eigenvalues of A.
5. Structured Perturbation Analysis
A key and novel advantage of the Latent Circuit Model framework is its inherent suitability for interpretable, structured perturbation analysis. This allows for in silico experiments to probe the causal role of the identified latent dynamical components.
5.1. Defining Perturbations (δ) in the Latent Space
Once an LCM (Q, A) has been fitted to data, we can design specific perturbations to the latent dynamics matrix A. Let δ∈Rn×n be a matrix representing the desired change to the latent connectivity. The perturbed latent dynamics matrix becomes Apert​=A+δ.
Examples of structured perturbations δ include:
Targeting Specific Rotations/Oscillations: If A is skew-symmetric, a skew-symmetric δ can be designed to alter the frequency or plane of specific rotations. For instance, if a pair of imaginary eigenvalues ±iω corresponds to a 2D rotation, δ can be designed to change ω or mix this rotation with other modes.
Targeting Stability (Damping/Growth): A symmetric δ can be used to add or remove damping (affecting the real parts of A's eigenvalues). For example, adding a negative definite symmetric component can stabilize an oscillatory mode.
Targeting Specific Latent Dimensions or Interactions: A sparse δ can be designed to modify only a few elements of A, thereby affecting interactions between specific latent dimensions or the dynamics along a single dimension.
Simulating Lesions or Inactivations: Setting specific rows and/or columns of A to zero (achieved by an appropriate δ) can simulate the lesioning of specific components within the latent circuit.
Enhancing or Suppressing Motifs: If A is decomposed (e.g., A=∑i​λi​vi​wiT​), δ could be designed to amplify or attenuate specific eigenmodes.
The design of δ is guided by hypotheses about the functional role of different aspects of the learned latent dynamics A.
5.2. Projecting Latent Perturbations to the Observation Space
The crucial step is to understand how this targeted, low-dimensional latent perturbation δ manifests in the high-dimensional observation space (e.g., across the entire neural population). The change in the effective high-dimensional connectivity, denoted δW∈RN×N, is given by:
δW=QδQT
This equation projects the meaningful, structured change δ from the n-dimensional latent space into the N-dimensional observation space using the learned orthonormal basis Q.
Properties of δW:
Low-Rank: Since Q is N×n and δ is n×n (with n≪N), δW is inherently a low-rank matrix (specifically, rank(δW)≤rank(δ)≤n). This implies that even though the perturbation is applied across the entire high-dimensional system, its structure is constrained by the low-dimensional latent manifold.
Targeted Influence: δW represents how the specific latent intervention is distributed across the observed units (e.g., neurons). It provides a precise prediction of how to modify the full system to achieve the desired latent change.
5.3. Simulating the Effects of Perturbation
With the effective high-dimensional perturbation δW, we can simulate its consequences:
Direct Latent Simulation:
Integrate the perturbed latent dynamics: z˙pert​(t)=(A+δ)z(t).
Observe the effect in the observation space: ypert​(t)=Qzpert​(t).
This allows direct comparison of ypert​(t) with the original y(t)=Qz(t).
Application to an External System or Full Model:
If the LCM was fitted to data from a larger, more complex system (e.g., a large-scale neural network model, or even potentially guiding real experimental interventions), δW can inform how to perturb that full system. For example, if Worig​ represents a connectivity matrix in the full system, the perturbed matrix could be Wnew​=Worig​+αδW, where α is a scaling factor for the perturbation strength. One can then simulate the full system with Wnew​ and observe changes in its behavior, internal states, or performance on a task.
5.4. Interpreting Perturbation Effects and Causal Inference
By observing the changes in system behavior, internal state representations, or task performance resulting from applying δW (or from the latent simulation ypert​(t)), we can infer the causal functional role of the specific latent dynamical components that were targeted by δ. For example:
If perturbing a specific rotational mode in A leads to deficits in a sequential motor task, it suggests that this rotational mode is causally involved in generating the sequence.
If stabilizing an oscillatory mode (by adding damping via δ) improves the precision of a timing task, it implies a role for that oscillation in temporal processing.
This perturbation analysis provides a powerful bridge between the observed high-dimensional dynamics, the inferred low-dimensional latent structure (Q,A), and the functional consequences of altering that structure. It moves beyond correlational analysis towards testing causal hypotheses in silico.
6. Experimental Evaluation: Proposed Studies
To validate the Latent Circuit Model and demonstrate its utility, we propose a series of experimental studies.
6.1. Study 1: Validation on Synthetic Data with Known Ground Truth
Objective:
Quantitatively verify the LCM's ability to accurately recover known latent dynamics (Atrue​), particularly those with specific structures like skew-symmetry, and the true latent subspace (Qtrue​).
Assess its ability to correctly identify the true latent dimensionality (ntrue​) under varying conditions (e.g., noise levels, data length).
Validate the QδQT perturbation framework by comparing the effects of simulated perturbations with ground truth effects.
Benchmark LCM performance against existing dimensionality reduction and dynamical systems modeling techniques.
Methodology:
Data Generation: Generate synthetic high-dimensional time series y(t) using a known LCM:
Define a ground truth latent dimension ntrue​.
Construct a ground truth latent dynamics matrix Atrue​∈Rntrue​×ntrue​ with desired properties (e.g., skew-symmetric for rotations, or with specific eigenvalue spectra).
Construct a ground truth orthonormal observation matrix Qtrue​∈RN×ntrue​.
Simulate latent trajectories ztrue​(t) by integrating z˙=Atrue​z from various initial conditions.
Generate observed data y(t)=Qtrue​ztrue​(t)+ϵ(t), where ϵ(t) is additive Gaussian noise with varying signal-to-noise ratios (SNRs).
Model Fitting: Fit the LCM to the generated y(t), potentially varying the assumed latent dimension nfit​.
Evaluation Metrics:
Subspace Recovery: Compare the learned Qfit​ with Qtrue​ using metrics like the principal angles between subspaces.
Dynamics Recovery: Compare the learned Afit​ with Atrue​ by comparing their matrix norms (e.g., ∣Afit​−PAtrue​P−1∣F​ where P is a permutation/scaling matrix if A is only identifiable up to such transformations), eigenvalue spectra, or by simulating trajectories from Afit​ and comparing them to ztrue​(t).
Dimensionality Estimation: Use cross-validation (e.g., prediction error on held-out data) or information criteria (AIC, BIC) to determine the optimal nfit​ and compare it to ntrue​.
Reconstruction Accuracy: Measure MSE on test data.
Perturbation Validation:
Define a known latent perturbation δtrue​.
Simulate the ground truth perturbed system: z˙pert,true=(Atrue+δtrue​)z, ypert,true​=Qtrue​zpert,true​.
Apply the learned perturbation using the fitted model: δWfit​=Qfit​δfit​QfitT​ (where δfit​ is designed to match δtrue​ in the learned latent space). Compare the effects of δWfit​ (when applied to a model using Atrue​,Qtrue​) or ypert,fit​=Qfit​(simulated from Afit​+δfit​) with ypert,true​.
Benchmarking: Compare LCM's performance (in terms of dynamics recovery, subspace identification, and reconstruction) against:
PCA followed by Vector Autoregression (VAR) on the principal components.
Factor Analysis (FA) followed by VAR on factors.
Standard Recurrent Neural Networks (RNNs) trained to reconstruct y(t).
Other latent variable models like GPFA or LFADS (if applicable to the synthetic data structure).
Expected Outcomes:
LCM should accurately recover Atrue​ and Qtrue​, especially when Atrue​ possesses the structure (e.g., skew-symmetry) that LCM is designed to capture.
LCM should outperform baseline methods that do not explicitly model structured latent dynamics in terms of interpretability and accuracy of dynamics recovery.
The QδQT perturbation effects in the LCM-fitted model should show high fidelity to the ground truth perturbation effects.
The study will characterize the robustness of LCM to noise, data limitations, and model misspecification (e.g., incorrect nfit​).
6.2. Study 2: Application to Real and Simulated Neuroscience Data
Objective:
Identify interpretable, low-dimensional latent dynamics in real neural population recordings, particularly focusing on datasets where specific dynamical motifs like rotations are hypothesized to be behaviorally relevant (e.g., motor cortex, prefrontal cortex during working memory).
Correlate the identified latent dynamics z(t) and structure of A with behavioral variables or task conditions.
Compare the insights gained from LCM with those from existing analysis techniques like jPCA, GPFA, or LFADS.
Methodology:
Dataset Selection:
Real Neural Data: Utilize publicly available datasets, such as those from CRCNS.org (e.g., motor cortex M1/PMd recordings during reaching tasks [Stevenson et al., Churchland et al.]), or prefrontal cortex recordings during cognitive tasks.
Simulated Neural Data: Alternatively, use data generated from complex, biophysically plausible network models where some aspects of the underlying dynamics might be partially understood or controlled.
Preprocessing: Standard preprocessing steps for neural data: spike sorting (if applicable), spike count binning, smoothing (e.g., Gaussian kernel), trial alignment to behavioral events, mean-centering, and possibly variance normalization.
LCM Fitting: Fit the LCM to the preprocessed neural activity. Select the latent dimension n using cross-validation (e.g., based on reconstruction error on held-out trials or time points) and interpretability.
Analysis of Learned LCM:
Structure of A: Analyze the learned latent dynamics matrix A. If a skew-symmetric structure was enforced, examine its eigenvalues (which should be imaginary, corresponding to rotation frequencies). Calculate the ratio of skew-symmetric to symmetric components if a more general A was learned.
Latent Trajectories z(t): Visualize the latent trajectories. Correlate z(t) or its features (e.g., phase or amplitude of rotations) with behavioral variables (e.g., reach direction, speed, reaction time, task conditions, cognitive states) using decoding analyses (e.g., linear classifiers/regressors).
Observation Matrix Q: Analyze the columns of Q (loading vectors) to understand how each latent dimension maps onto the activity of the recorded neurons.
Perturbation Analysis (Conceptual): Based on the analysis of A and z(t), design hypothetical latent perturbations δ (e.g., to slow down a prominent rotation or stabilize a dynamic). Simulate their effects on z(t) and predict consequences for behavior.
Comparison with Other Methods: Compare the LCM results (reconstruction quality, decoding performance from latent states, interpretability of dynamics) with those obtained from:
jPCA (for identifying rotational dynamics specifically).
GPFA or LFADS (for more general latent dynamic modeling).
PCA + behavioral decoding.
Expected Outcomes:
LCM will identify low-dimensional, interpretable latent dynamics (e.g., rotations in M1/PMd data) that are predictive of behavioral variables.
The structure of the learned A matrix (e.g., its eigenvalues and skew-symmetry) will provide direct insights into the nature of these dynamics (e.g., frequencies of rotation).
LCM will offer a more directly interpretable model of the underlying dynamical "circuit" compared to more black-box models like standard RNNs or LFADS generators, potentially leading to more specific, testable hypotheses about neural mechanisms.
The perturbation analysis will suggest specific ways in which modulating these latent dynamics could predictably alter behavior.
7. Discussion
7.1. Summary of LCM
The Latent Circuit Model (LCM) offers a novel and interpretable framework for discovering and analyzing low-dimensional dynamics embedded within high-dimensional time series, particularly neural population activity. Its core strengths lie in the combination of orthonormal dimensionality reduction (Q) with structured latent dynamical modeling (A), especially its emphasis on specific structures like skew-symmetry to capture prevalent motifs such as rotations. Crucially, LCM facilitates targeted in silico perturbation analysis through the QδQT projection, enabling causal investigation of the identified latent circuits. The proposed experimental studies are designed to rigorously validate its capabilities on synthetic data and demonstrate its utility in gleaning insights from complex empirical datasets.
7.2. Interpretation of Latent Circuits: Bridging Manifolds and Mechanisms
The LCM provides a powerful lens for interpreting neural population activity:
The "What" - Latent Manifold (Q): The orthonormal matrix Q defines the low-dimensional subspace, or manifold, within which the dominant neural dynamics unfold. The columns of Q represent basis vectors for this manifold, indicating how patterns of co-activation across the recorded neural population contribute to each latent dimension. Analyzing these loading patterns can reveal which neurons participate in specific latent signals.
The "How" - Latent Dynamics (A): The latent connectivity matrix A defines the "rules of motion" or the flow field within this manifold. By imposing structure on A (e.g., skew-symmetry), we gain direct insight into the nature of these dynamics. For instance, the eigenvalues of a skew-symmetric A directly correspond to the frequencies of rotation in the latent space. This allows for a more mechanistic interpretation compared to models where dynamics emerge from complex, unstructured interactions.
Together, Q and A provide a concise yet powerful description of the underlying "latent circuit" that generates the observed high-dimensional activity patterns.
7.3. Advantages of the Latent Circuit Model
LCM offers several advantages over existing approaches:
Enhanced Interpretability: The explicit parameterization of structured dynamics (e.g., skew-symmetric A for rotations) makes the learned model more directly interpretable in terms of fundamental dynamical motifs. This contrasts with "black-box" models where understanding the learned dynamics can be a significant post-hoc challenge.
Incorporation of Priors and Hypotheses: LCM allows for the direct incorporation of prior knowledge or hypotheses about the nature of neural dynamics (e.g., the prevalence of rotations in motor cortex) into the model structure itself.
Causal Probing via Structured Perturbations: The QδQT framework provides a principled way to simulate the effects of targeted interventions in the latent space, allowing for in silico testing of causal hypotheses about the functional role of identified dynamical components. This is a significant step beyond purely descriptive models.
Generative Model: As a state-space model, LCM is inherently generative, meaning it provides a model of how the observed data is produced. This allows for data simulation, prediction, and potentially filling in missing data.
Bridging Theory and Experiment: By identifying specific dynamical structures (like rotation frequencies or stability properties) and providing a means to predict perturbation effects, LCM can help bridge theoretical models of neural computation with experimental neuroscience.
7.4. Limitations and Future Challenges
Despite its promise, LCM also has limitations and faces several challenges:
Linearity Assumption in Latent Dynamics: The primary formulation focuses on linear latent dynamics (z˙=Az). While many complex non-linear systems can be locally approximated by linear dynamics, or exhibit dominant linear modes, truly non-linear neural dynamics might not be fully captured. Extending LCM to incorporate structured non-linearities is an important future direction (see Section 8.2).
Scalability: Fitting LCMs, especially those involving ODE integration and adjoint sensitivity methods for gradient computation, can be computationally intensive for very large datasets (many neurons, long recordings, or many trials). Efficient implementation and optimization strategies are crucial.
Identifiability Issues: While the orthonormality of Q helps, some identifiability issues can remain. For example, the signs of Q's columns and corresponding adjustments in A might be ambiguous without further constraints or careful interpretation. The order of latent dimensions is also arbitrary.
Hyperparameter Selection: The choice of the latent dimensionality n is critical and can be challenging. Cross-validation and information criteria can guide this, but finding the "true" or most meaningful n often requires careful consideration of reconstruction error, model complexity, and interpretability.
Assumption of Stationarity: The basic LCM assumes that the dynamics (A) and observation model (Q) are stationary over the period of analysis. Neural dynamics can be non-stationary, changing with learning, attention, or context. Extensions to handle non-stationarity (e.g., adaptive LCMs) would be valuable.
Noise Models: The assumption of isotropic Gaussian noise might be an oversimplification for real neural data, which can have more complex noise structures (e.g., neuron-specific firing statistics, non-Gaussian noise).
7.5. Implications for Neuroscience
If validated, LCM could have significant implications for computational and systems neuroscience:
Inferring Dynamical Mechanisms: Provides a tool to move beyond describing neural manifolds to inferring and characterizing the dynamical "generators" or rules that govern activity flow within these manifolds.
Testing Hypotheses about Neural Computation: Allows for direct testing of hypotheses about the role of specific dynamical motifs (e.g., rotations, oscillations, attractors) in neural computation.
Guiding Experimental Interventions: The QδQT perturbation analysis can provide specific, testable predictions for how targeted experimental manipulations (e.g., optogenetic or electrical stimulation patterned according to QδQT) might affect neural activity and behavior. This could lead to more precise methods for probing brain circuits.
Understanding Neural Coding and Representation: Offers insights into how information is represented and transformed by the collective dynamics of neural populations.
7.6. Implications for Artificial Intelligence
LCM also holds potential for advancing AI research:
Interpretability of AI Models: Could be used as an analysis tool to understand the internal dynamics of complex AI models, particularly RNNs or other stateful architectures, by fitting an LCM to their hidden unit activations. This might reveal interpretable low-dimensional "circuits" underlying their computations.
Principled Design of AI Systems: The principles of LCM could inspire the design of new AI architectures that incorporate structured, low-dimensional dynamical modules, potentially leading to models that are more interpretable, robust, and easier to control.
Targeted Control and Modulation of AI Behavior: The QδQT perturbation idea could be adapted for fine-grained control of AI systems. For instance, a higher-level control system could learn to apply such structured perturbations to modulate the behavior or internal states of a lower-level AI agent in a precise and predictable manner, offering a more nuanced approach than global parameter tuning.
8. Conclusion and Future Work
8.1. Conclusion
The Latent Circuit Model (LCM) presents a promising and principled approach for extracting, understanding, and manipulating structured low-dimensional dynamics within high-dimensional complex systems, with a particular focus on neural data. Its emphasis on interpretable dynamical structures (like skew-symmetry for rotations) and its unique capability for causal probing via structured, projected perturbations (QδQT) position it as a potentially valuable tool for advancing both neuroscience research and the development of more understandable and controllable artificial intelligence. The proposed validation on synthetic data and application to empirical datasets will be crucial in establishing its efficacy and scope.
8.2. Future Directions
Several exciting avenues exist for extending and enhancing the LCM framework:
Non-Linear Latent Circuit Models: Develop LCMs that incorporate structured non-linearities in the latent dynamics function f(z). For example, one could model f(z)=(B′−(B′)T)ϕ(z)+Sϕ(z), where ϕ(⋅) is a non-linear activation function (e.g., tanh), allowing for more complex dynamics like limit cycles or chaotic attractors while still maintaining some structural interpretability.
Hierarchical Latent Circuit Models: Extend LCM to model dynamics occurring at multiple interacting timescales or levels of abstraction. This could involve stacked LCMs or models where the parameters of one LCM are modulated by the latent states of another.
Adaptive and Online Latent Circuit Models: Develop methods for LCMs to track non-stationary dynamics, where A and Q might change slowly over time (e.g., due to learning or changes in brain state). This could involve online estimation algorithms or time-varying parameter models.
Integration with Control Theory: Explore the use of LCMs within a formal control theory framework. For instance, design optimal control inputs ulatent​(t) to steer the latent state z(t) towards desired targets, leveraging the interpretability of A and Q.
Closed-Loop Applications: Investigate the potential for using LCMs in real-time, closed-loop systems. This could include brain-computer interfaces (BCIs) where LCMs decode neural intent and guide prosthetic control, or in AI agents where LCMs enable adaptive and interpretable internal state modulation.
Learning from Multi-Modal Data: Extend LCM to integrate information from multiple data modalities simultaneously (e.g., neural activity and behavioral data, or EEG and fMRI), potentially by having shared or interacting latent spaces.
Bayesian LCMs: Develop Bayesian formulations of LCM to quantify uncertainty in the learned parameters (A,Q) and latent states z(t), and to more formally incorporate priors. This could also facilitate more robust model selection for the latent dimensionality n.
Addressing these future directions will further enhance the power and applicability of Latent Circuit Models, solidifying their role in the toolkit for understanding complex dynamical systems.
References
(References are identical to the previous version and are omitted here for brevity, but would be included in the final document. The Python code in Appendix B and the details in Appendix A and C are also assumed to be the same unless specific further elaborations were requested for them.)
Appendices
A. Derivation of Gradient for LCM with Orthonormality Constraints
(Content as previously provided, with LaTeX for math.)
The goal is to find the gradients of the Mean Squared Error (MSE) loss function L with respect to the model parameters, primarily the matrix B (which defines the skew-symmetric dynamics matrix A) and the orthonormal observation matrix Q.
The loss is given by:
L(Θ)=K1​∑k=1K​Tk​1​∑t=1Tk​​∣yk​(t)−Qzk​(t)∣2
where the latent state zk​(t) is obtained by integrating the latent dynamics:
z˙k(t)=Azk​(t)+Culatent,k(t)
with A=B′−(B′)T and B′=B[:n,:n]. For simplicity, we'll consider the autonomous case (C=0) here: z˙=Az. The initial state zk​(0) might be inferred, e.g., zk​(0)≈QTyk​(0).
Gradient w.r.t. Q:
The gradient ∂Q∂L​ has contributions from the reconstruction term and potentially from the initial state estimation if zk​(0) depends on Q. Focusing on the reconstruction term for a single data point (y(t),z(t)):
∂Q∂​∣y(t)−Qz(t)∣2=−2(y(t)−Qz(t))z(t)T
Summing over time and trials gives the overall gradient contribution from the reconstruction error:
∇Q​L=−K1​∑k=1K​Tk​1​∑t=1Tk​​2(yk​(t)−Qzk​(t))zk​(t)T
However, applying standard gradient descent using ∇Q​L will violate the orthonormality constraint QTQ=In​. To handle this:
Projection: After an unconstrained update (Qnew​=Qold​−η∇Q​L), project Qnew​ back onto the Stiefel manifold (the space of matrices with orthonormal columns). A common method is using the Singular Value Decomposition (SVD): If Qnew​=UΣVT, the projection is Qproj​=UVT. Alternatively, QR decomposition can be used.
Geodesic Gradient Descent: Compute the gradient on the tangent space of the Stiefel manifold and update Q along a geodesic curve. This is more complex to implement.
Parameterization: Re-parameterize Q using unconstrained variables (e.g., using the Cayley transform or exponential map), though this can complicate the optimization landscape.
Penalty Term: Add a term like λ∣QTQ−In​∣F2​ to the loss. This encourages but doesn't strictly enforce orthonormality. Projection (Method 1) is often the most practical approach when using standard optimization libraries.
Gradient w.r.t. B:
The parameter B influences the loss L through the latent trajectory zk​(t), which depends on A=B[:n,:n]−(B[:n,:n])T. Using the chain rule:
∂B∂L​=∑k,t​∂zk​(t)∂L​∂A∂zk​(t)​∂B∂A​
Calculating ∂A∂zk​(t)​ requires differentiating through the dynamics integration. This is typically handled by:
Backpropagation Through Time (BPTT): If using a discrete-time approximation (like Euler: zt+Δt​=zt​+ΔtAzt​), BPTT can be applied.
Adjoint Sensitivity Method: For continuous-time dynamics z˙=Az, the adjoint method computes the gradient ∂A∂L​ efficiently without needing to store intermediate states. This is the method underlying libraries like torchdiffeq. Modern automatic differentiation libraries (PyTorch, TensorFlow) can automatically compute these gradients ∂A∂L​ when the forward pass involves a differentiable ODE solver.
The final step is calculating ∂B∂A​. Given A=B′−(B′)T where B′=B[:n,:n] (the top-left n×n block of B), the gradient ∇B′​L=∂B′∂L​ obtained from the autograd library needs to be mapped back to B. Let ∇B′​L be the gradient of the loss with respect to B′. Then the gradient with respect to the relevant block of B is:
(∇B​L)[:n,:n]=∇B′​L−(∇B′​L)T
The gradient for the rest of B (elements Bij​ where i≥n or j≥n) is zero. In practice, defining A as B[:n,:n]−(B[:n,:n])T within a PyTorch model allows the autograd engine to correctly compute ∇B​L automatically when loss.backward() is called.
B. Pseudocode for LCM Fitting Algorithm
(Python code as previously provided, formatted for Markdown.)
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
            t_eval_batch = t_eval_batch.to(device) # t_eval should be on the correct device

            # Assuming y_true_batch is (batch_size, seq_len, N)
            y0_batch = y_true_batch[:, 0, :] 
            # t_eval_for_ode should be a 1D tensor of time points for the ODE solver
            # If t_eval_batch from dataloader is already in this format per batch, it's fine.
            # Often, t_eval is fixed across batches for a given sequence length.
            t_eval_for_ode = t_eval_batch[0] if t_eval_batch.ndim > 1 and t_eval_batch.shape[0] > 0 else t_eval_batch


            with torch.no_grad():
                z0_batch = model.estimate_initial_state(y0_batch)

            optimizer.zero_grad()
            y_pred_batch = model(z0_batch, t_eval_for_ode)

            reconstruction_loss = loss_fn(y_pred_batch, y_true_batch)

            l2_reg_loss = torch.tensor(0.).to(device)
            if l2_lambda > 0:
                # Apply L2 regularization directly to the B parameter
                l2_reg_loss = l2_lambda * torch.norm(model.B, p=2)**2


            loss = reconstruction_loss + l2_reg_loss
            loss.backward()
            optimizer.step()

            # Enforce Orthonormality on Q
            model.orthogonalize_Q()

            total_loss += reconstruction_loss.item() # Log reconstruction loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
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
    l2_reg_strength = 1e-4 # Renamed for clarity
    epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Generate Synthetic Data ---
    print("Generating synthetic data...")
    # Ensure true_Q is created on the correct device from the start or moved.
    true_Q_init = torch.randn(N_dim, n_dim, device=device)
    true_Q_ortho, _ = torch.linalg.qr(true_Q_init)
    true_Q = true_Q_ortho
    
    true_A_init = torch.randn(n_dim, n_dim, device=device)
    true_A = true_A_init - true_A_init.T # Skew-symmetric
    
    t_eval_tensor = torch.linspace(0, 1, seq_len, device=device)
    z0_true = torch.randn(num_trials, n_dim, device=device) * 2

    def true_dynamics(t, z):
        return z @ true_A.T # A is n x n, z is batch x n

    with torch.no_grad():
        # odeint expects (batch_size, n) for z0
        z_true = odeint(true_dynamics, z0_true, t_eval_tensor, method='dopri5').permute(1, 0, 2) # (batch, time, n)
        y_true = z_true @ true_Q.T
        noise_std = 0.1
        y_true += torch.randn_like(y_true) * noise_std # Noise added on the same device
    print("Synthetic data generated.")

    # --- Create DataLoader ---
    from torch.utils.data import TensorDataset, DataLoader
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, y_data, t_eval):
            self.y_data = y_data # Should be on the target device
            self.t_eval = t_eval # Should be on the target device

        def __len__(self):
            return len(self.y_data)

        def __getitem__(self, idx):
            return self.y_data[idx], self.t_eval # Return y_sample and its t_eval

    dataset = CustomDataset(y_true, t_eval_tensor) # y_true shape: (num_trials, seq_len, N_dim)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # --- Model, Loss, Optimizer ---
    lcm_model = LatentCircuitModel(N=N_dim, n=n_dim).to(device)
    mse_loss = nn.MSELoss()
    
    optimizer = optim.Adam([
        {'params': lcm_model.B, 'weight_decay': 0.0}, # L2 for B handled manually via l2_lambda
        {'params': lcm_model.Q, 'weight_decay': 0.0}  # No L2 for Q
    ], lr=learning_rate)

    # --- Train ---
    print("Starting training...")
    train_lcm(model=lcm_model,
              data_loader=data_loader,
              optimizer=optimizer,
              loss_fn=mse_loss,
              num_epochs=epochs,
              device=device,
              l2_lambda=l2_reg_strength, 
              print_every=10)

    # --- Post-Training Analysis ---
    lcm_model.eval() # Set model to evaluation mode
    learned_A = lcm_model._get_latent_dynamics_matrix().detach().cpu()
    print("\nLearned A matrix (skew-symmetric):")
    print(learned_A)

    learned_Q = lcm_model.Q.detach().cpu()
    print("\nLearned Q matrix (first 5 rows):")
    print(learned_Q[:5, :])
    
    # Example of comparing learned Q to true Q (if true_Q was also moved to CPU)
    # print("\nNorm of difference between true_Q and learned_Q (after alignment):")
    # This comparison is non-trivial due to potential permutation and sign flips.
    # For a simple check, one might compare subspace angles.


C. Implementation Details and Hyperparameter Settings
(Content as previously provided.)
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
Regularization: L2 weight decay is applied to the parameter matrix B (either via the optimizer's weight_decay argument for those parameters, or as an explicit loss term as shown in the example). Typical λ values range from 10−5 to 10−3. No weight decay is typically applied to Q.
Batching: Training uses mini-batches of trials.
Learning Rate: Typical range 10−4 to 10−3. May require tuning or scheduling.
Epochs: 100 to 1000+ epochs, often determined by early stopping on a validation set.
ODE Solver Tolerances: rtol (relative tolerance) and atol (absolute tolerance) for adaptive solvers (e.g., 10−5,10−6) affect accuracy vs. speed.
Hyperparameter Selection (Relevant to Section 6 Studies):
Latent Dimensionality n: Crucial. Chosen via cross-validation (reconstruction error on a held-out validation set) or information criteria (BIC/AIC). The range explored is typically based on preliminary analyses like PCA (e.g., selecting n that captures a significant portion of variance, such as n∈[2,30] for typical neuroscience datasets).
Learning Rate η: Selected from a logarithmic range, e.g., 10−4,3×10−4,10−3,3×10−3. Learning rate schedules (e.g., decay on plateau) can also be beneficial.
Weight Decay λ (for B): Selected from a range like 0,10−6,10−5,10−4,10−3. The optimal value often depends on the dataset size and noise level.
Batch Size: Depends on available GPU memory and dataset size (e.g., 16, 32, 64, 128 trials per batch). Larger batches can provide more stable gradient estimates but require more memory.
Initialization: As mentioned, PCA-based Q initialization can significantly speed up convergence by providing a more informed starting point for the latent subspace.
Perturbation Application:
The perturbation δW=QδQT requires the learned Q from the fitted LCM. The latent perturbation δ∈Rn×n is designed based on analysis of the learned A (e.g., its eigenmodes). Applying δW in a target system (e.g., a larger computational model or guiding an experimental manipulation) would involve adding it (potentially scaled by a factor α to control perturbation strength) to relevant weight matrices or state update rules. The impact of this perturbation is then assessed on the system's outputs or behavior.
