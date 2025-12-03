# graph_attention_fusion.py

"""
Enterprise-grade fusion module that constructs a graph over multiple modality embeddings.
It uses a Graph Attention Network (GAT) to propagate and refine modality-specific features,
and then aggregates these features via multi-head self-attention. The design is modular,
scalable, and production–ready.

Techniques used:
  • Graph–Based Fusion: Constructs a fully–connected graph (with learned attention) over modality embeddings.
  • Graph Attention (GAT): Updates each node's features by attending to its neighbors.
  • Self–Attention: Aggregates node features via multi–head self–attention using a configurable pooling strategy.

Example:
    $ python graph_attention_fusion.py --batch_size 2 --visual_dim 512 --text_dim 768 --audio_dim 128 --fusion_dim 256

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025 (Improved Mar 02 2025)
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Configure enterprise–level logging
logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up proper logging with configurable level and optional file output.
    
    Args:
        level (int): Logging level (default: logging.INFO)
        log_file (str, optional): Path to log file for persistent logging
    """
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class GraphAttentionLayer(nn.Module):
    """
    Batched Graph Attention Layer (GAT) with TorchScript support.

    Implements a single attention layer as described in:
      "Graph Attention Networks" (https://arxiv.org/abs/1710.10903).

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        dropout (float): Dropout probability.
        alpha (float): Negative slope for LeakyReLU.
        concat (bool): If True, applies ELU after aggregation; otherwise, returns linear output.
        use_layer_norm (bool): If True, applies layer normalization after attention.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout: float = 0.1, 
        alpha: float = 0.2, 
        concat: bool = True,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.dropout: float = dropout
        self.alpha: float = alpha
        self.concat: bool = concat
        self.use_layer_norm: bool = use_layer_norm

        # Linear transformation parameters
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(out_features) if use_layer_norm else nn.Identity()
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        logger.debug(f"Initialized GraphAttentionLayer with in_features={in_features}, out_features={out_features}")

    def _compute_attention(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute raw attention scores from concatenated node features.
        
        Args:
            z (torch.Tensor): Concatenated node features of shape (batch, N, N, 2*out_features)
            
        Returns:
            torch.Tensor: Raw attention scores of shape (batch, N, N)
        """
        return self.leakyrelu(torch.matmul(z, self.a).squeeze(-1))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batched inputs.

        Args:
            h (torch.Tensor): Node features of shape (batch, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch, N, N) or (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (batch, N, out_features).
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        
        # Linear projection: (batch, N, out_features)
        Wh = self.W(h)
        
        # Prepare concatenated pairs for attention
        Wh1 = Wh.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, self.out_features)
        Wh2 = Wh.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, self.out_features)
        attention_input = torch.cat([Wh1, Wh2], dim=-1)
        
        # Compute attention coefficients
        e = self._compute_attention(attention_input)
        
        # Apply adjacency mask
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            
        neg_inf = torch.finfo(e.dtype).min
        zero_vec = neg_inf * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize with softmax and apply dropout
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to get new node features
        h_prime = torch.bmm(attention, Wh)
        
        # Apply layer normalization if enabled
        h_prime = self.layer_norm(h_prime)
        
        if self.concat:
            h_prime = F.elu(h_prime)
            
        return h_prime


class GraphAttentionFusion(nn.Module):
    """
    Graph Attention Fusion Module.

    Fuses modality embeddings by:
      1. Projecting each embedding to a common fusion space.
      2. Constructing a complete graph over the modalities.
      3. Applying multiple Graph Attention Layers (GAT) to update node features.
      4. Refining the aggregated node features via multi-head self-attention.

    Args:
        input_dims (List[int]): List of input dimensions for each modality.
        fusion_dim (int): Target fusion dimension.
        gat_dropout (float): Dropout probability for GAT layers.
        gat_alpha (float): Negative slope for LeakyReLU in GAT layers.
        num_heads (int): Number of GAT layers (interpreted as attention heads).
        self_attn_heads (int): Number of heads for the subsequent self-attention layer.
        self_attn_dropout (float): Dropout probability for the self-attention layer.
        pooling (str): Pooling strategy for aggregating node features ("mean", "sum", or "learned").
        use_layer_norm (bool): Whether to use layer normalization in GAT layers.
        use_residual (bool): Whether to use residual connections between GAT layers.
        use_checkpointing (bool): Whether to use gradient checkpointing for memory efficiency.
        modality_names (List[str], optional): Names for each modality for better logging.
    """
    def __init__(
        self, 
        input_dims: List[int], 
        fusion_dim: int = 256, 
        gat_dropout: float = 0.1,
        gat_alpha: float = 0.2, 
        num_heads: int = 4, 
        self_attn_heads: int = 4,
        self_attn_dropout: float = 0.1, 
        pooling: str = "mean",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_checkpointing: bool = False,
        modality_names: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.num_modalities: int = len(input_dims)
        self.fusion_dim: int = fusion_dim
        self.num_heads: int = num_heads
        self.pooling: str = pooling.lower()
        self.use_residual: bool = use_residual
        self.use_checkpointing: bool = use_checkpointing
        
        if self.pooling not in {"mean", "sum", "learned"}:
            raise ValueError("Pooling must be one of: 'mean', 'sum', or 'learned'")
        
        self.modality_names = modality_names or [f"modality_{i}" for i in range(self.num_modalities)]
        if len(self.modality_names) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modality names, got {len(self.modality_names)}")

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])

        self.register_buffer("adjacency", self._create_full_adjacency(self.num_modalities))

        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                fusion_dim, 
                fusion_dim, 
                dropout=gat_dropout, 
                alpha=gat_alpha, 
                concat=True,
                use_layer_norm=use_layer_norm
            ) for _ in range(num_heads)
        ])
        
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        if self.pooling == "learned":
            self.pool_weights = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=self_attn_heads,
            dropout=self_attn_dropout, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
        logger.info(f"Initialized GraphAttentionFusion with {self.num_modalities} modalities: {self.modality_names}")
        logger.debug(f"Configuration: fusion_dim={fusion_dim}, num_heads={num_heads}, pooling={pooling}")

    def _create_full_adjacency(self, num_nodes: int) -> torch.Tensor:
        """
        Create a complete adjacency matrix with zeros on the diagonal.

        Args:
            num_nodes (int): Number of modalities/nodes.

        Returns:
            torch.Tensor: Adjacency matrix of shape (num_nodes, num_nodes).
        """
        adj = torch.ones((num_nodes, num_nodes))
        adj.fill_diagonal_(0)
        return adj

    def _apply_gat_layer(self, gat_layer: nn.Module, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Wrapper for GAT layer to support checkpointing"""
        if self.use_checkpointing and self.training:
            return checkpoint(gat_layer, x, adj)
        return gat_layer(x, adj)

    def _aggregate_modalities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool node features according to the selected strategy.
        
        Args:
            x (torch.Tensor): Node features of shape (batch, num_modalities, fusion_dim)
            
        Returns:
            torch.Tensor: Pooled features of shape (batch, 1, fusion_dim)
        """
        if self.pooling == "mean":
            return torch.mean(x, dim=1, keepdim=True)
        elif self.pooling == "sum":
            return torch.sum(x, dim=1, keepdim=True)
        elif self.pooling == "learned":
            weights = F.softmax(self.pool_weights, dim=0)
            weighted = x * weights.view(1, -1, 1)
            return torch.sum(weighted, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def forward(
        self, 
        embeddings: List[Optional[torch.Tensor]],
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Fuse the input modality embeddings.

        Args:
            embeddings (List[Optional[torch.Tensor]]): List of tensors (batch, input_dim).
            return_attention (bool): If True, return attention weights for interpretability.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Dict]: 
                - Fused embedding with shape (batch, fusion_dim)
                - Optionally, a dictionary of attention weights
        """
        if len(embeddings) != self.num_modalities:
            msg = f"Expected {self.num_modalities} embeddings, got {len(embeddings)}"
            logger.error(msg)
            raise ValueError(msg)

        projected = []
        for idx, (proj, emb, name) in enumerate(zip(self.projections, embeddings, self.modality_names)):
            if emb is None:
                msg = f"Embedding for modality '{name}' (index {idx}) is missing."
                logger.error(msg)
                raise ValueError(msg)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            projected.append(proj(emb))
        
        x = torch.stack(projected, dim=1)
        batch_size = x.size(0)
        
        adj = self.adjacency.expand(batch_size, -1, -1)
        
        node_features = x
        attentions = {}
        
        head_outputs = []
        for i, gat_layer in enumerate(self.gat_layers):
            gat_out = self._apply_gat_layer(gat_layer, node_features, adj)
            head_outputs.append(gat_out)
            if self.use_residual and i > 0:
                gat_out = gat_out + node_features
            node_features = gat_out
        
        norm_weights = F.softmax(self.head_weights, dim=0)
        head_stack = torch.stack(head_outputs, dim=0)
        aggregated = torch.sum(head_stack * norm_weights.view(-1, 1, 1, 1), dim=0)
        
        if return_attention:
            attentions["head_weights"] = norm_weights.detach()
            if self.pooling == "learned":
                attentions["modality_weights"] = F.softmax(self.pool_weights, dim=0).detach()
        
        query = self._aggregate_modalities(aggregated)
        
        attn_output, attn_weights = self.self_attn(query, aggregated, aggregated)
        
        if return_attention:
            attentions["self_attention"] = attn_weights.detach()
        
        refined = self.layer_norm(attn_output.squeeze(1))
        output = self.output_projection(refined)
        
        logger.debug(f"Forward pass completed with output shape: {output.shape}")
        
        if return_attention:
            return output, attentions
        return output

    def to_torchscript(self, example_inputs: List[torch.Tensor]) -> torch.jit.ScriptModule:
        """
        Convert the model to a TorchScript module for optimized deployment.

        Args:
            example_inputs (List[torch.Tensor]): Example input tensors for tracing.

        Returns:
            torch.jit.ScriptModule: TorchScript-compiled model.
        """
        logger.info("Converting GraphAttentionFusion to TorchScript")
        self.eval()
        original_checkpointing = self.use_checkpointing
        self.use_checkpointing = False
        
        try:
            scripted_module = torch.jit.trace(self, example_inputs)
            scripted_module = torch.jit.optimize_for_inference(scripted_module)
            logger.info("TorchScript conversion successful")
            return scripted_module
        except Exception as e:
            logger.exception(f"TorchScript conversion failed: {e}")
            raise
        finally:
            self.use_checkpointing = original_checkpointing


def save_model(module, path, example_inputs=None, use_scripting=False):
    """
    Save model to disk with proper error handling.
    
    Args:
        module (nn.Module): Module to save
        path (str): Path to save to
        example_inputs (List[torch.Tensor], optional): Example inputs for tracing
        use_scripting (bool): Whether to use scripting instead of tracing
    """
    try:
        if example_inputs is not None:
            if use_scripting:
                scripted = torch.jit.script(module)
            else:
                scripted = module.to_torchscript(example_inputs)
            torch.jit.save(scripted, f"{path}.pt")
            logger.info(f"Saved TorchScript model to {path}.pt")
            
        torch.save(module.state_dict(), f"{path}.pth")
        logger.info(f"Saved model state dict to {path}.pth")
    except Exception as e:
        logger.exception(f"Error saving model: {e}")


def main() -> None:
    """
    Entry point for running a test instance of the GraphAttentionFusion module.
    """
    parser = argparse.ArgumentParser(description="Enterprise Graph Attention Fusion Module")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy inputs")
    parser.add_argument("--visual_dim", type=int, default=512, help="Dimension of visual embeddings")
    parser.add_argument("--text_dim", type=int, default=768, help="Dimension of text embeddings")
    parser.add_argument("--audio_dim", type=int, default=128, help="Dimension of audio embeddings")
    parser.add_argument("--fusion_dim", type=int, default=256, help="Fusion space dimension")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "sum", "learned"], 
                        help="Pooling strategy")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of GAT attention heads")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections")
    parser.add_argument("--use_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model")
    args = parser.parse_args()
    
    setup_logging(level=getattr(logging, args.log_level))
    
    logger.info(f"Initializing with arguments: {args}")

    batch_size = args.batch_size
    visual_embedding = torch.randn(batch_size, args.visual_dim)
    text_embedding = torch.randn(batch_size, args.text_dim)
    audio_embedding = torch.randn(batch_size, args.audio_dim)

    fusion_module = GraphAttentionFusion(
        input_dims=[args.visual_dim, args.text_dim, args.audio_dim],
        fusion_dim=args.fusion_dim,
        gat_dropout=0.1,
        gat_alpha=0.2,
        num_heads=args.num_heads,
        self_attn_heads=4,
        self_attn_dropout=0.1,
        pooling=args.pooling,
        use_residual=args.use_residual,
        use_checkpointing=args.use_checkpointing,
        modality_names=["visual", "text", "audio"]
    )

    logger.info("Running forward pass")
    fused_output, attentions = fusion_module(
        [visual_embedding, text_embedding, audio_embedding], 
        return_attention=True
    )
    
    logger.info(f"Fused output shape: {fused_output.shape}")
    print("Fused output shape:", fused_output.shape)
    
    print("\nAttention Head Weights:")
    print(attentions["head_weights"])
    
    if args.pooling == "learned":
        print("\nModality Weights:")
        print(attentions["modality_weights"])
    
    if args.save_path:
        save_model(
            fusion_module, 
            args.save_path, 
            example_inputs=[visual_embedding, text_embedding, audio_embedding]
        )


if __name__ == "__main__":
    main()




## Files to Modify for Integration

### 1. Main HCDM.py
You'll need to modify the main HCDM.py file to:
- Import the new GraphAttentionFusion module
- Initialize the fusion module with appropriate parameters
- Connect it to relevant data streams in your system

### 2. sensory_processing_module.py
The Sensory Processing Module needs to be updated to:
- Replace the current CrossModalFusion with the new GraphAttentionFusion
- Ensure the output of the module is compatible with the new fusion architecture

### 3. dynamic_attention_routing.py (DAR)
The DAR module needs modifications to:
- Accept and process the outputs from the GraphAttentionFusion module
- Potentially adjust its routing decisions based on the new fusion approach

### 4. neural_cognitive_bus.py
The NCB may require updates to:
- Create additional channels for the GraphAttentionFusion's inputs/outputs
- Support the tensor shapes produced by the new fusion module

## Implementation Plan

### For HCDM.py

```python
# Add to the import section:
from graph_attention_fusion import GraphAttentionFusion

# In the main() function, after initializing core modules:
# Define fusion module parameters
visual_dim = config_dict.get("visual_dim", 512)
text_dim = config_dict.get("text_dim", 768)
audio_dim = config_dict.get("audio_dim", 128)
fusion_dim = config_dict.get("fusion_dim", 256)

# Initialize the fusion module
fusion_module = GraphAttentionFusion(
    input_dims=[visual_dim, text_dim, audio_dim],
    fusion_dim=fusion_dim,
    gat_dropout=0.1,
    gat_alpha=0.2,
    num_heads=4,
    self_attn_heads=4,
    self_attn_dropout=0.1,
    pooling="mean",
    use_layer_norm=True,
    use_residual=True,
    modality_names=["visual", "text", "audio"]
)

# Connect fusion module to sensory processing module
spm.set_fusion_module(fusion_module)
```

### For sensory_processing_module.py

# sensory_processing_module.py (SPM)

"""
This module implements an SPM that:
  • Separates modality processing into dedicated submodules:
      – TextProcessor uses spaCy for robust tokenization and the Hugging Face
        "feature-extraction" pipeline (e.g. DistilBERT) for generating high–quality
        embeddings.
      – VisionProcessor employs a pre–trained ResNet50 from torchvision (with proper
        image preprocessing and projection) for extracting image features.
      – AudioProcessor uses librosa to compute mel spectrograms from raw audio and
        a CNN–based network to project these features to a fixed dimension.
  • Fuses the modality–specific features via an attention–based CrossModalFusion module.
  • Supports advanced fusion via GraphAttentionFusion module.
  • Estimates salience (i.e. importance) of the fused features using a dedicated MLP
    (SalienceEstimator) with proper normalization.
  • Operates asynchronously, gathers inputs (which in a production system would be
    collected from real sensors or APIs), processes them concurrently, and publishes
    the fused features and salience score to a Neural Cognitive Bus (NCB).
  • Uses error handling and detailed logging to facilitate monitoring,
    debugging, and maintenance.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025
Updated: Mar 02 2025 - Added support for GraphAttentionFusion
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Import robust libraries for each modality processing
# -----------------------------------------------------------------------------

# Text processing using spaCy and transformers
try:
    import spacy
    # use a robust spaCy model (e.g., en_core_web_sm or larger)
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    raise ImportError(f"spaCy is required for text processing: {e}")

try:
    from transformers import pipeline
    # Use a state–of–the–art transformer for feature extraction
    feature_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")
except Exception as e:
    feature_extractor = None
    raise ImportError(f"Transformers feature extraction pipeline is required: {e}")

# Vision processing using torchvision and PIL
try:
    from torchvision import models, transforms
    from PIL import Image
except Exception as e:
    raise ImportError(f"Torchvision and PIL are required for vision processing: {e}")

# Audio processing using librosa
try:
    import librosa
except Exception as e:
    raise ImportError(f"Librosa is required for audio processing: {e}")

# Optional import for GraphAttentionFusion
try:
    from graph_attention_fusion import GraphAttentionFusion
except ImportError:
    pass  # Will be handled gracefully if needed

# -----------------------------------------------------------------------------
# Base Processor for modality processors
# -----------------------------------------------------------------------------
class BaseProcessor:
    def __init__(self, output_dim: int, logger: Optional[logging.Logger] = None):
        self.output_dim = output_dim
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    async def process(self, input_data: Any) -> torch.Tensor:
        """
        Process raw input data and return a feature tensor of shape (1, output_dim).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

# -----------------------------------------------------------------------------
# Text Processor
# -----------------------------------------------------------------------------
class TextProcessor(BaseProcessor):
    def __init__(self, output_dim: int = 768, logger: Optional[logging.Logger] = None):
        super().__init__(output_dim, logger)
        if feature_extractor is None:
            raise ImportError("Transformers feature_extraction pipeline is required for TextProcessor.")
        self.feature_extractor = feature_extractor
        if nlp is None:
            self.logger.warning("spaCy model not available; using basic tokenization.")
    
    async def process(self, input_data: Any) -> torch.Tensor:
        """
        Process text input:
          1. Clean and tokenize using spaCy (if available).
          2. Use a transformer (e.g., DistilBERT) to extract token–level features.
          3. Mean–pool the token embeddings to produce a fixed–size vector.
        Returns:
            A tensor of shape (1, output_dim).
        """
        try:
            if not isinstance(input_data, str):
                raise ValueError("TextProcessor expects a string input.")
            text = input_data.strip()
            if nlp:
                doc = nlp(text)
                # Remove stop words and punctuation for robust enterprise processing.
                cleaned_text = " ".join(token.text for token in doc if not token.is_stop and not token.is_punct)
            else:
                cleaned_text = text
            features = self.feature_extractor(cleaned_text)
            if not features:
                raise ValueError("No features extracted from text.")
            # Convert features to a numpy array (shape: [sequence_length, hidden_dim])
            features_np = np.array(features[0])
            # Mean pooling across tokens
            pooled = np.mean(features_np, axis=0)
            # If the pooled vector's dimension does not match output_dim, project it.
            if pooled.shape[0] != self.output_dim:
                projection = nn.Linear(pooled.shape[0], self.output_dim)
                pooled_tensor = torch.tensor(pooled, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pooled = projection(pooled_tensor).squeeze(0).numpy()
            feature_tensor = torch.tensor(pooled, dtype=torch.float32).unsqueeze(0)
            return feature_tensor
        except Exception as e:
            self.logger.error(f"Error in TextProcessor.process: {e}", exc_info=True)
            raise

# -----------------------------------------------------------------------------
# Vision Processor
# -----------------------------------------------------------------------------
class VisionProcessor(BaseProcessor):
    def __init__(self, output_dim: int = 512, logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
        super().__init__(output_dim, logger)
        self.device = device if device is not None else torch.device("cpu")
        try:
            # Load pre-trained ResNet50 and remove its classification head.
            model = models.resnet50(pretrained=True)
            modules = list(model.children())[:-1]  # Remove final FC layer.
            self.feature_extractor = nn.Sequential(*modules).to(self.device)
            self.feature_extractor.eval()
        except Exception as e:
            self.logger.error(f"Error initializing VisionProcessor model: {e}", exc_info=True)
            raise
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Use a projection layer to map the 2048–dim feature to output_dim.
        self.projection = nn.Linear(2048, self.output_dim).to(self.device)
    
    async def process(self, input_data: Any) -> torch.Tensor:
        """
        Process an image input:
          1. Ensure the image is in RGB mode.
          2. Apply the transformation pipeline.
          3. Extract features via the pre-trained ResNet50.
          4. Flatten and project the feature to output_dim.
        Returns:
            A tensor of shape (1, output_dim).
        """
        try:
            if not hasattr(input_data, "convert"):
                raise ValueError("VisionProcessor expects a PIL Image as input.")
            image = input_data.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)
            projected = self.projection(features)
            return projected
        except Exception as e:
            self.logger.error(f"Error in VisionProcessor.process: {e}", exc_info=True)
            raise

# -----------------------------------------------------------------------------
# Audio Processor
# -----------------------------------------------------------------------------
class AudioProcessor(BaseProcessor):
    def __init__(self, output_dim: int = 128, sample_rate: int = 22050, n_mels: int = 64, logger: Optional[logging.Logger] = None):
        super().__init__(output_dim, logger)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        # Define a CNN to process the mel spectrogram and output a fixed–size vector.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projection = nn.Linear(32, output_dim)
    
    async def process(self, input_data: Any) -> torch.Tensor:
        """
        Process an audio waveform:
          1. Compute the mel spectrogram using librosa.
          2. Convert to log–scale and normalize.
          3. Convert the spectrogram to a PyTorch tensor.
          4. Use a CNN to extract features and then project to output_dim.
        Returns:
            A tensor of shape (1, output_dim).
        """
        try:
            if not isinstance(input_data, np.ndarray):
                raise ValueError("AudioProcessor expects a NumPy array as input.")
            mel_spec = librosa.feature.melspectrogram(y=input_data, sr=self.sample_rate, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize the spectrogram
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-9)
            spec_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            features = self.cnn(spec_tensor)
            features = features.view(features.size(0), -1)
            projected = self.projection(features)
            return projected
        except Exception as e:
            self.logger.error(f"Error in AudioProcessor.process: {e}", exc_info=True)
            raise

# -----------------------------------------------------------------------------
# Cross–Modal Fusion Module using Multi–Head Self–Attention
# -----------------------------------------------------------------------------
class CrossModalFusion(nn.Module):
    def __init__(self, input_dims: List[int], fusion_dim: int = 256, num_heads: int = 4, logger: Optional[logging.Logger] = None):
        """
        Fuse modality features using multi–head self–attention.
        Parameters:
            input_dims (List[int]): List of dimensions for each modality's feature vector.
            fusion_dim (int): Target fusion dimension.
            num_heads (int): Number of attention heads.
        """
        super(CrossModalFusion, self).__init__()
        self.logger = logger or logging.getLogger("CrossModalFusion")
        self.num_modalities = len(input_dims)
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads

        # Project each modality feature to the fusion dimension.
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in input_dims
        ])
        # Multi–head attention: we treat the set of projected features as a sequence.
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        # Final projection and layer normalization.
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse the list of modality features.
        Each feature tensor is assumed to be of shape (1, modality_dim).
        Returns:
            A fused feature tensor of shape (1, fusion_dim).
        """
        try:
            projected = []
            for i, feat in enumerate(features):
                proj = self.projections[i](feat)  # (1, fusion_dim)
                projected.append(proj)
            # Stack the projected features into a sequence: shape (1, num_modalities, fusion_dim)
            stacked = torch.cat(projected, dim=0).unsqueeze(0)
            # Use the mean of the stacked features as the query.
            query = torch.mean(stacked, dim=1, keepdim=True)
            attn_output, _ = self.attention(query, stacked, stacked)
            out = self.output_proj(attn_output)
            fused = self.layer_norm(out.squeeze(1))
            return fused
        except Exception as e:
            self.logger.error(f"Error in CrossModalFusion.forward: {e}", exc_info=True)
            raise

# -----------------------------------------------------------------------------
# Salience Estimator
# -----------------------------------------------------------------------------
class SalienceEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, logger: Optional[logging.Logger] = None):
        """
        Estimates the salience (importance) of the fused sensory feature.
        Produces a scalar value in [0, 1].
        """
        super(SalienceEstimator, self).__init__()
        self.logger = logger or logging.getLogger("SalienceEstimator")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            A tensor of shape (1, 1) representing the salience score.
        """
        try:
            salience = self.mlp(fused_feature)
            return salience
        except Exception as e:
            self.logger.error(f"Error in SalienceEstimator.forward: {e}", exc_info=True)
            raise

# -----------------------------------------------------------------------------
# Sensory Processing Module (SPM)
# -----------------------------------------------------------------------------
class SensoryProcessingModule:
    def __init__(self, config: Dict[str, Any], ncb: Any, logger: Optional[logging.Logger] = None, device: Optional[torch.device] = None):
        """
        Initializes the Sensory Processing Module.
        Parameters:
            config (Dict[str, Any]): Configuration parameters (e.g., output dims, processing intervals).
            ncb (NeuralCognitiveBus): An instance of the Neural Cognitive Bus for publishing features.
            logger (logging.Logger, optional): Logger for logging messages.
            device (torch.device, optional): Computation device.
        """
        self.config = config
        self.logger = logger or logging.getLogger("SensoryProcessingModule")
        self.ncb = ncb
        self.device = device if device is not None else torch.device("cpu")

        # Modality output dimensions
        self.text_output_dim = config.get("text_output_dim", 768)
        self.vision_output_dim = config.get("vision_output_dim", 512)
        self.audio_output_dim = config.get("audio_output_dim", 128)
        self.fusion_dim = config.get("fusion_dim", 256)
        self.num_attention_heads = config.get("num_attention_heads", 4)
        self.salience_hidden_dim = config.get("salience_hidden_dim", 64)

        # Initialize modality processors
        self.text_processor = TextProcessor(output_dim=self.text_output_dim, logger=self.logger)
        self.vision_processor = VisionProcessor(output_dim=self.vision_output_dim, logger=self.logger, device=self.device)
        self.audio_processor = AudioProcessor(output_dim=self.audio_output_dim, logger=self.logger)

        # Initialize Cross–Modal Fusion
        self.cross_modal_fusion = CrossModalFusion(
            input_dims=[self.text_output_dim, self.vision_output_dim, self.audio_output_dim],
            fusion_dim=self.fusion_dim,
            num_heads=self.num_attention_heads,
            logger=self.logger
        ).to(self.device)

        # Initialize Salience Estimator
        self.salience_estimator = SalienceEstimator(
            input_dim=self.fusion_dim,
            hidden_dim=self.salience_hidden_dim,
            logger=self.logger
        ).to(self.device)

        # Set up the publishing channel on the NCB (e.g., "sensory_features")
        self.publish_channel = config.get("publish_channel", "sensory_features")
        if self.ncb:
            # The channel dimension is fusion_dim + 1 (for salience)
            self.ncb.create_channel(self.publish_channel, self.fusion_dim + 1)

        # Define processing interval (in seconds)
        self.processing_interval = config.get("processing_interval", 0.2)
        self.running = False
        self.update_task: Optional[asyncio.Task] = None

        # Initialize EFM reference to None (can be connected later)
        self.efm = None

        self.logger.info("SensoryProcessingModule initialized with multi–modal processing.")

    def set_fusion_module(self, fusion_module):
        """
        Replace the default fusion module with a custom fusion module (e.g., GraphAttentionFusion).
        
        Args:
            fusion_module: An instance of a custom fusion module that implements the appropriate interface
        """
        self.cross_modal_fusion = fusion_module.to(self.device)
        self.logger.info(f"Set custom fusion module: {fusion_module.__class__.__name__} as fusion mechanism")

    async def process_and_publish(self, inputs: Dict[str, Any]) -> None:
        """
        Asynchronously processes multi-modal inputs and publishes the fused feature vector
        along with its salience score to the NCB.
        
        Parameters:
            inputs (Dict[str, Any]): Dictionary with keys "text", "vision", "audio".
        """
        try:
            # Launch modality processing concurrently.
            tasks = []
            if "text" in inputs and inputs["text"]:
                tasks.append(asyncio.create_task(self.text_processor.process(inputs["text"])))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=torch.zeros((1, self.text_output_dim), dtype=torch.float32))))
            if "vision" in inputs and inputs["vision"]:
                tasks.append(asyncio.create_task(self.vision_processor.process(inputs["vision"])))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=torch.zeros((1, self.vision_output_dim), dtype=torch.float32))))
            if "audio" in inputs and inputs["audio"] is not None:
                tasks.append(asyncio.create_task(self.audio_processor.process(inputs["audio"])))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=torch.zeros((1, self.audio_output_dim), dtype=torch.float32))))
            
            raw_features = await asyncio.gather(*tasks)
            raw_features = [feat.to(self.device) for feat in raw_features]
            
            # Check if we're using the GraphAttentionFusion module
            if isinstance(self.cross_modal_fusion, GraphAttentionFusion):
                # Format features for GraphAttentionFusion as expected by its interface
                # The GraphAttentionFusion expects a list of tensors in a specific order (visual, text, audio)
                modality_embeddings = [
                    raw_features[1],  # Vision features (index 1)
                    raw_features[0],  # Text features (index 0)
                    raw_features[2]   # Audio features (index 2)
                ]
                
                # Get top-down signal from Executive Function Module (EFM) if available
                top_down_signal = None
                if hasattr(self, "efm") and self.efm is not None:
                    try:
                        top_down_signal = self.efm.get_gating_signal()
                        self.logger.debug(f"Retrieved top-down signal from EFM: {top_down_signal}")
                    except Exception as e:
                        self.logger.error(f"Error getting EFM gating signal: {e}", exc_info=True)
                        
                # Determine whether to return attention weights (for monitoring/debugging)
                return_attention = self.config.get("return_attention", False)
                
                # Fuse features using the graph attention fusion
                if return_attention:
                    fused_feature, attention_weights = self.cross_modal_fusion(
                        modality_embeddings, 
                        return_attention=True
                    )
                    # Log attention statistics for monitoring
                    self.logger.debug(f"GAF attention weights: {attention_weights}")
                else:
                    fused_feature = self.cross_modal_fusion(modality_embeddings)
            else:
                # Use the original CrossModalFusion implementation
                fused_feature = self.cross_modal_fusion(raw_features)

            # Estimate salience.
            salience_tensor = self.salience_estimator(fused_feature)  # (1, 1)
            salience_value = salience_tensor.item()

            # Prepare payload.
            payload = {
                "fused_feature": fused_feature.detach().cpu().numpy().tolist(),
                "salience": salience_value,
                "timestamp": time.time()
            }
            
            # Add modality information to payload for downstream processing
            payload["modalities"] = {
                "text": inputs["text"] is not None,
                "vision": inputs["vision"] is not None,
                "audio": inputs["audio"] is not None
            }
            
            # Publish to NCB.
            if self.ncb:
                await self.ncb.publish(self.publish_channel, payload)
                self.logger.debug(f"Published sensory features with salience: {salience_value:.4f}")
            else:
                self.logger.warning("NCB instance not available; cannot publish sensory features.")
        except Exception as e:
            self.logger.error(f"Error in process_and_publish: {e}", exc_info=True)
            raise

    def connect_to_efm(self, efm):
        """
        Connect to an Executive Function Module to enable top-down modulation
        of sensory processing.
        
        Args:
            efm: An instance of ExecutiveFunctionModule that provides gating signals
                 via a get_gating_signal() method
        """
        self.efm = efm
        self.logger.info(f"Connected to Executive Function Module for top-down modulation")
        
        # Verify that the EFM has the required interface
        if not hasattr(self.efm, "get_gating_signal"):
            self.logger.warning("EFM does not have get_gating_signal method - top-down modulation may not work")

    async def _gather_inputs(self) -> Dict[str, Any]:
        """
        Gathers sensory inputs from external sources. In production, this method would
        asynchronously retrieve data from sensors, cameras, microphones, or external APIs.
        Here, we assume that the implementation is fully integrated.
        Returns:
            Dict[str, Any]: Dictionary with keys "text", "vision", "audio".
        """
        try:
            # In an enterprise system, replace these stubs with actual data retrieval code.
            # For example, retrieve text from a message queue, images from a camera feed,
            # and audio from a microphone stream.
            text_input = "text input obtained from a real–time data source."
            from PIL import Image
            try:
                image_input = Image.open("enterprise_image.jpg")
            except Exception as e:
                self.logger.warning(f"Error loading image: {e}; using a blank image instead.")
                image_input = Image.new("RGB", (224, 224), color="white")
            audio_input = np.random.randn(self.config.get("audio_sample_length", 22050)).astype(np.float32)
            return {"text": text_input, "vision": image_input, "audio": audio_input}
        except Exception as e:
            self.logger.error(f"Error gathering inputs: {e}", exc_info=True)
            return {"text": "", "vision": None, "audio": None}

    async def _processing_loop(self) -> None:
        """
        Main asynchronous loop that gathers inputs from external sources, processes them,
        and publishes the results. In a production system, the _gather_inputs method would
        interface with real sensors or data streams.
        """
        while self.running:
            try:
                inputs = await self._gather_inputs()
                await self.process_and_publish(inputs)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
            await asyncio.sleep(self.processing_interval)

    async def start(self) -> None:
        """
        Starts the SPM processing loop asynchronously.
        """
        if self.running:
            self.logger.warning("SensoryProcessingModule is already running.")
            return
        self.running = True
        self.update_task = asyncio.create_task(self._processing_loop())
        self.logger.info("SensoryProcessingModule processing loop started.")

    async def stop(self) -> None:
        """
        Stops the SPM processing loop gracefully.
        """
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                self.logger.info("SensoryProcessingModule processing loop cancelled gracefully.")
        self.logger.info("SensoryProcessingModule stopped.")
```

### For dynamic_attention_routing.py

#!/usr/bin/env python3
"""
dynamic_attention_routing.py (DAR)

Dynamic Attention Routing (DAR)
=================================

This module implements a production–grade, dynamic multi–route decision mechanism that
integrates environmental context, high–level gating signals from the Executive Function Module (EFM),
and advanced exploration–exploitation modulation. It now optionally leverages a Graph Attention Fusion
module to fuse multi–modality embeddings (e.g. visual, text, audio) before routing decisions are made.
The fused representation is integrated with other continuous features before being processed
by an enhanced context network.

Enhancements:
  • Integration with EFM: External gating signals from EFM can modulate routing logits.
  • Multi–Module Fusion: Optionally fuses multi–modality inputs using a Graph Attention Fusion module.
  • Robust PPO Update: Uses a PPO–style update mechanism with a rollout buffer.
  • Scalability: Both routing and fusion are designed to scale to many channels and modalities.

Author: Jeremy Shows – Digital Hallucinations
Date: Feb 14 2025 (Integrated Mar 02 2025)
"""

import argparse
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Import the improved GraphAttentionFusion module.
# It is assumed to be in the same directory or installed as a package.
from graph_attention_fusion import GraphAttentionFusion

# -----------------------------------------------------------------------------
# Enhanced Context Network that integrates optional fusion features
# -----------------------------------------------------------------------------
class IntegratedEnvContextNet(nn.Module):
    """
    IntegratedEnvContextNet: Processes channel and source IDs, continuous features,
    environmental context, and (optionally) multi–modal fusion features to produce
    routing logits and a critic value.

    The input dimension is computed as:
       (2 * embed_dim) + cont_dim + context_dim + fusion_dim
    where fusion_dim is 0 if no fusion features are provided.
    """
    def __init__(
        self,
        max_channels: int,
        max_sources: int,
        embed_dim: int,
        cont_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_routes: int,
        fusion_dim: int = 0  # Additional dimension from fusion; 0 if not used.
    ):
        super().__init__()
        self.logger = logging.getLogger("IntegratedEnvContextNet")
        self.channel_embedding = nn.Embedding(max_channels, embed_dim)
        self.source_embedding = nn.Embedding(max_sources, embed_dim)
        self.fusion_dim = fusion_dim
        input_dim = (embed_dim * 2) + cont_dim + context_dim + fusion_dim
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.route_head = nn.Linear(hidden_dim, num_routes)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.logger.debug(f"IntegratedEnvContextNet initialized with input_dim={input_dim}")

    def forward(
        self,
        channel_ids: torch.Tensor,
        source_ids: torch.Tensor,
        cont_feats: torch.Tensor,
        env_ctx: torch.Tensor,
        fusion_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            channel_ids (torch.Tensor): Tensor of channel IDs.
            source_ids (torch.Tensor): Tensor of source IDs.
            cont_feats (torch.Tensor): Continuous features (e.g. salience), shape (batch, cont_dim).
            env_ctx (torch.Tensor): Environmental context, shape (batch, context_dim).
            fusion_features (Optional[torch.Tensor]): Fused modality features, shape (batch, fusion_dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (route_logits, value)
        """
        ch_emb = self.channel_embedding(channel_ids)
        src_emb = self.source_embedding(source_ids)
        # If fusion_features is provided, concatenate it; otherwise, ignore.
        if fusion_features is not None:
            x = torch.cat([ch_emb, src_emb, cont_feats, env_ctx, fusion_features], dim=-1)
        else:
            x = torch.cat([ch_emb, src_emb, cont_feats, env_ctx], dim=-1)
        h = self.relu(self.fc_in(x))
        h = self.relu(self.fc_hidden(h))
        route_logits = self.route_head(h)  # shape: (batch, num_routes)
        value = self.value_head(h)         # shape: (batch, 1)
        return route_logits, value

# -----------------------------------------------------------------------------
# Transition and RolloutBuffer for PPO updates
# -----------------------------------------------------------------------------
@dataclass
class Transition:
    """
    Transition: Stores one transition in the rollout buffer for PPO updates.
    """
    obs: Dict[str, Any]
    route: int
    logp: float
    value: float
    reward: float
    next_obs: Dict[str, Any]
    done: bool

class RolloutBuffer:
    """
    RolloutBuffer: Buffer to accumulate transitions for PPO updates.
    """
    def __init__(self, gamma: float, lam: float, capacity: int = 64):
        self.gamma = gamma
        self.lam = lam
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0

    def size(self) -> int:
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def compute_gae(self, final_value: float = 0.0) -> Tuple[List[float], List[float]]:
        advantages = []
        returns = []
        values = np.array([t.value for t in self.transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.bool_)
        next_values = np.concatenate([values[1:], np.array([final_value], dtype=np.float32)], axis=0)
        gae = 0.0
        for i in reversed(range(len(self.transitions))):
            mask = 1.0 - dones[i].astype(np.float32)
            delta = rewards[i] + self.gamma * next_values[i] * mask - values[i]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        for i in range(len(self.transitions)):
            returns.append(values[i] + advantages[i])
        return advantages, returns

# -----------------------------------------------------------------------------
# DAR Module with integrated GraphAttentionFusion support
# -----------------------------------------------------------------------------
class DAR(nn.Module):
    """
    Dynamic Attention Routing (DAR)
    ================================

    This module implements a dynamic multi–route decision mechanism that integrates environmental
    context, external gating signals from the EFM, and (optionally) fused multi–modality embeddings.
    The fused representation (if available) is concatenated with other features and processed by
    an IntegratedEnvContextNet to produce routing logits and a value estimate.

    Args:
        max_channels (int): Maximum channel index for embeddings.
        max_sources (int): Maximum source index for embeddings.
        embed_dim (int): Dimension for channel and source embeddings.
        cont_dim (int): Dimension of continuous features (e.g. salience).
        context_dim (int): Dimension of environmental context.
        hidden_dim (int): Hidden dimension for the context network.
        num_routes (int): Number of routing decisions.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        lam (float): GAE lambda.
        clip_eps (float): PPO clipping epsilon.
        n_epochs (int): Number of PPO update epochs.
        mini_batch_size (int): Mini-batch size for PPO updates.
        capacity (int): Rollout buffer capacity.
        ppo_update_interval (int): Number of transitions before an update.
        use_fusion (bool): Whether to integrate multi–modal fusion.
        fusion_input_dims (List[int]): List of input dimensions for each modality (if fusion is used).
        fusion_dim (int): Target fusion dimension.
        fusion_pooling (str): Pooling strategy for fusion ("mean", "sum", or "learned").
        efm (Optional[Any]): External gating signal provider.
    """
    def __init__(
        self,
        max_channels: int = 20,
        max_sources: int = 20,
        embed_dim: int = 16,
        cont_dim: int = 1,
        context_dim: int = 2,
        hidden_dim: int = 64,
        num_routes: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
        capacity: int = 256,
        ppo_update_interval: int = 32,
        use_fusion: bool = False,
        fusion_input_dims: Optional[List[int]] = None,
        fusion_dim: int = 256,
        fusion_pooling: str = "mean",
        config_manager: Optional[Any] = None,
        efm: Optional[Any] = None
    ):
        super(DAR, self).__init__()
        self.logger = (config_manager.setup_logger("DAR")
                       if config_manager else logging.getLogger("DAR"))
        self.device = torch.device("cpu")
        self.num_routes = num_routes
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.ppo_update_interval = ppo_update_interval
        self.capacity = capacity
        self.efm = efm  # External gating signal provider

        # Fusion-related attributes.
        self.use_fusion = use_fusion
        self.fusion_dim = fusion_dim
        if self.use_fusion:
            if fusion_input_dims is None:
                raise ValueError("fusion_input_dims must be provided when use_fusion is True.")
            self.fusion_module = GraphAttentionFusion(
                input_dims=fusion_input_dims,
                fusion_dim=fusion_dim,
                pooling=fusion_pooling
            ).to(self.device)
            self.logger.info(f"GraphAttentionFusion integrated with fusion_dim={fusion_dim}")
        else:
            self.fusion_module = None

        # Instantiate the context network.
        # If fusion is used, the context network input dimension increases by fusion_dim.
        self.context_net = IntegratedEnvContextNet(
            max_channels, max_sources, embed_dim, cont_dim, context_dim, hidden_dim, num_routes,
            fusion_dim=fusion_dim if self.use_fusion else 0
        ).to(self.device)
        self.optimizer = optim.Adam(self.context_net.parameters(), lr=lr)

        # Initialize a rollout buffer for PPO.
        self.rollout_buffer = RolloutBuffer(gamma=self.gamma, lam=self.lam, capacity=capacity)
        self.logger.info(f"DynamicAttentionRouting initialized with {num_routes} routes.")

    def forward(
        self,
        channel_ids: torch.Tensor,
        source_ids: torch.Tensor,
        salience: torch.Tensor,
        env_ctx: torch.Tensor,
        efm_gating: Optional[torch.Tensor] = None,
        modality_embeddings: Optional[List[torch.Tensor]] = None
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass for routing.

        Args:
            channel_ids (torch.Tensor): Channel IDs.
            source_ids (torch.Tensor): Source IDs.
            salience (torch.Tensor): Continuous feature (e.g. salience) of shape (batch, cont_dim).
            env_ctx (torch.Tensor): Environmental context of shape (batch, context_dim).
            efm_gating (Optional[torch.Tensor]): External gating signal, shape (batch, 1).
            modality_embeddings (Optional[List[torch.Tensor]]): List of modality tensors for fusion.

        Returns:
            Tuple[Categorical, torch.Tensor]: Distribution over routes and value estimate.
        """
        # If fusion is enabled, compute the fused features.
        fusion_features = None
        if self.use_fusion:
            if modality_embeddings is None:
                raise ValueError("modality_embeddings must be provided when use_fusion is True.")
            fusion_features = self.fusion_module(modality_embeddings)
            # fusion_features shape: (batch, fusion_dim)

        # Obtain routing logits and value from the integrated context network.
        route_logits, value = self.context_net(
            channel_ids, source_ids, salience, env_ctx, fusion_features
        )

        if efm_gating is not None:
            # Modulate logits with external gating signal.
            gating = efm_gating.unsqueeze(1)  # (batch, 1)
            route_logits = route_logits * gating

        dist = Categorical(logits=route_logits)
        return dist, value

    def route_data(self, obs: Dict[str, Any], next_obs: Optional[Dict[str, Any]] = None, done: bool = False,
                   modality_embeddings: Optional[List[torch.Tensor]] = None) -> int:
        """
        Process an observation and select a routing decision.
        """
        if next_obs is None:
            next_obs = obs

        channel_id = torch.tensor([obs.get("channel_id", 0)], dtype=torch.long, device=self.device)
        source_id = torch.tensor([obs.get("source_id", 0)], dtype=torch.long, device=self.device)
        sal_val = float(obs.get("salience", 0.0))
        sal_tensor = torch.tensor([[sal_val]], dtype=torch.float32, device=self.device)
        env_ctx_list = obs.get("env_context", [0.0, 0.0])
        env_ctx = torch.tensor([env_ctx_list], dtype=torch.float32, device=self.device)

        efm_gate = None
        if self.efm and hasattr(self.efm, "get_gating_signal"):
            try:
                gate_value = float(self.efm.get_gating_signal())
                efm_gate = torch.tensor([gate_value], dtype=torch.float32, device=self.device)
            except Exception as e:
                self.logger.error(f"Error obtaining gating signal from EFM: {e}", exc_info=True)

        with torch.no_grad():
            dist, value = self.forward(channel_id, source_id, sal_tensor, env_ctx, efm_gate, modality_embeddings)
            route_choice = dist.sample()
        logp = float(dist.log_prob(route_choice).item())
        route_int = int(route_choice.item())
        transition = Transition(
            obs=obs,
            route=route_int,
            logp=logp,
            value=value.item(),
            reward=0.0,
            next_obs=next_obs,
            done=done
        )
        self.rollout_buffer.add_transition(transition)
        if self.rollout_buffer.size() >= self.ppo_update_interval:
            self._ppo_update()
        self.logger.debug(f"Route selected: {route_int} for channel_id {channel_id.item()}")
        return route_int

    def give_reward(self, reward: float):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].reward += reward
            self.logger.debug(f"Reward {reward} assigned to latest transition.")

    def finalize_step(self, next_obs: Dict[str, Any], done: bool):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].next_obs = next_obs
            self.rollout_buffer.transitions[-1].done = done
            self.logger.debug("Finalized the latest transition.")

    def end_of_episode(self, final_value: float = 0.0):
        if self.rollout_buffer.transitions:
            self.rollout_buffer.transitions[-1].done = True
        self._ppo_update(final_value=final_value)

    def _ppo_update(self, final_value: float = 0.0):
        if self.rollout_buffer.is_empty():
            return
        self.logger.info("Starting PPO update for DAR.")
        advantages, returns = self.rollout_buffer.compute_gae(final_value=final_value)

        obs_channels = []
        obs_sources = []
        obs_saliences = []
        obs_env_ctxs = []
        old_logps = []
        old_values = []
        routes = []

        for t in self.rollout_buffer.transitions:
            obs_channels.append(t.obs.get("channel_id", 0))
            obs_sources.append(t.obs.get("source_id", 0))
            obs_saliences.append(t.obs.get("salience", 0.0))
            obs_env_ctxs.append(t.obs.get("env_context", [0.0, 0.0]))
            old_logps.append(t.logp)
            old_values.append(t.value)
            routes.append(t.route)

        channel_ids_t = torch.tensor(obs_channels, dtype=torch.long, device=self.device)
        source_ids_t = torch.tensor(obs_sources, dtype=torch.long, device=self.device)
        salience_t = torch.tensor(obs_saliences, dtype=torch.float32, device=self.device).unsqueeze(-1)
        env_ctx_t = torch.tensor(obs_env_ctxs, dtype=torch.float32, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        routes_t = torch.tensor(routes, dtype=torch.long, device=self.device)

        data_size = self.rollout_buffer.size()
        indices = np.arange(data_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, data_size, self.mini_batch_size):
                batch_idx = indices[start:start+self.mini_batch_size]
                ch_b = channel_ids_t[batch_idx]
                src_b = source_ids_t[batch_idx]
                sal_b = salience_t[batch_idx]
                ctx_b = env_ctx_t[batch_idx]
                old_logp_b = old_logps_t[batch_idx]
                adv_b = advantages_t[batch_idx]
                ret_b = returns_t[batch_idx]
                route_b = routes_t[batch_idx]

                # When fusion is used during PPO update, assume no multi-modal data; use zeros.
                if self.use_fusion:
                    dummy_fusion = torch.zeros((ch_b.shape[0], self.fusion_dim), device=self.device)
                else:
                    dummy_fusion = None

                route_logits, value_b = self.context_net(ch_b, src_b, sal_b, ctx_b, dummy_fusion)
                dist = Categorical(logits=route_logits)
                new_logps = dist.log_prob(route_b)
                ratio = torch.exp(new_logps - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = F.mse_loss(value_b.squeeze(-1), ret_b)
                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.logger.debug(f"Epoch {epoch+1}/{self.n_epochs} completed: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}")
        self.logger.info("PPO update for DAR completed.")
        self.rollout_buffer.clear()

    def get_gating_signal(self) -> float:
        if not self.rollout_buffer.transitions:
            return 1.0
        avg_reward = np.mean([t.reward for t in self.rollout_buffer.transitions])
        gating = 1.0 / (1.0 + np.exp(-avg_reward))
        self.logger.debug(f"Computed gating signal: {gating:.4f}")
        return gating

    async def async_route_data(self, obs: Dict[str, Any], efm_gating: Optional[float] = None,
                               modality_embeddings: Optional[List[torch.Tensor]] = None) -> int:
        return await asyncio.to_thread(self.route_data, obs, None, False, modality_embeddings)

    async def async_update(self, batch: Dict[str, torch.Tensor],
                           gamma: float = 0.99, lam: float = 0.95, ppo_epochs: int = 4) -> Dict[str, float]:
        return await asyncio.to_thread(self._ppo_update, 0.0)

# -----------------------------------------------------------------------------
# (Optional) Main function for testing the integrated DAR module.
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Attention Routing with Integrated Fusion")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy inputs")
    parser.add_argument("--visual_dim", type=int, default=512, help="Dimension of visual embeddings")
    parser.add_argument("--text_dim", type=int, default=768, help="Dimension of text embeddings")
    parser.add_argument("--audio_dim", type=int, default=128, help="Dimension of audio embeddings")
    parser.add_argument("--fusion_dim", type=int, default=256, help="Fusion space dimension")
    parser.add_argument("--use_fusion", action="store_true", help="Integrate multi-modal fusion")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger("DAR_Main")
    logger.info(f"Arguments: {args}")

    # Create dummy inputs for DAR.
    batch_size = args.batch_size
    channel_ids = torch.randint(0, 20, (batch_size,), dtype=torch.long)
    source_ids = torch.randint(0, 20, (batch_size,), dtype=torch.long)
    salience = torch.randn(batch_size, 1)
    env_ctx = torch.randn(batch_size, 2)
    # Dummy external gating signal.
    efm_gate = torch.ones(batch_size, dtype=torch.float32)

    # If fusion is enabled, create dummy modality embeddings.
    modality_embeddings = None
    if args.use_fusion:
        visual_embedding = torch.randn(batch_size, args.visual_dim)
        text_embedding = torch.randn(batch_size, args.text_dim)
        audio_embedding = torch.randn(batch_size, args.audio_dim)
        modality_embeddings = [visual_embedding, text_embedding, audio_embedding]
        # For this example, we assume fusion_input_dims match the modality dims.
        fusion_input_dims = [args.visual_dim, args.text_dim, args.audio_dim]
    else:
        fusion_input_dims = None

    # Initialize DAR.
    dar = DAR(
        max_channels=20,
        max_sources=20,
        embed_dim=16,
        cont_dim=1,
        context_dim=2,
        hidden_dim=64,
        num_routes=5,
        use_fusion=args.use_fusion,
        fusion_input_dims=fusion_input_dims,
        fusion_dim=args.fusion_dim,
    )

    # Run a forward pass.
    dist, value = dar.forward(channel_ids, source_ids, salience, env_ctx, efm_gate, modality_embeddings)
    route = dist.sample()
    logger.info(f"Sampled route: {route.item()}, Value estimate: {value.item()}")

if __name__ == "__main__":
    main()
w
```

### For neural_cognitive_bus.py

Ensure the NCB can handle the tensor shapes from GraphAttentionFusion:

```python
# In the HCDM.py initialization section:
ncb.create_channel("fusion_output", fusion_dim)

# In the sensory_processing_module.py after fusion:
await self.ncb.publish("fusion_output", fused_feature)
```

## Integration Testing

After implementing these changes, I recommend a phased integration testing approach:

1. First, test the GraphAttentionFusion module standalone with dummy inputs
2. Integrate it with just the sensory processing module and verify outputs
3. Connect it to the DAR module and test routing decisions
4. Finally, integrate with the full HCDM system

This approach will help isolate any integration issues that arise during the process.

The new fusion module brings significant advantages over simpler fusion methods, particularly in how it can combine information from multiple modalities and adapt its attention weights based on the data. The module's hierarchical structure and ability to incorporate top-down gating signals will integrate well with your existing cognitive architecture.