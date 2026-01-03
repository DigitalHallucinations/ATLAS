"""RAG capabilities detection and recommendation engine.

Detects hardware capabilities and database features to recommend optimal
RAG configuration settings during setup and runtime.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingTier(str, Enum):
    """Recommended embedding tier based on hardware."""
    
    API_ONLY = "api_only"  # No local GPU, use cloud APIs
    LOCAL_LIGHT = "local_light"  # Low VRAM, lightweight models
    LOCAL_STANDARD = "local_standard"  # Mid-range GPU
    LOCAL_HEAVY = "local_heavy"  # High-end GPU, large models
    HYBRID = "hybrid"  # Mix of local and API


@dataclass
class GPUInfo:
    """Detected GPU information."""
    
    name: str = ""
    vram_mb: int = 0
    cuda_available: bool = False
    cuda_version: str = ""
    mps_available: bool = False  # Apple Metal
    
    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024
    
    @property
    def supports_local_embeddings(self) -> bool:
        """Check if GPU can run local embedding models."""
        # Need at least 500MB for smallest models
        return self.vram_mb >= 500 or self.mps_available


@dataclass
class PgVectorInfo:
    """PostgreSQL pgvector extension information."""
    
    available: bool = False
    version: str = ""
    max_dimensions: int = 2000  # Default limit
    supports_hnsw: bool = False  # Requires pgvector >= 0.5.0
    supports_ivfflat: bool = True
    
    @classmethod
    def from_version(cls, version: str) -> "PgVectorInfo":
        """Create from version string."""
        if not version:
            return cls(available=False)
        
        try:
            parts = version.split(".")
            major = int(parts[0]) if parts else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            
            # HNSW support added in 0.5.0
            supports_hnsw = (major > 0) or (major == 0 and minor >= 5)
            
            return cls(
                available=True,
                version=version,
                supports_hnsw=supports_hnsw,
                supports_ivfflat=True,
            )
        except (ValueError, IndexError):
            return cls(available=True, version=version)


@dataclass
class EmbeddingModelRecommendation:
    """Recommended embedding model configuration."""
    
    provider: str  # "huggingface", "openai", "cohere"
    model: str
    dimensions: int
    vram_required_mb: int
    quality_tier: str  # "good", "better", "best"
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "vram_required_mb": self.vram_required_mb,
            "quality_tier": self.quality_tier,
            "reason": self.reason,
        }


# Model catalog with requirements
EMBEDDING_MODEL_CATALOG: List[EmbeddingModelRecommendation] = [
    # Local HuggingFace models (sorted by VRAM requirement)
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="all-MiniLM-L6-v2",
        dimensions=384,
        vram_required_mb=500,
        quality_tier="good",
        reason="Lightweight, fast, good for resource-constrained systems",
    ),
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="all-MiniLM-L12-v2",
        dimensions=384,
        vram_required_mb=600,
        quality_tier="good",
        reason="Slightly better quality than L6, still lightweight",
    ),
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="all-mpnet-base-v2",
        dimensions=768,
        vram_required_mb=1000,
        quality_tier="better",
        reason="Excellent balance of quality and speed",
    ),
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="e5-base-v2",
        dimensions=768,
        vram_required_mb=1200,
        quality_tier="better",
        reason="Strong performance on retrieval tasks",
    ),
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="e5-large-v2",
        dimensions=1024,
        vram_required_mb=2000,
        quality_tier="best",
        reason="High quality, requires more VRAM",
    ),
    EmbeddingModelRecommendation(
        provider="huggingface",
        model="bge-large-en-v1.5",
        dimensions=1024,
        vram_required_mb=2000,
        quality_tier="best",
        reason="Top-tier local model for English",
    ),
    # API-based models
    EmbeddingModelRecommendation(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        vram_required_mb=0,
        quality_tier="better",
        reason="Fast, cost-effective OpenAI embeddings",
    ),
    EmbeddingModelRecommendation(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=3072,
        vram_required_mb=0,
        quality_tier="best",
        reason="Highest quality OpenAI embeddings",
    ),
    EmbeddingModelRecommendation(
        provider="cohere",
        model="embed-english-v3.0",
        dimensions=1024,
        vram_required_mb=0,
        quality_tier="best",
        reason="Excellent Cohere embeddings with search optimization",
    ),
]


@dataclass
class RAGCapabilities:
    """Detected RAG capabilities and recommendations.
    
    Populated during preflight to inform setup wizard defaults.
    """
    
    # Hardware detection
    gpu_info: GPUInfo = field(default_factory=GPUInfo)
    available_ram_gb: float = 0.0
    available_disk_gb: float = 0.0
    
    # Database detection
    pgvector_info: PgVectorInfo = field(default_factory=PgVectorInfo)
    
    # Computed recommendations
    embedding_tier: EmbeddingTier = EmbeddingTier.API_ONLY
    recommended_models: List[EmbeddingModelRecommendation] = field(default_factory=list)
    recommended_chunk_size: int = 512
    recommended_chunk_overlap: int = 50
    reranking_recommended: bool = False
    max_safe_dimensions: int = 2000
    
    # Feature flags
    can_use_local_embeddings: bool = False
    can_use_reranking: bool = False
    can_use_hnsw_index: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Return serializable representation."""
        return {
            "gpu_info": {
                "name": self.gpu_info.name,
                "vram_mb": self.gpu_info.vram_mb,
                "cuda_available": self.gpu_info.cuda_available,
                "mps_available": self.gpu_info.mps_available,
            },
            "available_ram_gb": self.available_ram_gb,
            "available_disk_gb": self.available_disk_gb,
            "pgvector_info": {
                "available": self.pgvector_info.available,
                "version": self.pgvector_info.version,
                "supports_hnsw": self.pgvector_info.supports_hnsw,
            },
            "embedding_tier": self.embedding_tier.value,
            "recommended_models": [m.to_dict() for m in self.recommended_models],
            "recommended_chunk_size": self.recommended_chunk_size,
            "recommended_chunk_overlap": self.recommended_chunk_overlap,
            "reranking_recommended": self.reranking_recommended,
            "max_safe_dimensions": self.max_safe_dimensions,
            "can_use_local_embeddings": self.can_use_local_embeddings,
            "can_use_reranking": self.can_use_reranking,
            "can_use_hnsw_index": self.can_use_hnsw_index,
        }


class RAGCapabilitiesDetector:
    """Detects system capabilities for RAG configuration."""
    
    def __init__(self, database_dsn: Optional[str] = None) -> None:
        """Initialize detector.
        
        Args:
            database_dsn: Optional database connection string for pgvector check.
        """
        self._database_dsn = database_dsn
    
    def detect(self) -> RAGCapabilities:
        """Run full capability detection.
        
        Returns:
            RAGCapabilities with detected values and recommendations.
        """
        caps = RAGCapabilities()
        
        # Detect GPU
        caps.gpu_info = self._detect_gpu()
        
        # Detect system resources
        caps.available_ram_gb = self._detect_available_ram()
        caps.available_disk_gb = self._detect_available_disk()
        
        # Detect pgvector if database is configured
        if self._database_dsn:
            caps.pgvector_info = self._detect_pgvector()
        
        # Compute recommendations based on detected capabilities
        self._compute_recommendations(caps)
        
        logger.info(
            "RAG capabilities detected: tier=%s, gpu=%s (%.1fGB VRAM), pgvector=%s",
            caps.embedding_tier.value,
            caps.gpu_info.name or "none",
            caps.gpu_info.vram_gb,
            caps.pgvector_info.version or "not detected",
        )
        
        return caps
    
    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU capabilities."""
        info = GPUInfo()
        
        # Try CUDA detection via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                info.cuda_available = True
                info.cuda_version = torch.version.cuda or ""
                
                # Get GPU info
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    info.name = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    info.vram_mb = props.total_memory // (1024 * 1024)
            
            # Check for Apple Metal (MPS)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.mps_available = True
                if not info.name:
                    info.name = "Apple Metal (MPS)"
                    # Estimate available unified memory
                    info.vram_mb = int(self._detect_available_ram() * 1024 * 0.5)
                    
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
        except Exception as e:
            logger.debug("GPU detection failed: %s", e)
        
        # Fallback: try nvidia-smi
        if not info.cuda_available:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    line = result.stdout.strip().split("\n")[0]
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        info.name = parts[0].strip()
                        info.vram_mb = int(parts[1].strip())
                        info.cuda_available = True
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
        
        return info
    
    def _detect_available_ram(self) -> float:
        """Detect available system RAM in GB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except ImportError:
            # Fallback for Linux
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            return kb / (1024 ** 2)
            except Exception:
                pass
        return 0.0
    
    def _detect_available_disk(self) -> float:
        """Detect available disk space in GB."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return free / (1024 ** 3)
        except Exception:
            return 0.0
    
    def _detect_pgvector(self) -> PgVectorInfo:
        """Detect pgvector extension capabilities."""
        if not self._database_dsn:
            return PgVectorInfo()
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self._database_dsn)
            with engine.connect() as conn:
                # Check if pgvector is installed
                result = conn.execute(text(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                ))
                row = result.fetchone()
                if row:
                    version = row[0]
                    return PgVectorInfo.from_version(version)
                
        except Exception as e:
            logger.debug("pgvector detection failed: %s", e)
        
        return PgVectorInfo()
    
    def _compute_recommendations(self, caps: RAGCapabilities) -> None:
        """Compute recommendations based on detected capabilities."""
        gpu = caps.gpu_info
        pgvector = caps.pgvector_info
        
        # Determine embedding tier
        if gpu.vram_mb >= 2000:
            caps.embedding_tier = EmbeddingTier.LOCAL_HEAVY
            caps.can_use_local_embeddings = True
            caps.reranking_recommended = True
        elif gpu.vram_mb >= 1000:
            caps.embedding_tier = EmbeddingTier.LOCAL_STANDARD
            caps.can_use_local_embeddings = True
            caps.reranking_recommended = True
        elif gpu.vram_mb >= 500 or gpu.mps_available:
            caps.embedding_tier = EmbeddingTier.LOCAL_LIGHT
            caps.can_use_local_embeddings = True
            caps.reranking_recommended = False
        else:
            caps.embedding_tier = EmbeddingTier.API_ONLY
            caps.can_use_local_embeddings = False
            caps.reranking_recommended = False  # Reranking needs GPU too
        
        # Can do reranking if we have GPU
        caps.can_use_reranking = gpu.vram_mb >= 500 or gpu.mps_available
        
        # HNSW index support
        caps.can_use_hnsw_index = pgvector.supports_hnsw
        
        # Select recommended models
        caps.recommended_models = self._select_models(caps)
        
        # Compute safe dimension limit
        if pgvector.available:
            caps.max_safe_dimensions = pgvector.max_dimensions
        else:
            caps.max_safe_dimensions = 2000
        
        # Chunk size recommendations based on RAM
        if caps.available_ram_gb >= 16:
            caps.recommended_chunk_size = 512
            caps.recommended_chunk_overlap = 50
        elif caps.available_ram_gb >= 8:
            caps.recommended_chunk_size = 384
            caps.recommended_chunk_overlap = 40
        else:
            caps.recommended_chunk_size = 256
            caps.recommended_chunk_overlap = 30
    
    def _select_models(self, caps: RAGCapabilities) -> List[EmbeddingModelRecommendation]:
        """Select recommended models based on capabilities."""
        recommendations = []
        vram = caps.gpu_info.vram_mb
        
        # Add local models that fit in VRAM
        if caps.can_use_local_embeddings:
            for model in EMBEDDING_MODEL_CATALOG:
                if model.provider == "huggingface":
                    if model.vram_required_mb <= vram:
                        # Check dimension limit
                        if model.dimensions <= caps.max_safe_dimensions:
                            recommendations.append(model)
        
        # Always add API options
        for model in EMBEDDING_MODEL_CATALOG:
            if model.provider in ("openai", "cohere"):
                if model.dimensions <= caps.max_safe_dimensions:
                    recommendations.append(model)
        
        # Sort: local models first (by quality), then API
        def sort_key(m: EmbeddingModelRecommendation) -> Tuple[int, int, int]:
            provider_order = 0 if m.provider == "huggingface" else 1
            quality_order = {"best": 0, "better": 1, "good": 2}.get(m.quality_tier, 3)
            return (provider_order, quality_order, m.vram_required_mb)
        
        recommendations.sort(key=sort_key)
        
        # Keep top 5
        return recommendations[:5]


def detect_rag_capabilities(database_dsn: Optional[str] = None) -> RAGCapabilities:
    """Convenience function to detect RAG capabilities.
    
    Args:
        database_dsn: Optional database connection string.
        
    Returns:
        RAGCapabilities with detected values.
    """
    detector = RAGCapabilitiesDetector(database_dsn)
    return detector.detect()


__all__ = [
    "EmbeddingTier",
    "GPUInfo",
    "PgVectorInfo",
    "EmbeddingModelRecommendation",
    "RAGCapabilities",
    "RAGCapabilitiesDetector",
    "detect_rag_capabilities",
    "EMBEDDING_MODEL_CATALOG",
]
